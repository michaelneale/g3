use g3_providers::{LLMProvider, CompletionRequest, CompletionResponse, CompletionStream, CompletionChunk, Usage, Message, MessageRole};
use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, error};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

pub struct OpenAIProvider {
    client: Client,
    api_key: String,
    model: String,
    base_url: String,
}

#[derive(Debug, Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Debug, Serialize)]
struct OpenAIMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
    usage: OpenAIUsage,
    model: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: OpenAIResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIResponseMessage {
    content: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl OpenAIProvider {
    pub fn new(api_key: String, model: String, base_url: Option<String>) -> Result<Self> {
        let client = Client::new();
        let base_url = base_url.unwrap_or_else(|| "https://api.openai.com/v1".to_string());
        
        Ok(Self {
            client,
            api_key,
            model,
            base_url,
        })
    }
    
    fn convert_message(&self, message: &Message) -> OpenAIMessage {
        OpenAIMessage {
            role: match message.role {
                MessageRole::System => "system".to_string(),
                MessageRole::User => "user".to_string(),
                MessageRole::Assistant => "assistant".to_string(),
            },
            content: message.content.clone(),
        }
    }
}

#[async_trait::async_trait]
impl LLMProvider for OpenAIProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        debug!("Making OpenAI completion request");
        
        let openai_request = OpenAIRequest {
            model: self.model.clone(),
            messages: request.messages.iter().map(|m| self.convert_message(m)).collect(),
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            stream: Some(false),
        };
        
        let response = self
            .client
            .post(&format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&openai_request)
            .send()
            .await?;
        
        if !response.status().is_success() {
            let error_text = response.text().await?;
            error!("OpenAI API error: {}", error_text);
            anyhow::bail!("OpenAI API error: {}", error_text);
        }
        
        let openai_response: OpenAIResponse = response.json().await?;
        
        let content = openai_response
            .choices
            .first()
            .map(|choice| choice.message.content.clone())
            .unwrap_or_default();
        
        Ok(CompletionResponse {
            content,
            usage: Usage {
                prompt_tokens: openai_response.usage.prompt_tokens,
                completion_tokens: openai_response.usage.completion_tokens,
                total_tokens: openai_response.usage.total_tokens,
            },
            model: openai_response.model,
        })
    }
    
    async fn stream(&self, request: CompletionRequest) -> Result<CompletionStream> {
        debug!("Making OpenAI streaming request");
        
        let (tx, rx) = mpsc::channel(100);
        
        // For now, just send the complete response as a single chunk
        // In a real implementation, we'd handle Server-Sent Events
        let completion = self.complete(request).await?;
        
        let chunk = CompletionChunk {
            content: completion.content,
            finished: true,
            tool_calls: None,
        };
        
        tx.send(Ok(chunk)).await.map_err(|_| anyhow::anyhow!("Failed to send chunk"))?;
        
        Ok(ReceiverStream::new(rx))
    }
    
    fn name(&self) -> &str {
        "openai"
    }
    
    fn model(&self) -> &str {
        &self.model
    }
}
