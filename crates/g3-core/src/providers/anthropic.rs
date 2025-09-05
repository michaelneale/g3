use g3_providers::{LLMProvider, CompletionRequest, CompletionResponse, CompletionStream, CompletionChunk, Usage, Message, MessageRole};
use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, error};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

pub struct AnthropicProvider {
    client: Client,
    api_key: String,
    model: String,
}

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
    usage: AnthropicUsage,
    model: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicContent {
    #[serde(rename = "type")]
    content_type: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

impl AnthropicProvider {
    pub fn new(api_key: String, model: String) -> Result<Self> {
        let client = Client::new();
        
        Ok(Self {
            client,
            api_key,
            model,
        })
    }
    
    fn convert_message(&self, message: &Message) -> AnthropicMessage {
        AnthropicMessage {
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
impl LLMProvider for AnthropicProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        debug!("Making Anthropic completion request");
        
        // Separate system messages from other messages
        let mut system_content: Option<String> = None;
        let mut non_system_messages = Vec::new();
        
        for message in &request.messages {
            match message.role {
                MessageRole::System => {
                    // Combine multiple system messages if present
                    if let Some(existing) = &system_content {
                        system_content = Some(format!("{}\n\n{}", existing, message.content));
                    } else {
                        system_content = Some(message.content.clone());
                    }
                }
                _ => {
                    non_system_messages.push(self.convert_message(message));
                }
            }
        }
        
        let anthropic_request = AnthropicRequest {
            model: self.model.clone(),
            system: system_content,
            messages: non_system_messages,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
        };
        
        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .json(&anthropic_request)
            .send()
            .await?;
        
        if !response.status().is_success() {
            let error_text = response.text().await?;
            error!("Anthropic API error: {}", error_text);
            anyhow::bail!("Anthropic API error: {}", error_text);
        }
        
        let anthropic_response: AnthropicResponse = response.json().await?;
        
        let content = anthropic_response
            .content
            .first()
            .map(|content| content.text.clone())
            .unwrap_or_default();
        
        Ok(CompletionResponse {
            content,
            usage: Usage {
                prompt_tokens: anthropic_response.usage.input_tokens,
                completion_tokens: anthropic_response.usage.output_tokens,
                total_tokens: anthropic_response.usage.input_tokens + anthropic_response.usage.output_tokens,
            },
            model: anthropic_response.model,
        })
    }
    
    async fn stream(&self, request: CompletionRequest) -> Result<CompletionStream> {
        debug!("Making Anthropic streaming request");
        
        let (tx, rx) = mpsc::channel(100);
        
        // For now, just send the complete response as a single chunk
        // In a real implementation, we'd handle Server-Sent Events
        let completion = self.complete(request).await?;
        
        let chunk = CompletionChunk {
            content: completion.content,
            finished: true,
        };
        
        tx.send(Ok(chunk)).await.map_err(|_| anyhow::anyhow!("Failed to send chunk"))?;
        
        Ok(ReceiverStream::new(rx))
    }
    
    fn name(&self) -> &str {
        "anthropic"
    }
    
    fn model(&self) -> &str {
        &self.model
    }
}
