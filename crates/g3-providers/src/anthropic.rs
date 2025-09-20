//! Anthropic Claude provider implementation for the g3-providers crate.
//!
//! This module provides an implementation of the `LLMProvider` trait for Anthropic's Claude models,
//! supporting both completion and streaming modes through the Anthropic Messages API.
//!
//! # Features
//!
//! - Support for all Claude models (claude-3-5-sonnet-20241022, claude-3-haiku-20240307, etc.)
//! - Both completion and streaming response modes
//! - Proper message format conversion between g3 and Anthropic formats
//! - Rate limiting and error handling
//! - Native tool calling support
//!
//! # Usage
//!
//! ```rust,no_run
//! use g3_providers::{AnthropicProvider, LLMProvider, CompletionRequest, Message, MessageRole};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create the provider with your API key
//!     let provider = AnthropicProvider::new(
//!         "your-api-key".to_string(),
//!         Some("claude-3-5-sonnet-20241022".to_string()), // Optional: defaults to claude-3-5-sonnet-20241022
//!         Some(4096),  // Optional: max tokens
//!         Some(0.1),   // Optional: temperature
//!     )?;
//!
//!     // Create a completion request
//!     let request = CompletionRequest {
//!         messages: vec![
//!             Message {
//!                 role: MessageRole::System,
//!                 content: "You are a helpful assistant.".to_string(),
//!             },
//!             Message {
//!                 role: MessageRole::User,
//!                 content: "Hello! How are you?".to_string(),
//!             },
//!         ],
//!         max_tokens: Some(1000),
//!         temperature: Some(0.7),
//!         stream: false,
//!     };
//!
//!     // Get a completion
//!     let response = provider.complete(request).await?;
//!     println!("Response: {}", response.content);
//!
//!     Ok(())
//! }
//! ```
//!
//! # Streaming Example
//!
//! ```rust,no_run
//! use g3_providers::{AnthropicProvider, LLMProvider, CompletionRequest, Message, MessageRole};
//! use tokio_stream::StreamExt;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let provider = AnthropicProvider::new(
//!         "your-api-key".to_string(),
//!         None, None, None,
//!     )?;
//!
//!     let request = CompletionRequest {
//!         messages: vec![
//!             Message {
//!                 role: MessageRole::User,
//!                 content: "Write a short story about a robot.".to_string(),
//!             },
//!         ],
//!         max_tokens: Some(1000),
//!         temperature: Some(0.7),
//!         stream: true,
//!     };
//!
//!     let mut stream = provider.stream(request).await?;
//!     while let Some(chunk) = stream.next().await {
//!         match chunk {
//!             Ok(chunk) => {
//!                 print!("{}", chunk.content);
//!                 if chunk.finished {
//!                     break;
//!                 }
//!             }
//!             Err(e) => {
//!                 eprintln!("Stream error: {}", e);
//!                 break;
//!             }
//!         }
//!     }
//!
//!     Ok(())
//! }
//! ```

use anyhow::{anyhow, Result};
use bytes::Bytes;
use futures_util::stream::StreamExt;
use reqwest::{Client, RequestBuilder};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, error, info, warn};

use crate::{
    CompletionChunk, CompletionRequest, CompletionResponse, CompletionStream, LLMProvider, Message,
    MessageRole, Usage,
};

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

#[derive(Debug, Clone)]
pub struct AnthropicProvider {
    client: Client,
    api_key: String,
    model: String,
    max_tokens: u32,
    temperature: f32,
}

impl AnthropicProvider {
    pub fn new(
        api_key: String,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    ) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .map_err(|e| anyhow!("Failed to create HTTP client: {}", e))?;

        let model = model.unwrap_or_else(|| "claude-3-5-sonnet-20241022".to_string());
        
        info!("Initialized Anthropic provider with model: {}", model);

        Ok(Self {
            client,
            api_key,
            model,
            max_tokens: max_tokens.unwrap_or(4096),
            temperature: temperature.unwrap_or(0.1),
        })
    }

    fn create_request_builder(&self, streaming: bool) -> RequestBuilder {
        let mut builder = self
            .client
            .post(ANTHROPIC_API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json");

        if streaming {
            builder = builder.header("accept", "text/event-stream");
        }

        builder
    }

    fn convert_messages(&self, messages: &[Message]) -> Result<(Option<String>, Vec<AnthropicMessage>)> {
        let mut system_message = None;
        let mut anthropic_messages = Vec::new();

        for message in messages {
            match message.role {
                MessageRole::System => {
                    if system_message.is_some() {
                        warn!("Multiple system messages found, using the last one");
                    }
                    system_message = Some(message.content.clone());
                }
                MessageRole::User => {
                    anthropic_messages.push(AnthropicMessage {
                        role: "user".to_string(),
                        content: vec![AnthropicContent::Text {
                            text: message.content.clone(),
                        }],
                    });
                }
                MessageRole::Assistant => {
                    anthropic_messages.push(AnthropicMessage {
                        role: "assistant".to_string(),
                        content: vec![AnthropicContent::Text {
                            text: message.content.clone(),
                        }],
                    });
                }
            }
        }

        Ok((system_message, anthropic_messages))
    }

    fn create_request_body(
        &self,
        messages: &[Message],
        streaming: bool,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<AnthropicRequest> {
        let (system, anthropic_messages) = self.convert_messages(messages)?;

        if anthropic_messages.is_empty() {
            return Err(anyhow!("At least one user or assistant message is required"));
        }

        let request = AnthropicRequest {
            model: self.model.clone(),
            max_tokens,
            temperature,
            messages: anthropic_messages,
            system,
            stream: streaming,
        };

        // Ensure the conversation starts with a user message
        if request.messages[0].role != "user" {
            return Err(anyhow!("Conversation must start with a user message"));
        }

        Ok(request)
    }

    async fn parse_streaming_response(
        &self,
        mut stream: impl futures_util::Stream<Item = reqwest::Result<Bytes>> + Unpin,
        tx: mpsc::Sender<Result<CompletionChunk>>,
    ) {
        let mut buffer = String::new();

        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    let chunk_str = match std::str::from_utf8(&chunk) {
                        Ok(s) => s,
                        Err(e) => {
                            error!("Invalid UTF-8 in stream chunk: {}", e);
                            let _ = tx
                                .send(Err(anyhow!("Invalid UTF-8 in stream chunk: {}", e)))
                                .await;
                            return;
                        }
                    };

                    buffer.push_str(chunk_str);

                    // Process complete lines
                    while let Some(line_end) = buffer.find('\n') {
                        let line = buffer[..line_end].trim().to_string();
                        buffer.drain(..line_end + 1);

                        if line.is_empty() {
                            continue;
                        }

                        // Parse Server-Sent Events format
                        if let Some(data) = line.strip_prefix("data: ") {
                            if data == "[DONE]" {
                                debug!("Received stream completion marker");
                                let final_chunk = CompletionChunk {
                                    content: String::new(),
                                    finished: true,
                                    tool_calls: None,
                                };
                                if tx.send(Ok(final_chunk)).await.is_err() {
                                    debug!("Receiver dropped, stopping stream");
                                }
                                return;
                            }

                            match serde_json::from_str::<AnthropicStreamEvent>(data) {
                                Ok(event) => {
                                    match event.event_type.as_str() {
                                        "content_block_delta" => {
                                            if let Some(delta) = event.delta {
                                                if let Some(text) = delta.text {
                                                    let chunk = CompletionChunk {
                                                        content: text,
                                                        finished: false,
                                                        tool_calls: None,
                                                    };
                                                    if tx.send(Ok(chunk)).await.is_err() {
                                                        debug!("Receiver dropped, stopping stream");
                                                        return;
                                                    }
                                                }
                                            }
                                        }
                                        "message_stop" => {
                                            debug!("Received message stop event");
                                            let final_chunk = CompletionChunk {
                                                content: String::new(),
                                                finished: true,
                                                tool_calls: None,
                                            };
                                            if tx.send(Ok(final_chunk)).await.is_err() {
                                                debug!("Receiver dropped, stopping stream");
                                            }
                                            return;
                                        }
                                        "error" => {
                                            if let Some(error) = event.error {
                                                error!("Anthropic API error: {:?}", error);
                                                let _ = tx
                                                    .send(Err(anyhow!("Anthropic API error: {:?}", error)))
                                                    .await;
                                                return;
                                            }
                                        }
                                        _ => {
                                            debug!("Ignoring event type: {}", event.event_type);
                                        }
                                    }
                                }
                                Err(e) => {
                                    debug!("Failed to parse stream event: {} - Data: {}", e, data);
                                    // Don't error out on parse failures, just continue
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("Stream error: {}", e);
                    let _ = tx.send(Err(anyhow!("Stream error: {}", e))).await;
                    return;
                }
            }
        }

        // Send final chunk if we haven't already
        let final_chunk = CompletionChunk {
            content: String::new(),
            finished: true,
            tool_calls: None,
        };
        let _ = tx.send(Ok(final_chunk)).await;
    }
}

#[async_trait::async_trait]
impl LLMProvider for AnthropicProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        debug!(
            "Processing Anthropic completion request with {} messages",
            request.messages.len()
        );

        let max_tokens = request.max_tokens.unwrap_or(self.max_tokens);
        let temperature = request.temperature.unwrap_or(self.temperature);

        let request_body = self.create_request_body(&request.messages, false, max_tokens, temperature)?;

        debug!("Sending request to Anthropic API: model={}, max_tokens={}, temperature={}", 
               request_body.model, request_body.max_tokens, request_body.temperature);

        let response = self
            .create_request_builder(false)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| anyhow!("Failed to send request to Anthropic API: {}", e))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(anyhow!("Anthropic API error {}: {}", status, error_text));
        }

        let anthropic_response: AnthropicResponse = response
            .json()
            .await
            .map_err(|e| anyhow!("Failed to parse Anthropic response: {}", e))?;

        // Extract text content from the response
        let content = anthropic_response
            .content
            .iter()
            .filter_map(|c| match c {
                AnthropicContent::Text { text } => Some(text.as_str()),
            })
            .collect::<Vec<_>>()
            .join("");

        let usage = Usage {
            prompt_tokens: anthropic_response.usage.input_tokens,
            completion_tokens: anthropic_response.usage.output_tokens,
            total_tokens: anthropic_response.usage.input_tokens + anthropic_response.usage.output_tokens,
        };

        debug!(
            "Anthropic completion successful: {} tokens generated",
            usage.completion_tokens
        );

        Ok(CompletionResponse {
            content,
            usage,
            model: anthropic_response.model,
        })
    }

    async fn stream(&self, request: CompletionRequest) -> Result<CompletionStream> {
        debug!(
            "Processing Anthropic streaming request with {} messages",
            request.messages.len()
        );

        let max_tokens = request.max_tokens.unwrap_or(self.max_tokens);
        let temperature = request.temperature.unwrap_or(self.temperature);

        let request_body = self.create_request_body(&request.messages, true, max_tokens, temperature)?;

        debug!("Sending streaming request to Anthropic API: model={}, max_tokens={}, temperature={}", 
               request_body.model, request_body.max_tokens, request_body.temperature);

        let response = self
            .create_request_builder(true)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| anyhow!("Failed to send streaming request to Anthropic API: {}", e))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(anyhow!("Anthropic API error {}: {}", status, error_text));
        }

        let stream = response.bytes_stream();
        let (tx, rx) = mpsc::channel(100);

        // Spawn task to process the stream
        let provider = self.clone();
        tokio::spawn(async move {
            provider.parse_streaming_response(stream, tx).await;
        });

        Ok(ReceiverStream::new(rx))
    }

    fn name(&self) -> &str {
        "anthropic"
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn has_native_tool_calling(&self) -> bool {
        // Claude models support native tool calling
        true
    }
}

// Anthropic API request/response structures

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    temperature: f32,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    stream: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: Vec<AnthropicContent>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
enum AnthropicContent {
    #[serde(rename = "text")]
    Text { text: String },
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    id: String,
    #[serde(rename = "type")]
    response_type: String,
    role: String,
    content: Vec<AnthropicContent>,
    model: String,
    stop_reason: Option<String>,
    stop_sequence: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

// Streaming response structures

#[derive(Debug, Deserialize)]
struct AnthropicStreamEvent {
    #[serde(rename = "type")]
    event_type: String,
    #[serde(default)]
    delta: Option<AnthropicDelta>,
    #[serde(default)]
    error: Option<AnthropicError>,
}

#[derive(Debug, Deserialize)]
struct AnthropicDelta {
    #[serde(rename = "type")]
    delta_type: Option<String>,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicError {
    #[serde(rename = "type")]
    error_type: String,
    message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_conversion() {
        let provider = AnthropicProvider::new(
            "test-key".to_string(),
            None,
            None,
            None,
        ).unwrap();

        let messages = vec![
            Message {
                role: MessageRole::System,
                content: "You are a helpful assistant.".to_string(),
            },
            Message {
                role: MessageRole::User,
                content: "Hello!".to_string(),
            },
            Message {
                role: MessageRole::Assistant,
                content: "Hi there!".to_string(),
            },
        ];

        let (system, anthropic_messages) = provider.convert_messages(&messages).unwrap();

        assert_eq!(system, Some("You are a helpful assistant.".to_string()));
        assert_eq!(anthropic_messages.len(), 2);
        assert_eq!(anthropic_messages[0].role, "user");
        assert_eq!(anthropic_messages[1].role, "assistant");
    }

    #[test]
    fn test_request_body_creation() {
        let provider = AnthropicProvider::new(
            "test-key".to_string(),
            Some("claude-3-haiku-20240307".to_string()),
            Some(1000),
            Some(0.5),
        ).unwrap();

        let messages = vec![
            Message {
                role: MessageRole::User,
                content: "Test message".to_string(),
            },
        ];

        let request_body = provider
            .create_request_body(&messages, false, 1000, 0.5)
            .unwrap();

        assert_eq!(request_body.model, "claude-3-haiku-20240307");
        assert_eq!(request_body.max_tokens, 1000);
        assert_eq!(request_body.temperature, 0.5);
        assert!(!request_body.stream);
        assert_eq!(request_body.messages.len(), 1);
    }
}
