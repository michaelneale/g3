use g3_providers::{LLMProvider, CompletionRequest, CompletionResponse, CompletionStream, CompletionChunk, Usage, Message, MessageRole, ToolCall};
use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{debug, error, info};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use futures_util::stream::Stream;
use std::pin::Pin;

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
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
}

#[derive(Debug, Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: Value,
}

#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: String,
    content: AnthropicMessageContent,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum AnthropicMessageContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicResponseContent>,
    usage: AnthropicUsage,
    model: String,
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum AnthropicResponseContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
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
    #[serde(flatten)]
    data: Value,
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamDelta {
    #[serde(rename = "type")]
    delta_type: String,
    text: Option<String>,
    #[serde(flatten)]
    other: Value,
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
            content: AnthropicMessageContent::Text(message.content.clone()),
        }
    }

    fn create_tools() -> Vec<AnthropicTool> {
        vec![
            AnthropicTool {
                name: "shell".to_string(),
                description: "Execute a shell command and return the output".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute"
                        }
                    },
                    "required": ["command"]
                }),
            },
            AnthropicTool {
                name: "final_output".to_string(),
                description: "Provide a final summary or output for the task".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "A summary of what was accomplished"
                        }
                    },
                    "required": ["summary"]
                }),
            },
        ]
    }

    fn extract_content_and_tools(&self, response: &AnthropicResponse) -> (String, Vec<(String, String, Value)>) {
        let mut text_content = String::new();
        let mut tool_calls = Vec::new();

        for content in &response.content {
            match content {
                AnthropicResponseContent::Text { text } => {
                    if !text_content.is_empty() {
                        text_content.push('\n');
                    }
                    text_content.push_str(text);
                }
                AnthropicResponseContent::ToolUse { id, name, input } => {
                    tool_calls.push((id.clone(), name.clone(), input.clone()));
                }
            }
        }

        (text_content, tool_calls)
    }

    async fn execute_tool(&self, tool_name: &str, input: &Value) -> Result<String> {
        match tool_name {
            "shell" => {
                if let Some(command) = input.get("command").and_then(|v| v.as_str()) {
                    info!("Executing shell command via Anthropic tool: {}", command);
                    
                    // Import the CodeExecutor from g3-execution
                    use g3_execution::CodeExecutor;
                    
                    let executor = CodeExecutor::new();
                    match executor.execute_code("bash", command).await {
                        Ok(result) => {
                            if result.success {
                                Ok(if result.stdout.is_empty() {
                                    "✅ Command executed successfully".to_string()
                                } else {
                                    result.stdout
                                })
                            } else {
                                Ok(format!("❌ Command failed: {}", result.stderr))
                            }
                        }
                        Err(e) => {
                            error!("Shell execution error: {}", e);
                            Ok(format!("❌ Execution error: {}", e))
                        }
                    }
                } else {
                    Ok("❌ Missing command argument".to_string())
                }
            }
            "final_output" => {
                if let Some(summary) = input.get("summary").and_then(|v| v.as_str()) {
                    Ok(format!("📋 Final Output: {}", summary))
                } else {
                    Ok("📋 Task completed".to_string())
                }
            }
            _ => {
                error!("Unknown tool: {}", tool_name);
                Ok(format!("❓ Unknown tool: {}", tool_name))
            }
        }
    }

    async fn complete_with_tools(&self, request: CompletionRequest) -> Result<CompletionResponse> {
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
            tools: Some(Self::create_tools()),
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
        debug!("Anthropic response: {:?}", anthropic_response);
        
        let (text_content, tool_calls) = self.extract_content_and_tools(&anthropic_response);
        
        // For the completion API, we'll execute tools and return the combined result
        let final_content = if !tool_calls.is_empty() {
            info!("Anthropic response contains {} tool calls", tool_calls.len());
            
            let mut content_with_tools = text_content.clone();
            for (_id, name, input) in tool_calls {
                // Execute the tool call
                let tool_result = match self.execute_tool(&name, &input).await {
                    Ok(result) => result,
                    Err(e) => format!("Error executing tool {}: {}", name, e),
                };
                
                // Append tool execution info to content
                content_with_tools.push_str(&format!(
                    "\n\nTool executed: {} -> {}\n",
                    name, tool_result
                ));
            }
            content_with_tools
        } else {
            text_content
        };
        
        Ok(CompletionResponse {
            content: final_content,
            usage: Usage {
                prompt_tokens: anthropic_response.usage.input_tokens,
                completion_tokens: anthropic_response.usage.output_tokens,
                total_tokens: anthropic_response.usage.input_tokens + anthropic_response.usage.output_tokens,
            },
            model: anthropic_response.model,
        })
    }
}

#[async_trait::async_trait]
impl LLMProvider for AnthropicProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        debug!("Making Anthropic completion request with tools");
        
        // This is a simplified implementation - for full tool support,
        // we should use the streaming method with proper tool handling
        self.complete_with_tools(request).await
    }
    
    async fn stream(&self, request: CompletionRequest) -> Result<CompletionStream> {
        debug!("Making Anthropic streaming request with tools");
        
        let (tx, rx) = mpsc::channel(100);
        
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
            tools: Some(Self::create_tools()),
        };
        
        // Add stream parameter
        let mut request_json = serde_json::to_value(&anthropic_request)?;
        request_json["stream"] = serde_json::Value::Bool(true);
        
        let client = self.client.clone();
        let api_key = self.api_key.clone();
        
        tokio::spawn(async move {
            debug!("Sending Anthropic streaming request with tools: {:?}", request_json);
            let response = client
                .post("https://api.anthropic.com/v1/messages")
                .header("x-api-key", &api_key)
                .header("Content-Type", "application/json")
                .header("anthropic-version", "2023-06-01")
                .json(&request_json)
                .send()
                .await;
                
            let response = match response {
                Ok(resp) => {
                    if !resp.status().is_success() {
                        let error_text = resp.text().await.unwrap_or_default();
                        let _ = tx.send(Err(anyhow::anyhow!("Anthropic API error: {}", error_text))).await;
                        return;
                    }
                    resp
                }
                Err(e) => {
                    let _ = tx.send(Err(e.into())).await;
                    return;
                }
            };
            
            // Handle Server-Sent Events
            let mut stream = response.bytes_stream();
            let mut buffer = String::new();
            let mut pending_tool_calls = Vec::new();
            
            while let Some(chunk_result) = stream.next().await {
                let chunk = match chunk_result {
                    Ok(bytes) => bytes,
                    Err(e) => {
                        let _ = tx.send(Err(e.into())).await;
                        break;
                    }
                };
                
                let chunk_str = match std::str::from_utf8(&chunk) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                
                buffer.push_str(chunk_str);
                
                // Process complete lines
                while let Some(line_end) = buffer.find('\n') {
                    let line = buffer[..line_end].trim().to_string();
                    buffer.drain(..line_end + 1);
                    
                    if line.is_empty() {
                        continue;
                    }
                    
                    // Parse SSE format: "data: {...}"
                    if let Some(data) = line.strip_prefix("data: ") {
                        debug!("Raw SSE data: {}", data);
                        if data == "[DONE]" {
                            // Send any pending tool calls first
                            if !pending_tool_calls.is_empty() {
                                let tool_chunk = CompletionChunk {
                                    content: String::new(),
                                    finished: false,
                                    tool_calls: Some(pending_tool_calls.clone()),
                                };
                                let _ = tx.send(Ok(tool_chunk)).await;
                                pending_tool_calls.clear();
                            }
                            
                            // Send final chunk
                            let final_chunk = CompletionChunk {
                                content: String::new(),
                                finished: true,
                                tool_calls: None,
                            };
                            let _ = tx.send(Ok(final_chunk)).await;
                            break;
                        }
                        
                        // Parse the JSON event
                        match serde_json::from_str::<AnthropicStreamEvent>(data) {
                            Ok(event) => {
                                debug!("Received Anthropic event: type={}, data={:?}", event.event_type, event.data);
                                match event.event_type.as_str() {
                                    "content_block_start" => {
                                        // Check if this is a tool use block
                                        if let Some(content_block) = event.data.get("content_block") {
                                            if let Some(block_type) = content_block.get("type").and_then(|t| t.as_str()) {
                                                if block_type == "tool_use" {
                                                    // Extract tool call information immediately
                                                    if let (Some(id), Some(name), Some(input)) = (
                                                        content_block.get("id").and_then(|v| v.as_str()),
                                                        content_block.get("name").and_then(|v| v.as_str()),
                                                        content_block.get("input")
                                                    ) {
                                                        let tool_call = ToolCall {
                                                            id: id.to_string(),
                                                            tool: name.to_string(),
                                                            args: input.clone(),
                                                        };
                                                        debug!("Added tool call from content_block_start: {:?}", tool_call);
                                                        pending_tool_calls.push(tool_call);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    "content_block_delta" => {
                                        // Extract text from delta
                                        if let Some(delta) = event.data.get("delta") {
                                            if let Some(text) = delta.get("text").and_then(|t| t.as_str()) {
                                                let chunk = CompletionChunk {
                                                    content: text.to_string(),
                                                    finished: false,
                                                    tool_calls: None,
                                                };
                                                if tx.send(Ok(chunk)).await.is_err() {
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                    "content_block_stop" => {
                                        // Check if we have a complete tool use block
                                        if let Some(content_block) = event.data.get("content_block") {
                                            if let Some(block_type) = content_block.get("type").and_then(|t| t.as_str()) {
                                                if block_type == "tool_use" {
                                                    // Extract tool call information
                                                    if let (Some(id), Some(name), Some(input)) = (
                                                        content_block.get("id").and_then(|v| v.as_str()),
                                                        content_block.get("name").and_then(|v| v.as_str()),
                                                        content_block.get("input")
                                                    ) {
                                                        let tool_call = ToolCall {
                                                            id: id.to_string(),
                                                            tool: name.to_string(),
                                                            args: input.clone(),
                                                        };
                                                        pending_tool_calls.push(tool_call);
                                                    }
                                                }
                                            }
                                        }
                                        debug!("Content block finished");
                                    }
                                    "message_stop" => {
                                        // Send any pending tool calls first
                                        if !pending_tool_calls.is_empty() {
                                            let tool_chunk = CompletionChunk {
                                                content: String::new(),
                                                finished: false,
                                                tool_calls: Some(pending_tool_calls.clone()),
                                            };
                                            let _ = tx.send(Ok(tool_chunk)).await;
                                        }
                                        
                                        // Message finished
                                        let final_chunk = CompletionChunk {
                                            content: String::new(),
                                            finished: true,
                                            tool_calls: None,
                                        };
                                        let _ = tx.send(Ok(final_chunk)).await;
                                        break;
                                    }
                                    _ => {
                                        debug!("Unhandled event type: {}", event.event_type);
                                    }
                                }
                            }
                            Err(e) => {
                                debug!("Failed to parse streaming event: {} - Data: {}", e, data);
                            }
                        }
                    }
                }
            }
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
        true
    }
}
