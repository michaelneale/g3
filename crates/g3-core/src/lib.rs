use anyhow::Result;
use g3_config::Config;
use g3_execution::CodeExecutor;
use g3_providers::{CompletionRequest, Message, MessageRole, ProviderRegistry};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub tool: String,
    pub args: serde_json::Value,
}

#[derive(Debug, Clone)]
pub enum StreamState {
    Generating,
    ToolDetected(ToolCall),
    Executing,
    Resuming,
}

#[derive(Debug)]
pub struct StreamingToolParser {
    buffer: String,
    brace_count: i32,
    in_tool_call: bool,
    tool_start_pos: Option<usize>,
}

impl StreamingToolParser {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            brace_count: 0,
            in_tool_call: false,
            tool_start_pos: None,
        }
    }

    pub fn add_chunk(&mut self, chunk: &str) -> Option<(ToolCall, usize)> {
        self.buffer.push_str(chunk);
        //info!("Parser buffer now: {:?}", self.buffer);
        self.detect_tool_call()
    }

    fn detect_tool_call(&mut self) -> Option<(ToolCall, usize)> {
        //info!("Detecting tool call in buffer: {:?}", self.buffer);

        // Look for the start of a tool call pattern: {"tool":
        if !self.in_tool_call {
            // Look for JSON tool call pattern - check both raw JSON and inside code blocks
            if let Some(pos) = self.buffer.rfind(r#"{"tool":"#) {
                //info!("Found tool call pattern at position: {}", pos);

                // Check if this is inside a code block
                let before_pos = &self.buffer[..pos];
                let code_block_count = before_pos.matches("```").count();

                //info!("Code block count before position {}: {}", pos, code_block_count);

                // Accept tool calls both inside and outside code blocks
                // The LLM might use either format despite our instructions
                //info!("Starting tool call parsing (code block status: {})", code_block_count % 2 == 1);
                self.in_tool_call = true;
                self.tool_start_pos = Some(pos);
                self.brace_count = 0; // Start counting from 0, we'll count the opening brace in parsing

                // Continue parsing from after the opening brace
                return self.parse_from_start_pos(pos);
            }
        } else {
            //info!("Already in tool call, continuing parsing");
            // We're already in a tool call, continue parsing
            let start_pos = self.tool_start_pos.unwrap();
            return self.parse_from_start_pos(start_pos);
        }

        None
    }

    fn parse_from_start_pos(&mut self, start_pos: usize) -> Option<(ToolCall, usize)> {
        let remaining = self.buffer[start_pos..].to_string();
        self.parse_from_position(&remaining, start_pos)
    }

    fn parse_from_position(&mut self, text: &str, start_pos: usize) -> Option<(ToolCall, usize)> {
        let mut current_brace_count = 0; // Always start fresh for each parsing attempt

        //info!("Parsing from position {} with text: {:?}", start_pos, text);
        //info!("Starting brace count: {}", current_brace_count);

        for (i, ch) in text.char_indices() {
            match ch {
                '{' => current_brace_count += 1,
                '}' => {
                    current_brace_count -= 1;
                    //info!("Found '}}' at position {}, brace count now: {}", i, current_brace_count);
                    if current_brace_count == 0 {
                        // Found complete JSON object
                        let end_pos = start_pos + i + 1;
                        let json_str = &self.buffer[start_pos..end_pos];

                        //info!("Complete JSON found: {}", json_str);

                        if let Ok(tool_call) = serde_json::from_str::<ToolCall>(json_str) {
                            info!("Successfully parsed tool call: {:?}", tool_call);
                            // Reset parser state
                            self.in_tool_call = false;
                            self.tool_start_pos = None;
                            self.brace_count = 0;

                            return Some((tool_call, end_pos));
                        } else {
                            info!("Failed to parse JSON: {}", json_str);
                            // Invalid JSON, reset and continue looking
                            self.in_tool_call = false;
                            self.tool_start_pos = None;
                            self.brace_count = 0;
                        }
                    }
                }
                _ => {}
            }
        }

        // Update brace count for next iteration
        self.brace_count = current_brace_count;
        //info!("End of parsing, final brace count: {}", current_brace_count);
        None
    }

    pub fn get_content_before_tool(&self, tool_end_pos: usize) -> String {
        if tool_end_pos <= self.buffer.len() {
            self.buffer[..tool_end_pos].to_string()
        } else {
            self.buffer.clone()
        }
    }

    pub fn get_remaining_content(&self, from_pos: usize) -> String {
        if from_pos < self.buffer.len() {
            self.buffer[from_pos..].to_string()
        } else {
            String::new()
        }
    }
}

#[derive(Debug, Clone)]
pub struct ContextWindow {
    pub used_tokens: u32,
    pub total_tokens: u32,
    pub conversation_history: Vec<Message>,
}

impl ContextWindow {
    pub fn new(total_tokens: u32) -> Self {
        Self {
            used_tokens: 0,
            total_tokens,
            conversation_history: Vec::new(),
        }
    }

    pub fn add_message(&mut self, message: Message) {
        // Simple token estimation: ~4 characters per token
        let estimated_tokens = (message.content.len() as f32 / 4.0).ceil() as u32;
        self.used_tokens += estimated_tokens;
        self.conversation_history.push(message);
    }

    pub fn update_usage(&mut self, usage: &g3_providers::Usage) {
        // Update with actual token usage from the provider
        self.used_tokens = usage.total_tokens;
    }

    pub fn percentage_used(&self) -> f32 {
        if self.total_tokens == 0 {
            0.0
        } else {
            (self.used_tokens as f32 / self.total_tokens as f32) * 100.0
        }
    }

    pub fn remaining_tokens(&self) -> u32 {
        self.total_tokens.saturating_sub(self.used_tokens)
    }
}

pub struct Agent {
    providers: ProviderRegistry,
    config: Config,
    context_window: ContextWindow,
}

impl Agent {
    pub async fn new(config: Config) -> Result<Self> {
        let mut providers = ProviderRegistry::new();

        // Register providers based on configuration
        if let Some(openai_config) = &config.providers.openai {
            let openai_provider = crate::providers::openai::OpenAIProvider::new(
                openai_config.api_key.clone(),
                openai_config.model.clone(),
                openai_config.base_url.clone(),
            )?;
            providers.register(openai_provider);
        }

        if let Some(anthropic_config) = &config.providers.anthropic {
            let anthropic_provider = crate::providers::anthropic::AnthropicProvider::new(
                anthropic_config.api_key.clone(),
                anthropic_config.model.clone(),
            )?;
            providers.register(anthropic_provider);
        }

        if let Some(embedded_config) = &config.providers.embedded {
            let embedded_provider = crate::providers::embedded::EmbeddedProvider::new(
                embedded_config.model_path.clone(),
                embedded_config.model_type.clone(),
                embedded_config.context_length,
                embedded_config.max_tokens,
                embedded_config.temperature,
                embedded_config.gpu_layers,
                embedded_config.threads,
            )?;
            providers.register(embedded_provider);
        }

        // Set default provider
        providers.set_default(&config.providers.default_provider)?;

        // Determine context window size based on active provider
        let context_length = Self::determine_context_length(&config, &providers)?;
        let context_window = ContextWindow::new(context_length);

        Ok(Self {
            providers,
            config,
            context_window,
        })
    }

    fn determine_context_length(config: &Config, providers: &ProviderRegistry) -> Result<u32> {
        // Get the active provider to determine context length
        let provider = providers.get(None)?;
        let provider_name = provider.name();
        let model_name = provider.model();

        // Use provider-specific context length if available, otherwise fall back to agent config
        let context_length = match provider_name {
            "embedded" => {
                // For embedded models, use the configured context_length or model-specific defaults
                if let Some(embedded_config) = &config.providers.embedded {
                    embedded_config.context_length.unwrap_or_else(|| {
                        // Model-specific defaults for embedded models
                        match embedded_config.model_type.to_lowercase().as_str() {
                            "codellama" => 16384, // CodeLlama supports 16k context
                            "llama" => 4096,      // Base Llama models
                            "mistral" => 8192,    // Mistral models
                            _ => 4096,            // Conservative default
                        }
                    })
                } else {
                    config.agent.max_context_length as u32
                }
            }
            "openai" => {
                // OpenAI model-specific context lengths
                match model_name {
                    m if m.contains("gpt-4") => 128000, // GPT-4 models have 128k context
                    m if m.contains("gpt-3.5") => 16384, // GPT-3.5-turbo has 16k context
                    _ => 4096,                          // Conservative default
                }
            }
            "anthropic" => {
                // Anthropic model-specific context lengths
                match model_name {
                    m if m.contains("claude-3") => 200000, // Claude-3 has 200k context
                    m if m.contains("claude-2") => 100000, // Claude-2 has 100k context
                    _ => 100000,                           // Conservative default for Claude
                }
            }
            _ => config.agent.max_context_length as u32,
        };

        info!(
            "Using context length: {} tokens for provider: {} (model: {})",
            context_length, provider_name, model_name
        );

        Ok(context_length)
    }

    pub fn get_provider_info(&self) -> Result<(String, String)> {
        let provider = self.providers.get(None)?;
        Ok((provider.name().to_string(), provider.model().to_string()))
    }

    pub async fn execute_task(
        &mut self,
        description: &str,
        language: Option<&str>,
        _auto_execute: bool,
    ) -> Result<String> {
        self.execute_task_with_options(description, language, false, false, false)
            .await
    }

    pub async fn execute_task_with_options(
        &mut self,
        description: &str,
        language: Option<&str>,
        _auto_execute: bool,
        show_prompt: bool,
        show_code: bool,
    ) -> Result<String> {
        self.execute_task_with_timing(
            description,
            language,
            _auto_execute,
            show_prompt,
            show_code,
            false,
        )
        .await
    }

    pub async fn execute_task_with_timing(
        &mut self,
        description: &str,
        language: Option<&str>,
        _auto_execute: bool,
        show_prompt: bool,
        show_code: bool,
        show_timing: bool,
    ) -> Result<String> {
        // Create a cancellation token that never cancels for backward compatibility
        let cancellation_token = CancellationToken::new();
        self.execute_task_with_timing_cancellable(
            description,
            language,
            _auto_execute,
            show_prompt,
            show_code,
            show_timing,
            cancellation_token,
        )
        .await
    }

    pub async fn execute_task_with_timing_cancellable(
        &mut self,
        description: &str,
        _language: Option<&str>,
        _auto_execute: bool,
        show_prompt: bool,
        show_code: bool,
        show_timing: bool,
        cancellation_token: CancellationToken,
    ) -> Result<String> {
        info!("Executing task: {}", description);

        let _provider = self.providers.get(None)?;

        let system_prompt = format!(
            "You are G3, a general-purpose AI agent. Your goal is to analyze and solve problems step by step.

# Tool Call Format

When you need to execute a tool, write ONLY the JSON tool call on a new line:

{{\"tool\": \"tool_name\", \"args\": {{\"param\": \"value\"}}}}

The tool will execute immediately and you'll receive the result to continue with.

# Available Tools

- **shell**: Execute shell commands
  - Format: {{\"tool\": \"shell\", \"args\": {{\"command\": \"your_command_here\"}}}}
  - Example: {{\"tool\": \"shell\", \"args\": {{\"command\": \"ls ~/Downloads\"}}}}

- **final_output**: Signal task completion
  - Format: {{\"tool\": \"final_output\", \"args\": {{\"summary\": \"what_was_accomplished\"}}}}

# Instructions

1. Break down tasks into small steps
2. Execute ONE tool at a time
3. Wait for the result before proceeding
4. Use the actual file paths on the system (like ~/Downloads for Downloads folder)
5. End with final_output when done

Let's start with the first step of your task.
");

        if show_prompt {
            println!("🔍 System Prompt:");
            println!("================");
            println!("{}", system_prompt);
            println!("================");
            println!();
        }

        // Add system message to context window
        let system_message = Message {
            role: MessageRole::System,
            content: system_prompt.clone(),
        };
        self.context_window.add_message(system_message.clone());

        // Add user message to context window
        let user_message = Message {
            role: MessageRole::User,
            content: format!("Task: {}", description),
        };
        self.context_window.add_message(user_message.clone());

        let messages = vec![system_message, user_message];

        let request = CompletionRequest {
            messages,
            max_tokens: Some(2048),
            temperature: Some(0.2),
            stream: true, // Enable streaming
        };

        // Time the LLM call with cancellation support and streaming
        let llm_start = Instant::now();
        let (response_content, think_time) = tokio::select! {
            result = self.stream_completion(request) => result?,
            _ = cancellation_token.cancelled() => {
                return Err(anyhow::anyhow!("Operation cancelled by user"));
            }
        };
        let llm_duration = llm_start.elapsed();

        // Create a mock usage for now (we'll need to track this during streaming)
        let mock_usage = g3_providers::Usage {
            prompt_tokens: 100,                                   // Estimate
            completion_tokens: response_content.len() as u32 / 4, // Rough estimate
            total_tokens: 100 + (response_content.len() as u32 / 4),
        };

        // Update context window with estimated token usage
        self.context_window.update_usage(&mock_usage);

        // Add assistant response to context window
        let assistant_message = Message {
            role: MessageRole::Assistant,
            content: response_content.clone(),
        };
        self.context_window.add_message(assistant_message);

        // With streaming tool execution, we don't need separate code execution
        // The tools are already executed during streaming
        if show_timing {
            let timing_summary = format!(
                "\n⏱️ {} | 💭 {}",
                Self::format_duration(llm_duration),
                Self::format_duration(think_time)
            );
            Ok(format!("{}\n{}", response_content, timing_summary))
        } else {
            Ok(response_content)
        }
    }

    pub fn get_context_window(&self) -> &ContextWindow {
        &self.context_window
    }

    async fn stream_completion(&self, request: CompletionRequest) -> Result<(String, Duration)> {
        self.stream_completion_with_tools(request).await
    }

    async fn stream_completion_with_tools(
        &self,
        mut request: CompletionRequest,
    ) -> Result<(String, Duration)> {
        use std::io::{self, Write};
        use tokio_stream::StreamExt;

        let mut full_response = String::new();
        let mut first_token_time: Option<Duration> = None;
        let stream_start = Instant::now();
        let mut total_execution_time = Duration::new(0, 0);
        let mut iteration_count = 0;
        const MAX_ITERATIONS: usize = 10; // Prevent infinite loops

        print!("🤖 "); // Start the response indicator
        io::stdout().flush()?;

        loop {
            iteration_count += 1;
            if iteration_count > MAX_ITERATIONS {
                warn!("Maximum iterations reached, stopping stream");
                break;
            }

            // Add a small delay between iterations to prevent "model busy" errors
            if iteration_count > 1 {
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }

            let provider = self.providers.get(None)?;
            let mut stream = match provider.stream(request.clone()).await {
                Ok(s) => s,
                Err(e) => {
                    if iteration_count > 1 && e.to_string().contains("busy") {
                        warn!("Model busy on iteration {}, retrying in 500ms", iteration_count);
                        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                        match provider.stream(request.clone()).await {
                            Ok(s) => s,
                            Err(e2) => {
                                error!("Failed to start stream after retry: {}", e2);
                                return Err(e2);
                            }
                        }
                    } else {
                        return Err(e);
                    }
                }
            };
            let mut parser = StreamingToolParser::new();
            let mut current_response = String::new();
            let mut tool_executed = false;

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        // Record time to first token
                        if first_token_time.is_none() && !chunk.content.is_empty() {
                            first_token_time = Some(stream_start.elapsed());
                        }

                        // Check for tool calls in the streaming content
                        if let Some((tool_call, tool_end_pos)) = parser.add_chunk(&chunk.content) {
                            info!(
                                "🔧 Detected tool call: {:?} at position {}",
                                tool_call, tool_end_pos
                            );
                            // Found a complete tool call! Stop streaming and execute it
                            let content_before_tool = parser.get_content_before_tool(tool_end_pos);

                            // Display content up to the tool call (excluding the JSON)
                            let display_content = if let Some(json_start) =
                                content_before_tool.rfind(r#"{"tool":"#)
                            {
                                &content_before_tool[..json_start]
                            } else {
                                &content_before_tool
                            };

                            // Safely get the new content to display
                            let new_content = if current_response.len() <= display_content.len() {
                                // Use char indices to avoid UTF-8 boundary issues
                                let chars_already_shown = current_response.chars().count();
                                display_content.chars().skip(chars_already_shown).collect::<String>()
                            } else {
                                String::new()
                            };
                            print!("{}", new_content);
                            io::stdout().flush()?;

                            // Execute the tool
                            println!(); // New line before tool execution
                            let exec_start = Instant::now();
                            let tool_result = self.execute_tool(&tool_call).await?;
                            let exec_duration = exec_start.elapsed();
                            total_execution_time += exec_duration;

                            // Display tool execution result
                            println!("🔧 {}: {}", tool_call.tool, tool_result);
                            print!("🤖 "); // Continue response indicator
                            io::stdout().flush()?;

                            // Update the conversation with the tool call and result
                            let tool_message = Message {
                                role: MessageRole::Assistant,
                                content: format!(
                                    "{}\n\n{{\"tool\": \"{}\", \"args\": {}}}",
                                    display_content.trim(),
                                    tool_call.tool,
                                    tool_call.args
                                ),
                            };
                            let result_message = Message {
                                role: MessageRole::User, // Tool results come back as user messages
                                content: format!("Tool result: {}", tool_result),
                            };

                            request.messages.push(tool_message);
                            request.messages.push(result_message);

                            full_response.push_str(display_content);
                            full_response.push_str(&format!(
                                "\n\nTool executed: {} -> {}\n\n",
                                tool_call.tool, tool_result
                            ));

                            tool_executed = true;
                            // Break out of current stream to start a new one with updated context
                            break;
                        } else {
                            // No tool call detected, continue streaming normally
                            print!("{}", chunk.content);
                            io::stdout().flush()?;
                            current_response.push_str(&chunk.content);
                        }

                        if chunk.finished {
                            // Stream finished naturally without tool calls
                            full_response.push_str(&current_response);
                            println!(); // New line after streaming completes
                            let ttft = first_token_time.unwrap_or_else(|| stream_start.elapsed());
                            return Ok((full_response, ttft));
                        }
                    }
                    Err(e) => {
                        error!("Streaming error: {}", e);

                        // If we executed a tool, try to continue with a new stream
                        if tool_executed {
                            warn!("Stream error after tool execution, attempting to continue");
                            break; // Break to outer loop to start new stream
                        } else {
                            return Err(e);
                        }
                    }
                }
            }

            // If we get here and no tool was executed, we're done
            if !tool_executed {
                full_response.push_str(&current_response);
                println!(); // New line after streaming completes
                let ttft = first_token_time.unwrap_or_else(|| stream_start.elapsed());
                return Ok((full_response, ttft));
            }

            // Continue the loop to start a new stream with updated context
            info!(
                "Starting new stream iteration {} with {} messages",
                iteration_count,
                request.messages.len()
            );
        }

        // If we exit the loop due to max iterations
        let ttft = first_token_time.unwrap_or_else(|| stream_start.elapsed());
        Ok((full_response, ttft))
    }

    async fn execute_tool(&self, tool_call: &ToolCall) -> Result<String> {
        match tool_call.tool.as_str() {
            "shell" => {
                if let Some(command) = tool_call.args.get("command") {
                    if let Some(command_str) = command.as_str() {
                        let executor = CodeExecutor::new();
                        match executor.execute_code("bash", command_str).await {
                            Ok(result) => {
                                if result.success {
                                    Ok(if result.stdout.is_empty() {
                                        "✅ Command executed successfully".to_string()
                                    } else {
                                        result.stdout.trim().to_string()
                                    })
                                } else {
                                    Ok(format!("❌ Command failed: {}", result.stderr.trim()))
                                }
                            }
                            Err(e) => Ok(format!("❌ Execution error: {}", e)),
                        }
                    } else {
                        Ok("❌ Invalid command argument".to_string())
                    }
                } else {
                    Ok("❌ Missing command argument".to_string())
                }
            }
            "final_output" => {
                if let Some(summary) = tool_call.args.get("summary") {
                    if let Some(summary_str) = summary.as_str() {
                        Ok(format!("📋 Final Output: {}", summary_str))
                    } else {
                        Ok("📋 Task completed".to_string())
                    }
                } else {
                    Ok("📋 Task completed".to_string())
                }
            }
            _ => {
                warn!("Unknown tool: {}", tool_call.tool);
                Ok(format!("❓ Unknown tool: {}", tool_call.tool))
            }
        }
    }

    fn format_duration(duration: Duration) -> String {
        let total_ms = duration.as_millis();

        if total_ms < 1000 {
            format!("{}ms", total_ms)
        } else if total_ms < 60_000 {
            let seconds = duration.as_secs_f64();
            format!("{:.1}s", seconds)
        } else {
            let minutes = total_ms / 60_000;
            let remaining_seconds = (total_ms % 60_000) as f64 / 1000.0;
            format!("{}m {:.1}s", minutes, remaining_seconds)
        }
    }
}

pub mod providers {
    pub mod anthropic;
    pub mod embedded;
    pub mod openai;
}
