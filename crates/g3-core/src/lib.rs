use anyhow::Result;
use g3_config::Config;
use g3_execution::CodeExecutor;
use g3_providers::{CompletionRequest, Message, MessageRole, ProviderRegistry, Tool};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

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
            // Also handle malformed patterns and whitespace variations
            let patterns = [
                r#"{"tool":"#,     // Normal pattern
                r#"{ "tool":"#,    // Pattern with space after opening brace
                r#"{"tool" :"#,    // Pattern with space before colon
                r#"{ "tool" :"#,   // Pattern with spaces around tool
                r#"{"{""tool"":"#, // Malformed pattern with extra brace and doubled quotes
                r#"{{""tool"":"#,  // Alternative malformed pattern
            ];

            for pattern in &patterns {
                if let Some(pos) = self.buffer.rfind(pattern) {
                    // Check if this is inside a code block
                    let before_pos = &self.buffer[..pos];
                    let _code_block_count = before_pos.matches("```").count();

                    // Accept tool calls both inside and outside code blocks
                    // The LLM might use either format despite our instructions
                    //info!("Starting tool call parsing (code block status: {})", code_block_count % 2 == 1);
                    self.in_tool_call = true;
                    self.tool_start_pos = Some(pos);
                    self.brace_count = 0; // Start counting from 0, we'll count the opening brace in parsing

                    // Continue parsing from after the opening brace
                    return self.parse_from_start_pos(pos);
                }
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
                        let mut json_str = self.buffer[start_pos..end_pos].to_string();

                        // Clean up malformed JSON patterns
                        json_str = json_str
                            .replace(r#"{"{""#, r#"{"#) // Fix {"{" -> {"
                            .replace(r#"""}"#, r#""}"#) // Fix ""} -> "}
                            .replace(r#"{{""#, r#"{"#) // Fix {{" -> {"
                            .replace(r#"""}"#, r#""}"#); // Fix ""} -> "}

                        // First, try to parse the JSON as-is
                        if let Ok(tool_call) = serde_json::from_str::<ToolCall>(&json_str) {
                            info!("Successfully parsed tool call: {:?}", tool_call);
                            // Reset parser state
                            self.in_tool_call = false;
                            self.tool_start_pos = None;
                            self.brace_count = 0;

                            return Some((tool_call, end_pos));
                        }

                        // If parsing failed and this is a shell command, try to fix nested quotes
                        if json_str.contains(r#""tool": "shell""#) {
                            let fixed_json = fix_nested_quotes_in_shell_command(&json_str);
                            if let Ok(tool_call) = serde_json::from_str::<ToolCall>(&fixed_json) {
                                // Reset parser state
                                self.in_tool_call = false;
                                self.tool_start_pos = None;
                                self.brace_count = 0;

                                return Some((tool_call, end_pos));
                            } else {
                                info!(
                                    "Failed to parse JSON even after fixing nested quotes: {}",
                                    fixed_json
                                );
                            }
                        }

                        // Try to fix mixed quote issues (single quotes in JSON)
                        let fixed_mixed_quotes = fix_mixed_quotes_in_json(&json_str);
                        if fixed_mixed_quotes != json_str {
                            if let Ok(tool_call) =
                                serde_json::from_str::<ToolCall>(&fixed_mixed_quotes)
                            {
                                info!(
                                    "Successfully parsed tool call after fixing mixed quotes: {:?}",
                                    tool_call
                                );
                                // Reset parser state
                                self.in_tool_call = false;
                                self.tool_start_pos = None;
                                self.brace_count = 0;

                                return Some((tool_call, end_pos));
                            } else {
                                info!(
                                    "Failed to parse JSON even after fixing mixed quotes: {}",
                                    fixed_mixed_quotes
                                );
                            }
                        } else {
                            info!("Failed to parse JSON (no fixes applied): {}", json_str);
                        }

                        // Invalid JSON, reset and continue looking
                        self.in_tool_call = false;
                        self.tool_start_pos = None;
                        self.brace_count = 0;
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
        // Skip messages with empty content to avoid API errors
        if message.content.trim().is_empty() {
            warn!("Skipping empty message to avoid API error");
            return;
        }

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
    context_window: ContextWindow,
    session_id: Option<String>,
}

impl Agent {
    pub async fn new(config: Config) -> Result<Self> {
        let mut providers = ProviderRegistry::new();

        // Only register providers that are configured AND selected as the default provider
        // This prevents unnecessary initialization of heavy providers like embedded models

        // Register embedded provider if configured AND it's the default provider
        if let Some(embedded_config) = &config.providers.embedded {
            if config.providers.default_provider == "embedded" {
                info!("Initializing embedded provider (selected as default)");
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
            } else {
                info!("Embedded provider configured but not selected as default, skipping initialization");
            }
        }

        // Register Anthropic provider if configured AND it's the default provider
        if let Some(anthropic_config) = &config.providers.anthropic {
            if config.providers.default_provider == "anthropic" {
                info!("Initializing Anthropic provider (selected as default)");
                let anthropic_provider = g3_providers::AnthropicProvider::new(
                    anthropic_config.api_key.clone(),
                    Some(anthropic_config.model.clone()),
                    anthropic_config.max_tokens,
                    anthropic_config.temperature,
                )?;
                providers.register(anthropic_provider);
            } else {
                info!("Anthropic provider configured but not selected as default, skipping initialization");
            }
        }

        // Set default provider
        debug!(
            "Setting default provider to: {}",
            config.providers.default_provider
        );
        providers.set_default(&config.providers.default_provider)?;
        debug!("Default provider set successfully");

        // Determine context window size based on active provider
        let context_length = Self::determine_context_length(&config, &providers)?;
        let context_window = ContextWindow::new(context_length);

        Ok(Self {
            providers,
            context_window,
            session_id: None,
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
                            "qwen" => 32768,      // Qwen2.5 supports 32k context
                            _ => 4096,            // Conservative default
                        }
                    })
                } else {
                    config.agent.max_context_length as u32
                }
            }
            "anthropic" => {
                // Claude models have large context windows
                200000 // Default for Claude models
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
        // Execute the task directly without splitting
        self.execute_single_task(
            description,
            show_prompt,
            show_code,
            show_timing,
            cancellation_token,
        )
        .await
    }

    async fn execute_single_task(
        &mut self,
        description: &str,
        show_prompt: bool,
        _show_code: bool,
        show_timing: bool,
        cancellation_token: CancellationToken,
    ) -> Result<String> {
        // Generate session ID based on the initial prompt if this is a new session
        if self.session_id.is_none() {
            self.session_id = Some(self.generate_session_id(description));
        }

        // Only add system message if this is the first interaction (empty conversation history)
        if self.context_window.conversation_history.is_empty() {
            let provider = self.providers.get(None)?;
            let system_prompt = if provider.has_native_tool_calling() {
                // For native tool calling providers, use a more explicit system prompt
                "You are G3, a general-purpose AI agent. Your goal is to analyze and solve problems by writing code. The current directory always contains a project that the user is working on and likely referring to.

You have access to tools. When you need to accomplish a task, you MUST use the appropriate tool immediately. Do not just describe what you would do - actually use the tools.

IMPORTANT: You must call tools to complete tasks. When you receive a request:
1. Identify what needs to be done
2. Immediately call the appropriate tool with the required parameters
3. Wait for the tool result
4. Continue or complete the task based on the result
5. Call the final_output task with a detailed summary when done with all tasks.

For shell commands: Use the shell tool with the exact command needed. Avoid commands that produce a large amount of output, and consider piping those outputs to files. Example: If asked to list files, immediately call the shell tool with command parameter \"ls\".

IMPORTANT: If the user asks you to just respond with text (like \"just say hello\" or \"tell me about X\"), do NOT use tools. Simply respond with the requested text directly. Only use tools when you need to execute commands or complete tasks that require action.

Do not explain what you're going to do - just do it by calling the tools.

# Response Guidelines

- Use Markdown formatting for all responses except tool calls.
- Whenever taking actions, use the pronoun 'I'
".to_string()
            } else {
                // For non-native providers (embedded models), use JSON format instructions
                "You are G3, a general-purpose AI agent. Your goal is to analyze and solve problems by writing code.

# Tool Call Format

When you need to execute a tool, write ONLY the JSON tool call on a new line:

{\"tool\": \"tool_name\", \"args\": {\"param\": \"value\"}}

The tool will execute immediately and you'll receive the result (success or error) to continue with.

# Available Tools

- **shell**: Execute shell commands
  - Format: {\"tool\": \"shell\", \"args\": {\"command\": \"your_command_here\"}}
  - Example: {\"tool\": \"shell\", \"args\": {\"command\": \"ls ~/Downloads\"}}

- **read_file**: Read the contents of a file
  - Format: {\"tool\": \"read_file\", \"args\": {\"file_path\": \"path/to/file\"}}
  - Example: {\"tool\": \"read_file\", \"args\": {\"file_path\": \"src/main.rs\"}}

- **write_file**: Write content to a file (creates or overwrites)
  - Format: {\"tool\": \"write_file\", \"args\": {\"file_path\": \"path/to/file\", \"content\": \"file content\"}}
  - Example: {\"tool\": \"write_file\", \"args\": {\"file_path\": \"src/lib.rs\", \"content\": \"pub fn hello() {}\"}}

- **edit_file**: Edit a specific range of lines in a file
  - Format: {\"tool\": \"edit_file\", \"args\": {\"file_path\": \"path/to/file\", \"start_line\": 1, \"end_line\": 3, \"new_text\": \"replacement text\"}}
  - Example: {\"tool\": \"edit_file\", \"args\": {\"file_path\": \"src/main.rs\", \"start_line\": 5, \"end_line\": 7, \"new_text\": \"println!(\\\"Hello, world!\\\");\"}}

- **final_output**: Signal task completion with a detailed summary of work done in markdown format
  - Format: {\"tool\": \"final_output\", \"args\": {\"summary\": \"what_was_accomplished\"}}

# Instructions

1. Analyze the request and break down into smaller tasks if appropriate
2. Execute ONE tool at a time
3. STOP when the original request was satisfied
4. Call the final_output tool when done

# Response Guidelines

- Use Markdown formatting for all responses except tool calls.
- Whenever taking actions, use the pronoun 'I'

".to_string()
            };

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
                content: system_prompt,
            };
            self.context_window.add_message(system_message);
        }

        // Add user message to context window
        let user_message = Message {
            role: MessageRole::User,
            content: format!("Task: {}", description),
        };
        self.context_window.add_message(user_message);

        // Use the complete conversation history for the request
        let messages = self.context_window.conversation_history.clone();

        // Check if provider supports native tool calling and add tools if so
        let provider = self.providers.get(None)?;
        let tools = if provider.has_native_tool_calling() {
            Some(Self::create_tool_definitions())
        } else {
            None
        };

        let request = CompletionRequest {
            messages,
            max_tokens: Some(2048),
            temperature: Some(0.1),
            stream: true, // Enable streaming
            tools,
        };

        // Time the LLM call with cancellation support and streaming
        let llm_start = Instant::now();
        let result = tokio::select! {
            result = self.stream_completion(request) => result,
            _ = cancellation_token.cancelled() => {
                // Save context window on cancellation
                self.save_context_window("cancelled");
                Err(anyhow::anyhow!("Operation cancelled by user"))
            }
        };

        let (response_content, think_time) = match result {
            Ok(content) => content,
            Err(e) => {
                // Save context window on error
                self.save_context_window("error");
                return Err(e);
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

        // Save context window at the end of successful interaction
        self.save_context_window("completed");

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

    /// Generate a session ID based on the initial prompt
    fn generate_session_id(&self, description: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Clean and truncate the description for a readable filename
        let clean_description = description
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace() || *c == '-' || *c == '_')
            .collect::<String>()
            .split_whitespace()
            .take(5) // Take first 5 words
            .collect::<Vec<_>>()
            .join("_")
            .to_lowercase();

        // Create a hash for uniqueness
        let mut hasher = DefaultHasher::new();
        description.hash(&mut hasher);
        let hash = hasher.finish();

        // Format: clean_description_hash
        format!("{}_{:x}", clean_description, hash)
    }

    /// Save the entire context window to a per-session file
    fn save_context_window(&self, status: &str) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Use session-based filename if we have a session ID, otherwise fall back to timestamp
        let filename = if let Some(ref session_id) = self.session_id {
            format!("g3_session_{}.json", session_id)
        } else {
            format!("g3_context_{}.json", timestamp)
        };

        let context_data = serde_json::json!({
            "session_id": self.session_id,
            "timestamp": timestamp,
            "status": status,
            "context_window": {
                "used_tokens": self.context_window.used_tokens,
                "total_tokens": self.context_window.total_tokens,
                "percentage_used": self.context_window.percentage_used(),
                "conversation_history": self.context_window.conversation_history
            }
        });

        match serde_json::to_string_pretty(&context_data) {
            Ok(json_content) => {
                if let Err(e) = fs::write(&filename, json_content) {
                    error!("Failed to save context window to {}: {}", filename, e);
                }
            }
            Err(e) => {
                error!("Failed to serialize context window: {}", e);
            }
        }
    }

    pub fn get_context_window(&self) -> &ContextWindow {
        &self.context_window
    }

    async fn stream_completion(
        &mut self,
        request: CompletionRequest,
    ) -> Result<(String, Duration)> {
        self.stream_completion_with_tools(request).await
    }

    /// Create tool definitions for native tool calling providers
    fn create_tool_definitions() -> Vec<Tool> {
        vec![
            Tool {
                name: "shell".to_string(),
                description: "Execute shell commands".to_string(),
                input_schema: json!({
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
            Tool {
                name: "read_file".to_string(),
                description: "Read the contents of a file".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path to the file to read"
                        }
                    },
                    "required": ["file_path"]
                }),
            },
            Tool {
                name: "write_file".to_string(),
                description: "Write content to a file (creates or overwrites)".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path to the file to write"
                        },
                        "content": {
                            "type": "string",
                            "description": "The content to write to the file"
                        }
                    },
                    "required": ["file_path", "content"]
                }),
            },
            // Tool {
            //     name: "edit_file".to_string(),
            //     description: "Edit a specific range of lines in a file".to_string(),
            //     input_schema: json!({
            //         "type": "object",
            //         "properties": {
            //             "file_path": {
            //                 "type": "string",
            //                 "description": "The path to the file to edit"
            //             },
            //             "start_line": {
            //                 "type": "integer",
            //                 "description": "The starting line number (1-based) of the range to replace"
            //             },
            //             "end_line": {
            //                 "type": "integer",
            //                 "description": "The ending line number (1-based) of the range to replace"
            //             },
            //             "new_text": {
            //                 "type": "string",
            //                 "description": "The new text to replace the specified range"
            //             }
            //         },
            //         "required": ["file_path", "start_line", "end_line", "new_text"]
            //     }),
            // },
            Tool {
                name: "final_output".to_string(),
                description: "Signal task completion with a detailed summary".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "A detailed summary of what was accomplished"
                        }
                    },
                    "required": ["summary"]
                }),
            },
        ]
    }

    async fn stream_completion_with_tools(
        &mut self,
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
        let mut response_started = false;

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
                        warn!(
                            "Model busy on iteration {}, retrying in 500ms",
                            iteration_count
                        );
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

                        // Check for tool calls - prioritize native tool calls over JSON parsing
                        let mut detected_tool_call = None;
                        let mut is_text_based_tool_call = false;

                        // First check for native tool calls in the chunk
                        if let Some(ref tool_calls) = chunk.tool_calls {
                            debug!("Found native tool calls in chunk: {:?}", tool_calls);
                            if let Some(first_tool) = tool_calls.first() {
                                // Convert native tool call to our internal format
                                detected_tool_call = Some((
                                    crate::ToolCall {
                                        tool: first_tool.tool.clone(),
                                        args: first_tool.args.clone(),
                                    },
                                    current_response.len(), // Position doesn't matter for native calls
                                ));
                                debug!("Converted native tool call: {:?}", detected_tool_call);
                            }
                        } else {
                            debug!("No native tool calls in chunk, chunk.tool_calls is None");
                        }

                        // Always try JSON parsing as fallback, even for native providers
                        // This handles cases where Anthropic returns tool calls as text instead of native format
                        if detected_tool_call.is_none() {
                            // Try to parse JSON tool calls from text content
                            detected_tool_call = parser.add_chunk(&chunk.content);
                            if detected_tool_call.is_some() {
                                debug!("Found JSON tool call in text content for native provider");
                                is_text_based_tool_call = true;
                            }
                        }

                        if let Some((tool_call, tool_end_pos)) = detected_tool_call {
                            // Found a complete tool call! Stop streaming and execute it
                            let content_before_tool = parser.get_content_before_tool(tool_end_pos);

                            // Display content up to the tool call (excluding the JSON and any stop tokens)
                            let display_content = if let Some(json_start) =
                                content_before_tool.rfind(r#"{"tool":"#)
                            {
                                // Only show content before the JSON tool call
                                content_before_tool[..json_start].trim()
                            } else {
                                // Fallback: clean any stop tokens from the content
                                content_before_tool.trim()
                            };

                            // Clean stop tokens from display content
                            let clean_display_content = display_content
                                .replace("<|im_end|>", "")
                                .replace("</s>", "")
                                .replace("[/INST]", "")
                                .replace("<</SYS>>", "");
                            let final_display_content = clean_display_content.trim();

                            // Safely get the new content to display
                            let new_content =
                                if current_response.len() <= final_display_content.len() {
                                    // Use char indices to avoid UTF-8 boundary issues
                                    let chars_already_shown = current_response.chars().count();
                                    final_display_content
                                        .chars()
                                        .skip(chars_already_shown)
                                        .collect::<String>()
                                } else {
                                    String::new()
                                };

                            // Only print if there's actually new content to show
                            if !new_content.trim().is_empty() {
                                // Replace thinking indicator with response indicator if not already done
                                if !response_started {
                                    print!("\r🤖 "); // Clear thinking indicator and show response indicator
                                    response_started = true;
                                }
                                print!("{}", new_content);
                                io::stdout().flush()?;
                            }

                            // Check if this was a JSON tool call detected from text (not native)
                            // If so, show a brief indicator that we detected a text-based tool call
                            let provider = self.providers.get(None)?;
                            if provider.has_native_tool_calling() && is_text_based_tool_call {
                                // This means we detected a JSON tool call in text for a native provider
                                // Show a brief indicator instead of the raw JSON
                                if !response_started {
                                    print!("\r🤖 "); // Clear thinking indicator and show response indicator
                                    response_started = true;
                                }
                                print!("🔧 "); // Brief tool call indicator
                                io::stdout().flush()?;
                            }

                            // Execute the tool with formatted output
                            println!(); // New line before tool execution

                            // Tool call header
                            println!("┌─ {}", tool_call.tool);
                            if let Some(args_obj) = tool_call.args.as_object() {
                                for (_key, value) in args_obj {
                                    let value_str = match value {
                                        serde_json::Value::String(s) => {
                                            // For shell commands, truncate at newlines to keep display clean
                                            if tool_call.tool == "shell" && _key == "command" {
                                                if let Some(first_line) = s.lines().next() {
                                                    if s.lines().count() > 1 {
                                                        format!("{}...", first_line)
                                                    } else {
                                                        first_line.to_string()
                                                    }
                                                } else {
                                                    s.clone()
                                                }
                                            } else {
                                                s.clone()
                                            }
                                        }
                                        _ => value.to_string(),
                                    };
                                    println!("│ {}", value_str);
                                }
                            }
                            println!("├─ output:");

                            let exec_start = Instant::now();
                            let tool_result = self.execute_tool(&tool_call).await?;
                            let exec_duration = exec_start.elapsed();
                            total_execution_time += exec_duration;

                            // Display tool execution result with proper indentation
                            let output_lines: Vec<&str> = tool_result.lines().collect();
                            const MAX_LINES: usize = 5;

                            if output_lines.len() <= MAX_LINES {
                                // Show all lines if within limit
                                for line in output_lines {
                                    println!("│ {}", line);
                                }
                            } else {
                                // Show first MAX_LINES and add truncation note
                                for line in output_lines.iter().take(MAX_LINES) {
                                    println!("│ {}", line);
                                }
                                let hidden_count = output_lines.len() - MAX_LINES;
                                println!(
                                    "│ ... ({} more line{} hidden)",
                                    hidden_count,
                                    if hidden_count == 1 { "" } else { "s" }
                                );
                            }

                            // Check if this was a final_output tool call - if so, stop the conversation
                            if tool_call.tool == "final_output" {
                                // For final_output, don't add the tool call and result to context
                                // Just add the display content and return immediately
                                full_response.push_str(final_display_content);
                                if let Some(summary) = tool_call.args.get("summary") {
                                    if let Some(summary_str) = summary.as_str() {
                                        full_response.push_str(&format!("\n\n=> {}", summary_str));
                                    }
                                }
                                println!(); // New line after final output
                                let ttft =
                                    first_token_time.unwrap_or_else(|| stream_start.elapsed());
                                return Ok((full_response, ttft));
                            }

                            // Closure marker with timing
                            println!("└─ ⚡️ {}", Self::format_duration(exec_duration));
                            println!();
                            print!("🤖 "); // Continue response indicator
                            io::stdout().flush()?;

                            // Add the tool call and result to the context window immediately
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

                            // Add to context window for persistence
                            self.context_window.add_message(tool_message);
                            self.context_window.add_message(result_message);

                            // Update the request with the new context for next iteration
                            request.messages = self.context_window.conversation_history.clone();

                            // Ensure tools are included for native providers in subsequent iterations
                            if provider.has_native_tool_calling() {
                                request.tools = Some(Self::create_tool_definitions());
                            }

                            full_response.push_str(final_display_content);
                            // full_response.push_str(&format!(
                            //     "\n\nTool executed: {} -> {}\n\n",
                            //     tool_call.tool, tool_result
                            // ));

                            tool_executed = true;
                            // Break out of current stream to start a new one with updated context
                            break;
                        } else {
                            // No tool call detected, continue streaming normally
                            // But first check if we need to filter JSON tool calls from display
                            let clean_content = chunk
                                .content
                                .replace("<|im_end|>", "")
                                .replace("</s>", "")
                                .replace("[/INST]", "")
                                .replace("<</SYS>>", "");

                            if !clean_content.is_empty() {
                                // Filter out JSON tool calls from display
                                let filtered_content = filter_json_tool_calls(&clean_content);
                                
                                // If we have any content to display
                                if !filtered_content.is_empty() {
                                    // Replace thinking indicator with response indicator on first content
                                    if !response_started {
                                        print!("\r🤖 "); // Clear thinking indicator and show response indicator
                                        response_started = true;
                                    }

                                    debug!("Printing filtered content: '{}'", filtered_content);
                                    print!("{}", filtered_content);
                                    let _ = io::stdout().flush(); // Force immediate output
                                    debug!("Flushed {} characters to stdout", filtered_content.len());
                                    current_response.push_str(&filtered_content);
                                }
                            }
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
        }

        // If we exit the loop due to max iterations
        let ttft = first_token_time.unwrap_or_else(|| stream_start.elapsed());
        Ok((full_response, ttft))
    }

    async fn execute_tool(&self, tool_call: &ToolCall) -> Result<String> {
        debug!("Executing tool: {}", tool_call.tool);
        debug!("Tool call args: {:?}", tool_call.args);
        debug!(
            "Tool call args JSON: {}",
            serde_json::to_string(&tool_call.args)
                .unwrap_or_else(|_| "failed to serialize".to_string())
        );

        match tool_call.tool.as_str() {
            "shell" => {
                debug!("Processing shell tool call");
                if let Some(command) = tool_call.args.get("command") {
                    debug!("Found command parameter: {:?}", command);
                    if let Some(command_str) = command.as_str() {
                        debug!("Command string: {}", command_str);
                        // Use shell escaping to handle filenames with spaces and special characters
                        let escaped_command = shell_escape_command(command_str);

                        let executor = CodeExecutor::new();
                        match executor.execute_code("bash", &escaped_command).await {
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
                        debug!("Command parameter is not a string: {:?}", command);
                        Ok("❌ Invalid command argument".to_string())
                    }
                } else {
                    debug!("No command parameter found in args: {:?}", tool_call.args);
                    debug!(
                        "Available keys: {:?}",
                        tool_call
                            .args
                            .as_object()
                            .map(|obj| obj.keys().collect::<Vec<_>>())
                    );
                    Ok("❌ Missing command argument".to_string())
                }
            }
            "read_file" => {
                debug!("Processing read_file tool call");
                if let Some(file_path) = tool_call.args.get("file_path") {
                    if let Some(path_str) = file_path.as_str() {
                        debug!("Reading file: {}", path_str);
                        match std::fs::read_to_string(path_str) {
                            Ok(content) => {
                                let line_count = content.lines().count();
                                Ok(format!(
                                    "📄 File content ({} lines):\n{}",
                                    line_count, content
                                ))
                            }
                            Err(e) => Ok(format!("❌ Failed to read file '{}': {}", path_str, e)),
                        }
                    } else {
                        Ok("❌ Invalid file_path argument".to_string())
                    }
                } else {
                    Ok("❌ Missing file_path argument".to_string())
                }
            }
            "write_file" => {
                debug!("Processing write_file tool call");
                let file_path = tool_call.args.get("file_path");
                let content = tool_call.args.get("content");

                if let (Some(path_val), Some(content_val)) = (file_path, content) {
                    if let (Some(path_str), Some(content_str)) =
                        (path_val.as_str(), content_val.as_str())
                    {
                        debug!("Writing to file: {}", path_str);

                        // Create parent directories if they don't exist
                        if let Some(parent) = std::path::Path::new(path_str).parent() {
                            if let Err(e) = std::fs::create_dir_all(parent) {
                                return Ok(format!(
                                    "❌ Failed to create parent directories for '{}': {}",
                                    path_str, e
                                ));
                            }
                        }

                        match std::fs::write(path_str, content_str) {
                            Ok(()) => {
                                let line_count = content_str.lines().count();
                                Ok(format!(
                                    "✅ Successfully wrote {} lines to '{}'",
                                    line_count, path_str
                                ))
                            }
                            Err(e) => {
                                Ok(format!("❌ Failed to write to file '{}': {}", path_str, e))
                            }
                        }
                    } else {
                        Ok("❌ Invalid file_path or content argument".to_string())
                    }
                } else {
                    Ok("❌ Missing file_path or content argument".to_string())
                }
            }
            "edit_file" => {
                debug!("Processing edit_file tool call");
                debug!("Raw tool_call.args: {:?}", tool_call.args);

                let file_path = tool_call.args.get("file_path");
                let start_line = tool_call.args.get("start_line");
                let end_line = tool_call.args.get("end_line");
                let new_text = tool_call.args.get("new_text");

                debug!("Extracted values - file_path: {:?}, start_line: {:?}, end_line: {:?}, new_text: {:?}",
                       file_path, start_line, end_line, new_text);

                if let (Some(path_val), Some(start_val), Some(end_val), Some(text_val)) =
                    (file_path, start_line, end_line, new_text)
                {
                    debug!("All required arguments present");
                    debug!(
                        "path_val: {:?}, start_val: {:?}, end_val: {:?}, text_val: {:?}",
                        path_val, start_val, end_val, text_val
                    );

                    if let (Some(path_str), Some(start_num), Some(end_num), Some(text_str)) = (
                        path_val.as_str(),
                        start_val.as_i64(),
                        end_val.as_i64(),
                        text_val.as_str(),
                    ) {
                        debug!("Successfully converted types - path: {}, start: {}, end: {}, text_len: {}",
                               path_str, start_num, end_num, text_str.len());

                        // Validate line numbers
                        if start_num < 1 || end_num < 1 || start_num > end_num {
                            return Ok("❌ Invalid line numbers: start_line and end_line must be >= 1 and start_line <= end_line".to_string());
                        }

                        // Read the current file content
                        let original_content = match std::fs::read_to_string(path_str) {
                            Ok(content) => content,
                            Err(e) => {
                                return Ok(format!("❌ Failed to read file '{}': {}", path_str, e))
                            }
                        };

                        let lines: Vec<&str> = original_content.lines().collect();
                        let total_lines = lines.len();
                        debug!("File has {} lines", total_lines);

                        // Convert to 0-based indexing
                        let start_idx = (start_num - 1) as usize;
                        let end_idx = (end_num - 1) as usize;
                        debug!(
                            "Using 0-based indices: start_idx={}, end_idx={}",
                            start_idx, end_idx
                        );

                        // Validate line ranges
                        if start_idx >= total_lines {
                            return Ok(format!(
                                "❌ start_line {} is beyond file length ({} lines)",
                                start_num, total_lines
                            ));
                        }
                        if end_idx >= total_lines {
                            return Ok(format!(
                                "❌ end_line {} is beyond file length ({} lines)",
                                end_num, total_lines
                            ));
                        }

                        // Split new_text into lines
                        let new_lines: Vec<&str> = if text_str.is_empty() {
                            vec![]
                        } else {
                            text_str.lines().collect()
                        };

                        let new_lines_count = new_lines.len();
                        debug!("New text has {} lines", new_lines_count);

                        // Create the new content
                        let mut new_content_lines = Vec::new();

                        // Add lines before the edit range
                        new_content_lines.extend_from_slice(&lines[..start_idx]);

                        // Add the new lines
                        new_content_lines.extend(new_lines);

                        // Add lines after the edit range
                        if end_idx + 1 < lines.len() {
                            new_content_lines.extend_from_slice(&lines[end_idx + 1..]);
                        }

                        // Join the lines back together
                        let new_content = new_content_lines.join("\n");
                        debug!("New content length: {} characters", new_content.len());

                        // Write the modified content back to the file
                        match std::fs::write(path_str, &new_content) {
                            Ok(()) => {
                                let old_range_size = end_idx - start_idx + 1;
                                Ok(format!("✅ Successfully edited '{}': replaced {} lines ({}:{}) with {} lines",
                                    path_str, old_range_size, start_num, end_num, new_lines_count))
                            }
                            Err(e) => Ok(format!(
                                "❌ Failed to write edited content to '{}': {}",
                                path_str, e
                            )),
                        }
                    } else {
                        debug!("Type conversion failed:");
                        debug!("  path_val.as_str(): {:?}", path_val.as_str());
                        debug!("  start_val.as_i64(): {:?}", start_val.as_i64());
                        debug!("  end_val.as_i64(): {:?}", end_val.as_i64());
                        debug!("  text_val.as_str(): {:?}", text_val.as_str());
                        Ok("❌ Invalid argument types: file_path must be string, start_line and end_line must be integers, new_text must be string".to_string())
                    }
                } else {
                    debug!("Missing required arguments:");
                    debug!("  file_path present: {}", file_path.is_some());
                    debug!("  start_line present: {}", start_line.is_some());
                    debug!("  end_line present: {}", end_line.is_some());
                    debug!("  new_text present: {}", new_text.is_some());
                    Ok(
                        "❌ Missing required arguments: file_path, start_line, end_line, new_text"
                            .to_string(),
                    )
                }
            }
            "final_output" => {
                if let Some(summary) = tool_call.args.get("summary") {
                    if let Some(summary_str) = summary.as_str() {
                        Ok(format!("{}", summary_str))
                    } else {
                        Ok("✅ Task completed".to_string())
                    }
                } else {
                    Ok("✅ Task completed".to_string())
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

// Helper function to filter JSON tool calls from display content
fn filter_json_tool_calls(content: &str) -> String {
    // Check if content contains any JSON tool call patterns
    let patterns = [
        r#"{"tool":"#,
        r#"{ "tool":"#,
        r#"{"tool" :"#,
        r#"{ "tool" :"#,
    ];
    
    // Check if any pattern is found in the content
    let has_tool_call_pattern = patterns.iter().any(|pattern| content.contains(pattern));
    
    if has_tool_call_pattern {
        // If we detect a JSON tool call pattern anywhere in the content, 
        // replace the entire content with the indicator
        "<<tool call detected>>".to_string()
    } else {
        // Check for partial JSON patterns that might be split across chunks
        let trimmed = content.trim();
        
        // Check for partial patterns that might indicate the start of a JSON tool call
        if trimmed.starts_with(r#"{"tool"#) ||
           trimmed.starts_with(r#"{ "tool"#) ||
           trimmed.starts_with(r#"{"#) && (trimmed.contains("tool") || trimmed.contains("args")) ||
           trimmed.contains(r#""tool":"#) ||
           trimmed.contains(r#""args":"#) ||
           trimmed.contains(r#"file_path"#) ||
           trimmed.contains(r#"command"#) ||
           (trimmed.starts_with('{') && trimmed.len() < 50 && (trimmed.contains("tool") || trimmed.contains("args"))) {
            // This looks like part of a JSON tool call, suppress it
            "".to_string()
        } else {
            // Regular content, return as-is
            content.to_string()
        }
    }
}

// Helper function to properly escape shell commands
fn shell_escape_command(command: &str) -> String {
    // Simple approach: if the command contains file paths with spaces,
    // we need to be more intelligent about escaping

    // For now, let's use a basic approach that handles common cases
    // This is a simplified version - a full implementation would use proper shell parsing

    let parts: Vec<&str> = command.split_whitespace().collect();
    if parts.is_empty() {
        return command.to_string();
    }

    let cmd = parts[0];
    let _args = &parts[1..];

    // Commands that typically take file paths as arguments
    let file_commands = [
        "cat", "ls", "cp", "mv", "rm", "chmod", "chown", "file", "head", "tail", "wc", "grep",
    ];

    if file_commands.contains(&cmd) {
        // For file commands, we need to be smarter about escaping
        // Let's use a different approach: use the original command but wrap it in quotes if needed

        // Check if the command already has proper quoting
        if command.contains('"') || command.contains('\'') {
            // Already has some quoting, use as-is
            return command.to_string();
        }

        // Look for file paths that need escaping (contain spaces but aren't quoted)
        let mut escaped_command = String::new();
        let mut in_quotes = false;
        let mut current_word = String::new();
        let mut words = Vec::new();

        for ch in command.chars() {
            match ch {
                ' ' if !in_quotes => {
                    if !current_word.is_empty() {
                        words.push(current_word.clone());
                        current_word.clear();
                    }
                }
                '"' => {
                    in_quotes = !in_quotes;
                    current_word.push(ch);
                }
                _ => {
                    current_word.push(ch);
                }
            }
        }

        if !current_word.is_empty() {
            words.push(current_word);
        }

        // Reconstruct the command with proper escaping
        for (i, word) in words.iter().enumerate() {
            if i > 0 {
                escaped_command.push(' ');
            }

            // If this word looks like a file path (contains / or ~) and has spaces, quote it
            if word.contains('/') || word.starts_with('~') {
                if word.contains(' ') && !word.starts_with('"') && !word.starts_with('\'') {
                    escaped_command.push_str(&format!("\"{}\"", word));
                } else {
                    escaped_command.push_str(word);
                }
            } else {
                escaped_command.push_str(word);
            }
        }

        escaped_command
    } else {
        // For non-file commands, use the original command
        command.to_string()
    }
}

// Helper function to fix nested quotes in shell command JSON
fn fix_nested_quotes_in_shell_command(json_str: &str) -> String {
    // This handles cases where shell commands contain nested quotes that break JSON parsing
    // Example: {"tool": "shell", "args": {"command": "python -c 'import os; print("hello")'"}}

    // Look for the pattern: "command": "
    if let Some(command_start) = json_str.find(r#""command": ""#) {
        let command_value_start = command_start + r#""command": ""#.len();

        // Find the end of the command string by looking for the pattern "}
        // We need to be careful about nested quotes
        if let Some(end_marker) = json_str[command_value_start..].find(r#"" }"#) {
            let command_end = command_value_start + end_marker;

            let before = &json_str[..command_value_start];
            let command_content = &json_str[command_value_start..command_end];
            let after = &json_str[command_end..];

            // Fix the command content by properly escaping quotes
            let mut fixed_command = String::new();
            let mut chars = command_content.chars().peekable();

            while let Some(ch) = chars.next() {
                match ch {
                    '"' => {
                        // Check if this quote is already escaped
                        if fixed_command.ends_with('\\') {
                            fixed_command.push(ch); // Already escaped, keep as-is
                        } else {
                            fixed_command.push_str(r#"\""#); // Escape the quote
                        }
                    }
                    '\\' => {
                        // Check what follows the backslash
                        if let Some(&_next_ch) = chars.peek() {
                            if _next_ch == '"' {
                                // This is an escaped quote, keep the backslash
                                fixed_command.push(ch);
                            } else {
                                // Regular backslash, escape it
                                fixed_command.push_str(r#"\\"#);
                            }
                        } else {
                            // Backslash at end, escape it
                            fixed_command.push_str(r#"\\"#);
                        }
                    }
                    _ => fixed_command.push(ch),
                }
            }

            return format!("{}{}{}", before, fixed_command, after);
        }
    }

    // Fallback: if we can't parse the structure, try some basic replacements
    json_str.to_string()
}

// Helper function to fix mixed quotes in JSON (single quotes where double quotes should be)
fn fix_mixed_quotes_in_json(json_str: &str) -> String {
    // This handles cases where the LLM uses single quotes in JSON values
    // Example: {"tool": "shell", "args": {"command": 'echo "hello"'}}

    let mut result = String::new();
    let mut chars = json_str.chars().peekable();
    let mut in_string = false;
    let mut string_delimiter = '"';

    while let Some(ch) = chars.next() {
        match ch {
            '"' if !in_string => {
                // Start of a double-quoted string
                in_string = true;
                string_delimiter = '"';
                result.push(ch);
            }
            '\'' if !in_string => {
                // Start of a single-quoted string - convert to double quotes
                in_string = true;
                string_delimiter = '\'';
                result.push('"'); // Convert single quote to double quote
            }
            c if in_string && c == string_delimiter => {
                // End of current string
                if string_delimiter == '\'' {
                    result.push('"'); // Convert single quote to double quote
                } else {
                    result.push(c);
                }
                in_string = false;
            }
            '"' if in_string && string_delimiter == '\'' => {
                // Double quote inside single-quoted string - escape it
                result.push_str(r#"\""#);
            }
            '\\' if in_string => {
                // Escape sequence - preserve it
                result.push(ch);
                if let Some(&_next_ch) = chars.peek() {
                    result.push(chars.next().unwrap());
                }
            }
            _ => {
                result.push(ch);
            }
        }
    }

    result
}

pub mod providers {
    pub mod embedded;
}
