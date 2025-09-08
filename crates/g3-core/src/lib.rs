use anyhow::Result;
use g3_config::Config;
use g3_execution::CodeExecutor;
use g3_providers::{CompletionRequest, Message, MessageRole, ProviderRegistry};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;
use tracing::{error, field::debug, info};

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
        language: Option<&str>,
        _auto_execute: bool,
        show_prompt: bool,
        show_code: bool,
        show_timing: bool,
        cancellation_token: CancellationToken,
    ) -> Result<String> {
        info!("Executing task: {}", description);

        let provider = self.providers.get(None)?;

        let system_prompt = format!(
            "You are G3, a general-purpose AI agent. Your goal is to analyze and write code to solve given problems.

            G3 uses LLMs with tool calling capability.
            Tools allow external systems to provide context and data to G3. You solve higher level problems using
            tools, and can interact with multiple at once. When you want to perform an action, use 'I' as the pronoun.

# Available Tools
- shell:
    Execute a command in the shell.

    This will return the output and error concatenated into a single string, as
    you would see from running on the command line. There will also be an indication
    of if the command succeeded or failed.

    Avoid commands that produce a large amount of output, and consider piping those outputs to files.

    **Important**: Each shell command runs in its own process. Things like directory changes or
    sourcing files do not persist between tool calls. So you may need to repeat them each time by
    stringing together commands, e.g. `cd example && ls` or `source env/bin/activate && pip install numpy`

    Multiple commands: Use ; or && to chain commands, avoid newlines
    Pathnames: Use absolute paths and avoid cd unless explicitly requested

    Usage:
    - Call the `shell` tool with the desired bash/shell commands.

- search:
    Search the web for information about any topic.

- final_output:
    This tool signals the final output for a user in a conversation and MUST be used for the final message to the user. You must
    pass in a detailed summary of the work done to this tool call.

    Purpose:
    - Collects the final output for a user
    - Provides clear validation feedback when output isn't valid

    Usage:
    - Call the `final_output` tool with a summary of the work performed.

# Response Guidelines
- Use Markdown formatting for all responses.
- Follow best practices for Markdown, including:
    - Using headers for organization.
    - Bullet points for lists.
    - Links formatted correctly, either as linked text (e.g., [this is linked text](https://example.com)) or automatic links using angle brackets (e.g., <http://example.com/>).
- For code, use fenced code blocks by placing triple backticks (` ``` `) before and after the code. Include the language identifier after the opening backticks (e.g., ` ```python `) to enable syntax highlighting.
- Ensure clarity, conciseness, and proper formatting to enhance readability and usability.

IMPORTANT INSTRUCTIONS:

Please keep going until the user's query is completely resolved, before ending your turn and yielding back to the user.
Only terminate your turn when you are sure that the problem is solved.

If you are not sure about file content or codebase structure, or other information pertaining to the user's request,
use your tools to read files and gather the relevant information: do NOT guess or make up an answer. It is important
you use tools that can assist with providing the right context.
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
        let response_content = tokio::select! {
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

        // Time the code execution with cancellation support
        let exec_start = Instant::now();
        let executor = CodeExecutor::new();
        let result = tokio::select! {
            result = executor.execute_from_response_with_options(&response_content, show_code) => result?,
            _ = cancellation_token.cancelled() => {
                return Err(anyhow::anyhow!("Operation cancelled by user"));
            }
        };
        let exec_duration = exec_start.elapsed();

        if show_timing {
            let timing_summary = format!(
                "\n💭 {} | ⚡️ {}",
                Self::format_duration(llm_duration),
                Self::format_duration(exec_duration)
            );
            Ok(format!("{}\n{}", result, timing_summary))
        } else {
            Ok(result)
        }
    }

    pub fn get_context_window(&self) -> &ContextWindow {
        &self.context_window
    }

    async fn stream_completion(&self, request: CompletionRequest) -> Result<String> {
        use tokio_stream::StreamExt;

        let provider = self.providers.get(None)?;
        let mut stream = provider.stream(request).await?;

        let mut full_content = String::new();
        print!("🤖 "); // Start the response indicator
        use std::io::{self, Write};
        io::stdout().flush()?;

        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    print!("{}", chunk.content);
                    io::stdout().flush()?;
                    full_content.push_str(&chunk.content);

                    if chunk.finished {
                        break;
                    }
                }
                Err(e) => {
                    error!("Streaming error: {}", e);
                    return Err(e);
                }
            }
        }

        println!(); // New line after streaming completes
        Ok(full_content)
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
