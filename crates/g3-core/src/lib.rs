use anyhow::Result;
use g3_config::Config;
use g3_execution::CodeExecutor;
use g3_providers::{CompletionRequest, Message, MessageRole, ProviderRegistry};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::{Duration, Instant};
use tracing::field::debug;
use tracing::info;
use tokio_util::sync::CancellationToken;

pub struct Agent {
    providers: ProviderRegistry,
    config: Config,
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

        Ok(Self { providers, config })
    }

    pub async fn execute_task(
        &self,
        description: &str,
        language: Option<&str>,
        _auto_execute: bool,
    ) -> Result<String> {
        self.execute_task_with_options(description, language, false, false, false)
            .await
    }

    pub async fn execute_task_with_options(
        &self,
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
        &self,
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
        &self,
        description: &str,
        language: Option<&str>,
        _auto_execute: bool,
        show_prompt: bool,
        show_code: bool,
        show_timing: bool,
        cancellation_token: CancellationToken,
    ) -> Result<String> {
        info!("Executing task: {}", description);

        let total_start = Instant::now();

        let provider = self.providers.get(None)?;

        let system_prompt = format!(
            "You are G3, a code-first AI agent. Your goal is to solve problems by writing and executing code autonomously.

When given a task:
1. Analyze what needs to be done
2. Choose the most appropriate programming language{}
3. Include any necessary imports/dependencies
4. Add error handling where appropriate
5. EXECUTE the code immediately to solve the user's problem

Prefer these languages:
- Bash/Shell: File operations, system administration, simple tasks
- Python: Complex data processing, when libraries are needed
- Rust: Performance-critical tasks, system programming

Only use Rust/Python when you need libraries or complex logic that bash can't handle easily.

Format your response as:
```[language]
[code]
```

with nothing afterwards.",
            if let Some(lang) = language {
                format!(" (prefer {})", lang)
            } else {
                " based on the task type".to_string()
            }
        );

        if show_prompt {
            println!("🔍 System Prompt:");
            println!("================");
            println!("{}", system_prompt);
            println!("================");
            println!();
        }

        let messages = vec![
            Message {
                role: MessageRole::System,
                content: system_prompt,
            },
            Message {
                role: MessageRole::User,
                content: format!("Task: {}", description),
            },
        ];

        let request = CompletionRequest {
            messages,
            max_tokens: Some(2048),
            temperature: Some(0.2),
            stream: false,
        };

        // Time the LLM call with cancellation support
        let llm_start = Instant::now();
        let response = tokio::select! {
            result = provider.complete(request) => result?,
            _ = cancellation_token.cancelled() => {
                return Err(anyhow::anyhow!("Operation cancelled by user"));
            }
        };
        let llm_duration = llm_start.elapsed();

        // Time the code execution with cancellation support
        let exec_start = Instant::now();
        let executor = CodeExecutor::new();
        let result = tokio::select! {
            result = executor.execute_from_response_with_options(&response.content, show_code) => result?,
            _ = cancellation_token.cancelled() => {
                return Err(anyhow::anyhow!("Operation cancelled by user"));
            }
        };
        let exec_duration = exec_start.elapsed();

        let total_duration = total_start.elapsed();

        if show_timing {
            let timing_summary = format!(
                "\n{} [💡: {} ⚡️: {}]",
                Self::format_duration(total_duration),
                Self::format_duration(llm_duration),
                Self::format_duration(exec_duration)
            );
            Ok(format!("{}\n{}", result, timing_summary))
        } else {
            Ok(result)
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
