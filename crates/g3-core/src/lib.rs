use g3_providers::{ProviderRegistry, CompletionRequest, Message, MessageRole};
use g3_config::Config;
use g3_execution::CodeExecutor;
use anyhow::Result;
use serde::{Serialize, Deserialize};
use std::path::Path;
use std::time::{Duration, Instant};
use tracing::info;

pub struct Agent {
    providers: ProviderRegistry,
    config: Config,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub summary: String,
    pub issues: Vec<Issue>,
    pub suggestions: Vec<Suggestion>,
    pub metrics: CodeMetrics,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Issue {
    pub severity: IssueSeverity,
    pub message: String,
    pub line: Option<u32>,
    pub column: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IssueSeverity {
    Error,
    Warning,
    Info,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Suggestion {
    pub description: String,
    pub code_example: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CodeMetrics {
    pub lines_of_code: u32,
    pub complexity_score: f32,
    pub maintainability_index: f32,
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
        
        // Set default provider
        providers.set_default(&config.providers.default_provider)?;
        
        Ok(Self { providers, config })
    }
    
    pub async fn analyze(&self, path: &str) -> Result<AnalysisResult> {
        info!("Analyzing path: {}", path);
        
        let content = self.read_file_or_directory(path)?;
        let provider = self.providers.get(None)?;
        
        let messages = vec![
            Message {
                role: MessageRole::System,
                content: "You are a code analysis expert. Analyze the provided code and return a detailed analysis including issues, suggestions, and metrics.".to_string(),
            },
            Message {
                role: MessageRole::User,
                content: format!("Please analyze this code:\n\n{}", content),
            },
        ];
        
        let request = CompletionRequest {
            messages,
            max_tokens: Some(2048),
            temperature: Some(0.1),
            stream: false,
        };
        
        let response = provider.complete(request).await?;
        
        // For now, return a simplified analysis
        // In a real implementation, we'd parse the LLM response into structured data
        Ok(AnalysisResult {
            summary: response.content,
            issues: vec![],
            suggestions: vec![],
            metrics: CodeMetrics {
                lines_of_code: content.lines().count() as u32,
                complexity_score: 1.0,
                maintainability_index: 85.0,
            },
        })
    }
    
    pub async fn generate(&self, description: &str) -> Result<String> {
        info!("Generating code for: {}", description);
        
        let provider = self.providers.get(None)?;
        
        let messages = vec![
            Message {
                role: MessageRole::System,
                content: "You are a code generation expert. Generate clean, well-documented code based on the user's description.".to_string(),
            },
            Message {
                role: MessageRole::User,
                content: description.to_string(),
            },
        ];
        
        let request = CompletionRequest {
            messages,
            max_tokens: Some(2048),
            temperature: Some(0.2),
            stream: false,
        };
        
        let response = provider.complete(request).await?;
        Ok(response.content)
    }
    
    pub async fn review(&self, path: &str) -> Result<String> {
        info!("Reviewing path: {}", path);
        
        let content = self.read_file_or_directory(path)?;
        let provider = self.providers.get(None)?;
        
        let messages = vec![
            Message {
                role: MessageRole::System,
                content: "You are a code review expert. Review the provided code and suggest improvements focusing on best practices, performance, and maintainability.".to_string(),
            },
            Message {
                role: MessageRole::User,
                content: format!("Please review this code:\n\n{}", content),
            },
        ];
        
        let request = CompletionRequest {
            messages,
            max_tokens: Some(2048),
            temperature: Some(0.1),
            stream: false,
        };
        
        let response = provider.complete(request).await?;
        Ok(response.content)
    }
    
    pub async fn execute_task(&self, description: &str, language: Option<&str>, _auto_execute: bool) -> Result<String> {
        self.execute_task_with_options(description, language, false, false, false).await
    }
    
    pub async fn execute_task_with_options(&self, description: &str, language: Option<&str>, _auto_execute: bool, show_prompt: bool, show_code: bool) -> Result<String> {
        self.execute_task_with_timing(description, language, _auto_execute, show_prompt, show_code, false).await
    }
    
    pub async fn execute_task_with_timing(&self, description: &str, language: Option<&str>, _auto_execute: bool, show_prompt: bool, show_code: bool, show_timing: bool) -> Result<String> {
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

Prefer these languages for different tasks (in order of preference):
- Bash/Shell: File operations, system administration, simple tasks, process management, text processing
- Python: Complex data processing, web scraping, APIs, when libraries are needed
- Rust: Performance-critical tasks, system programming

For simple tasks like listing files, checking processes, basic text manipulation, etc. - prefer bash/shell.
Only use Rust/Python when you need libraries or complex logic that bash can't handle easily.

Format your response as:
```[language]
[code]
```
Then execute it and show the output.",
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
        
        // Time the LLM call
        let llm_start = Instant::now();
        let response = provider.complete(request).await?;
        let llm_duration = llm_start.elapsed();
        
        // Time the code execution
        let exec_start = Instant::now();
        let executor = CodeExecutor::new();
        let result = executor.execute_from_response_with_options(&response.content, show_code).await?;
        let exec_duration = exec_start.elapsed();
        
        let total_duration = total_start.elapsed();
        
        if show_timing {
            let timing_summary = format!(
                "\n⏱️  Task Summary:\n   LLM call: {}\n   Code execution: {}\n   Total time: {}",
                Self::format_duration(llm_duration),
                Self::format_duration(exec_duration),
                Self::format_duration(total_duration)
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
    
    pub async fn create_automation(&self, workflow: &str) -> Result<String> {
        info!("Creating automation for: {}", workflow);
        
        let provider = self.providers.get(None)?;
        
        let messages = vec![
            Message {
                role: MessageRole::System,
                content: "You are G3, a code-first AI agent. Create automation scripts that can be saved and reused. Focus on creating robust, well-documented scripts with error handling and logging.".to_string(),
            },
            Message {
                role: MessageRole::User,
                content: format!("Create an automation script for: {}", workflow),
            },
        ];
        
        let request = CompletionRequest {
            messages,
            max_tokens: Some(2048),
            temperature: Some(0.1),
            stream: false,
        };
        
        let response = provider.complete(request).await?;
        Ok(response.content)
    }
    
    pub async fn process_data(&self, operation: &str, input_file: Option<&str>) -> Result<String> {
        info!("Processing data: {}", operation);
        
        let provider = self.providers.get(None)?;
        
        let context = if let Some(file) = input_file {
            format!("Operation: {}\nInput file: {}", operation, file)
        } else {
            format!("Operation: {}", operation)
        };
        
        let messages = vec![
            Message {
                role: MessageRole::System,
                content: "You are G3, a code-first AI agent specializing in data processing. Write Python code using pandas, numpy, or other appropriate libraries to process and analyze data. Always include data validation and error handling.".to_string(),
            },
            Message {
                role: MessageRole::User,
                content: context,
            },
        ];
        
        let request = CompletionRequest {
            messages,
            max_tokens: Some(2048),
            temperature: Some(0.1),
            stream: false,
        };
        
        let response = provider.complete(request).await?;
        Ok(response.content)
    }
    
    pub async fn execute_web_task(&self, task: &str, url: Option<&str>) -> Result<String> {
        info!("Executing web task: {}", task);
        
        let provider = self.providers.get(None)?;
        
        let context = if let Some(url) = url {
            format!("Task: {}\nURL: {}", task, url)
        } else {
            format!("Task: {}", task)
        };
        
        let messages = vec![
            Message {
                role: MessageRole::System,
                content: "You are G3, a code-first AI agent for web tasks. Write code for web scraping, API calls, downloads, or web automation. Use appropriate libraries like requests, BeautifulSoup, selenium, or similar. Always respect robots.txt and rate limits.".to_string(),
            },
            Message {
                role: MessageRole::User,
                content: context,
            },
        ];
        
        let request = CompletionRequest {
            messages,
            max_tokens: Some(2048),
            temperature: Some(0.2),
            stream: false,
        };
        
        let response = provider.complete(request).await?;
        Ok(response.content)
    }
    
    pub async fn execute_file_operation(&self, operation: &str, path: Option<&str>) -> Result<String> {
        info!("Executing file operation: {}", operation);
        
        let provider = self.providers.get(None)?;
        
        let context = if let Some(path) = path {
            format!("Operation: {}\nPath: {}", operation, path)
        } else {
            format!("Operation: {}", operation)
        };
        
        let messages = vec![
            Message {
                role: MessageRole::System,
                content: "You are G3, a code-first AI agent for file operations. Write scripts (bash, Python, or other appropriate languages) for file management, organization, backup, compression, format conversion, etc. Always include safety checks and confirmation prompts for destructive operations.".to_string(),
            },
            Message {
                role: MessageRole::User,
                content: context,
            },
        ];
        
        let request = CompletionRequest {
            messages,
            max_tokens: Some(2048),
            temperature: Some(0.1),
            stream: false,
        };
        
        let response = provider.complete(request).await?;
        Ok(response.content)
    }
    
    fn read_file_or_directory(&self, path: &str) -> Result<String> {
        let path = Path::new(path);
        
        if path.is_file() {
            Ok(std::fs::read_to_string(path)?)
        } else if path.is_dir() {
            // For directories, read multiple files and combine them
            let mut content = String::new();
            self.read_directory_recursive(path, &mut content)?;
            Ok(content)
        } else {
            anyhow::bail!("Path does not exist: {}", path.display())
        }
    }
    
    fn read_directory_recursive(&self, dir: &Path, content: &mut String) -> Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    // Only read common code files
                    if matches!(ext.to_str(), Some("rs" | "py" | "js" | "ts" | "go" | "java" | "cpp" | "c" | "h")) {
                        content.push_str(&format!("\n--- {} ---\n", path.display()));
                        if let Ok(file_content) = std::fs::read_to_string(&path) {
                            content.push_str(&file_content);
                        }
                    }
                }
            } else if path.is_dir() && !path.file_name().unwrap().to_str().unwrap().starts_with('.') {
                self.read_directory_recursive(&path, content)?;
            }
        }
        Ok(())
    }
}

impl std::fmt::Display for AnalysisResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Code Analysis Results")?;
        writeln!(f, "====================")?;
        writeln!(f)?;
        writeln!(f, "Summary:")?;
        writeln!(f, "{}", self.summary)?;
        writeln!(f)?;
        writeln!(f, "Metrics:")?;
        writeln!(f, "- Lines of Code: {}", self.metrics.lines_of_code)?;
        writeln!(f, "- Complexity Score: {:.2}", self.metrics.complexity_score)?;
        writeln!(f, "- Maintainability Index: {:.2}", self.metrics.maintainability_index)?;
        Ok(())
    }
}

pub mod providers {
    pub mod openai;
    pub mod anthropic;
}
