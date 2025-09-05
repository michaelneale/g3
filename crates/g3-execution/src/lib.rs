use anyhow::Result;
use regex::Regex;
use std::process::Command;
use tempfile::NamedTempFile;
use std::io::Write;
use tracing::{info, debug, error};

pub struct CodeExecutor {
    // Future: add configuration for execution limits, sandboxing, etc.
}

#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    pub success: bool,
}

impl CodeExecutor {
    pub fn new() -> Self {
        Self {}
    }
    
    /// Extract code blocks from LLM response and execute them
    pub async fn execute_from_response(&self, response: &str) -> Result<String> {
        self.execute_from_response_with_options(response, true).await
    }
    
    /// Extract code blocks from LLM response and execute them with UI options
    pub async fn execute_from_response_with_options(&self, response: &str, show_code: bool) -> Result<String> {
        debug!("CodeExecutor received response ({} chars): {}", response.len(), response);
        let code_blocks = self.extract_code_blocks(response)?;
        
        if code_blocks.is_empty() {
            if show_code {
                return Ok(format!("⚠️  No executable code blocks found in response.\n\n{}", response));
            } else {
                return Ok("⚠️  No executable code found.".to_string());
            }
        }
        
        let mut results = Vec::new();
        
        // Only show the original LLM response if show_code is true
        if show_code {
            results.push(response.to_string());
            results.push("\n🚀 Executing code...\n".to_string());
        }
        
        for (language, code) in code_blocks {
            info!("Executing {} code", language);
            
            if show_code {
                results.push(format!("📋 Running {} code:", language));
            }
            
            match self.execute_code(&language, &code).await {
                Ok(result) => {
                    if result.success {
                        if show_code {
                            results.push("✅ Success".to_string());
                        }
                        // Always show stdout if there is any, regardless of show_code
                        if !result.stdout.is_empty() {
                            results.push(result.stdout.trim().to_string());
                        }
                    } else {
                        results.push("❌ Failed".to_string());
                        if !result.stderr.is_empty() {
                            results.push(format!("Error: {}", result.stderr.trim()));
                        }
                    }
                }
                Err(e) => {
                    error!("Failed to execute {} code: {}", language, e);
                    results.push(format!("❌ Execution failed: {}", e));
                }
            }
        }
        
        // If no results were added (e.g., successful execution with no output), 
        // return a simple success message when show_code is false
        if results.is_empty() && !show_code {
            Ok("✅ Done".to_string())
        } else {
            Ok(results.join("\n"))
        }
    }
    
    /// Extract code blocks from markdown-formatted text
    fn extract_code_blocks(&self, text: &str) -> Result<Vec<(String, String)>> {
        let re = Regex::new(r"(?s)```(\w+)?\n(.*?)```")?;
        let mut blocks = Vec::new();
        
        debug!("Extracting code blocks from text: {}", text);
        
        for cap in re.captures_iter(text) {
            let language = cap.get(1)
                .map(|m| m.as_str().to_lowercase())
                .unwrap_or_else(|| "bash".to_string()); // Default to bash
            let code = cap.get(2).map(|m| m.as_str()).unwrap_or("").trim();
            
            debug!("Found code block - language: '{}', code: '{}'", language, code);
            
            if !code.is_empty() {
                blocks.push((language, code.to_string()));
            }
        }
        
        debug!("Total code blocks found: {}", blocks.len());
        Ok(blocks)
    }
    
    /// Execute code in the specified language
    async fn execute_code(&self, language: &str, code: &str) -> Result<ExecutionResult> {
        match language.to_lowercase().as_str() {
            "python" | "py" => self.execute_python(code).await,
            "bash" | "shell" | "sh" => self.execute_bash(code).await,
            "javascript" | "js" => self.execute_javascript(code).await,
            _ => {
                // Try to execute as bash by default
                debug!("Unknown language '{}', trying as bash", language);
                self.execute_bash(code).await
            }
        }
    }
    
    /// Execute Python code
    async fn execute_python(&self, code: &str) -> Result<ExecutionResult> {
        let mut temp_file = NamedTempFile::new()?;
        temp_file.write_all(code.as_bytes())?;
        let temp_path = temp_file.path();
        
        let output = Command::new("python3")
            .arg(temp_path)
            .output()?;
        
        Ok(ExecutionResult {
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            exit_code: output.status.code().unwrap_or(-1),
            success: output.status.success(),
        })
    }
    
    /// Execute Bash code
    async fn execute_bash(&self, code: &str) -> Result<ExecutionResult> {
        let output = Command::new("bash")
            .arg("-c")
            .arg(code)
            .output()?;
        
        Ok(ExecutionResult {
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            exit_code: output.status.code().unwrap_or(-1),
            success: output.status.success(),
        })
    }
    
    /// Execute JavaScript code (requires Node.js)
    async fn execute_javascript(&self, code: &str) -> Result<ExecutionResult> {
        let mut temp_file = NamedTempFile::new()?;
        temp_file.write_all(code.as_bytes())?;
        let temp_path = temp_file.path();
        
        let output = Command::new("node")
            .arg(temp_path)
            .output()?;
        
        Ok(ExecutionResult {
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            exit_code: output.status.code().unwrap_or(-1),
            success: output.status.success(),
        })
    }
}

impl Default for CodeExecutor {
    fn default() -> Self {
        Self::new()
    }
}
