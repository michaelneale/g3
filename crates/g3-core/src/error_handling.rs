//! Error handling module for G3 with retry logic and detailed logging
//!
//! This module provides:
//! - Classification of errors as recoverable or non-recoverable
//! - Retry logic with exponential backoff and jitter for recoverable errors
//! - Detailed error logging with context information
//! - Request/response capture for debugging

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{error, info, warn};

/// Maximum number of retry attempts for recoverable errors
const MAX_RETRY_ATTEMPTS: u32 = 3;

/// Base delay for exponential backoff (in milliseconds)
const BASE_RETRY_DELAY_MS: u64 = 1000;

/// Maximum delay between retries (in milliseconds)
const MAX_RETRY_DELAY_MS: u64 = 10000;

/// Jitter factor (0.0 to 1.0) to randomize retry delays
const JITTER_FACTOR: f64 = 0.3;

/// Error context information for detailed logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    /// The operation that was being performed
    pub operation: String,
    /// The provider being used
    pub provider: String,
    /// The model being used
    pub model: String,
    /// The last prompt sent (truncated for logging)
    pub last_prompt: String,
    /// Raw request data (if available)
    pub raw_request: Option<String>,
    /// Raw response data (if available)
    pub raw_response: Option<String>,
    /// Stack trace
    pub stack_trace: String,
    /// Timestamp
    pub timestamp: u64,
    /// Number of tokens in context
    pub context_tokens: u32,
    /// Session ID if available
    pub session_id: Option<String>,
}

impl ErrorContext {
    pub fn new(
        operation: String,
        provider: String,
        model: String,
        last_prompt: String,
        session_id: Option<String>,
        context_tokens: u32,
    ) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Capture stack trace
        let stack_trace = std::backtrace::Backtrace::force_capture().to_string();

        Self {
            operation,
            provider,
            model,
            last_prompt: truncate_for_logging(&last_prompt, 1000),
            raw_request: None,
            raw_response: None,
            stack_trace,
            timestamp,
            context_tokens,
            session_id,
        }
    }

    pub fn with_request(mut self, request: String) -> Self {
        self.raw_request = Some(truncate_for_logging(&request, 5000));
        self
    }

    pub fn with_response(mut self, response: String) -> Self {
        self.raw_response = Some(truncate_for_logging(&response, 5000));
        self
    }

    /// Log the error context with ERROR level
    pub fn log_error(&self, error: &anyhow::Error) {
        error!("=== G3 ERROR DETAILS ===");
        error!("Operation: {}", self.operation);
        error!("Provider: {} | Model: {}", self.provider, self.model);
        error!("Error: {}", error);
        error!("Timestamp: {}", self.timestamp);
        error!("Session ID: {:?}", self.session_id);
        error!("Context Tokens: {}", self.context_tokens);
        error!("Last Prompt: {}", self.last_prompt);
        
        if let Some(ref req) = self.raw_request {
            error!("Raw Request: {}", req);
        }
        
        if let Some(ref resp) = self.raw_response {
            error!("Raw Response: {}", resp);
        }
        
        error!("Stack Trace:\n{}", self.stack_trace);
        error!("=== END ERROR DETAILS ===");

        // Also save to error log file
        self.save_to_file();
    }

    /// Save error context to a file for later analysis
    fn save_to_file(&self) {
        let logs_dir = std::path::Path::new("logs/errors");
        if !logs_dir.exists() {
            if let Err(e) = std::fs::create_dir_all(logs_dir) {
                error!("Failed to create error logs directory: {}", e);
                return;
            }
        }

        let filename = format!(
            "logs/errors/error_{}_{}.json",
            self.timestamp,
            self.session_id.as_deref().unwrap_or("unknown")
        );

        match serde_json::to_string_pretty(self) {
            Ok(json_content) => {
                if let Err(e) = std::fs::write(&filename, json_content) {
                    error!("Failed to save error context to {}: {}", filename, e);
                } else {
                    info!("Error details saved to: {}", filename);
                }
            }
            Err(e) => {
                error!("Failed to serialize error context: {}", e);
            }
        }
    }
}

/// Classification of error types
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorType {
    /// Recoverable errors that should be retried
    Recoverable(RecoverableError),
    /// Non-recoverable errors that should terminate execution
    NonRecoverable,
}

/// Types of recoverable errors
#[derive(Debug, Clone, PartialEq)]
pub enum RecoverableError {
    /// Rate limit exceeded
    RateLimit,
    /// Temporary network error
    NetworkError,
    /// Server error (5xx)
    ServerError,
    /// Model is busy/overloaded
    ModelBusy,
    /// Timeout
    Timeout,
    /// Token limit exceeded (might be recoverable with summarization)
    TokenLimit,
}

/// Classify an error as recoverable or non-recoverable
pub fn classify_error(error: &anyhow::Error) -> ErrorType {
    let error_str = error.to_string().to_lowercase();

    // Check for recoverable error patterns
    if error_str.contains("rate limit") || error_str.contains("rate_limit") || error_str.contains("429") {
        return ErrorType::Recoverable(RecoverableError::RateLimit);
    }

    if error_str.contains("network") || error_str.contains("connection") || 
       error_str.contains("dns") || error_str.contains("refused") {
        return ErrorType::Recoverable(RecoverableError::NetworkError);
    }

    if error_str.contains("500") || error_str.contains("502") || 
       error_str.contains("503") || error_str.contains("504") ||
       error_str.contains("server error") || error_str.contains("internal error") {
        return ErrorType::Recoverable(RecoverableError::ServerError);
    }

    if error_str.contains("busy") || error_str.contains("overloaded") || 
       error_str.contains("capacity") || error_str.contains("unavailable") {
        return ErrorType::Recoverable(RecoverableError::ModelBusy);
    }

    if error_str.contains("timeout") || error_str.contains("timed out") {
        return ErrorType::Recoverable(RecoverableError::Timeout);
    }

    if error_str.contains("token") && (error_str.contains("limit") || error_str.contains("exceeded")) {
        return ErrorType::Recoverable(RecoverableError::TokenLimit);
    }

    // Default to non-recoverable for unknown errors
    ErrorType::NonRecoverable
}

/// Calculate retry delay with exponential backoff and jitter
pub fn calculate_retry_delay(attempt: u32) -> Duration {
    use rand::Rng;
    
    // Exponential backoff: delay = base * 2^attempt
    let base_delay = BASE_RETRY_DELAY_MS * (2_u64.pow(attempt.saturating_sub(1)));
    let capped_delay = base_delay.min(MAX_RETRY_DELAY_MS);
    
    // Add jitter to prevent thundering herd
    let mut rng = rand::thread_rng();
    let jitter = (capped_delay as f64 * JITTER_FACTOR * rng.gen::<f64>()) as u64;
    let final_delay = if rng.gen_bool(0.5) {
        capped_delay + jitter
    } else {
        capped_delay.saturating_sub(jitter)
    };
    
    Duration::from_millis(final_delay)
}

/// Retry logic for async operations
pub async fn retry_with_backoff<F, Fut, T>(
    operation_name: &str,
    mut operation: F,
    context: &ErrorContext,
) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let mut attempt = 0;
    let mut _last_error = None;

    loop {
        attempt += 1;
        
        match operation().await {
            Ok(result) => {
                if attempt > 1 {
                    info!(
                        "Operation '{}' succeeded after {} attempts",
                        operation_name, attempt
                    );
                }
                return Ok(result);
            }
            Err(error) => {
                let error_type = classify_error(&error);
                
                match error_type {
                    ErrorType::Recoverable(recoverable_type) => {
                        if attempt >= MAX_RETRY_ATTEMPTS {
                            error!(
                                "Operation '{}' failed after {} attempts. Giving up.",
                                operation_name, attempt
                            );
                            context.clone().log_error(&error);
                            return Err(error);
                        }
                        
                        let delay = calculate_retry_delay(attempt);
                        warn!(
                            "Recoverable error ({:?}) in '{}' (attempt {}/{}). Retrying in {:?}...",
                            recoverable_type, operation_name, attempt, MAX_RETRY_ATTEMPTS, delay
                        );
                        warn!("Error details: {}", error);
                        
                        // Special handling for token limit errors
                        if matches!(recoverable_type, RecoverableError::TokenLimit) {
                            info!("Token limit error detected. Consider triggering summarization.");
                        }
                        
                        tokio::time::sleep(delay).await;
                        _last_error = Some(error);
                    }
                    ErrorType::NonRecoverable => {
                        error!(
                            "Non-recoverable error in '{}' (attempt {}). Terminating.",
                            operation_name, attempt
                        );
                        context.clone().log_error(&error);
                        return Err(error);
                    }
                }
            }
        }
    }
}

/// Helper function to truncate strings for logging
fn truncate_for_logging(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}... (truncated, {} total chars)", &s[..max_len], s.len())
    }
}

/// Macro for creating error context easily
#[macro_export]
macro_rules! error_context {
    ($operation:expr, $provider:expr, $model:expr, $prompt:expr, $session_id:expr, $tokens:expr) => {
        $crate::error_handling::ErrorContext::new(
            $operation.to_string(),
            $provider.to_string(),
            $model.to_string(),
            $prompt.to_string(),
            $session_id,
            $tokens,
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_classification() {
        // Rate limit errors
        let error = anyhow!("Rate limit exceeded");
        assert_eq!(classify_error(&error), ErrorType::Recoverable(RecoverableError::RateLimit));
        
        let error = anyhow!("HTTP 429 Too Many Requests");
        assert_eq!(classify_error(&error), ErrorType::Recoverable(RecoverableError::RateLimit));
        
        // Network errors
        let error = anyhow!("Network connection failed");
        assert_eq!(classify_error(&error), ErrorType::Recoverable(RecoverableError::NetworkError));
        
        // Server errors
        let error = anyhow!("HTTP 503 Service Unavailable");
        assert_eq!(classify_error(&error), ErrorType::Recoverable(RecoverableError::ServerError));
        
        // Model busy
        let error = anyhow!("Model is busy, please try again");
        assert_eq!(classify_error(&error), ErrorType::Recoverable(RecoverableError::ModelBusy));
        
        // Timeout
        let error = anyhow!("Request timed out");
        assert_eq!(classify_error(&error), ErrorType::Recoverable(RecoverableError::Timeout));
        
        // Token limit
        let error = anyhow!("Token limit exceeded");
        assert_eq!(classify_error(&error), ErrorType::Recoverable(RecoverableError::TokenLimit));
        
        // Non-recoverable
        let error = anyhow!("Invalid API key");
        assert_eq!(classify_error(&error), ErrorType::NonRecoverable);
        
        let error = anyhow!("Malformed request");
        assert_eq!(classify_error(&error), ErrorType::NonRecoverable);
    }

    #[test]
    fn test_retry_delay_calculation() {
        // Test that delays increase exponentially
        let delay1 = calculate_retry_delay(1);
        let delay2 = calculate_retry_delay(2);
        let delay3 = calculate_retry_delay(3);
        
        // Due to jitter, we can't test exact values, but the base should increase
        assert!(delay1.as_millis() >= (BASE_RETRY_DELAY_MS as f64 * 0.7) as u128);
        assert!(delay1.as_millis() <= (BASE_RETRY_DELAY_MS as f64 * 1.3) as u128);
        
        // Delay 2 should be roughly 2x delay 1 (minus jitter)
        assert!(delay2.as_millis() >= delay1.as_millis());
        
        // Delay 3 should be roughly 2x delay 2 (minus jitter)
        assert!(delay3.as_millis() >= delay2.as_millis());
        
        // Test max cap
        let delay_max = calculate_retry_delay(10);
        assert!(delay_max.as_millis() <= (MAX_RETRY_DELAY_MS as f64 * 1.3) as u128);
    }

    #[test]
    fn test_truncate_for_logging() {
        let short_text = "Hello, world!";
        assert_eq!(truncate_for_logging(short_text, 20), "Hello, world!");
        
        let long_text = "This is a very long text that should be truncated for logging purposes";
        let truncated = truncate_for_logging(long_text, 20);
        assert!(truncated.starts_with("This is a very long "));
        assert!(truncated.contains("truncated"));
        assert!(truncated.contains("total chars"));
    }
}
