use serde::{Deserialize, Serialize};
use anyhow::Result;
use std::collections::HashMap;

/// Trait for LLM providers
#[async_trait::async_trait]
pub trait LLMProvider: Send + Sync {
    /// Generate a completion for the given messages
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse>;
    
    /// Stream a completion for the given messages
    async fn stream(&self, request: CompletionRequest) -> Result<CompletionStream>;
    
    /// Get the provider name
    fn name(&self) -> &str;
    
    /// Get the model name
    fn model(&self) -> &str;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub messages: Vec<Message>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub stream: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub content: String,
    pub usage: Usage,
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

pub type CompletionStream = tokio_stream::wrappers::ReceiverStream<Result<CompletionChunk>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionChunk {
    pub content: String,
    pub finished: bool,
}

/// Provider registry for managing multiple LLM providers
pub struct ProviderRegistry {
    providers: HashMap<String, Box<dyn LLMProvider>>,
    default_provider: String,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            default_provider: String::new(),
        }
    }
    
    pub fn register<P: LLMProvider + 'static>(&mut self, provider: P) {
        let name = provider.name().to_string();
        self.providers.insert(name.clone(), Box::new(provider));
        
        if self.default_provider.is_empty() {
            self.default_provider = name;
        }
    }
    
    pub fn set_default(&mut self, provider_name: &str) -> Result<()> {
        if !self.providers.contains_key(provider_name) {
            anyhow::bail!("Provider '{}' not found", provider_name);
        }
        self.default_provider = provider_name.to_string();
        Ok(())
    }
    
    pub fn get(&self, provider_name: Option<&str>) -> Result<&dyn LLMProvider> {
        let name = provider_name.unwrap_or(&self.default_provider);
        self.providers
            .get(name)
            .map(|p| p.as_ref())
            .ok_or_else(|| anyhow::anyhow!("Provider '{}' not found", name))
    }
    
    pub fn list_providers(&self) -> Vec<&str> {
        self.providers.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}
