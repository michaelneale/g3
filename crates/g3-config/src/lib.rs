use serde::{Deserialize, Serialize};
use anyhow::Result;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub providers: ProvidersConfig,
    pub agent: AgentConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvidersConfig {
    pub openai: Option<OpenAIConfig>,
    pub anthropic: Option<AnthropicConfig>,
    pub embedded: Option<EmbeddedConfig>,
    pub default_provider: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIConfig {
    pub api_key: String,
    pub model: String,
    pub base_url: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicConfig {
    pub api_key: String,
    pub model: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddedConfig {
    pub model_path: String,
    pub model_type: String, // e.g., "llama", "mistral", "codellama"
    pub context_length: Option<u32>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub gpu_layers: Option<u32>, // Number of layers to offload to GPU
    pub threads: Option<u32>,    // Number of CPU threads to use
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub max_context_length: usize,
    pub enable_streaming: bool,
    pub timeout_seconds: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            providers: ProvidersConfig {
                openai: None,
                anthropic: None,
                embedded: None,
                default_provider: "anthropic".to_string(),
            },
            agent: AgentConfig {
                max_context_length: 8192,
                enable_streaming: true,
                timeout_seconds: 60,
            },
        }
    }
}

impl Config {
    pub fn load(config_path: Option<&str>) -> Result<Self> {
        let mut settings = config::Config::builder();
        
        // Load default configuration
        settings = settings.add_source(config::Config::try_from(&Config::default())?);
        
        // Load from config file if provided
        if let Some(path) = config_path {
            if Path::new(path).exists() {
                settings = settings.add_source(config::File::with_name(path));
            }
        } else {
            // Try to load from default locations
            let default_paths = [
                "./g3.toml",
                "~/.config/g3/config.toml",
                "~/.g3.toml",
            ];
            
            for path in &default_paths {
                let expanded_path = shellexpand::tilde(path);
                if Path::new(expanded_path.as_ref()).exists() {
                    settings = settings.add_source(config::File::with_name(expanded_path.as_ref()));
                    break;
                }
            }
        }
        
        // Override with environment variables
        settings = settings.add_source(
            config::Environment::with_prefix("G3")
                .separator("_")
        );
        
        let config = settings.build()?.try_deserialize()?;
        Ok(config)
    }
    
    pub fn save(&self, path: &str) -> Result<()> {
        let toml_string = toml::to_string_pretty(self)?;
        std::fs::write(path, toml_string)?;
        Ok(())
    }
}
