# Analysis: Lifting Retry Logic Above Provider Level

## Current State

### Retry Implementation in Providers

1. **Anthropic Provider** (`anthropic.rs`)
   - Has comprehensive retry logic with exponential backoff
   - Constants defined:
     - `MAX_RETRIES: u32 = 3`
     - `INITIAL_RETRY_DELAY_MS: u64 = 1000`
     - `MAX_RETRY_DELAY_MS: u64 = 10000`
     - `RETRY_MULTIPLIER: u64 = 2`
   - Generic `execute_with_retry` method that handles:
     - Timeout errors
     - Connection errors (reset, closed)
     - HTTP status codes (429, 502, 503, 504)
   - Applies retry to both `complete()` and `stream()` operations

2. **Embedded Provider** (`embedded.rs`)
   - Has simple retry logic for session lock acquisition
   - Retries up to 5 times with increasing delays (100ms * attempt)
   - No HTTP retry logic (doesn't make network calls)
   - Retry is specific to resource contention (session lock)

3. **Databricks Provider** (`databricks.rs`)
   - No retry logic implemented
   - Only has timeout configuration (`DEFAULT_TIMEOUT_SECS`)
   - Would benefit from retry logic for network operations

## Benefits of Lifting Retry Logic

### 1. **Consistency Across Providers**
- All providers would have the same retry behavior
- Uniform handling of transient failures
- Predictable behavior for users regardless of provider choice

### 2. **Centralized Configuration**
- Single place to configure retry parameters
- Easy to adjust retry behavior globally
- Could expose retry configuration to users via API

### 3. **Reduced Code Duplication**
- No need to implement retry logic in each provider
- Easier maintenance and bug fixes
- Consistent error handling patterns

### 4. **Provider-Agnostic Retry Policies**
- Could implement different retry strategies (exponential, linear, circuit breaker)
- Easy to add new retry conditions without modifying providers
- Better separation of concerns

### 5. **Enhanced Observability**
- Centralized logging and metrics for retries
- Easier to track retry patterns across all providers
- Better debugging capabilities

## Implementation Approaches

### Option 1: Wrapper Trait Implementation
```rust
// Create a retry wrapper that implements LLMProvider
pub struct RetryProvider<P: LLMProvider> {
    inner: P,
    config: RetryConfig,
}

impl<P: LLMProvider> LLMProvider for RetryProvider<P> {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        retry_with_backoff(
            || self.inner.complete(request.clone()),
            &self.config
        ).await
    }
    
    async fn stream(&self, request: CompletionRequest) -> Result<CompletionStream> {
        // Streaming retry is more complex, might need special handling
        retry_with_backoff(
            || self.inner.stream(request.clone()),
            &self.config
        ).await
    }
}
```

### Option 2: Middleware Pattern
```rust
// Add retry as middleware in the provider registry
impl ProviderRegistry {
    pub fn register_with_retry<P: LLMProvider + 'static>(
        &mut self, 
        provider: P,
        retry_config: RetryConfig
    ) {
        let wrapped = RetryProvider::new(provider, retry_config);
        self.register(wrapped);
    }
}
```

### Option 3: Built-in Registry Support
```rust
// Make retry a first-class feature of the registry
pub struct ProviderRegistry {
    providers: HashMap<String, Box<dyn LLMProvider>>,
    retry_configs: HashMap<String, RetryConfig>,
    default_provider: String,
}

impl ProviderRegistry {
    pub async fn complete_with_retry(
        &self, 
        provider_name: Option<&str>,
        request: CompletionRequest
    ) -> Result<CompletionResponse> {
        let provider = self.get(provider_name)?;
        let config = self.get_retry_config(provider_name);
        
        retry_with_backoff(
            || provider.complete(request.clone()),
            config
        ).await
    }
}
```

## Challenges and Considerations

### 1. **Provider-Specific Error Handling**
- Different providers may have different retryable error conditions
- Some errors should never be retried (e.g., authentication failures)
- Need a way for providers to indicate which errors are retryable

### 2. **Streaming Complexity**
- Streaming responses are harder to retry
- Once streaming starts, can't easily restart from the beginning
- May need to buffer initial response to enable retry
- Connection drops mid-stream need special handling

### 3. **Resource Management**
- Embedded provider's session lock retry is different from network retry
- Need to distinguish between resource contention and network failures
- Some providers might have their own internal retry (avoid double retry)

### 4. **Configuration Flexibility**
- Different providers might need different retry configurations
- Some operations might need different retry behavior (e.g., streaming vs non-streaming)
- Users might want to override retry behavior per-request

### 5. **Backward Compatibility**
- Need to maintain compatibility with existing code
- Providers with existing retry logic need migration path
- Should be able to disable centralized retry for specific providers

## Recommendations

### Short Term (Minimal Change)
1. **Extract Retry Logic to Shared Module**
   - Create a `retry` module in `g3-providers`
   - Move Anthropic's `execute_with_retry` to shared module
   - Make it generic and reusable by all providers
   - Providers can opt-in to use shared retry logic

### Medium Term (Recommended)
2. **Implement Wrapper Pattern**
   - Create `RetryProvider<P>` wrapper
   - Allow providers to declare retryable errors via trait method
   - Make retry configurable per provider
   - Gradually migrate providers to remove internal retry logic

### Long Term (Full Solution)
3. **Registry-Level Retry Management**
   - Build retry into the `ProviderRegistry`
   - Support per-provider and per-request retry configuration
   - Implement circuit breaker pattern for repeated failures
   - Add comprehensive retry metrics and observability

## Proposed Implementation Plan

### Phase 1: Shared Retry Module
```rust
// crates/g3-providers/src/retry.rs
pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
    pub multiplier: f64,
}

pub async fn execute_with_retry<F, Fut, T>(
    operation: F,
    config: &RetryConfig,
) -> Result<T>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    // Implementation here
}

pub fn is_retryable_error(error: &anyhow::Error) -> bool {
    // Common retryable error detection
}
```

### Phase 2: Provider Trait Extension
```rust
pub trait LLMProvider: Send + Sync {
    // ... existing methods ...
    
    /// Determine if an error is retryable for this provider
    fn is_retryable(&self, error: &anyhow::Error) -> bool {
        // Default implementation uses common logic
        is_retryable_error(error)
    }
    
    /// Get recommended retry configuration for this provider
    fn retry_config(&self) -> Option<RetryConfig> {
        None // Providers can override to provide defaults
    }
}
```

### Phase 3: Wrapper Implementation
```rust
pub struct RetryProvider<P: LLMProvider> {
    inner: P,
    config: RetryConfig,
}

impl<P: LLMProvider> RetryProvider<P> {
    pub fn new(provider: P, config: Option<RetryConfig>) -> Self {
        let config = config.unwrap_or_else(|| {
            provider.retry_config().unwrap_or_default()
        });
        Self { inner: provider, config }
    }
}
```

## Conclusion

Lifting retry logic above the provider level offers significant benefits:
- **Consistency**: Uniform retry behavior across all providers
- **Maintainability**: Single implementation to maintain and test
- **Flexibility**: Easy to configure and customize retry behavior
- **Extensibility**: Simple to add new retry strategies or conditions

The recommended approach is to:
1. Start with a shared retry module that providers can use
2. Gradually migrate to a wrapper-based approach
3. Eventually integrate retry management into the provider registry

This migration can be done incrementally without breaking existing code, allowing for a smooth transition while immediately providing value to providers that currently lack retry logic.