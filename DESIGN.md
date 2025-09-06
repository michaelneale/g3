# G3 General Purpose AI Agent - Design Document

## Overview
G3 is a **code-first AI agent** that helps you complete tasks by writing and executing code or scripts. Instead of just giving advice, G3 solves problems by generating executable code in the appropriate language.

## Core Principles
1. **Code-First Philosophy**: Always try to solve problems with executable code
2. **Multi-Language Support**: Generate scripts in Python, Bash, JavaScript, Rust, etc.
3. **Unix Philosophy**: Small, focused tools that do one thing well
4. **Modularity**: Clear separation of concerns
5. **Composability**: Components can be combined in different ways
6. **Performance**: Blazing fast execution

## Architecture

### High-Level Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI Module    │    │  Core Engine    │    │ LLM Providers   │
│                 │    │                 │    │                 │
│ - Task commands │◄──►│ - Task          │◄──►│ - OpenAI        │
│ - Interactive   │    │   interpretation│    │ - Anthropic     │
│   mode          │    │ - Code          │    │ - Embedded      │
│ - Code exec     │    │   generation    │    │   (llama.cpp)   │
│   approval      │    │ - Script        │    │ - Custom APIs   │
│                 │    │   execution     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Execution     │
                    │   Engine        │
                    │                 │
                    │ - Python        │
                    │ - Bash/Shell    │
                    │ - JavaScript    │
                    │ - Rust          │
                    │ - Sandboxing    │
                    └─────────────────┘
```

### Module Breakdown

#### 1. CLI Module (`g3-cli`)
- **Responsibility**: User interface and task interpretation
- **New Features**:
  - Progress indicators for script execution

#### 2. Core Engine (`g3-core`)
- **Responsibility**: Task interpretation and code generation
- **New Features**:
  - Task analysis and decomposition
  - Language selection based on task type
  - Code generation with execution context
  - Script template system
  - Autonomous execution of generated code

#### 3. LLM Providers (`g3-providers`)
- **Responsibility**: LLM communication and model abstraction
- **Supported Providers**:
  - **OpenAI**: GPT-4, GPT-3.5-turbo via API
  - **Anthropic**: Claude models via API  
  - **Embedded**: Local open-weights models via llama.cpp
- **Enhanced Prompts**:
  - Code-first system prompts
  - Language-specific generation instructions

#### 5. Embedded Provider (`g3-core/providers/embedded`) - NEW
- **Responsibility**: Local model inference using llama.cpp
- **Features**:
  - GGUF model support (Llama, CodeLlama, Mistral, etc.)
  - GPU acceleration via CUDA/Metal
  - Configurable context length and generation parameters
  - Async-compatible inference without blocking
  - Thread-safe model access
  - Stop sequence detection

#### 4. Execution Engine (`g3-execution`) - NEW
- **Responsibility**: Safe code execution
- **Features**:
  - Multi-language script execution
  - Sandboxing and security
  - Resource limits
  - Output capture and formatting
  - Error handling and recovery

### Task Types and Language Selection

| Task Type | Preferred Language | Use Cases |
|-----------|-------------------|-----------|
| Data Processing | Python | CSV/JSON analysis, data transformation |
| File Operations | Bash/Shell | File manipulation, backups, organization |
| System Admin | Bash/Shell | Process management, system monitoring |
| Text Processing | Python/Bash | Log analysis, text transformation |
| Database | Python/SQL | Data migration, queries, reporting |
| Image/Media | Python | Image processing, format conversion |
| Development | Rust | Code generation, project setup |

## Implementation Plan

### Phase 1: Core Refactoring ✅
1. ✅ Update CLI commands for task-oriented interface
2. ✅ Enhance system prompts for code-first approach
3. ✅ Add basic code execution capabilities
4. ✅ Update interactive mode messaging

### Phase 2: Enhanced Provider Support ✅
1. ✅ Implement embedded model provider using llama.cpp
2. ✅ Add GGUF model support for local inference
3. ✅ Configure GPU acceleration and performance optimization
4. ✅ Add comprehensive logging and debugging support

### Phase 3: Advanced Features (Future)
1. Model quantization and optimization
2. Multi-model ensemble support
3. Advanced code execution sandboxing
4. Plugin system for custom providers
5. Web interface for remote access

## Provider Comparison

| Feature | OpenAI | Anthropic | Embedded |
|---------|--------|-----------|----------|
| **Cost** | Pay per token | Pay per token | Free after download |
| **Privacy** | Data sent to API | Data sent to API | Completely local |
| **Performance** | Very fast | Very fast | Depends on hardware |
| **Model Quality** | Excellent | Excellent | Good (varies by model) |
| **Offline Support** | No | No | Yes |
| **Setup Complexity** | API key only | API key only | Model download required |
| **Hardware Requirements** | None | None | 4-16GB RAM, optional GPU |

## Configuration Examples

### Cloud-First Setup
```toml
[providers]
default_provider = "openai"

[providers.openai]
api_key = "sk-..."
model = "gpt-4"
```

### Privacy-First Setup  
```toml
[providers]
default_provider = "embedded"

[providers.embedded]
model_path = "~/.cache/g3/models/codellama-7b-instruct.Q4_K_M.gguf"
model_type = "codellama"
gpu_layers = 32
```

### Hybrid Setup
```toml
[providers]
default_provider = "embedded"

# Use embedded for most tasks
[providers.embedded]
model_path = "~/.cache/g3/models/codellama-7b-instruct.Q4_K_M.gguf"
model_type = "codellama"
gpu_layers = 32

# Fallback to cloud for complex tasks
[providers.openai]
api_key = "sk-..."
model = "gpt-4"
```
