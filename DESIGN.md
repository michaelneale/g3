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
│   mode          │    │ - Code          │    │ - Local models  │
│ - Code exec     │    │   generation    │    │ - Custom APIs   │
│   approval      │    │ - Script        │    │                 │
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
- **Responsibility**: LLM communication (unchanged)
- **Enhanced Prompts**:
  - Code-first system prompts
  - Language-specific generation instructions

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

### Phase 1: Core Refactoring
1. Update CLI commands for task-oriented interface
2. Enhance system prompts for code-first approach
3. Add basic code execution capabilities
4. Update interactive mode messaging
