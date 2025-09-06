# G3 General Purpose AI Agent

A code-first AI agent that helps you complete tasks by writing and executing code or scripts.

## Philosophy

G3 doesn't just give you advice - **it writes and runs code to solve your problems**. Whether you need to:
- Process data files
- Automate workflows
- Scrape websites
- Manipulate files
- Analyze logs
- Generate reports
- Set up environments

G3 will write the appropriate scripts (Python, Bash, JavaScript, etc.) and can execute them for you.

## Features

- **Code-First Approach**: Always tries to solve problems with executable code
- **Multi-Language Support**: Generates Python, Bash, JavaScript, Rust, and more
- **Modular Architecture**: Clean separation between CLI, core engine, and LLM providers
- **Multiple LLM Providers**: Support for OpenAI, Anthropic, and embedded open-weights models
- **Local Model Support**: Run completely offline with embedded GGUF models via llama.cpp
- **Interactive Mode**: Chat with the AI and watch it solve problems in real-time
- **Task Automation**: Create reusable automation scripts

## Installation

```bash
cargo install --path .
```

## Configuration

Create a configuration file at `~/.config/g3/config.toml`:

### Cloud Providers

```toml
[providers]
default_provider = "openai"

[providers.openai]
api_key = "your-openai-api-key"
model = "gpt-4"
max_tokens = 2048
temperature = 0.1

[providers.anthropic]
api_key = "your-anthropic-api-key"
model = "claude-3-sonnet-20240229"
max_tokens = 2048
temperature = 0.1
```

### Local Embedded Models

For completely offline operation with open-weights models:

```toml
[providers]
default_provider = "embedded"

[providers.embedded]
# Path to your GGUF model file
model_path = "~/.cache/g3/models/codellama-7b-instruct.Q4_K_M.gguf"
model_type = "codellama"
context_length = 4096
max_tokens = 2048
temperature = 0.1
# Number of layers to offload to GPU (0 for CPU only)
gpu_layers = 32
# Number of CPU threads to use
threads = 8
```

**Getting Models**: Download GGUF models from [Hugging Face](https://huggingface.co/models?library=gguf) (search for "GGUF"). Popular options:
- [CodeLlama 7B Instruct](https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF)
- [Llama 2 7B Chat](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)  
- [Mistral 7B Instruct](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF)

### Agent Settings

```toml
[agent]
max_context_length = 8192
enable_streaming = true
timeout_seconds = 60
```

## Usage

### Interactive Mode (Default)

Simply run G3 to start interactive mode:

```bash
g3
```

Example interactions:
```
g3> Process this CSV file and show me the top 10 customers by revenue
# G3 writes Python pandas script to analyze the CSV

g3> Set up a backup script for my home directory
# G3 creates a bash script with rsync/tar commands

g3> Download all images from this webpage
# G3 writes a Python script with requests/BeautifulSoup

g3> Convert these JSON files to a SQLite database
# G3 creates a Python script to parse JSON and insert into SQLite
```

### Direct Commands

You can also use G3 with direct commands:

```bash
# Solve any task with code
g3 task "merge these PDF files into one"
g3 task "find all TODO comments in my codebase"

# Create automation scripts
g3 automate "daily backup of my projects folder"
g3 automate "resize all images in a folder to 800px width"

# Data processing
g3 data "analyze this log file for error patterns"
g3 data "convert CSV to JSON with validation"

# Legacy code commands (still supported)
g3 analyze src/main.rs
g3 generate "fibonacci function" --output fib.py
g3 review src/lib.rs
```

## Architecture

G3 follows the Unix philosophy with modular, composable components:

- **g3-cli**: Command-line interface
- **g3-core**: Core agent logic and orchestration
- **g3-providers**: LLM provider abstractions
- **g3-config**: Configuration management

See [DESIGN.md](DESIGN.md) for detailed architecture documentation.

## Development

```bash
# Build all crates
cargo build

# Run tests
cargo test

# Run with debug logging
RUST_LOG=debug cargo run -- analyze src/
```

## License

MIT
