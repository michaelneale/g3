# G3 ACP - Agent Client Protocol Implementation

This crate implements the Agent Client Protocol (ACP) for the G3 AI coding agent, allowing it to be exposed as a server that clients can connect to via stdio.

## Overview

The ACP protocol is a JSON-RPC based protocol that enables structured communication between AI agents and clients. This implementation allows G3 to:

- Accept prompts from ACP-compatible clients
- Stream responses back to clients in real-time
- Handle tool calls and report their execution status
- Support session management and cancellation
- Provide capability negotiation

## Usage

### Running the ACP Server

To start G3 in ACP server mode:

```bash
g3 acp
```

With custom configuration:

```bash
g3 acp --config /path/to/config.toml
```

With verbose logging:

```bash
g3 acp --verbose
```

### Protocol Flow

1. **Initialize**: Client sends an initialize request to discover agent capabilities
2. **Authenticate**: Client authenticates (currently a no-op)
3. **New Session**: Client requests a new session, receives a session ID
4. **Prompt**: Client sends prompts with the session ID
5. **Streaming**: Agent streams back text chunks, tool calls, and results
6. **Cancel**: Client can cancel active operations

## Supported ACP Methods

### Agent Methods (Required)

- ✅ `initialize` - Advertises G3's capabilities
- ✅ `authenticate` - No-op authentication
- ✅ `new_session` - Creates a new conversation session
- ✅ `prompt` - Executes a prompt and streams responses
- ✅ `cancel` - Cancels an active prompt

### Optional Methods

- ❌ `load_session` - Not yet supported (returns method_not_found)
- ❌ `set_session_mode` - Not yet supported
- ❌ `ext_method` - Extension methods not supported
- ✅ `ext_notification` - Extension notifications accepted (no-op)

## Capabilities

G3's ACP implementation advertises the following capabilities:

```json
{
  "load_session": false,
  "prompt_capabilities": {
    "image": false,
    "audio": false,
    "embedded_context": true
  },
  "mcp_capabilities": {
    "http": false,
    "sse": false
  }
}
```

## Session Management

Each session maintains:
- **Conversation history**: All messages exchanged
- **Tool call tracking**: Mapping between internal and ACP tool call IDs
- **Cancellation token**: Allows graceful cancellation

## Tool Call Handling

When G3 executes tools, they are reported to the client with:
- Unique ACP tool call IDs
- Human-friendly tool names (e.g., "Read File" instead of "read_file")
- Execution status (Pending → Completed/Failed)
- Tool output as text content

## Architecture

```
Client (stdio) <--JSON-RPC--> ACP Server <--> G3 Agent <--> Tools
```

### Key Components

- **G3AcpAgent**: Implements the ACP Agent trait
- **G3Session**: Tracks individual session state
- **Message Conversion**: Converts between ACP and G3 message formats
- **Tool Tracking**: Maps tool IDs between protocols
- **Streaming**: Sends incremental updates to clients

## Implementation Details

### Non-Send Futures

The ACP protocol crate uses non-Send futures, so we use `tokio::task::LocalSet` to execute them on a single thread.

### Stdio Communication

- **Input**: Reads JSON-RPC messages from stdin
- **Output**: Writes JSON-RPC responses and notifications to stdout
- **Errors**: Logged to stderr

### Tool Name Formatting

Tool names are formatted for better readability:
- `shell` → "Shell"
- `read_file` → "Read File"
- `write_file` → "Write File"
- `final_output` → "Final Output"

## Testing

Run tests with:

```bash
cargo test -p g3-acp
```

## Future Enhancements

- [ ] Session persistence (load_session support)
- [ ] Image content support
- [ ] Better streaming with incremental tool call updates
- [ ] File location extraction for file-related tools
- [ ] MCP (Model Context Protocol) integration
- [ ] Session modes support

## Example Client

A simple Python client example:

```python
import json
import subprocess

# Start G3 in ACP mode
proc = subprocess.Popen(
    ['g3', 'acp'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Send initialize request
init_request = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {}
}
proc.stdin.write(json.dumps(init_request) + '\n')
proc.stdin.flush()

# Read initialize response
response = json.loads(proc.stdout.readline())
print("Initialize response:", response)

# Create new session
session_request = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "new_session",
    "params": {}
}
proc.stdin.write(json.dumps(session_request) + '\n')
proc.stdin.flush()

# Read session response
response = json.loads(proc.stdout.readline())
session_id = response['result']['session_id']
print("Session ID:", session_id)

# Send a prompt
prompt_request = {
    "jsonrpc": "2.0",
    "id": 3,
    "method": "prompt",
    "params": {
        "session_id": session_id,
        "prompt": [
            {
                "type": "text",
                "text": "List the files in the current directory"
            }
        ]
    }
}
proc.stdin.write(json.dumps(prompt_request) + '\n')
proc.stdin.flush()

# Read streaming responses
while True:
    line = proc.stdout.readline()
    if not line:
        break
    msg = json.loads(line)
    print("Response:", msg)
    if 'result' in msg and msg['result'].get('stop_reason'):
        break
```

## Dependencies

- `agent-client-protocol` - The ACP protocol specification
- `g3-core` - G3 agent implementation
- `g3-config` - Configuration management
- `tokio` - Async runtime
- `tokio-util` - Utilities for async I/O
- `futures` - Async abstractions

## License

MIT
