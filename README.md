# AI Proxy for Kimi-K2.5 / K2

A Go-based HTTP proxy that transforms Kimi-K2.5 and K2's proprietary tool call format into OpenAI-compatible `tool_calls` format, enabling seamless integration with OpenAI-compatible clients and SDKs.

## Problem Statement

Kimi-K2.5 and K2 models output tool/function calls using special delimiter tokens embedded in the SSE `reasoning` field, rather than the standard OpenAI `tool_calls` format. Cloud providers typically fix this server-side, but self-hosted deployments or direct API access expose this incompatibility, breaking OpenAI-compatible clients, SDKs, and tools.

### Non-Standard Tool Call Format (Example)

Kimi-K2.5 and K2 use special delimiter tokens instead of OpenAI's structured JSON:

```
<|tool_calls_section_begin|>
<|tool_call_begin|>functions.bash:15<|tool_call_argument_begin|>{"command": "ls -la"}<|tool_call_end|>
<|tool_call_begin|>functions.task:16<|tool_call_argument_begin|>{"description": "..."}<|tool_call_end|>
<|tool_calls_section_end|>
```

**Note:** This behavior is intermittent—it occurs only sometimes depending on the model's response. The proxy handles both cases: when tool calls appear in reasoning tokens (transforms them) and when they use standard format (passes through unchanged).

### OpenAI's Expected Format

OpenAI-compatible clients expect tool calls in this structured format:

```json
{
  "choices": [{
    "delta": {
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "bash",
          "arguments": "{\"command\": \"ls -la\"}"
        }
      }]
    }
  }]
}
```

## Solution

This proxy sits between your application and the Kimi-K2.5/K2 upstream API, transforming the non-standard tool call format in real-time during SSE streaming:

```
┌─────────────┐      ┌──────────────────────┐      ┌────────────────────┐
│   Client    │ ───▶ │   AI Proxy           │ ───▶ │  Kimi-K2.5/K2 API  │
│ (OpenAI SDK)│ ◀─── │ (ToolCallTransformer)│ ◀─── │(e.g. llm.chutes.ai)│
└─────────────┘      └──────────────────────┘      └────────────────────┘
```

### Key Features

- **Real-time transformation**: Converts tool call tokens to OpenAI format during streaming
- **Token reassembly**: Handles special tokens split across multiple SSE chunks via state machine buffering
- **Full OpenAI compatibility**: Exposes standard `/v1/chat/completions`, `/v1/models`, and `/health` endpoints
- **Pass-through for non-tool responses**: Regular text completions pass through unchanged

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/v1/models` | List available models |
| `POST` | `/v1/chat/completions` | Chat completions (streaming supported) |

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `PORT` | `8080` | Server port |
| `UPSTREAM_URL` | `https://llm.chutes.ai/v1/chat/completions` | Upstream API URL |
| `UPSTREAM_API_KEY` | (empty) | Default API key for upstream |
| `SSELOG_DIR` | (empty) | Directory for SSE debug logs |

## Usage

```bash
# Build
go build -o ai-proxy .

# Run with default settings
./ai-proxy

# Run with custom configuration
UPSTREAM_API_KEY=your-key PORT=3000 ./ai-proxy
```

## Technical Details

### Tool Call Transformation

The `ToolCallTransformer` implements a 5-state machine (`IDLE → IN_SECTION → READING_ID → READING_ARGS → TRAILING`) that:

1. Buffers incoming reasoning text across SSE chunks
2. Detects special delimiter tokens
3. Extracts function name and arguments
4. Emits properly formatted OpenAI `tool_calls` deltas

### Supported Special Tokens

| Token | Description |
|-------|-------------|
| `<|tool_calls_section_begin|>` | Starts the tool calls section |
| `<|tool_call_begin|>` | Starts a function call (ID/name follows) |
| `<|tool_call_argument_begin|>` | Starts the JSON arguments |
| `<|tool_call_end|>` | Ends the current tool call |
| `<|tool_calls_section_end|>` | Ends the tool calls section |

## Development

```bash
# Run tests
go test ./...

# Run tests with coverage
go test -cover ./...

# Format code
go fmt ./...

# Static analysis
go vet ./...
```

## License

MIT