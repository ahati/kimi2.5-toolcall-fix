# AI Proxy

A Go-based HTTP proxy for LLM APIs with OpenAI and Anthropic compatibility. Provides format transformation, tool call normalization, and seamless integration between different API formats.

## Features

- **Multi-format support**: OpenAI Chat Completions, Anthropic Messages, and OpenAI Responses API
- **Bidirectional conversion**: Convert between OpenAI and Anthropic formats in both directions
- **Tool call normalization**: Transforms Kimi-K2.5/K2's proprietary tool call format into standard formats
- **Streaming support**: Real-time SSE streaming with format transformation
- **Request capture**: Optional logging of all requests/responses for debugging
- **Model-based routing**: Route requests to different providers based on model name

## API Endpoints

| Method | Path | Request Format | Response Format | Description |
|--------|------|----------------|-----------------|-------------|
| `GET` | `/health` | N/A | N/A | Health check |
| `GET` | `/v1/models` | OpenAI | OpenAI | List available models |
| `POST` | `/v1/chat/completions` | OpenAI | OpenAI | Chat completions (routes by model) |
| `POST` | `/v1/messages` | Anthropic | Anthropic | Anthropic messages (routes by model) |
| `POST` | `/v1/messages/count_tokens` | Anthropic | Anthropic | Count tokens in messages |
| `POST` | `/v1/responses` | OpenAI Responses | OpenAI Responses | OpenAI Responses API (routes by model) |

## Architecture

The proxy uses a unified handler architecture where each endpoint intelligently routes requests based on model configuration:

```
                                    ┌──────────────────────────────────────────────────┐
                                    │                  AI Proxy                        │
                                    │                                                  │
                                    │  ┌────────────────────────────────────────────┐  │
┌──────────────┐                    │  │         /v1/chat/completions               │  │
│  OpenAI SDK  │──▶ POST /v1/chat/completions ──▶│                            │──┼──▶ Provider (by model)
│              │◀── OpenAI Response ─────────────│  Routes based on model:    │◀─┼─── Provider Response
└──────────────┘                    │  │  • OpenAI provider → pass through          │  │
                                    │  │  • Anthropic provider → convert O→A→O      │  │
                                    │  └────────────────────────────────────────────┘  │
┌──────────────┐                    │  ┌────────────────────────────────────────────┐  │
│ Anthropic SDK│──▶ POST /v1/messages ──────────▶│                            │──┼──▶ Provider (by model)
│              │◀─ Anthropic Response ───────────│  Routes based on model:    │◀─┼─── Provider Response
└──────────────┘                    │  │  • Anthropic provider → pass through       │  │
                                    │  │  • OpenAI provider → convert A→O→A         │  │
                                    │  └────────────────────────────────────────────┘  │
┌──────────────┐                    │  ┌────────────────────────────────────────────┐  │
│ OpenAI SDK   │──▶ POST /v1/responses ─────────▶│                            │──┼──▶ Provider (by model)
│ (Responses)  │◀── OpenAI Response ─────────────│  Routes based on model:    │◀─┼─── Provider Response
└──────────────┘                    │  │  • OpenAI provider → convert R→C→R         │  │
                                    │  │  • Anthropic provider → convert R→A→R      │  │
                                    │  └────────────────────────────────────────────┘  │
                                    │                                                  │
                                    └──────────────────────────────────────────────────┘

Legend: O=OpenAI Chat, A=Anthropic Messages, R=OpenAI Responses, C=OpenAI Chat
```

### Request Flow

1. Client sends request to unified endpoint with a model name
2. Router resolves model name to provider configuration
3. Handler transforms request format if needed (e.g., OpenAI → Anthropic)
4. Request is forwarded to the appropriate upstream provider
5. Response is transformed back to the client's expected format

## Tool Call Transformation

Kimi-K2.5 and K2 models output tool/function calls using special delimiter tokens embedded in the SSE `reasoning` field, rather than standard formats. This proxy transforms these in real-time during streaming.

### Non-Standard Format (Input)

```
<|tool_calls_section_begin|>
<|tool_call_begin|>functions.bash:15<|tool_call_argument_begin|>{"command": "ls -la"}<|tool_call_end|>
<|tool_call_begin|>functions.task:16<|tool_call_argument_begin|>{"description": "..."}<|tool_call_end|>
<|tool_calls_section_end|>
```

### OpenAI Format (Output)

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

### Anthropic Format (Output)

```json
{
  "type": "content_block_delta",
  "index": 1,
  "delta": {
    "type": "input_json_delta",
    "partial_json": "{\"command\": \"ls -la\"}"
  }
}
```

## Configuration

### Configuration File

The proxy requires a JSON configuration file that defines providers and model mappings.

```json
{
  "providers": [
    {
      "name": "kimi-chutes",
      "type": "openai",
      "base_url": "https://llm.chutes.ai/v1/chat/completions",
      "envApiKey": "CHUTES_API_KEY"
    },
    {
      "name": "alibaba-coding",
      "type": "anthropic",
      "base_url": "https://coding-intl.dashscope.aliyuncs.com/apps/anthropic/v1/messages",
      "envApiKey": "ALIBABA_ANTHROPIC_API_KEY"
    }
  ],
  "models": {
    "kimi-k2.5": {
      "provider": "kimi-chutes",
      "model": "moonshotai/Kimi-K2.5-TEE",
      "tool_call_transform": true
    },
    "claude-3-opus": {
      "provider": "alibaba-coding",
      "model": "claude-3-opus-20240229",
      "tool_call_transform": false
    }
  },
  "fallback": {
    "enabled": true,
    "provider": "kimi-chutes",
    "model": "{model}",
    "tool_call_transform": true
  }
}
```

### Configuration Schema

| Field | Description |
|-------|-------------|
| `providers[]` | List of upstream API providers |
| `providers[].name` | Unique identifier for the provider |
| `providers[].type` | API format: `"openai"` or `"anthropic"` |
| `providers[].base_url` | API endpoint URL |
| `providers[].apiKey` | Direct API key (optional) |
| `providers[].envApiKey` | Environment variable name for API key |
| `models{}` | Map of model aliases to configurations |
| `models[].provider` | Provider name to use for this model |
| `models[].model` | Actual model identifier on upstream |
| `models[].tool_call_transform` | Enable tool call transformation |
| `fallback.enabled` | Enable fallback for unknown models |
| `fallback.provider` | Default provider for unknown models |
| `fallback.model` | Model pattern (use `{model}` placeholder) |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | Server port |
| `SSELOG_DIR` | (empty) | Directory for request/response logging |
| `CONFIG_FILE` | (empty) | Path to JSON configuration file |

### Command-Line Flags

| Flag | Description |
|------|-------------|
| `--config-file` | Path to JSON configuration file |
| `--sse-log-dir` | Directory for request/response logging |
| `--port` | Server port |

## Usage

### Build and Run

```bash
# Build
go build -o ai-proxy .

# Run with config file
./ai-proxy --config-file config.json

# Run with environment variable for config
CONFIG_FILE=config.json ./ai-proxy

# Run with request logging
./ai-proxy --config-file config.json --sse-log-dir ./logs

# Run with custom port
./ai-proxy --config-file config.json --port 3000
```

### Example Requests

#### OpenAI Chat Completions

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k2.5",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

#### Anthropic Messages

```bash
curl -X POST http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "Anthropic-Version: 2023-06-01" \
  -d '{
    "model": "claude-3-opus",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

#### OpenAI Responses API

```bash
curl -X POST http://localhost:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-opus",
    "input": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

#### Count Tokens

```bash
curl -X POST http://localhost:8080/v1/messages/count_tokens \
  -H "Content-Type: application/json" \
  -H "Anthropic-Version: 2023-06-01" \
  -d '{
    "model": "claude-3-opus",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Cross-Provider Examples

The unified architecture enables seamless cross-provider calls:

**Use Anthropic SDK with an OpenAI backend:**

```bash
# Request uses Anthropic format, routes to OpenAI provider based on model
curl -X POST http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "Anthropic-Version: 2023-06-01" \
  -d '{
    "model": "kimi-k2.5",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

**Use OpenAI SDK with an Anthropic backend:**

```bash
# Request uses OpenAI Chat format, routes to Anthropic provider based on model
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-opus",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

## Request Capture

When `SSELOG_DIR` is set, all requests are captured to structured JSON files:

```bash
./ai-proxy --config-file config.json --sse-log-dir ./logs

# Logs are organized by date
ls logs/$(date +%Y-%m-%d)/
```

**Captured data** (4 capture points):
1. **Downstream TX** - Client request to proxy
2. **Upstream TX** - Proxy request to LLM API
3. **Upstream RX** - LLM API response to proxy
4. **Downstream RX** - Proxy response to client

## Project Structure

```
ai-proxy/
├── main.go                 # Entry point, server initialization
├── api/                    # HTTP server and routing
│   ├── server.go           # Server setup and route registration
│   ├── middleware.go       # Capture middleware
│   └── handlers/           # HTTP request handlers
│       ├── health.go       # Health check endpoint
│       ├── models.go       # Models listing endpoint
│       ├── completions.go  # OpenAI chat completions (unified)
│       ├── messages.go     # Anthropic messages (unified)
│       ├── responses.go    # OpenAI Responses API (unified)
│       ├── count_tokens.go # Token counting endpoint
│       ├── common.go       # Shared handler utilities
│       └── interface.go    # Handler interface definition
├── config/                 # Configuration loading
│   ├── config.go           # Config struct and accessors
│   ├── schema.go           # JSON schema definitions
│   ├── loader.go           # Config file loading
│   └── cli.go              # CLI flag parsing
├── router/                 # Model routing
│   └── router.go           # Model-to-provider resolution
├── convert/                # Format conversion
│   ├── interface.go        # Converter interface
│   ├── common.go           # Shared conversion utilities
│   ├── chat_to_anthropic.go    # OpenAI Chat → Anthropic
│   ├── anthropic_to_chat.go    # Anthropic → OpenAI Chat
│   ├── responses_to_anthropic.go # Responses → Anthropic
│   └── responses_to_chat.go    # Responses → OpenAI Chat
├── logging/                # Logging utilities
│   └── logging.go
├── proxy/                  # Upstream API client
│   ├── client.go           # HTTP client for upstream APIs
│   └── request.go          # Request building utilities
├── transform/              # Response transformation
│   ├── interface.go        # Transformer interface
│   ├── passthrough.go      # Pass-through transformer
│   └── toolcall/           # Tool call format transformation
│       ├── openai_transformer.go      # OpenAI format transformer
│       ├── anthropic_transformer.go   # Anthropic format transformer
│       ├── responses_transformer.go   # Responses API transformer
│       ├── parser.go       # Token parsing
│       ├── tokens.go       # Special token definitions
│       ├── formatter.go    # Output formatting
│       ├── openai.go       # OpenAI format support
│       ├── anthropic.go    # Anthropic format support
│       └── common.go       # Shared utilities
├── types/                  # Type definitions
│   ├── openai.go           # OpenAI API types
│   ├── openai_responses.go # OpenAI Responses API types
│   ├── anthropic.go        # Anthropic API types
│   └── sse.go              # Server-Sent Events types
├── tokens/                 # Token counting
│   └── counter.go          # Token counter implementation
└── capture/                # Request/response capture
    ├── storage.go          # Log storage management
    ├── writer.go           # JSON log writer
    ├── recorder.go         # Request/response recording
    └── context.go          # Context utilities
```

## Development

```bash
# Run tests
go test ./...

# Run tests with coverage
go test -cover ./...

# Run specific test
go test -v -run TestFunctionName ./...

# Format code
go fmt ./...

# Static analysis
go vet ./...

# Tidy dependencies
go mod tidy
```

## Technical Details

### Tool Call Transformation

The `ToolCallTransformer` implements a 5-state machine (`IDLE → IN_SECTION → READING_ID → READING_ARGS → TRAILING`) that:

1. Buffers incoming reasoning text across SSE chunks
2. Detects special delimiter tokens
3. Extracts function name and arguments
4. Emits properly formatted tool calls in the target format

### Supported Special Tokens

| Token | Description |
|-------|-------------|
| `<|tool_calls_section_begin|>` | Starts the tool calls section |
| `<|tool_call_begin|>` | Starts a function call (ID/name follows) |
| `<|tool_call_argument_begin|>` | Starts the JSON arguments |
| `<|tool_call_end|>` | Ends the current tool call |
| `<|tool_calls_section_end|>` | Ends the tool calls section |

### Format Conversions

| Endpoint | Provider Type | Request Transform | Response Transform |
|----------|---------------|-------------------|-------------------|
| `/v1/chat/completions` | OpenAI | None (pass-through) | Tool call normalization* |
| `/v1/chat/completions` | Anthropic | OpenAI → Anthropic | Anthropic → OpenAI |
| `/v1/messages` | Anthropic | None (pass-through) | Tool call normalization* |
| `/v1/messages` | OpenAI | Anthropic → OpenAI | OpenAI → Anthropic |
| `/v1/responses` | OpenAI | Responses → Chat | Chat → Responses |
| `/v1/responses` | Anthropic | Responses → Anthropic | Anthropic → Responses |

*Tool call normalization only applies when `tool_call_transform: true` is set for the model.

## License

GPL v3