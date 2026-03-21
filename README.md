# AI Proxy

A Go-based HTTP proxy that enables Codex with Alibaba and reliable Kimi. Provides format transformation, tool call normalization, and seamless integration between OpenAI, Anthropic, and other API formats.

## What This Project Solves

**Model mapping** and **provider aggregation** solve the routing problem. Applications need to route different models to different providers—Alibaba's Qwen for cost, Anthropic's Claude for reasoning, Kimi for specialized tasks. Without a routing layer, provider selection logic gets hardcoded throughout codebases. This proxy makes "use qwen3-max" a config change, not a code refactor.

**Protocol translation** handles format incompatibility:

| Client Format | OpenAI Provider | Anthropic Provider |
|---------------|-----------------|---------------------|
| OpenAI Chat | ✓ Pass-through | Chat → Messages → Chat |
| Anthropic Messages | Messages → Chat → Messages | ✓ Pass-through |
| OpenAI Responses | Responses → Chat → Chat | Responses → Messages → Responses |

Each conversion handles message structure, tool call formats, streaming semantics, and edge cases around system prompts and multi-modal inputs.

**Alibaba model access** unlocks cost-effective alternatives. Alibaba's Qwen and hosted Kimi models only support OpenAI-compatible endpoints. Codex users can't access them without rewriting integration code. This proxy acts as a universal adapter: Codex talks to Alibaba through OpenAI responses protocol.

**Kimi tool-call extraction** makes Kimi-K2.5/K2 usable. These models embed tool calls in reasoning tokens using proprietary delimiters (`<|tool_calls_section_begin|>`, `<|tool_call_begin|>`) rather than standard formats. Without real-time extraction from the SSE `delta.reasoning` stream, agents receive malformed tool calls and function-calling breaks. This proxy's state-machine parser extracts and reformats tool calls into standard OpenAI/Anthropic structures—making Kimi viable for production agents.

## Features

- **Multi-format support**: OpenAI Chat Completions, Anthropic Messages, and OpenAI Responses API
- **Bidirectional conversion**: Convert between OpenAI and Anthropic formats in both directions
- **Tool call normalization**: Transforms Kimi-K2.5/K2's proprietary tool call format into standard formats
- **Streaming support**: Real-time SSE streaming with format transformation
- **Request capture**: Optional logging of all requests/responses for debugging
- **Model-based routing**: Route requests to different providers based on model name

## User Guide

### Installation

```bash
# Build from source
go build -o ai-proxy .

# Or install via Makefile
make build
make install  # Installs binary to ~/.local/bin and config to ~/.config/ai-proxy/config.json
```

### Configuration

The proxy requires a JSON configuration file. By default, it searches for `config.json` in XDG standard locations:

1. `--config-file` flag or `CONFIG_FILE` env
2. `$XDG_CONFIG_HOME/ai-proxy/config.json`
3. `$HOME/.config/ai-proxy/config.json`
4. `$XDG_CONFIG_DIRS/ai-proxy/config.json` (default: `/etc/xdg`)

#### Configuration Example

```json
{
  "providers": [
    {
      "name": "alibaba",
      "type": "openai",
      "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
      "envApiKey": "ALIBABA_API_KEY"
    },
    {
      "name": "anthropic",
      "type": "anthropic",
      "base_url": "https://api.anthropic.com",
      "envApiKey": "ANTHROPIC_API_KEY"
    }
  ],
  "models": {
    "qwen3-max": {
      "provider": "alibaba",
      "model": "qwen3-max-2026-01-23",
      "tool_call_transform": false
    },
    "kimi-k2.5": {
      "provider": "alibaba",
      "model": "kimi-k2.5",
      "tool_call_transform": true
    }
  },
  "fallback": {
    "enabled": true,
    "provider": "alibaba",
    "model": "{model}"
  }
}
```

#### Provider Configuration

| Field | Description |
|-------|-------------|
| `name` | Unique identifier for the provider |
| `type` | API format: `"openai"` or `"anthropic"` |
| `base_url` | Provider's API endpoint URL |
| `apiKey` | Direct API key (optional) |
| `envApiKey` | Environment variable name for API key |

#### Model Configuration

| Field | Description |
|-------|-------------|
| `provider` | Provider name to route requests to |
| `model` | Actual model identifier on the provider |
| `tool_call_transform` | Enable Kimi tool-call extraction (default: `false`) |

### Running the Proxy

```bash
# Run with config in XDG default location (~/.config/ai-proxy/config.json)
./ai-proxy

# Run with explicit config file
./ai-proxy --config-file /path/to/config.json

# Run with environment variable
CONFIG_FILE=/path/to/config.json ./ai-proxy

# Run with additional options
./ai-proxy --port 8080 --sse-log-dir ./logs

# Run with conversation store tuning
./ai-proxy --conversation-store-size 2000 --conversation-store-ttl 48h
```

### Command-Line Options

| Flag | Environment | Default | Description |
|------|-------------|---------|-------------|
| `--config-file` | `CONFIG_FILE` | XDG discovery | Path to configuration file |
| `--port` | `PORT` | `8080` | Server listen port |
| `--sse-log-dir` | `SSELOG_DIR` | (disabled) | Directory for request logging |
| `--conversation-store-size` | - | `1000` | Max cached conversations |
| `--conversation-store-ttl` | - | `24h` | Conversation cache TTL |

### Using with Codex

Set the OpenAI base URL to point to the proxy:

```bash
# In your environment or .env
OPENAI_API_BASE=http://localhost:8080/v1
OPENAI_API_KEY=any-key
```

Codex will now route all requests through the proxy, enabling access to Alibaba models, Kimi, and any other configured providers.

### Fallback Behavior

When `fallback.enabled` is `true`, failed requests are automatically retried with the fallback provider. Use `{model}` as a placeholder to preserve the original model name:

```json
{
  "fallback": {
    "enabled": true,
    "provider": "alibaba",
    "model": "{model}",
    "tool_call_transform": false
  }
}
```

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
<|tool_call_begin|>get_weather<|tool_call_argument_begin|>{"location": "San Francisco"}<|tool_call_end|>
<|tool_calls_section_end|>
```

### Standard Format (Output - OpenAI)

```json
{
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": "{\"location\": \"San Francisco\"}"
      }
    }
  ]
}
```

### Standard Format (Output - Anthropic)

```json
{
  "content": [
    {
      "type": "tool_use",
      "id": "toolu_abc123",
      "name": "get_weather",
      "input": {"location": "San Francisco"}
    }
  ]
}
```

### Configuration

Enable tool call transformation per model in your config:

```json
{
  "models": {
    "kimi-k2.5": {
      "provider": "alibaba",
      "model": "kimi-k2.5",
      "tool_call_transform": true
    }
  }
}
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
├── main.go                     # Entry point, server initialization
├── api/                        # HTTP server and routing
│   ├── server.go               # Server setup and route registration
│   ├── middleware.go           # Capture middleware
│   └── handlers/               # HTTP request handlers
│       ├── interface.go        # Handler interface definition
│       ├── health.go           # Health check endpoint
│       ├── models.go           # Models listing endpoint
│       ├── completions.go      # OpenAI chat completions
│       ├── messages.go         # Anthropic messages
│       ├── responses.go        # OpenAI Responses API
│       ├── count_tokens.go     # Token counting endpoint
│       └── response_recorder.go # Response recording utilities
├── config/                     # Configuration loading
│   ├── cli.go                  # CLI flag parsing, XDG discovery
│   ├── config.go               # Config struct and accessors
│   ├── loader.go               # Config file loading and validation
│   └── schema.go               # JSON schema definitions
├── router/                     # Model routing
│   └── router.go               # Model-to-provider resolution
├── convert/                    # Format conversion
│   ├── interface.go            # Converter interface
│   ├── common.go               # Shared conversion utilities
│   ├── param_convert.go        # Parameter conversion
│   ├── finish_reason.go        # Finish reason mapping
│   ├── anthropic_to_chat.go    # Anthropic → OpenAI Chat
│   ├── anthropic_to_responses.go # Anthropic → OpenAI Responses
│   ├── chat_to_anthropic.go    # OpenAI Chat → Anthropic
│   ├── chat_to_responses.go    # OpenAI Chat → Responses
│   ├── responses_to_anthropic.go # Responses → Anthropic
│   ├── responses_to_anthropic_streaming.go # Streaming variant
│   └── responses_to_chat.go    # Responses → OpenAI Chat
├── transform/                  # Response transformation
│   ├── interface.go            # Transformer interface
│   ├── passthrough.go          # Pass-through transformer
│   ├── sse_writer.go           # SSE streaming writer
│   └── toolcall/               # Tool call format transformation
│       ├── parser.go           # Token parsing
│       ├── tokens.go           # Special token definitions
│       ├── state.go            # State machine
│       ├── formatter.go        # Output formatting
│       ├── common.go           # Shared utilities
│       ├── openai.go           # OpenAI format support
│       ├── anthropic.go        # Anthropic format support
│       ├── openai_transformer.go    # OpenAI format transformer
│       ├── anthropic_transformer.go # Anthropic format transformer
│       └── responses_transformer.go # Responses API transformer
├── types/                      # Type definitions
│   ├── openai.go               # OpenAI Chat API types
│   ├── openai_responses.go     # OpenAI Responses API types
│   ├── anthropic.go            # Anthropic API types
│   └── sse.go                  # Server-Sent Events types
├── proxy/                      # Upstream API client
│   ├── client.go               # HTTP client for upstream APIs
│   └── request.go              # Request building utilities
├── tokens/                     # Token counting
│   └── counter.go              # Token counter implementation
├── conversation/               # Conversation storage
│   └── store.go                # In-memory conversation cache
├── capture/                    # Request/response capture
│   ├── storage.go              # Log storage management
│   ├── writer.go               # JSON log writer
│   ├── recorder.go             # Request/response recording
│   └── context.go              # Context utilities
└── logging/                    # Logging utilities
    └── logging.go
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
