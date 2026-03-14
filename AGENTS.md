# AGENTS.md - Developer Guide

Go-based HTTP proxy for LLM APIs with OpenAI and Anthropic compatibility.

## Build Commands

```bash
# Build and run
go build -o ai-proxy . && ./ai-proxy --config-file=config.example.json

# Run with environment variable for config
CONFIG_FILE=config.example.json ./ai-proxy

# Run tests
go test ./...
go test -v ./...
go test -v -run TestFunctionName ./...
go test -cover ./...

# Format and lint
go fmt ./...
go vet ./...
go mod tidy
```

## Configuration

### Configuration File

The proxy requires a JSON configuration file specified via `--config-file` flag or `CONFIG_FILE` environment variable.

**Priority**: CLI flag > Environment variable > Error

**Example** (`config.example.json`):
```json
{
  "providers": [
    {
      "name": "kimi-chutes",
      "type": "openai",
      "base_url": "https://llm.chutes.ai/v1",
      "envApiKey": "CHUTES_API_KEY"
    },
    {
      "name": "anthropic-direct",
      "type": "anthropic",
      "base_url": "https://api.anthropic.com/v1",
      "envApiKey": "ANTHROPIC_API_KEY"
    }
  ],
  "models": {
    "kimi-k2.5": {
      "provider": "kimi-chutes",
      "model": "moonshotai/Kimi-K2.5-TEE",
      "tool_call_transform": true
    },
    "claude-3-opus": {
      "provider": "anthropic-direct",
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

### Provider Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique identifier for the provider |
| `type` | string | Yes | Protocol type: `"openai"` or `"anthropic"` |
| `base_url` | string | Yes | Base URL for the provider API |
| `apiKey` | string | No | Plaintext API key (use sparingly) |
| `envApiKey` | string | No | Environment variable name containing the API key |

### Model Mapping Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `provider` | string | Yes | Name of the provider to use |
| `model` | string | Yes | Actual model name to send to the provider |
| `tool_call_transform` | bool | No | Enable Kimi-style tool call parsing (default: false) |

### Fallback Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `enabled` | bool | Yes | Enable fallback for unknown models |
| `provider` | string | Yes | Default provider name |
| `model` | string | Yes | Model template (supports `{model}` placeholder) |
| `tool_call_transform` | bool | No | Enable tool call parsing for fallback |

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `CONFIG_FILE` | Path to JSON config file | Yes (or `--config-file`) |
| `PORT` | Server port (default: 8080) | No |
| `SSELOG_DIR` | Directory for request logging | No |
| Provider API keys | Referenced by `envApiKey` in config | Per-provider |

## Protocol Support

### Input Protocols (3 types)

| Protocol | Endpoint | Format |
|----------|----------|--------|
| OpenAI Chat Completions | `/v1/chat/completions` | OpenAI `ChatCompletionRequest` |
| OpenAI Responses | `/v1/responses` | OpenAI `ResponsesRequest` |
| Anthropic Messages | `/v1/messages` | Anthropic `MessageRequest` |

### Output Protocols (2 types)

| Protocol | Provider Type | Format |
|----------|---------------|--------|
| OpenAI Chat Completions | `openai` | OpenAI `ChatCompletionRequest/Response` |
| Anthropic Messages | `anthropic` | Anthropic `MessageRequest/Response` |

### Conversion Matrix (6 combinations)

| Input Protocol | Output Protocol | Conversion |
|----------------|-----------------|------------|
| OpenAI Chat | OpenAI Chat | Pass-through |
| OpenAI Chat | Anthropic | `convert.ChatToAnthropicConverter` |
| OpenAI Responses | OpenAI Chat | `convert.ResponsesToChatConverter` |
| OpenAI Responses | Anthropic | Existing bridge logic |
| Anthropic | OpenAI Chat | `transformAnthropicToOpenAI` |
| Anthropic | Anthropic | Pass-through |

## Code Style Guidelines

### General Principles

- **Simplicity**: Prefer simple, readable code over clever abstractions
- **Idiomatic Go**: Follow standard Go conventions and patterns
- **Minimal dependencies**: Only add external dependencies when necessary
- **No comments unless requested**: Do not add comments unless explicitly asked

### Imports

Group imports in order: standard library, external packages (github.com), then internal packages (ai-proxy/...). Use blank lines between groups. No import aliases unless necessary.

```go
import (
    "context"
    "fmt"
    "io"
    "net/http"

    "github.com/gin-gonic/gin"
    "github.com/stretchr/testify/require"

    "ai-proxy/api"
    "ai-proxy/config"
    "ai-proxy/router"
)
```

### Formatting

- Use `go fmt` for all code formatting
- Indent with tabs, not spaces
- Maximum line length: 100 characters (soft limit)
- Add blank line between top-level declarations

### Naming Conventions

- **Variables/Functions**: `camelCase` (e.g., `apiKey`, `streamResponse`)
- **Constants**: `PascalCase` or `camelCase` for unexported
- **Types/Interfaces**: `PascalCase` (e.g., `Config`)
- **Packages**: lowercase, short, no underscores (e.g., `ai-proxy`)
- **Files**: lowercase with underscores (e.g., `completions.go`)
- **Exported/Unexported**: Uppercase/lowercase first letter
- **Acronyms**: Use all caps for acronyms > 2 letters (e.g., `APIKey`)

### Types

- Use specific types rather than generic ones
- Define custom types for domain concepts
- Pointer receivers for methods that modify state; value receivers for read-only

### Error Handling

- Always handle errors explicitly; never ignore with `_`
- Return errors early; avoid deep nesting
- Use `fmt.Errorf` with `%w` for wrapping errors
- Log errors before returning when appropriate

```go
func readBody(c *gin.Context) ([]byte, error) {
    body, err := io.ReadAll(c.Request.Body)
    if err != nil {
        return nil, fmt.Errorf("read body: %w", err)
    }
    return body, nil
}
```

### Context

- Pass `context.Context` as the first parameter to functions that make HTTP requests or do I/O
- Use `c.Request.Context()` to get the request's context
- Respect context cancellation in long-running operations

### HTTP Handlers

- Use Gin framework for HTTP handlers
- Return appropriate HTTP status codes
- Log errors with the logging package
- Always close response bodies

### Testing

- Tests in `*_test.go` files in the same package
- Use `testify` library for assertions and mocks
- Use table-driven tests when testing multiple cases
- Test naming: `TestFunctionName_Scenario`
- Target coverage: >90%

### Logging

- Use `ai-proxy/logging` package: `logging.InfoMsg` and `logging.ErrorMsg`

### Dependencies

- Run `go mod tidy` after adding/removing dependencies
- Avoid dependencies for trivial functionality

### Project Structure

```
ai-proxy/
├── main.go                    # Entry point, CLI flags
├── config.example.json        # Example configuration
├── api/                       # HTTP server, handlers, and middleware
│   ├── handlers/              # HTTP endpoint handlers
│   │   ├── completions.go     # /v1/chat/completions
│   │   ├── messages.go        # /v1/messages
│   │   ├── responses.go       # /v1/responses
│   │   ├── bridge.go          # /v1/openai-to-anthropic/messages
│   │   ├── interface.go       # Handler interfaces
│   │   └── common.go          # Shared handler utilities
│   ├── middleware.go          # Capture middleware
│   └── server.go              # Server setup and routing
├── capture/                   # Request/response capture and logging
├── config/                    # Configuration loading
│   ├── config.go              # Config struct and accessors
│   ├── schema.go              # JSON schema structs
│   └── loader.go              # JSON loading and validation
├── convert/                   # Protocol converters
│   ├── interface.go           # Converter interfaces
│   ├── common.go              # Shared conversion utilities
│   ├── chat_to_anthropic.go   # OpenAI Chat → Anthropic
│   └── responses_to_chat.go   # OpenAI Responses → OpenAI Chat
├── logging/                   # Logging utilities
├── proxy/                     # Upstream API client
├── router/                    # Model-to-provider routing
│   └── router.go              # Router implementation
├── transform/                 # Format transformations
│   ├── interface.go           # Transformer interface
│   ├── passthrough.go         # Pass-through transformer
│   └── toolcall/              # Tool call transformations
└── types/                     # Shared types
    ├── anthropic.go           # Anthropic API types
    ├── openai.go              # OpenAI API types
    ├── openai_responses.go    # OpenAI Responses API types
    └── sse.go                 # SSE types
```

### Git Conventions

- Do NOT commit unless explicitly asked
- Run `go vet ./...` and `go fmt ./...` before commits
- Ensure build passes: `go build ./...`

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/v1/models` | List available models |
| POST | `/v1/chat/completions` | OpenAI Chat Completions (routes to any provider) |
| POST | `/v1/messages` | Anthropic Messages (routes to any provider) |
| POST | `/v1/responses` | OpenAI Responses API (routes to any provider) |
| POST | `/v1/openai-to-anthropic/messages` | Anthropic→OpenAI bridge |
| POST | `/v1/anthropic-to-openai/responses` | OpenAI Responses→Anthropic bridge |

## Common Tasks

### Adding a new provider

1. Add provider configuration to JSON config file
2. Set the API key in environment variable
3. Add model mappings as needed

### Adding a new endpoint

1. Create handler in `api/handlers/` implementing `RoutingHandler` interface
2. Register route in `api/server.go`

### Modifying SSE handling

- SSE parsing via `github.com/tmaxmax/go-sse`
- Transformers implement `transform.SSETransformer` interface

## Testing

### Manual Testing with curl

#### Start server

```bash
# Set required environment variables
export CHUTES_API_KEY=your-key
export ANTHROPIC_API_KEY=your-key

# Start server
./ai-proxy --config-file=config.example.json
```

#### OpenAI Chat Completions (`/v1/chat/completions`)

```bash
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k2.5",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

#### Anthropic Messages (`/v1/messages`)

```bash
curl -s -X POST http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "Anthropic-Version: 2023-06-01" \
  -d '{
    "model": "claude-3-opus",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

#### OpenAI Responses API (`/v1/responses`)

```bash
curl -s -X POST http://localhost:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k2.5",
    "input": "Hello",
    "stream": true
  }'
```

### Request Capture/Logging

When `SSELOG_DIR` is set, all requests are captured to structured JSON files:

```bash
SSELOG_DIR=./test_logs ./ai-proxy --config-file=config.example.json
```

**Captured data** (4 capture points):
1. **Downstream TX** - Client request to proxy
2. **Upstream TX** - Proxy request to LLM API
3. **Upstream RX** - LLM API response to proxy
4. **Downstream RX** - Proxy response to client