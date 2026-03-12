# AGENTS.md - Developer Guide

Go-based HTTP proxy for LLM APIs with OpenAI and Anthropic compatibility.

## Build Commands

```bash
# Build and run
go build -o ai-proxy . && ./ai-proxy

# Run with environment variables
PORT=8081 UPSTREAM_API_KEY=key ./ai-proxy

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
    "github.com/tmaxmax/go-sse"

    "ai-proxy/config"
    "ai-proxy/logging"
    "ai-proxy/types"
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
- **Files**: lowercase with underscores (e.g., `unified_handler.go`)
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
- Use table-driven tests when testing multiple cases
- Test naming: `TestFunctionName_Scenario`
- Use `t.Fatalf` or `t.Errorf` for failures

### Configuration

- Use environment variables for configuration
- Provide sensible defaults
- Use a config struct loaded at startup

### Logging

- Use `ai-proxy/logging` package: `logging.InfoMsg` and `logging.ErrorMsg`

### Dependencies

- Run `go mod tidy` after adding/removing dependencies
- Avoid dependencies for trivial functionality

### Project Structure

```
ai-proxy/
├── main.go                    # Entry point, route setup
├── config/                    # Configuration loading
├── types/                     # Shared type definitions
│   ├── openai.go              # OpenAI streaming types
│   ├── anthropic.go           # Anthropic streaming types
│   └── common.go              # ToolCall, normalization utilities
├── downstream/                # HTTP handlers (client-facing)
│   ├── unified_handler.go     # Unified stream handler
│   ├── protocols/             # Protocol adapters
│   │   ├── adapter.go         # ProtocolAdapter interface
│   │   ├── openai.go          # OpenAI adapter
│   │   ├── anthropic.go       # Anthropic adapter
│   │   └── bridge.go          # OpenAI→Anthropic bridge
│   ├── health.go              # Health check endpoint
│   ├── models.go              # Models listing
│   ├── response_recorder.go   # Response capture wrapper
│   └── sse_capture.go         # SSE capture utilities
├── transform/                 # Transformation pipeline
│   └── toolcall/              # Tool call transformation
│       ├── parser.go          # Shared state machine parser
│       ├── openai_output.go   # OpenAI output formatter
│       └── anthropic_output.go# Anthropic output formatter
├── upstream/                  # Upstream API client
│   ├── client.go              # HTTPClient + Client interface
│   └── mock.go                # Mock client for testing
└── logging/                   # Logging and capture utilities
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
| POST | `/v1/chat/completions` | Chat completions (streaming) |
| POST | `/v1/messages` | Native Anthropic messages |
| POST | `/v1/openai-to-anthropic/messages` | OpenAI→Anthropic bridge |

## Architecture

### Protocol Adapter Pattern

The proxy uses a protocol adapter pattern to handle different API formats:

```go
type ProtocolAdapter interface {
    TransformRequest(body []byte) ([]byte, error)
    ValidateRequest(body []byte) error
    CreateTransformer(w io.Writer, base types.StreamChunk) *ToolCallTransformer
    UpstreamURL(cfg *config.Config) string
    UpstreamAPIKey(cfg *config.Config) string
    ForwardHeaders(src, dst http.Header)
    SendError(c *gin.Context, status int, msg string)
    IsStreamingRequest(body []byte) bool
}
```

### Tool Call Transformation

Tool calls are transformed using a shared parser with format-specific output formatters:

1. **Parser** (`transform/toolcall/parser.go`): State machine that extracts tool calls from special tokens
2. **OpenAI Output** (`transform/toolcall/openai_output.go`): Emits OpenAI-format `tool_calls` deltas
3. **Anthropic Output** (`transform/toolcall/anthropic_output.go`): Emits Anthropic-format `tool_use` content blocks

## Common Tasks

### Adding a new endpoint

1. Create a new protocol adapter in `downstream/protocols/`
2. Register route in `main.go` using `StreamHandler(cfg, adapter)`

### Adding a new output format

1. Create a new output formatter implementing `toolcall.EventHandler`
2. Create a new protocol adapter that uses the formatter

### Modifying SSE handling

- SSE parsing via `github.com/tmaxmax/go-sse`
- Use `sse.Read()` in `downstream/unified_handler.go`

## Testing

### Manual Testing with curl

#### OpenAI Compatible Endpoint (`/v1/chat/completions`)

```bash
# Start server with logging
PORT=8081 SSELOG_DIR=./test_logs ./ai-proxy

# Basic chat request
curl -s -X POST http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "moonshotai/Kimi-K2.5-TEE",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'

# With tool calls
curl -s -X POST http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "moonshotai/Kimi-K2.5-TEE",
    "messages": [{"role": "user", "content": "List files in current directory"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "bash",
        "description": "Execute bash commands",
        "parameters": {
          "type": "object",
          "properties": {
            "command": {"type": "string", "description": "The bash command"}
          },
          "required": ["command"]
        }
      }
    }],
    "stream": true
  }'
```

**Model**: `moonshotai/Kimi-K2.5-TEE`

#### Anthropic Endpoint (`/v1/messages`)

```bash
# Start server with logging
PORT=8081 SSELOG_DIR=./test_logs ./ai-proxy

# Basic chat request
curl -s -X POST http://localhost:8081/v1/messages \
  -H "Content-Type: application/json" \
  -H "Anthropic-Version: 2023-06-01" \
  -d '{
    "model": "kimi-k2.5",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'

# With tool calls
curl -s -X POST http://localhost:8081/v1/messages \
  -H "Content-Type: application/json" \
  -H "Anthropic-Version: 2023-06-01" \
  -d '{
    "model": "kimi-k2.5",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "List files using bash"}],
    "tools": [{
      "name": "bash",
      "description": "Execute bash commands",
      "input_schema": {
        "type": "object",
        "properties": {
          "command": {"type": "string", "description": "The bash command"}
        },
        "required": ["command"]
      }
    }],
    "stream": true
  }'
```

**Model**: `kimi-k2.5`

### Request Capture/Logging

When `SSELOG_DIR` is set, all requests are captured to structured JSON files:

```bash
# Start with logging enabled
PORT=8081 SSELOG_DIR=./test_logs ./ai-proxy

# Make requests (no X-Request-ID needed - ID extracted from SSE response)
curl -X POST http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "moonshotai/Kimi-K2.5-TEE", "messages": [{"role": "user", "content": "test"}], "stream": true}'

# Check captured logs
ls -la test_logs/$(date +%Y-%m-%d)/
cat test_logs/$(date +%Y-%m-%d)/*.json
```

**Captured data** (4 capture points):
1. **Downstream TX** - Client request to proxy
2. **Upstream TX** - Proxy request to LLM API
3. **Upstream RX** - LLM API response to proxy
4. **Downstream RX** - Proxy response to client

**Log format**: Structured JSON with:
- Request metadata (ID, timestamps, duration)
- Headers (sanitized - auth masked)
- Body (parsed JSON)
- SSE chunks (structured JSON in `data` field, raw string in `raw` if invalid)