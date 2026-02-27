# AGENTS.md - Developer Guide for This Repository

## Project Overview

Go-based HTTP proxy server for OpenAI-compatible LLM APIs. Proxies requests to llm.chutes.ai while exposing an OpenAI-compatible interface.

## Build Commands

```bash
# Build the project
go build -o proxy .

# Run the server
./proxy

# Run with custom port
PORT=8080 ./proxy

# Run with custom upstream URL and API key
UPSTREAM_URL=https://llm.chutes.ai/v1/chat/completions \
UPSTREAM_API_KEY=your-api-key \
./proxy

# Run tests
go test ./...

# Run tests with verbose output
go test -v ./...

# Run a single test by name
go test -v -run TestFunctionName ./...

# Run tests with coverage
go test -cover ./...

# Format code
go fmt ./...

# Vet code (static analysis)
go vet ./...

# Tidy go.mod
go mod tidy

# View dependencies
go list -m all
```

## Code Style Guidelines

### General Principles

- **Simplicity**: Prefer simple, readable code over clever abstractions
- **Idiomatic Go**: Follow standard Go conventions and patterns
- **Minimal dependencies**: Only add external dependencies when necessary
- **No comments unless requested**: Do not add comments unless explicitly asked

### Imports

Group imports in order: standard library, external packages (github.com), then internal packages (proxy/...). Use blank lines between groups. No import aliases unless necessary.

```go
import (
    "context"
    "fmt"
    "io"
    "net/http"

    "github.com/gin-gonic/gin"
    "github.com/tmaxmax/go-sse"

    "proxy/config"
    "proxy/logging"
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
- **Packages**: lowercase, short, no underscores (e.g., `proxy`)
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
- Use table-driven tests when testing multiple cases
- Test naming: `TestFunctionName_Scenario`
- Use `t.Fatalf` or `t.Errorf` for failures

### Configuration

- Use environment variables for configuration
- Provide sensible defaults
- Use a config struct loaded at startup

### Logging

- Use `proxy/logging` package: `logging.InfoMsg` and `logging.ErrorMsg`

### Dependencies

- Run `go mod tidy` after adding/removing dependencies
- Avoid dependencies for trivial functionality

### Project Structure

```
proxy/
├── main.go           # Entry point, route setup
├── config/           # Configuration loading
├── downstream/       # HTTP handlers (client-facing)
├── upstream/         # Upstream API client
└── logging/          # Logging utilities
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

## Common Tasks

### Adding a new endpoint

1. Add handler in `downstream/`
2. Register route in `main.go`

### Modifying SSE handling

- SSE parsing via `github.com/tmaxmax/go-sse`
- Use `sse.Read()` in `downstream/completions.go`
