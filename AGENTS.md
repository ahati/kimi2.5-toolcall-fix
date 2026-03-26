# AGENTS.md - Developer Guide

Go-based HTTP proxy for LLM APIs with OpenAI and Anthropic compatibility.

## Build Commands

```bash
# Standard build (no llama.cpp, no CGo required)
go build

# Build with llama.cpp for local summarization
make build

# Build with CUDA GPU support
make build-cuda

# Build with specific llama.cpp version
LLAMA_VERSION=b5508 make build

# Clean binary only
make clean

# Clean llama.cpp cache (rebuilds on next build)
make clean-cache

# Run tests
go test ./...
go test -v ./...
go test -v -run TestFunctionName ./...    # Run single test
go test -cover ./...

# Format and lint
go fmt ./...
go vet ./...
go mod tidy
```

### Build Variants

| Command | CGo | llama.cpp | Use Case |
|---------|-----|-----------|----------|
| `go build` | No | No | Standard deployment, lighter binary |
| `make build` | Yes | CPU | Local summarization with CPU inference |
| `make build-cuda` | Yes | GPU | Local summarization with GPU acceleration |

### Build Cache

llama.cpp is cached at `.build/llama-cpp-<version>/` (gitignored).
- First build: ~2 minutes to compile llama.cpp
- Subsequent builds: instant (uses cached libraries)
- Clean with `make clean-cache` or `rm -rf .build`
- CUDA builds use separate cache: `.build/llama-cpp-<version>-cuda/`

## Code Style Guidelines

### General Principles

- **Simplicity**: Prefer simple, readable code over clever abstractions
- **Idiomatic Go**: Follow standard Go conventions and patterns
- **Minimal dependencies**: Only add external dependencies when necessary
- **Documentation**: Keep code well documented with clear doc comments on exported functions and types
- **DRY**: Do not repeat yourself - extract common logic into reusable functions
- **SOLID**: Follow SOLID principles (Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion)

### Quality Requirements

- **Code coverage**: Must be >90% for all new code
- **Defect fixing**: Always reproduce the defect with a test before fixing. Clarify the expected behavior with the user if ambiguous

### Imports

Group imports: standard library, external packages (github.com), then internal packages (ai-proxy/...). Blank lines between groups. No import aliases.

```go
import (
    "context"
    "encoding/json"
    "io"

    "github.com/gin-gonic/gin"
    "github.com/tmaxmax/go-sse"

    "ai-proxy/config"
    "ai-proxy/types"
)
```

### Formatting

- Use `go fmt` for all code formatting
- Indent with tabs, not spaces
- Maximum line length: 100 characters (soft limit)
- Blank line between top-level declarations

### Naming Conventions

- **Variables/Functions**: `camelCase` (e.g., `apiKey`, `streamResponse`)
- **Types/Interfaces**: `PascalCase` (e.g., `Config`, `SSETransformer`)
- **Constants**: `PascalCase` or `camelCase` for unexported
- **Packages**: lowercase, short, no underscores
- **Files**: lowercase with underscores (e.g., `chat_to_responses.go`)
- **Exported/Unexported**: Uppercase/lowercase first letter
- **Acronyms**: All caps for acronyms > 2 letters (e.g., `APIKey`, `SSETransformer`)

### Types

- Use specific types rather than generic ones
- Define custom types for domain concepts
- Pointer receivers for methods that modify state; value receivers for read-only

### Error Handling

- Always handle errors explicitly; never ignore with `_`
- Return errors early; avoid deep nesting
- Use `fmt.Errorf` with `%w` for wrapping errors

```go
func readBody(c *gin.Context) ([]byte, error) {
    body, err := io.ReadAll(c.Request.Body)
    if err != nil {
        return nil, fmt.Errorf("read body: %w", err)
    }
    return body, nil
}
```

### Documentation Comments

Use structured doc comments with annotations on exported functions:

```go
// FunctionName does something.
//
// @param name - description
// @return description
// @pre precondition if any
// @post postcondition if any
// @note additional notes
```

### Context

- Pass `context.Context` as first parameter for functions making HTTP requests or I/O
- Use `c.Request.Context()` to get request context in handlers

### HTTP Handlers

- Use Gin framework for HTTP handlers
- Handlers implement the `Handler` interface with methods: `ValidateRequest`, `TransformRequest`, `UpstreamURL`, `ResolveAPIKey`, `ForwardHeaders`, `CreateTransformer`, `WriteError`
- Always close response bodies

### Testing

- Tests in `*_test.go` files in same package
- Use table-driven tests when testing multiple cases
- Test naming: `TestFunctionName_Scenario` (e.g., `TestChatToResponsesTransformer_ToolCalls`)
- Use `t.Fatalf` for fatal failures, `t.Errorf` for non-fatal

```go
func TestFunctionName_Scenario(t *testing.T) {
    tests := []struct {
        name string
        input string
        want string
    }{
        {name: "case 1", input: "foo", want: "bar"},
    }
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            if got := function(tt.input); got != tt.want {
                t.Errorf("function() = %v, want %v", got, tt.want)
            }
        })
    }
}
```

### Configuration

- Configuration loaded from JSON file (required) via `--config-file` flag or `CONFIG_FILE` env var
- Environment variables: `PORT`, `SSELOG_DIR`
- Use config struct loaded at startup via `config.Load()`

### Logging

- Use `ai-proxy/logging` package
- Functions: `logging.InfoMsg`, `logging.ErrorMsg`, `logging.DebugMsg`

## Project Structure

```
├── main.go              # Entry point
├── api/
│   ├── server.go        # Server setup and routing
│   ├── middleware.go    # Capture middleware
│   └── handlers/        # HTTP endpoint handlers
├── capture/             # Request/response capture and logging
├── config/              # Configuration loading (JSON + flags + env)
├── convert/             # Format conversions (OpenAI↔Anthropic↔Responses)
├── conversation/        # Conversation store for previous_response_id
├── llama/               # CGo bindings to llama.cpp (build tag: llama)
├── logging/             # Logging utilities
├── proxy/               # Upstream API client
├── router/              # Model-to-provider routing
├── summarizer/          # Reasoning summarization (HTTP or local llama.cpp)
├── tokens/              # Token counting utilities
├── transform/           # SSE stream transformations
│   └── toolcall/        # Tool call parsing and formatting
├── websearch/           # Web search service (Exa, Brave, DDG)
└── types/               # Shared types (OpenAI, Anthropic, SSE)
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/v1/models` | List available models |
| POST | `/v1/chat/completions` | OpenAI-compatible chat completions |
| POST | `/v1/messages` | Anthropic Messages API |
| POST | `/v1/responses` | OpenAI Responses API |

## Git Conventions

- Do NOT commit unless explicitly asked
- Run `go vet ./...` and `go fmt ./...` before commits
- Ensure build passes: `make build`