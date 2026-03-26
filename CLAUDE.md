# AI Proxy - Claude Development Guide

Go-based HTTP proxy for LLM APIs with OpenAI and Anthropic compatibility.

## Quick Commands

```bash
go build            # Standard build (no llama.cpp)
make build          # Build with llama.cpp (cached at .build/)
make build-cuda     # Build with CUDA GPU support
make clean-cache    # Clear llama.cpp cache (rebuilds next time)
go test ./...       # Run tests (>90% coverage required)
go fmt ./...        # Format code
go vet ./...        # Lint
```

## Project Structure

```
├── api/              # HTTP server, routing, handlers
├── capture/          # Request/response logging
├── config/           # JSON config loading (XDG paths)
├── convert/          # Format conversions (OpenAI↔Anthropic↔Responses)
├── conversation/     # In-memory conversation cache
├── llama/            # CGo bindings to llama.cpp (build tag: llama)
├── logging/          # Logging utilities
├── proxy/            # Upstream API client
├── router/           # Model-to-provider routing
├── summarizer/       # Reasoning summarization (HTTP or local llama.cpp)
├── tokens/           # Token counting
├── transform/        # SSE stream transformations
│   └── toolcall/     # Tool call parsing (Kimi K2.5, GLM-5 format)
├── websearch/        # Web search service (Exa, Brave, DDG)
└── types/            # API type definitions
```

## Code Style

- **Imports**: stdlib, external (github.com), internal (ai-proxy/...), blank lines between groups
- **Naming**: camelCase vars/funcs, PascalCase types, ALL_CAPS for acronyms >2 letters
- **Comments**: Structured with `@param`, `@return`, `@pre`, `@post`, `@note` annotations
- **Tests**: `TestFunctionName_Scenario` pattern, table-driven, >90% coverage required
- **Line length**: 100 chars soft limit
- **Error handling**: Return early, wrap with `fmt.Errorf("%w", err)`

## Key Architecture

| Endpoint | Provider | Request | Response |
|----------|----------|---------|----------|
| /v1/chat/completions | OpenAI | pass-through | tool call norm* |
| /v1/chat/completions | Anthropic | O→A | A→O |
| /v1/messages | Anthropic | pass-through | tool call norm* |
| /v1/messages | OpenAI | A→O | O→A |
| /v1/responses | OpenAI | R→Chat | Chat→R |
| /v1/responses | Anthropic | R→A | A→R |

*Tool call norm only when `kimi_tool_call_transform: true` or `glm5_tool_call_transform: true`

## Tool Call Transformations

### Kimi K2.5/K2
Embeds tool calls in reasoning tokens using special delimiters:
- <|tool_call_section_begin|> - section markers
- <|tool_call|> - call markers

### GLM-5
Uses XML-based tool call format that requires extraction and transformation to standard OpenAI/Anthropic structures.

The `ToolCallTransformer` uses a state machine to extract and reformat during SSE streaming.

## Configuration

Config loaded from JSON via `--config-file` or XDG paths:
1. `$XDG_CONFIG_HOME/ai-proxy/config.json`
2. `~/.config/ai-proxy/config.json`

Key fields per model:
- `provider`: Provider name to route to
- `model`: Actual model identifier on provider
- `kimi_tool_call_transform`: Enable Kimi tool-call extraction
- `glm5_tool_call_transform`: Enable GLM-5 XML tool-call extraction
- `reasoning_split`: Enable separate reasoning output

## Testing Requirements

- All new code >90% coverage
- Reproduce defects with tests before fixing
- Use table-driven tests for multiple cases
