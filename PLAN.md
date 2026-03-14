# Multi-Provider Multi-Protocol Proxy Implementation Plan

## Overview

This document outlines the implementation plan for transforming the AI-Proxy from a hardcoded single-provider proxy to a flexible, configuration-driven multi-provider multi-protocol proxy.

## Problem Statement

### Current Limitations

1. **Hardcoded providers**: OpenAI endpoint → chutes.ai only; Anthropic endpoint → Alibaba only
2. **Fixed routing**: Each endpoint talks to exactly one upstream
3. **No model mapping**: Model name passes through unchanged
4. **Protocol tied to endpoint**: `/v1/chat/completions` always uses OpenAI upstream, `/v1/messages` always uses Anthropic upstream

### Required Solution

A flexible, configuration-driven proxy where:

1. **Multiple providers**: Define multiple upstream LLM providers in JSON config
2. **Model mapping**: Incoming model names map to specific provider + model combinations
3. **Protocol-agnostic routing**: Any input protocol can route to any output protocol based on provider config
4. **Runtime configuration**: Change providers/models via JSON config file

---

## Configuration Design

### Configuration File Schema

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
      "name": "alibaba-coding-plan",
      "type": "anthropic",
      "base_url": "https://coding-intl.dashscope.aliyuncs.com/apps/anthropic/v1",
      "envApiKey": "ALIBABA_API_KEY"
    }
  ],
  "models": {
    "kimi-k2.5": {
      "provider": "kimi-chutes",
      "model": "moonshotai/Kimi-K2.5-TEE",
      "tool_call_transform": true
    },
    "claude-3-opus": {
      "provider": "alibaba-coding-plan"
      "model": "glm-5",
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

### Configuration Loading

- **CLI flag**: `--config-file=/path/to/config.json`
- **Environment variable**: `CONFIG_FILE=/path/to/config.json`
- **Priority**: CLI flag > Environment variable > Error (no default)

### API Key Resolution

- `envApiKey`: Environment variable name containing the key (preferred for security)
- `apiKey`: Plaintext API key (use sparingly)
- Resolution: `envApiKey` takes precedence, falls back to `apiKey`

### Model Resolution Logic

1. Look up model in `models` map
2. If found, use mapped provider and model name
3. If not found and `fallback.enabled` is true:
   - Use fallback provider
   - Replace `{model}` placeholder in fallback model name
4. If not found and no fallback, return error

---

## Protocol Support

### Input Protocols (3 types)

| Protocol | Endpoint | Format |
|----------|----------|--------|
| OpenAI Chat Completions | `/v1/chat/completions` | OpenAI ChatCompletionRequest |
| OpenAI Responses | `/v1/responses` | OpenAI ResponsesRequest |
| Anthropic Messages | `/v1/messages` | Anthropic MessageRequest |

### Output Protocols (2 types)

| Protocol | Upstream Type | Format |
|----------|---------------|--------|
| OpenAI Chat Completions | `openai` | OpenAI ChatCompletionRequest/Response |
| Anthropic Messages | `anthropic` | Anthropic MessageRequest/Response |

### Conversion Matrix (6 combinations)

| Input | Output | Status | Implementation |
|-------|--------|--------|----------------|
| OpenAI Chat → OpenAI Chat | Pass-through | Existing | No changes |
| OpenAI Chat → Anthropic | Convert | **NEW** | `convert/chat_to_anthropic.go` |
| OpenAI Responses → OpenAI Chat | Convert | **NEW** | `convert/responses_to_chat.go` |
| OpenAI Responses → Anthropic | Convert | Existing | Refactor `anthropic_to_openai.go` |
| Anthropic → OpenAI Chat | Convert | Existing | Refactor `bridge.go` |
| Anthropic → Anthropic | Pass-through | Existing | No changes |

---

## Architecture

### Request Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLIENT REQUEST                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ENDPOINT DETECTION                                                          │
│  /v1/chat/completions → inputProtocol = "openai_chat"                        │
│  /v1/responses       → inputProtocol = "openai_responses"                    │
│  /v1/messages        → inputProtocol = "anthropic"                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  MODEL RESOLUTION                                                            │
│  1. Extract model from request body                                          │
│  2. router.Resolve(model) → Provider, TargetModel, Settings                  │
│  3. outputProtocol = provider.type (openai/anthropic)                        │
│  4. toolCallTransform = settings.tool_call_transform                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  REQUEST CONVERSION (if inputProtocol ≠ outputProtocol)                      │
│                                                                              │
│  openai_chat    → openai_chat     : Pass-through                            │
│  openai_chat    → anthropic       : convert.ChatToAnthropic()               │
│  openai_responses → openai_chat   : convert.ResponsesToChat()               │
│  openai_responses → anthropic     : convert.ResponsesToAnthropic()          │
│  anthropic      → openai_chat     : convert.AnthropicToChat()               │
│  anthropic      → anthropic       : Pass-through                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  UPSTREAM REQUEST                                                            │
│  - URL: provider.base_url + endpoint path                                    │
│  - Auth: provider.apiKey or os.Getenv(provider.envApiKey)                    │
│  - Headers: Forward relevant headers for protocol                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  RESPONSE CONVERSION (SSE Streaming)                                         │
│                                                                              │
│  Based on: outputProtocol, inputProtocol, toolCallTransform                  │
│                                                                              │
│  If toolCallTransform: Apply toolcall parser + appropriate transformer       │
│  Else: Use passthrough or format converter                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  CLIENT RESPONSE                                                             │
│  Stream in inputProtocol format                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                   main.go                                    │
│   Entry Point: Parse CLI → Load Config → Init Router → Start Server         │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              config/                                          │
│   schema.go    - Config struct definitions                                   │
│   loader.go    - JSON loading, validation, env var resolution                │
│   config.go    - Config accessor methods                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              router/                                          │
│   router.go    - Router interface and implementation                         │
│   resolver.go  - Model resolution with fallback logic                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              convert/                                         │
│   interface.go            - Converter interfaces                             │
│   common.go               - Shared conversion utilities                      │
│   chat_to_anthropic.go    - OpenAI Chat → Anthropic                          │
│   responses_to_chat.go    - OpenAI Responses → OpenAI Chat                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           api/handlers/                                       │
│   completions.go  - /v1/chat/completions (modified)                          │
│   messages.go     - /v1/messages (modified)                                  │
│   responses.go    - /v1/responses (NEW)                                      │
│   bridge.go       - Existing bridge handler (refactored)                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            transform/                                         │
│   passthrough.go  - Pass-through SSE transformer (NEW)                       │
│   toolcall/       - Existing tool call transformers                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Tasks

### Batch 1: Foundation Layer (Parallel)

#### Task T1.1: Config Schema

**File**: `config/schema.go`

**Description**: Define configuration structs with JSON tags.

**Structs**:
```go
type Provider struct {
    Name      string `json:"name"`
    Type      string `json:"type"`       // "openai" or "anthropic"
    BaseURL   string `json:"base_url"`
    APIKey    string `json:"apiKey,omitempty"`
    EnvAPIKey string `json:"envApiKey,omitempty"`
}

type ModelConfig struct {
    Provider          string `json:"provider"`
    Model             string `json:"model"`
    ToolCallTransform bool   `json:"tool_call_transform"`
}

type FallbackConfig struct {
    Enabled           bool   `json:"enabled"`
    Provider          string `json:"provider"`
    Model             string `json:"model"`
    ToolCallTransform bool   `json:"tool_call_transform"`
}

type Config struct {
    Providers []Provider             `json:"providers"`
    Models    map[string]ModelConfig `json:"models"`
    Fallback  FallbackConfig         `json:"fallback"`
}
```

**Tests**: `config/schema_test.go`
- Test JSON unmarshaling
- Test field presence/absence

---

#### Task T1.2: CLI Flag Parsing

**File**: Modify `main.go`

**Description**: Add `--config-file` CLI flag support.

**Requirements**:
- Use standard `flag` package
- Flag name: `--config-file`
- Resolve config path from flag or `CONFIG_FILE` env var
- Return error if neither provided

**Tests**: `main_test.go`
- Test flag parsing
- Test env var fallback
- Test error when neither provided

---

### Batch 2: Configuration Loader (Sequential)

#### Task T2.1: Config Loader

**File**: `config/loader.go`

**Description**: Implement JSON config loading with validation.

**Functions**:
```go
type Loader struct{}

func NewLoader() *Loader
func (l *Loader) Load(path string) (*Config, error)
func (l *Loader) validate(cfg *Config) error
func (l *Loader) resolveEnvVars(cfg *Config) error
func (p *Provider) GetAPIKey() string
```

**Validation Rules**:
- At least one provider required
- Each provider must have name, type, base_url
- Provider type must be "openai" or "anthropic"
- At least one API key source (apiKey or envApiKey)
- Model mappings must reference existing providers
- If fallback enabled, provider must exist

**Tests**: `config/loader_test.go`
- Table-driven tests for all validation cases
- Test env var resolution
- Test invalid JSON handling
- Test missing file handling

---

#### Task T2.2: Refactor config.go

**File**: Modify `config/config.go`

**Description**: Remove old hardcoded fields, integrate with new loader.

**Changes**:
- Remove `OpenAIUpstreamURL`, `OpenAIUpstreamAPIKey`, `AnthropicUpstreamURL`, `AnthropicAPIKey`
- Add `ConfigFile` field for path
- Add `AppConfig` field holding `*config.Config`
- Update accessor methods to use new config

**Tests**: Update existing tests in `config/config_test.go`

---

### Batch 3: Router Layer

#### Task T3.1: Router Implementation

**File**: `router/router.go`

**Description**: Implement model resolution with fallback.

**Interface**:
```go
type Router interface {
    Resolve(modelName string) (*ResolvedRoute, error)
    GetProvider(name string) (Provider, bool)
    ListModels() []string
}

type ResolvedRoute struct {
    Provider          Provider
    Model             string
    OutputProtocol    string  // "openai" or "anthropic"
    ToolCallTransform bool
}
```

**Implementation**:
```go
func NewRouter(cfg *config.Config) (*Router, error)
func (r *Router) Resolve(modelName string) (*ResolvedRoute, error)
func (r *Router) GetProvider(name string) (Provider, bool)
```

**Tests**: `router/router_test.go`
- Test exact model match
- Test fallback with `{model}` placeholder
- Test fallback disabled
- Test unknown model error
- Test missing provider error

---

### Batch 4: Transform Layer (Parallel)

#### Task T4.1: Passthrough Transformer

**File**: `transform/passthrough.go`

**Description**: Simple pass-through SSE transformer.

**Interface**:
```go
type PassthroughTransformer struct {
    w io.Writer
}

func NewPassthroughTransformer(w io.Writer) *PassthroughTransformer
func (t *PassthroughTransformer) Transform(event *sse.Event) error
func (t *PassthroughTransformer) Flush() error
func (t *PassthroughTransformer) Close() error
```

**Tests**: `transform/passthrough_test.go`
- Test event passthrough
- Test flush/close

---

### Batch 5: Conversion Layer (Sequential)

#### Task T5.1: Converter Interfaces

**File**: `convert/interface.go`

**Description**: Define converter interfaces.

```go
type RequestConverter interface {
    Convert(body []byte) ([]byte, error)
}

type ResponseTransformer interface {
    transform.SSETransformer
}

type ConverterPair struct {
    Request  RequestConverter
    Response func(io.Writer) ResponseTransformer
}
```

**Tests**: N/A (interface only)

---

#### Task T5.2: Common Conversion Utilities

**File**: `convert/common.go`

**Description**: Shared helper functions for all converters.

**Functions**:
```go
// Message conversion
func ConvertAnthropicMessagesToOpenAI(anthMsgs []types.MessageInput) []types.Message
func ConvertOpenAIMessagesToAnthropic(openMsgs []types.Message) []types.MessageInput

// Tool conversion
func ConvertAnthropicToolsToOpenAI(anthTools []types.ToolDef) []types.Tool
func ConvertOpenAIToolsToAnthropic(openTools []types.Tool) []types.ToolDef

// Content extraction
func ExtractTextFromContent(content interface{}) string
func ConvertContentBlocks(blocks []interface{}) (string, []types.ToolCall, string)

// System message
func ExtractSystemMessage(system interface{}) string
```

**Tests**: `convert/common_test.go`
- Table-driven tests for each function
- Edge cases: nil, empty, nested content

---

#### Task T5.3: OpenAI Chat → Anthropic Converter

**File**: `convert/chat_to_anthropic.go`

**Description**: Convert OpenAI Chat Completions format to Anthropic Messages format.

**Request Conversion**:
```go
type ChatToAnthropicConverter struct{}

func NewChatToAnthropicConverter() *ChatToAnthropicConverter
func (c *ChatToAnthropicConverter) Convert(body []byte) ([]byte, error)
```

**Request Transformation**:
- `ChatCompletionRequest` → `MessageRequest`
- `messages[].content` (string) → `content` (string or array)
- `messages[].tool_calls` → `content` array with `tool_use` blocks
- `messages[].tool_call_id` → `content` array with `tool_result` blocks
- `tools` → `tools` with `input_schema`
- `system` message → `system` field

**Response Conversion (SSE)**:
```go
type ChatToAnthropicTransformer struct {
    w io.Writer
    // state for tracking tool calls
}

func NewChatToAnthropicTransformer(w io.Writer) *ChatToAnthropicTransformer
func (t *ChatToAnthropicTransformer) Transform(event *sse.Event) error
func (t *ChatToAnthropicTransformer) Flush() error
func (t *ChatToAnthropicTransformer) Close() error
```

**Response Transformation**:
- OpenAI `data: {"choices":[{"delta":{"content":"..."}}]}` → Anthropic `data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"..."}}`
- OpenAI `tool_calls` delta → Anthropic `content_block_delta` with `input_json_delta`

**Tests**: `convert/chat_to_anthropic_test.go`
- Test request conversion with various message types
- Test response streaming conversion
- Test tool call conversion

---

#### Task T5.4: OpenAI Responses → OpenAI Chat Converter

**File**: `convert/responses_to_chat.go`

**Description**: Convert OpenAI Responses API format to OpenAI Chat Completions format.

**Request Conversion**:
```go
type ResponsesToChatConverter struct{}

func NewResponsesToChatConverter() *ResponsesToChatConverter
func (c *ResponsesToChatConverter) Convert(body []byte) ([]byte, error)
```

**Request Transformation**:
- `ResponsesRequest` → `ChatCompletionRequest`
- `input` (string or array) → `messages` array
- `instructions` → `system` or first message with role "system"
- `tools` (ResponsesTool) → `tools` (Tool)
- `max_output_tokens` → `max_tokens`

**Response Conversion (SSE)**:
```go
type ResponsesToChatTransformer struct {
    w io.Writer
    // state for tracking response
}

func NewResponsesToChatTransformer(w io.Writer) *ResponsesToChatTransformer
```

**Response Transformation**:
- Responses API events → Chat Completions SSE format
- `response.output_text.delta` → `choices[0].delta.content`
- `response.function_call_arguments.delta` → `choices[0].delta.tool_calls`

**Tests**: `convert/responses_to_chat_test.go`
- Test request conversion
- Test response streaming conversion
- Test tool handling

---

### Batch 6: Handler Layer (Sequential)

#### Task T6.1: Responses Handler

**File**: `api/handlers/responses.go`

**Description**: New handler for `/v1/responses` endpoint.

**Implementation**:
- Parse `ResponsesRequest`
- Use Router to resolve model
- Determine conversion based on provider type
- Apply appropriate converter

**Tests**: `api/handlers/responses_test.go`
- Test routing to OpenAI provider
- Test routing to Anthropic provider
- Test error handling

---

#### Task T6.2: Refactor Completions Handler

**File**: Modify `api/handlers/completions.go`

**Description**: Use Router for model resolution and conversion selection.

**Changes**:
- Add Router dependency
- Extract model from request
- Use Router.Resolve() to get provider and settings
- Select appropriate converter based on input/output protocol
- Apply tool call transform if configured

**Tests**: Update existing tests
- Test routing to different providers
- Test tool call transform enabled/disabled

---

#### Task T6.3: Refactor Messages Handler

**File**: Modify `api/handlers/messages.go`

**Description**: Use Router for model resolution and conversion selection.

**Changes**:
- Add Router dependency
- Extract model from request
- Use Router.Resolve() to get provider and settings
- Select appropriate converter based on input/output protocol
- Apply tool call transform if configured

**Tests**: Update existing tests
- Test routing to different providers
- Test tool call transform enabled/disabled

---

#### Task T6.4: Refactor Bridge Handler

**File**: Modify `api/handlers/bridge.go`

**Description**: Use Router for model resolution.

**Changes**:
- Add Router dependency
- Use Router.Resolve() instead of hardcoded config
- May deprecate if functionality fully covered by messages handler

**Tests**: Update existing tests

---

### Batch 7: Integration (Sequential)

#### Task T7.1: Update Server Routes

**File**: Modify `api/server.go`

**Description**: Add new routes and pass Router to handlers.

**Changes**:
- Add `/v1/responses` route
- Modify handler constructors to accept Router
- Remove hardcoded config references

**Tests**: Update route registration tests

---

#### Task T7.2: Final main.go Integration

**File**: Modify `main.go`

**Description**: Wire up all components.

**Changes**:
- Parse CLI flags
- Load config file
- Initialize Router
- Pass Router to Server

**Tests**: `main_test.go`
- Smoke test for startup
- Test config loading

---

#### Task T7.3: Example Config File

**File**: `config.example.json`

**Description**: Example configuration for documentation.

---

#### Task T7.4: Integration Test

**File**: `integration_test.go`

**Description**: End-to-end tests for all 6 conversion paths.

**Tests**:
- OpenAI Chat → OpenAI Chat (pass-through)
- OpenAI Chat → Anthropic
- OpenAI Responses → OpenAI Chat
- OpenAI Responses → Anthropic
- Anthropic → OpenAI Chat
- Anthropic → Anthropic (pass-through)

---

## Quality Requirements

### Code Style

- Follow AGENTS.md guidelines
- No comments unless requested
- Functions < 30 lines
- Group imports: stdlib → external → internal
- Use tabs for indentation

### Test Coverage

- Target: > 90% coverage
- Use `testify` library for assertions and mocks
- Table-driven tests for all test cases
- Test both happy path and error cases

### Design Principles

**DRY (Don't Repeat Yourself)**:
- Extract common utilities to `convert/common.go`
- Share message/content transformation functions

**SOLID**:
- Single Responsibility: Each converter handles one protocol pair
- Open/Closed: New protocols via new converters
- Liskov Substitution: All transformers implement `transform.SSETransformer`
- Interface Segregation: Separate request/response converter interfaces
- Dependency Inversion: Handlers depend on Router interface

### Validation Commands

```bash
# Build
go build ./...

# Test
go test ./...

# Test with coverage
go test -cover ./...
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out

# Lint
go vet ./...
go fmt ./...
```

---

## File Summary

### New Files (12)

| File | Description |
|------|-------------|
| `config/schema.go` | Config struct definitions |
| `config/schema_test.go` | Schema tests |
| `config/loader.go` | Config loading and validation |
| `config/loader_test.go` | Loader tests |
| `router/router.go` | Router implementation |
| `router/router_test.go` | Router tests |
| `convert/interface.go` | Converter interfaces |
| `convert/common.go` | Shared conversion utilities |
| `convert/common_test.go` | Common utilities tests |
| `convert/chat_to_anthropic.go` | OpenAI Chat → Anthropic converter |
| `convert/chat_to_anthropic_test.go` | Converter tests |
| `convert/responses_to_chat.go` | OpenAI Responses → OpenAI Chat converter |
| `convert/responses_to_chat_test.go` | Converter tests |
| `api/handlers/responses.go` | /v1/responses handler |
| `api/handlers/responses_test.go` | Responses handler tests |
| `transform/passthrough.go` | Pass-through SSE transformer |
| `transform/passthrough_test.go` | Passthrough tests |
| `config.example.json` | Example configuration |

### Modified Files (7)

| File | Changes |
|------|---------|
| `config/config.go` | Remove old fields, integrate new loader |
| `main.go` | Add CLI flags, load JSON config |
| `api/server.go` | Add /v1/responses route |
| `api/handlers/completions.go` | Use Router |
| `api/handlers/messages.go` | Use Router |
| `api/handlers/bridge.go` | Use Router |
| `AGENTS.md` | Update documentation |

---

## Task Dependency Graph

```
Batch 1 (Parallel):
T1.1 (schema) ─────────────────────────────────────┐
T1.2 (CLI flags) ──────────────────────────────────┤
                                                   │
Batch 2 (Sequential):                              │
T2.1 (loader) ◄────────────────────────────────────┤
T2.2 (config.go) ◄─────────────────────────────────┤
                                                   │
Batch 3:                                           │
T3.1 (router) ◄────────────────────────────────────┤
                                                   │
Batch 4 (Parallel with Batch 3):                   │
T4.1 (passthrough) ────────────────────────────────┤
                                                   │
Batch 5 (Sequential):                              │
T5.1 (interfaces) ─────────────────────────────────┤
T5.2 (common) ◄────────────────────────────────────┤
T5.3 (chat→anthropic) ◄────────────────────────────┤
T5.4 (responses→chat) ◄────────────────────────────┤
                                                   │
Batch 6 (Sequential):                              │
T6.1 (responses handler) ◄─────────────────────────┤
T6.2 (completions refactor) ◄──────────────────────┤
T6.3 (messages refactor) ◄─────────────────────────┤
T6.4 (bridge refactor) ◄───────────────────────────┤
                                                   │
Batch 7 (Sequential):                              │
T7.1 (server routes) ◄─────────────────────────────┤
T7.2 (main integration) ◄──────────────────────────┤
T7.3 (example config) ─────────────────────────────┤
T7.4 (integration test) ◄──────────────────────────┘
```

---

## Estimated Timeline

| Batch | Tasks | Est. Time |
|-------|-------|-----------|
| 1 | T1.1, T1.2 | ~10 min |
| 2 | T2.1, T2.2 | ~10 min |
| 3 | T3.1 | ~5 min |
| 4 | T4.1 | ~5 min |
| 5 | T5.1-T5.4 | ~15 min |
| 6 | T6.1-T6.4 | ~15 min |
| 7 | T7.1-T7.4 | ~10 min |
| **Total** | | **~60-70 min** |

---

## Success Criteria

- [ ] JSON configuration file support via `--config-file` and `CONFIG_FILE`
- [ ] Multiple providers with different types (openai/anthropic)
- [ ] Model mapping with provider and model name
- [ ] Fallback configuration for unknown models
- [ ] Per-model tool call transform configuration
- [ ] All 6 protocol conversion paths working
- [ ] Test coverage > 90%
- [ ] All tests passing
- [ ] Code follows AGENTS.md style
- [ ] `config.example.json` documentation
- [ ] Updated AGENTS.md with configuration docs
