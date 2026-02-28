# Tool Call Transformation Design

## Overview

This document describes the design for transforming Kimi-K2.5's proprietary tool call format (special delimiter tokens) into OpenAI's standard `tool_calls` format.

## Architecture

### Component Structure

```
                    ┌─────────────────────────────┐
                    │     streamResponse()        │
                    │     (completions.go)        │
                    └──────────┬──────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                     ToolCallTransformer                      │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  ToolCallParser (owned by Transformer)                  │ │
│  │  - State machine                                        │ │
│  │  - Buffers tokens/content                               │ │
│  │  - Returns TransformedEvents                            │ │
│  └─────────────────────────────────────────────────────────┘ │
│                              │                               │
│                              ▼                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  ToolCallEventBuilder (owned by Transformer)            │ │
│  │  - Builds OpenAI JSON format                            │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    ┌────────────────────────┐
                    │  LoggingTransformer    │
                    └────────────────────────┘
```

### Class/Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| `ToolCallTransformer` | Implements `SSETransformer` interface, owns `ToolCallParser` and `ToolCallEventBuilder`, coordinates transformation, writes output |
| `ToolCallParser` | **Accumulate-then-parse** strategy: buffers tool call section content, then parses to extract tool calls |
| `ToolCallEventBuilder` | Constructs OpenAI-format JSON for tool_calls |

---

## Component Designs

### 1. ToolCallTransformer

```go
type ToolCallTransformer struct {
    parser  *ToolCallParser
    builder *ToolCallEventBuilder
    output  io.Writer
}
```

**Responsibilities:**
- Implements `SSETransformer` interface
- **Owns** the `ToolCallParser` and `ToolCallEventBuilder` instances
- Receives SSE events via `Transform(event *sse.Event)` - **calls parser.Process()**
- Writes transformed output to downstream client
- Handles SSE formatting (`data: ...\n\n`)

**Public API:**
```go
func NewToolCallTransformer(output io.Writer) *ToolCallTransformer
func (t *ToolCallTransformer) Transform(event *sse.Event)
func (t *ToolCallTransformer) Flush() error  // flush any buffered content at end of stream
```

**Flow:**
```
Transform() → parser.Process() → builder.Build*() → write to output
```

### 2. ToolCallParser

```go
type ParserState int

const (
    StateIdle ParserState = iota
    StateAccumulating  // Accumulating tool call section content
    StateParsing       // Parsing accumulated content
    StateTrailing
)

type ToolCallParser struct {
    state              ParserState
    accumulatedContent strings.Builder  // Buffer for tool section content
    pendingContent     strings.Builder  // Content before tool section
    toolCalls          []ToolCall       // Parsed tool calls
    metadata           StreamMetadata
}

type ToolCall struct {
    ID        string
    Name      string
    Arguments string
}

type StreamMetadata struct {
    ID        string
    Model     string
    Created   int64
    // other fields from original chunk
}
```

**Responsibilities:**
- **Accumulate-then-parse strategy**: Buffers all content between `<|tool_calls_section_begin|>` and `<|tool_calls_section_end|>`
- Emits content before/after tool call section immediately
- After section ends, parses accumulated content to extract tool calls using simple string operations
- Handles multiple tool calls in a single section
- Preserves stream metadata (id, model, created, etc.)

**Why This Approach?**
- Real-time token detection fails when tokens arrive in separate chunks (see Appendix A)
- Accumulating first gives us complete data to work with
- Simple string splitting is more robust than complex state machines

**Public API:**
```go
func NewToolCallParser() *ToolCallParser
func (p *ToolCallParser) Process(event *sse.Event) []TransformedEvent
func (p *ToolCallParser) Flush() []TransformedEvent  // call at end of stream
```

**TransformedEvent Types:**
```go
type TransformedEvent struct {
    Type         EventType
    Content      string        // for content chunks
    ToolCall     *ToolCall     // for tool_call deltas
    Metadata     StreamMetadata
    FinishReason string
    Usage        *Usage        // for final usage chunk
}

type EventType int
const (
    EventContent EventType = iota
    EventToolCall
    EventFinish
    EventUsage
    EventPassthrough  // for non-tool-call chunks
)
```

### 3. ToolCallEventBuilder

```go
type ToolCallEventBuilder struct{}
```

**Responsibilities:**
- Converts internal `ToolCall` struct to OpenAI JSON format
- Handles partial arguments (streaming scenarios)
- Preserves stream metadata (id, model, created, etc.)

**Public API:**
```go
func (b *ToolCallEventBuilder) BuildToolCallDelta(
    meta StreamMetadata, 
    toolCall *ToolCall,
) []byte

func (b *ToolCallEventBuilder) BuildContentDelta(
    meta StreamMetadata, 
    content string,
) []byte

func (b *ToolCallEventBuilder) BuildFinishEvent(
    meta StreamMetadata, 
    reason string,
) []byte
```

---

## Design Decision: Accumulate-then-Parse Strategy

### Why This Approach?

The original design used a complex state machine that attempted to detect tokens in real-time during streaming. However, analysis of actual SSE logs revealed a critical issue:

**Token Pattern from Real Logs:**

```
Line 38:  <|tool_calls_section_begin|>      → arrives ALONE in chunk
Line 40:  <|tool_call_begin|>               → arrives ALONE in chunk  
Line 42:  " functions"                       → function name streamed character by character
Line 44:  ".task"
Line 46:  ":"
Line 48:  "45"
Line 50:  <|tool_call_argument_begin|>       → arrives ALONE in chunk (BUG: parser doesn't handle this!)
Line 52:  " {\""
Line 54:  "description"
...
Line 358: <|tool_call_end|>                   → arrives ALONE in chunk
Line 360: <|tool_call_begin|>                → next tool call starts
```

**Key Observation:** Tokens arrive in **separate chunks** from the content they delimit. The `<|tool_call_argument_begin|>` token comes in its own SSE chunk, not mixed with the function name.

**Problem with Real-Time Token Detection:**
The previous state machine in `StateInToolCallsSection` only checked for `<|tool_call_begin|>` and `<|tool_calls_section_end|>`. When `<|tool_call_argument_begin|>` arrived in a separate chunk, it fell through to the default case which buffered it as part of the function name, breaking the entire parsing.

**Solution:** 
1. Accumulate ALL content between `<|tool_calls_section_begin|>` and `<|tool_calls_section_end|>` without trying to parse tokens during streaming
2. After the section ends, parse the accumulated string to extract tool calls

This approach is:
- **Simpler**: No complex state transitions during streaming
- **More Robust**: Handles any token chunking pattern
- **Easier to Debug**: Parsing happens on complete data

### Updated State Machine

```
                    ┌─────────────────────┐
                    │       IDLE          │
                    │ (emit content)      │
                    └──────────┬──────────┘
                               │
         content before section│
                               ▼
              ┌────────────────────────────────┐
              │      ACCUMULATING              │
              │ (buffer ALL content)           │
              │ - Emit content before          │
              │ - Buffer until section_end     │
              └────────────┬───────────────────┘
                           │
 <|tool_calls_section_end|>│
                           │
                           ▼
              ┌────────────────────────────────┐
              │      PARSING                   │
              │ (parse accumulated content)    │
              │ - Split by tool_call_end       │
              │ - Extract name/ID/args         │
              │ - Emit tool_calls              │
              └────────────┬───────────────────┘
                           │
                           ▼
              ┌────────────────────────────────┐
              │       TRAILING                 │
              │ (emit remaining content)       │
              └────────────────────────────────┘
```

### Parsing Algorithm (After Accumulation)

Once the entire tool call section is accumulated, parsing is straightforward:

```go
// Accumulated content example:
// " functions.task:45 <|tool_call_argument_begin|> {JSON} <|tool_call_end|> <|tool_call_begin|> ..."

func parseToolCalls(accumulated string) []ToolCall {
    // Step 1: Split by tool_call_end to get individual tool calls
    parts := strings.Split(accumulated, "<|tool_call_end|>")
    
    var toolCalls []ToolCall
    for _, part := range parts {
        // Skip empty parts
        part = strings.TrimSpace(part)
        if part == "" {
            continue
        }
        
        // Step 2: Find the arguments section
        argBeginIdx := strings.Index(part, "<|tool_call_argument_begin|>")
        if argBeginIdx < 0 {
            continue
        }
        
        // Extract function name/ID (everything before argument begin)
        nameAndID := strings.TrimSpace(part[:argBeginIdx])
        
        // Extract ID if present (format: "name:ID" or just "name")
        var name, id string
        if colonIdx := strings.LastIndex(nameAndID, ":"); colonIdx >= 0 {
            name = strings.TrimSpace(nameAndID[:colonIdx])
            id = strings.TrimSpace(nameAndID[colonIdx+1:])
        } else {
            name = nameAndID
        }
        
        // Extract arguments (everything after argument begin)
        argsStart := argBeginIdx + len("<|tool_call_argument_begin|>")
        arguments := part[argsStart:]
        
        toolCalls = append(toolCalls, ToolCall{
            ID:   id,
            Name: name,
            Arguments: arguments,
        })
    }
    
    return toolCalls
}
```

### Updated ToolCallParser Structure

```go
type ParserState int

const (
    StateIdle ParserState = iota
    StateAccumulating    // Accumulating tool call section content
    StateParsing         // Parsing accumulated content
    StateTrailing        // Emitting content after tool calls
)

type ToolCallParser struct {
    state              ParserState
    accumulatedContent strings.Builder  // Buffer for tool section content
    pendingContent     strings.Builder  // Content before tool section
    toolCalls          []ToolCall       // Parsed tool calls
    metadata           StreamMetadata
}
```

### Updated Process() Logic

```go
func (p *ToolCallParser) Process(event *sse.Event) []TransformedEvent {
    // ... extract reasoning from event ...
    
    switch p.state {
    case StateIdle:
        if strings.Contains(reasoning, "<|tool_calls_section_begin|>") {
            // Content before section: emit as content
            beforeSection := extractContentBefore(reasoning, "<|tool_calls_section_begin|>")
            if beforeSection != "" {
                events = append(events, TransformedEvent{Type: EventContent, Content: beforeSection})
            }
            // Start accumulating
            p.state = StateAccumulating
            p.accumulatedContent.WriteString(extractContentAfter(reasoning, "<|tool_calls_section_begin|>"))
        } else {
            // Regular content, emit immediately
            events = append(events, TransformedEvent{Type: EventContent, Content: reasoning})
        }
        
    case StateAccumulating:
        if strings.Contains(reasoning, "<|tool_calls_section_end|>") {
            // Accumulate content before section end
            beforeEnd := extractContentBefore(reasoning, "<|tool_calls_section_end|>")
            if beforeEnd != "" {
                p.accumulatedContent.WriteString(beforeEnd)
            }
            // Switch to parsing state
            p.state = StateParsing
            events = append(events, p.parseAccumulatedContent()...)
            
            // Handle trailing content
            trailing := extractContentAfter(reasoning, "<|tool_calls_section_end|>")
            if trailing != "" {
                events = append(events, TransformedEvent{Type: EventContent, Content: trailing})
                p.state = StateTrailing
            }
        } else {
            // Continue accumulating
            p.accumulatedContent.WriteString(reasoning)
        }
        
    case StateTrailing:
        events = append(events, TransformedEvent{Type: EventContent, Content: reasoning})
    }
    
    return events
}
```

### Output: OpenAI Format

The parser emits OpenAI-format tool_calls:

```json
{
  "delta": {
    "tool_calls": [
      {
        "id": "call_abc123",
        "type": "function", 
        "function": {
          "name": "functions.task",
          "arguments": "{\"description\": \"...\", \"prompt\": \"...\"}"
        }
      }
    ]
  }
}
```

---

## Appendix A: Log Analysis

### Source: sse_2026-02-28_01-40-51.log

#### Extracted Reasoning Stream (lines 38-75)

```
[ 38] >>> SECTION_BEGIN >>> (content: " <|tool_calls_section_begin|>")
[ 40] >>> CALL_BEGIN >>> (name_buf: empty)  (content: " <|tool_call_begin|>")
[ 42] TOOL_NAME: " functions"
[ 44] TOOL_NAME: ".task"
[ 46] TOOL_NAME: ":"
[ 48] TOOL_NAME: "45"
[ 50] >>> ARG_BEGIN >>> (content: " <|tool_call_argument_begin|>")
       FINAL FUNCTION NAME: " functions.task:45"  <-- This was buffered incorrectly!
[ 52] TOOL_NAME: " {"
[ 54] TOOL_NAME: "description"
[ 56] TOOL_NAME: "\":"
[ 58] TOOL_NAME: " \""
[ 60] TOOL_NAME: "Explore"
[ 62] TOOL_NAME: " core"
[ 64] TOOL_NAME: " C"
[ 66] TOOL_NAME: " headers"
[ 68] TOOL_NAME: "\","
[ 70] TOOL_NAME: " \""
[ 72] TOOL_NAME: "prompt"
[ 74] TOOL_NAME: "\":"
```

#### Accumulated Tool Section Content

When everything between `<|tool_calls_section_begin|>` and `<|tool_calls_section_end|>` is accumulated:

```
<|tool_call_begin|> functions.task:45 <|tool_call_argument_begin|> {"description": "Explore core C headers", "prompt": "Recursively explore and analyze all core C standard library headers in /usr/include...", "subagent_type": "explore"} <|tool_call_end|> <|tool_call_begin|> functions.task:46 <|tool_call_argument_begin|> {"description": "Explore network headers", ...} <|tool_call_end|>
```

This clearly shows the format is:
```
<|tool_call_begin|>NAME:ID<|tool_call_argument_begin|>JSON_ARGS<|tool_call_end|>
```

### Why Accumulate-then-Parse Works Better

1. **Complete Data**: We have the full tool call section before attempting to parse
2. **Simple Split**: Just split by `<|tool_call_end|>` to get individual calls
3. **Reliable Extraction**: Use string operations to extract name/ID/arguments
4. **No State Complexity**: No need to track state during streaming

---

## Thread Safety

- Each request gets its own `ToolCallTransformer` instance
- Parser is not thread-safe (single-threaded stream processing)
- No shared state between concurrent requests

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Invalid JSON in event | Log error, passthrough original chunk |
| Unknown token format | Treat as regular content, passthrough |
| Buffer overflow | Limit buffer to 1MB, flush and reset |
| Writer error | Return error up the stack |

---

## Testing Strategy

### Unit Tests
- State machine transitions
- Token detection regex
- JSON building

### Integration Tests
- Use existing logs in `sse_logs/` as test fixtures
- Compare input (raw SSE) with expected output (OpenAI format)

### Test Fixtures
```go
// Example test case structure
var testCases = []struct {
    name     string
    input    []string // raw SSE data lines
    expected []string // expected transformed output
}{
    {
        name: "simple_tool_call",
        input: []string{
            `{"delta": {"reasoning": "<|tool_calls_section_begin|>"}}`,
            `{"delta": {"reasoning": "<|tool_call_begin|>call_1<|tool_call_argument_begin|>{"command":"ls"}<|tool_call_end|>"}}`,
            `{"delta": {"reasoning": "<|tool_calls_section_end|>"}}`,
            `[DONE]`,
        },
        expected: []string{
            `{"delta":{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"bash","arguments":"{\"command\":\"ls\""}}},"choices":[...]}`,
            // ...
        },
    },
}
```
