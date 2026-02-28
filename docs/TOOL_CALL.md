# Tool Call Transformation Specification

## Problem Statement

The Kimi-K2.5 model outputs tool calls using special delimiter tokens in the SSE `reasoning` and `reasoning_content` fields. These tokens must be transformed into proper OpenAI `tool_calls` format before being sent to downstream clients.

### Current Flow

```
Upstream (Kimi-K2.5) → ai-proxy → Downstream Client
                     ↑
              ToolCallTransformer
           (currently passes tokens through)
```

### Issue

The current `ToolCallTransformer` only filters `[DONE]` events. It does not parse or transform the special tokens, resulting in malformed tool_call responses to clients expecting OpenAI format.

---

## Log Samples (Current Behavior)

### SSE Event Structure from Kimi-K2.5

Each chunk contains special tokens in the `reasoning` field:

```json
{
  "id": "chatcmpl-8c3707e154df23bb",
  "object": "chat.completion.chunk",
  "model": "moonshotai/Kimi-K2.5-TEE",
  "choices": [{
    "index": 0,
    "delta": {
      "reasoning": " <|tool_calls_section_begin|>",
      "reasoning_content": " <|tool_calls_section_begin|>"
    }
  }]
}
```

### Special Tokens

| Token | Description |
|-------|-------------|
| `<|tool_calls_section_begin|>` | Starts the tool calls section |
| `<|tool_call_begin|>` | Starts a function call (function name follows) |
| `<|tool_call_argument_begin|>` | Starts the JSON arguments |
| `<|tool_call_end|>` | Ends the current tool call |
| `<|tool_calls_section_end|>` | Ends the tool calls section |

### Example Tool Call Sequence (from sse_2026-02-27_23-27-36.log)

```
1.  {"delta": {"reasoning": " <|tool_calls_section_begin|>"}}
2.  {"delta": {"reasoning": " <|tool_call_begin|>"}}
3.  {"delta": {"reasoning": " functions"}}
4.  {"delta": {"reasoning": ".b"}}
5.  {"delta": {"reasoning": "ash"}}
6.  {"delta": {"reasoning": ":"}}
7.  {"delta": {"reasoning": "15"}}
8.  {"delta": {"reasoning": " <|tool_call_argument_begin|>"}}
9.  {"delta": {"reasoning": " {\""}}
10. {"delta": {"reasoning": "command"}}
11. {"delta": {"reasoning": "\": "}}
12. {"delta": {"reasoning": " \"}}
13. {"delta": {"reasoning": "ls"}}
14. {"delta": {"reasoning": " -la /usr/include | grep asm\""}}
15. {"delta": {"reasoning": "}"}}
16. {"delta": {"reasoning": " <|tool_call_end|>"}}
17. {"delta": {"reasoning": " <|tool_call_begin|>"}}
    ... (second tool call)
18. {"delta": {"reasoning": " <|tool_call_end|>"}}
19. {"delta": {"reasoning": " <|tool_calls_section_end|>"}}
20. {"delta": {"reasoning": "", "finish_reason": "stop"}}
```

### Key Observations

1. Tokens arrive **split across multiple SSE chunks** (e.g., "functions. bash:15" is built character by character)
2. Both `reasoning` and `reasoning_content` fields contain identical content
3. Function name and arguments are streamed as plain text between special tokens
4. Multiple tool calls can be present in a single response

---

## Desired OpenAI Format

The transformed output should follow OpenAI's tool_calls specification:

### Tool Call Start

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion.chunk",
  "model": "moonshotai/Kimi-K2.5-TEE",
  "choices": [{
    "index": 0,
    "delta": {
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "bash",
          "arguments": ""
        }
      }]
    }
  }]
}
```

### Tool Call Arguments (chunked)

```json
{
  "choices": [{
    "delta": {
      "tool_calls": [{
        "id": "call_abc123",
        "function": {
          "arguments": "{\"command\": \""
        }
      }]
    }
  }]
}
```

```json
{
  "choices": [{
    "delta": {
      "tool_calls": [{
        "id": "call_abc123",
        "function": {
          "arguments": "ls -la /usr/include | grep asm\"}"
        }
      }]
    }
  }]
}
```

### Tool Call End

No special marker needed - the next chunk will either start a new tool call or end the section.

### Final Chunk (usage)

```json
{
  "choices": [],
  "usage": {
    "prompt_tokens": 43206,
    "completion_tokens": 133
  }
}
```

---

## Jinja Template (Upstream Format)

The upstream uses this Jinja template to generate tool calls:

```jinja
{%- macro render_toolcalls(message) -%}
  <|tool_calls_section_begin|>
  {%- for tool_call in message['tool_calls'] -%}
    {%- set formatted_id = tool_call['id'] -%}
    <|tool_call_begin|>{{ formatted_id }}<|tool_call_argument_begin|>{% if tool_call['function']['arguments'] is string %}{{ tool_call['function']['arguments'] }}{% else %}{{ tool_call['function']['arguments'] | tojson }}{% endif %}<|tool_call_end|>
  {%- endfor -%}
  <|tool_calls_section_end|>
{%- endmacro -%}
```

### Format Structure

```
<|tool_calls_section_begin|>
<|tool_call_begin|>call_123<|tool_call_argument_begin|>{"command": "ls"}<|tool_call_end|>
<|tool_call_begin|>call_456<|tool_call_argument_begin|>{"arg": "value"}<|tool_call_end|>
<|tool_calls_section_end|>
```

### Key Points

1. **ID follows immediately after** `<|tool_call_begin|>` (no space)
2. **Arguments follow immediately after** `<|tool_call_argument_begin|>` (no space)
3. **Tokens can be mixed** in a single SSE packet:
   - Content + tool call section + more content → split into 3 separate SSE events
   - Content + `<|tool_calls_section_end|>` + trailing content → split into 2 events
4. **Streaming**: The entire sequence is streamed as separate tokens/chunks

### Handling Mixed Content in Single SSE Packet

When a single SSE packet contains both regular content AND tool call tokens:

1. **Before section begin**: Output as separate `content` chunk
2. **Tool call tokens**: Transform to `tool_calls` format
3. **After section end**: Output as separate `content` chunk (if any trailing text)

Example input:
```json
{"delta": {"reasoning": "Hello <|tool_calls_section_begin|>world"}}
```

Expected output (2 SSE events):
```
data: {"delta": {"content": "Hello "}}

data: {"delta": {"reasoning": " <|tool_calls_section_begin|>world"}}
```

### State Machine

| State | Input | Action |
|-------|-------|--------|
| IDLE | Any text before section | Buffer as pending content |
| IDLE | `<|tool_calls_section_begin|>` | Flush pending content, enter TOOL_CALLS_SECTION |
| TOOL_CALLS_SECTION | `<|tool_call_begin|>` + ID + `<|tool_call_argument_begin|>` | Extract ID, enter ARGUMENTS |
| ARGUMENTS | text | Buffer arguments, output delta with partial args |
| ARGUMENTS | `<|tool_call_end|>` | Close function, stay in TOOL_CALLS_SECTION |
| TOOL_CALLS_SECTION | `<|tool_call_begin|>` | Start next tool call |
| TOOL_CALLS_SECTION | `<|tool_calls_section_end|>` | Exit section, enter TRAILING |
| TRAILING | any text | Output as content chunk |
| any | `finish_reason` | Pass through unchanged |

### ID Handling

- **From upstream**: ID is embedded right after `<|tool_call_begin|>` per Jinja template
- **Example**: `<|tool_call_begin|>call_abc123<|tool_call_argument_begin|>{...}`
- If ID is missing/invalid: Generate fallback ID `call_<index>_<timestamp>`

### Output Requirements

1. Preserve all metadata (id, object, model, created, etc.)
2. Output valid SSE format: `data: <json>\n\n`
3. Buffer and reassemble split tokens across chunks
4. Only output when tool_calls delta has meaningful content
5. Pass through non-tool-call chunks unchanged

---

## Edge Cases

1. **Empty arguments**: Function with no arguments
2. **Multiple tool calls**: Two or more functions in one response
3. **Nested tool calls**: Tool calling another tool
4. **Token boundary**: Function name or arguments split across SSE chunks
5. **No tool calls**: Regular text completion (passthrough)
6. **Error responses**: Non-200 status codes (passthrough)
7. **Mixed content**: Regular text + tool call tokens in same SSE packet
8. **Trailing content**: Text after `<|tool_calls_section_end|>` should be output as content

---

## Implementation Notes

### Files to Modify

- `downstream/tool_call_transformer.go` - Main transformation logic
- `downstream/completions.go` - May need adjustments for transformed output

### Dependencies

- `github.com/tmaxmax/go-sse` - Already in use for SSE parsing

### Testing

Use existing log files in `sse_logs/` for regression testing.
