# LLM API Protocol Conversion Reference Guide
### Anthropic Messages ↔ OpenAI Chat Completions ↔ OpenAI Responses
*Request & Response · Streaming · Tools · Content Blocks*

---

## Table of Contents

1. [Introduction](#1-introduction)
   - 1.1 [Protocol Overview](#11-protocol-overview)
   - 1.2 [Conversion Matrix](#12-conversion-matrix)
   - 1.3 [Terminology](#13-terminology)
2. [Protocol Field Reference](#2-protocol-field-reference)
   - 2.1 [Anthropic Messages API](#21-anthropic-messages-api)
   - 2.2 [OpenAI Chat Completions API](#22-openai-chat-completions-api)
   - 2.3 [OpenAI Responses API](#23-openai-responses-api)
3. [Anthropic → Chat Completions](#3-anthropic--chat-completions)
4. [Chat Completions → Anthropic](#4-chat-completions--anthropic)
5. [Responses → Anthropic](#5-responses--anthropic)
6. [Anthropic → Responses](#6-anthropic--responses)
7. [Chat Completions → Responses](#7-chat-completions--responses)
8. [Responses → Chat Completions](#8-responses--chat-completions)
9. [Cross-Protocol Corner Cases](#9-cross-protocol-corner-cases)
10. [Implementation Reference](#10-implementation-reference)

---

## 1. Introduction

This document is a comprehensive technical reference for converting API payloads between three LLM protocols:

- **Anthropic Messages API** (`claude-3-*`, `claude-opus-4`, etc.)
- **OpenAI Chat Completions API** (`/v1/chat/completions`)
- **OpenAI Responses API** (`/v1/responses`) — the stateful, event-driven successor to Chat Completions

Each section covers both directions of conversion, handling non-streaming and streaming (SSE) variants. Every field mapping is described with:

1. The source field name and type
2. The target field name and transformation
3. Behaviour when the field is absent, null, or has no target equivalent
4. Concrete JSON examples
5. Test cases including edge cases and corner cases

---

### 1.1 Protocol Overview

| Protocol | Endpoint | Paradigm | Statefulness | Streaming Format |
|---|---|---|---|---|
| Anthropic Messages | `POST /v1/messages` | Request/Response | Stateless | SSE — named events |
| Chat Completions | `POST /v1/chat/completions` | Request/Response | Stateless | SSE — `data:` lines |
| Responses | `POST /v1/responses` | Event-driven | Optional (`previous_response_id`) | SSE — named event types |

---

### 1.2 Conversion Matrix

| # | Source Protocol | Target Protocol | Section |
|---|---|---|---|
| 1 | Anthropic | Chat Completions | §3 |
| 2 | Chat Completions | Anthropic | §4 |
| 3 | Responses | Anthropic | §5 |
| 4 | Anthropic | Responses | §6 |
| 5 | Chat Completions | Responses | §7 |
| 6 | Responses | Chat Completions | §8 |

---

### 1.3 Terminology

- **SSE** — Server-Sent Events. A `text/event-stream` HTTP response where the server pushes event lines to the client.
- **Content block** — Anthropic's unit of message content (`text`, `image`, `tool_use`, `tool_result`, `thinking`).
- **Delta** — an incremental SSE chunk containing a partial content update.
- **Tool / Function call** — a structured request from the model to invoke a caller-defined function.
- **Stop reason / Finish reason** — why the model stopped generating (`end_turn`, `tool_use`, `max_tokens`, `stop`, `length`, `tool_calls`).

---

## 2. Protocol Field Reference

### 2.1 Anthropic Messages API

#### 2.1.1 Request Fields

| Field | Type | Notes |
|---|---|---|
| `model` | `string` | Required. e.g. `"claude-opus-4-5"`, `"claude-3-5-sonnet-20241022"` |
| `max_tokens` | `integer` | Required. Maximum tokens to generate. |
| `messages` | `array<Message>` | Required. Conversation history. |
| `messages[].role` | `"user" \| "assistant"` | Required per message. |
| `messages[].content` | `string \| array<Block>` | String = single text block. Array = multiple typed blocks. |
| `system` | `string \| array<SystemBlock>` | Optional. System prompt. Array form allows `cache_control`. |
| `tools` | `array<Tool>` | Optional. Enables tool/function calling. |
| `tools[].name` | `string` | Tool name, snake_case convention. |
| `tools[].description` | `string` | Description for the model. |
| `tools[].input_schema` | `JSON Schema` | JSON Schema describing tool parameters. |
| `tool_choice` | `"auto" \| "any" \| {type:"tool",name:string}` | `"auto"`=model decides, `"any"`=must use some tool, object=specific tool. |
| `temperature` | `float 0-1` | Optional. Sampling temperature. |
| `top_p` | `float 0-1` | Optional. Nucleus sampling. |
| `top_k` | `integer` | Optional. Top-k sampling. No OpenAI equivalent. |
| `stop_sequences` | `array<string>` | Optional. Strings that halt generation. |
| `stream` | `boolean` | Optional. If true, returns SSE stream. |
| `metadata.user_id` | `string` | Optional. User identifier for abuse tracking. |
| `thinking` | `{type:"enabled",budget_tokens:int}` | Optional. Extended thinking (claude-3-7+). |
| `betas` | `array<string>` | Optional. Feature flags (header: `anthropic-beta`). |

#### 2.1.2 Response Fields

| Field | Type | Notes |
|---|---|---|
| `id` | `string` | `"msg_..."` |
| `type` | `"message"` | Always `"message"`. |
| `role` | `"assistant"` | Always `"assistant"`. |
| `model` | `string` | Model that produced the response. |
| `content` | `array<Block>` | Array of content blocks. |
| `content[].type` | `"text" \| "tool_use" \| "thinking"` | Block type. |
| `content[].text` | `string` | For `type="text"`. |
| `content[].id` | `string` | For `tool_use`: `"toolu_..."` |
| `content[].name` | `string` | For `tool_use`: tool name. |
| `content[].input` | `object` | For `tool_use`: parsed tool arguments. |
| `stop_reason` | `"end_turn" \| "tool_use" \| "max_tokens" \| "stop_sequence"` | Why generation stopped. |
| `stop_sequence` | `string \| null` | Which stop sequence triggered stop (if any). |
| `usage.input_tokens` | `integer` | Input token count. |
| `usage.output_tokens` | `integer` | Output token count. |
| `usage.cache_read_input_tokens` | `integer` | Prompt cache read tokens (optional). |
| `usage.cache_creation_input_tokens` | `integer` | Prompt cache write tokens (optional). |

#### 2.1.3 Streaming SSE Events

| Event | Shape | Notes |
|---|---|---|
| `message_start` | `{type, message:{id,type,role,model,content:[],usage:{input_tokens,...}}}` | First event. Contains initial message metadata. |
| `content_block_start` | `{type, index:int, content_block:{type,text?}}` | Signals start of a content block. |
| `content_block_delta` | `{type, index:int, delta:{type,text? \| partial_json?}}` | Incremental content. `type="text_delta"` or `"input_json_delta"`. |
| `content_block_stop` | `{type, index:int}` | End of a content block. |
| `message_delta` | `{type, delta:{stop_reason,stop_sequence?}, usage:{output_tokens}}` | Final metadata update. |
| `message_stop` | `{type}` | Stream complete. |
| `ping` | `{type:"ping"}` | Keep-alive. Can be safely ignored. |
| `error` | `{type:"error", error:{type,message}}` | Stream-level error. |

---

### 2.2 OpenAI Chat Completions API

#### 2.2.1 Request Fields

| Field | Type | Notes |
|---|---|---|
| `model` | `string` | Required. e.g. `"gpt-4o"`, `"gpt-4-turbo"`. |
| `messages` | `array<Message>` | Required. |
| `messages[].role` | `"system" \| "user" \| "assistant" \| "tool"` | Required per message. |
| `messages[].content` | `string \| array<Part>` | For user: can be array with `text`/`image_url` parts. |
| `messages[].tool_calls` | `array<ToolCall>` | In assistant messages when model requested tools. |
| `messages[].tool_call_id` | `string` | In tool messages referencing which call this responds to. |
| `messages[].name` | `string` | Optional. Disambiguates participants with same role. |
| `max_tokens` | `integer` | Deprecated alias; use `max_completion_tokens`. |
| `max_completion_tokens` | `integer` | Maximum output tokens. |
| `temperature` | `float 0-2` | Sampling temperature. Note: OAI range is 0-2, Anthropic is 0-1. |
| `top_p` | `float` | Nucleus sampling. |
| `n` | `integer` | Number of choices. Anthropic does not support `n>1`. |
| `stop` | `string \| array<string>` | Stop sequences. Up to 4 strings. |
| `stream` | `boolean` | If true, returns SSE stream. |
| `stream_options` | `{include_usage:bool}` | Whether to include usage in stream. |
| `tools` | `array<Tool>` | Function tools. |
| `tools[].type` | `"function"` | Currently only `"function"`. |
| `tools[].function.name` | `string` | Function name. |
| `tools[].function.description` | `string` | Function description. |
| `tools[].function.parameters` | `JSON Schema` | Parameter schema. |
| `tool_choice` | `"auto" \| "none" \| "required" \| {type:"function",function:{name}}` | Tool selection strategy. |
| `response_format` | `{type:"text" \| "json_object" \| "json_schema"}` | Output format constraint. |
| `frequency_penalty` | `float -2 to 2` | No Anthropic equivalent. |
| `presence_penalty` | `float -2 to 2` | No Anthropic equivalent. |
| `logprobs` | `boolean` | No Anthropic equivalent. |
| `seed` | `integer` | No Anthropic equivalent. |
| `user` | `string` | End-user ID. Maps to `metadata.user_id`. |

#### 2.2.2 Response Fields

| Field | Type | Notes |
|---|---|---|
| `id` | `string` | `"chatcmpl-..."` |
| `object` | `"chat.completion"` | Always this value. |
| `created` | `integer` | Unix timestamp. |
| `model` | `string` | Model used. |
| `choices` | `array<Choice>` | Always array; usually 1 element. |
| `choices[].index` | `integer` | Choice index (0 when `n=1`). |
| `choices[].message.role` | `"assistant"` | Always assistant. |
| `choices[].message.content` | `string \| null` | Text content. Null when `tool_calls` present. |
| `choices[].message.tool_calls` | `array<ToolCall> \| null` | Tool calls if model invoked functions. |
| `choices[].message.tool_calls[].id` | `string` | `"call_..."` |
| `choices[].message.tool_calls[].type` | `"function"` | Always `"function"`. |
| `choices[].message.tool_calls[].function.name` | `string` | Function name. |
| `choices[].message.tool_calls[].function.arguments` | `string` | JSON-encoded arguments string. |
| `choices[].finish_reason` | `"stop" \| "length" \| "tool_calls" \| "content_filter"` | Why generation ended. |
| `usage.prompt_tokens` | `integer` | Input token count. |
| `usage.completion_tokens` | `integer` | Output token count. |
| `usage.total_tokens` | `integer` | Sum of both. |

#### 2.2.3 Streaming SSE Format

| Field | Notes |
|---|---|
| `data: {chunk}` | Each chunk is a `ChatCompletionChunk` JSON. Lines prefixed with `"data: "`. |
| `chunk.choices[].delta.role` | `"assistant"` — only in first chunk. |
| `chunk.choices[].delta.content` | Incremental text (`string \| null`). |
| `chunk.choices[].delta.tool_calls` | Incremental tool call data. |
| `chunk.choices[].delta.tool_calls[].index` | Which tool call this delta belongs to. |
| `chunk.choices[].finish_reason` | Non-null only in final content chunk. |
| `chunk.usage` | Only when `stream_options.include_usage=true`. |
| `data: [DONE]` | Final line indicating stream end. |

---

### 2.3 OpenAI Responses API

#### 2.3.1 Request Fields

| Field | Type | Notes |
|---|---|---|
| `model` | `string` | Required. e.g. `"gpt-4o"`. |
| `input` | `string \| array<InputItem>` | Required. String = single user message. Array = multi-turn. |
| `input[].type` | `"message"` | Message container. |
| `input[].role` | `"user" \| "assistant" \| "system"` | Message role. |
| `input[].content` | `string \| array<ContentPart>` | Content parts. |
| `input[].content[].type` | `"input_text" \| "input_image" \| "input_file"` | Content part type. |
| `instructions` | `string` | System prompt (top-level, replaces system message). |
| `tools` | `array<Tool>` | Function tools, `file_search`, `web_search_preview`, `computer_use_preview`. |
| `tools[].type` | `"function" \| "file_search" \| "web_search_preview" \| "computer_use_preview"` | Tool type. |
| `tools[].name` | `string` | For function tools. |
| `tools[].description` | `string` | For function tools. |
| `tools[].parameters` | `JSON Schema` | For function tools. |
| `tool_choice` | `"auto" \| "none" \| "required" \| {type:"function",name:string}` | Tool selection. |
| `max_output_tokens` | `integer` | Maximum tokens. Note: field name differs from Chat Completions. |
| `temperature` | `float` | Sampling temperature. |
| `top_p` | `float` | Nucleus sampling. |
| `stream` | `boolean` | Enable SSE streaming. |
| `previous_response_id` | `string` | For multi-turn state; replaces full history. |
| `truncation` | `"auto" \| "disabled"` | Context truncation strategy. |
| `metadata` | `object` | Arbitrary key-value metadata. |
| `store` | `boolean` | Whether to store the response for later retrieval. |
| `reasoning` | `{effort:"low" \| "medium" \| "high"}` | Reasoning effort for o-series models. |

#### 2.3.2 Response Fields

| Field | Type | Notes |
|---|---|---|
| `id` | `string` | `"resp_..."` |
| `object` | `"response"` | Always `"response"`. |
| `created_at` | `integer` | Unix timestamp. |
| `model` | `string` | Model used. |
| `status` | `"completed" \| "failed" \| "in_progress" \| "incomplete"` | Response lifecycle status. |
| `output` | `array<OutputItem>` | Array of output items. |
| `output[].type` | `"message" \| "function_call" \| "file_search_call" \| "web_search_call"` | Output item type. |
| `output[].id` | `string` | Item ID. |
| `output[].role` | `"assistant"` | For message items. |
| `output[].content` | `array<ContentPart>` | For message items. |
| `output[].content[].type` | `"output_text" \| "refusal"` | Content part type. |
| `output[].content[].text` | `string` | For `output_text`. |
| `output[].name` | `string` | For `function_call` items. |
| `output[].call_id` | `string` | For `function_call` items. |
| `output[].arguments` | `string` | For `function_call`: JSON-encoded arguments. |
| `usage.input_tokens` | `integer` | Input token count. |
| `usage.output_tokens` | `integer` | Output token count. |
| `usage.total_tokens` | `integer` | Total tokens. |
| `error` | `object \| null` | Error if `status=failed`. |

#### 2.3.3 Streaming SSE Events

| Event | Shape | Notes |
|---|---|---|
| `response.created` | `{type, response:{id,status:"in_progress",...}}` | Response object created. |
| `response.in_progress` | `{type, response}` | Response being processed. |
| `response.output_item.added` | `{type, output_index, item}` | New output item started. |
| `response.content_part.added` | `{type, item_id, output_index, content_index, part}` | New content part started. |
| `response.output_text.delta` | `{type, item_id, output_index, content_index, delta}` | Incremental text. |
| `response.output_text.done` | `{type, item_id, output_index, content_index, text}` | Text part complete. |
| `response.function_call_arguments.delta` | `{type, item_id, output_index, delta}` | Incremental function args. |
| `response.function_call_arguments.done` | `{type, item_id, output_index, arguments}` | Function args complete. |
| `response.output_item.done` | `{type, output_index, item}` | Output item complete. |
| `response.completed` | `{type, response}` | Final complete response. Contains full response object. |
| `response.failed` | `{type, response}` | Response failed. |
| `response.incomplete` | `{type, response}` | Response incomplete (e.g. `max_output_tokens` hit). |

---

## 3. Anthropic → Chat Completions

Convert Anthropic Messages API requests/responses to OpenAI Chat Completions format.

### 3.1 Request Conversion

#### 3.1.1 Field Mapping

| Source Field | Target Field | Conversion Notes |
|---|---|---|
| `model` | `model` | Pass through or remap via model map. |
| `max_tokens` | `max_completion_tokens` | Direct mapping. Required in both. |
| `messages` | `messages` | See §3.1.2 for message-level conversion. |
| `system` (string) | `messages[0]` with `role:"system"` | Prepend a system message. |
| `system` (array) | `messages[0]` with `role:"system"`, joined text | Join text of all system blocks. |
| `tools` | `tools` | See §3.1.3 for tool conversion. |
| `tool_choice: "auto"` | `tool_choice: "auto"` | Direct mapping. |
| `tool_choice: "any"` | `tool_choice: "required"` | `"any"` → `"required"`. |
| `tool_choice: {type:"tool",name}` | `tool_choice: {type:"function",function:{name}}` | Wrap in function object. |
| `temperature` | `temperature` | Direct mapping (OAI range is 0-2; Anthropic 0-1 is safe). |
| `top_p` | `top_p` | Direct mapping. |
| `top_k` | *(drop)* | No Chat Completions equivalent. Drop silently. |
| `stop_sequences` | `stop` | Direct mapping. OAI accepts string or array. |
| `stream` | `stream` | Direct mapping. |
| `metadata.user_id` | `user` | Map nested field to top-level `user`. |
| `thinking` | *(drop)* | No Chat Completions equivalent. Drop. |
| *(absent)* | `stream_options: {include_usage:true}` | Always add when `stream=true` to get usage in stream. |

#### 3.1.2 Message Conversion

| Source | Target | Notes |
|---|---|---|
| `role:"user", content:string` | `{role:"user",content:string}` | Direct mapping. |
| `role:"user", content:[{type:"text",text}]` | `{role:"user",content:string}` | Extract text if only one text block. |
| `role:"user", content:[{type:"image",source:{type:"base64",...}}]` | `{role:"user",content:[{type:"image_url",image_url:{url:"data:...;base64,..."}}]}` | Convert base64 image to data URI. |
| `role:"user", content:[{type:"image",source:{type:"url",url}}]` | `{role:"user",content:[{type:"image_url",image_url:{url}}]}` | URL images map directly. |
| `role:"user", content:[{type:"tool_result",tool_use_id,content}]` | `{role:"tool",tool_call_id,content:string}` | Convert tool result to `tool` role. |
| `role:"assistant", content:string` | `{role:"assistant",content:string}` | Direct mapping. |
| `role:"assistant", content:[{type:"text"}]` | `{role:"assistant",content:string}` | Extract text. |
| `role:"assistant", content:[{type:"tool_use",id,name,input}]` | `{role:"assistant",content:null,tool_calls:[{id,type:"function",function:{name,arguments:JSON.stringify(input)}}]}` | Convert `tool_use` to `tool_calls`. |
| `role:"assistant", content:[text, tool_use]` | `{role:"assistant",content:text,tool_calls:[...]}` | Mixed: keep text, add `tool_calls`. |
| `role:"assistant", content:[{type:"thinking"}]` | *(drop thinking block)* | No Chat Completions equivalent. |

#### 3.1.3 Tool Conversion

| Source Field | Target Field | Notes |
|---|---|---|
| `tools[].name` | `tools[].function.name` | Direct. |
| `tools[].description` | `tools[].function.description` | Direct. |
| `tools[].input_schema` | `tools[].function.parameters` | Direct (both are JSON Schema). |
| *(absent type)* | `tools[].type = "function"` | Always set `type` to `"function"`. |

#### 3.1.4 Request Example

```json
// ANTHROPIC REQUEST
{
  "model": "claude-3-5-sonnet-20241022",
  "max_tokens": 1024,
  "system": "You are a helpful assistant.",
  "messages": [
    { "role": "user", "content": "What is 2+2?" },
    {
      "role": "assistant",
      "content": [
        { "type": "text", "text": "Let me check." },
        {
          "type": "tool_use",
          "id": "toolu_01XyZ",
          "name": "calculator",
          "input": { "expression": "2+2" }
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "tool_result",
          "tool_use_id": "toolu_01XyZ",
          "content": "4"
        }
      ]
    }
  ],
  "tools": [{
    "name": "calculator",
    "description": "Evaluates math expressions.",
    "input_schema": {
      "type": "object",
      "properties": { "expression": { "type": "string" } },
      "required": ["expression"]
    }
  }],
  "tool_choice": { "type": "tool", "name": "calculator" },
  "temperature": 0.7,
  "stop_sequences": ["END"],
  "metadata": { "user_id": "user_abc" }
}
```

```json
// CONVERTED CHAT COMPLETIONS REQUEST
{
  "model": "gpt-4o",
  "max_completion_tokens": 1024,
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "What is 2+2?" },
    {
      "role": "assistant",
      "content": "Let me check.",
      "tool_calls": [{
        "id": "toolu_01XyZ",
        "type": "function",
        "function": {
          "name": "calculator",
          "arguments": "{\"expression\":\"2+2\"}"
        }
      }]
    },
    {
      "role": "tool",
      "tool_call_id": "toolu_01XyZ",
      "content": "4"
    }
  ],
  "tools": [{
    "type": "function",
    "function": {
      "name": "calculator",
      "description": "Evaluates math expressions.",
      "parameters": {
        "type": "object",
        "properties": { "expression": { "type": "string" } },
        "required": ["expression"]
      }
    }
  }],
  "tool_choice": { "type": "function", "function": { "name": "calculator" } },
  "temperature": 0.7,
  "stop": ["END"],
  "user": "user_abc"
}
```

---

### 3.2 Response Conversion (Non-Streaming)

#### 3.2.1 Field Mapping

| Source Field | Target Field | Conversion Notes |
|---|---|---|
| `id` (`chatcmpl-...`) | `id` | Remap or pass through. |
| `created` | *(synthetic)* | Anthropic responses have no `created`; generate one. |
| `model` | `model` | Direct. |
| `choices[0].message.content` (string) | `content:[{type:"text",text}]` | Wrap in text block. |
| `choices[0].message.content` (null) | `content:[]` | Empty content array. |
| `choices[0].message.tool_calls[].function.name` | `content[].type="tool_use", .name` | Convert each tool call. |
| `choices[0].message.tool_calls[].function.arguments` | `content[].input = JSON.parse(arguments)` | Parse string to object. |
| `choices[0].message.tool_calls[].id` | `content[].id` | Preserve call ID. |
| `choices[0].finish_reason:"stop"` | `stop_reason:"end_turn"` | Map finish reason. |
| `choices[0].finish_reason:"length"` | `stop_reason:"max_tokens"` | Map finish reason. |
| `choices[0].finish_reason:"tool_calls"` | `stop_reason:"tool_use"` | Map finish reason. |
| `choices[0].finish_reason:"content_filter"` | `stop_reason:"end_turn"` | Best-effort approximation. |
| `usage.prompt_tokens` | `usage.input_tokens` | Rename. |
| `usage.completion_tokens` | `usage.output_tokens` | Rename. |
| `usage.total_tokens` | *(drop)* | Anthropic does not include `total_tokens`. |
| *(absent)* | `type:"message"` | Add `type` field. |
| *(absent)* | `role:"assistant"` | Add `role` field. |

#### 3.2.2 Response Example

```json
// CHAT COMPLETIONS RESPONSE
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1730000000,
  "model": "gpt-4o",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_xyz",
        "type": "function",
        "function": {
          "name": "calculator",
          "arguments": "{\"expression\":\"2+2\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }],
  "usage": {
    "prompt_tokens": 85,
    "completion_tokens": 22,
    "total_tokens": 107
  }
}
```

```json
// CONVERTED ANTHROPIC RESPONSE
{
  "id": "chatcmpl-abc123",
  "type": "message",
  "role": "assistant",
  "model": "gpt-4o",
  "content": [{
    "type": "tool_use",
    "id": "call_xyz",
    "name": "calculator",
    "input": { "expression": "2+2" }
  }],
  "stop_reason": "tool_use",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 85,
    "output_tokens": 22
  }
}
```

---

### 3.3 Streaming Conversion (Anthropic → Chat Completions)

#### 3.3.1 Event Mapping

| Source Event | Target | Notes |
|---|---|---|
| `message_start` | First delta chunk with `role:"assistant"` | Emit chunk with `delta.role="assistant"`, `delta.content=""`. |
| `content_block_start` (text) | *(buffer)* | No chunk emitted. Record block index. |
| `content_block_delta` (`text_delta`) | chunk with `delta.content=text` | Emit one chunk per delta. |
| `content_block_start` (tool_use) | *(buffer, emit tool_calls start)* | Emit chunk with `delta.tool_calls=[{index,id,type:"function",function:{name,arguments:""}}]`. |
| `content_block_delta` (`input_json_delta`) | chunk with `delta.tool_calls=[{index,function:{arguments:partial_json}}]` | Emit partial argument JSON. |
| `content_block_stop` | *(buffer)* | No chunk needed. |
| `message_delta` (stop_reason) | Final chunk with `finish_reason` | Emit chunk with `finish_reason` translated. |
| `message_stop` | `data: [DONE]` | Emit DONE sentinel. |
| `ping` | *(drop)* | Ignore ping events. |

> ⚠️ **Note:** Chat Completions `tool_calls` index must be maintained correctly. Each `tool_use` block from Anthropic maps to one tool call entry, with `index` matching insertion order starting from 0.

#### 3.3.2 Streaming Example

```
// ANTHROPIC STREAM EVENTS:
event: message_start
data: {"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","model":"claude-3-5-sonnet-20241022","content":[],"usage":{"input_tokens":45}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"The answer"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" is 4."}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":10}}

event: message_stop
data: {"type":"message_stop"}

// CONVERTED CHAT COMPLETIONS STREAM:
data: {"id":"msg_01","object":"chat.completion.chunk","created":1730000000,"model":"claude-3-5-sonnet-20241022","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"msg_01","object":"chat.completion.chunk","created":1730000000,"model":"claude-3-5-sonnet-20241022","choices":[{"index":0,"delta":{"content":"The answer"},"finish_reason":null}]}

data: {"id":"msg_01","object":"chat.completion.chunk","created":1730000000,"model":"claude-3-5-sonnet-20241022","choices":[{"index":0,"delta":{"content":" is 4."},"finish_reason":null}]}

data: {"id":"msg_01","object":"chat.completion.chunk","created":1730000000,"model":"claude-3-5-sonnet-20241022","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":45,"completion_tokens":10,"total_tokens":55}}

data: [DONE]
```

---

### 3.4 Test Cases

#### 3.4.1 Request Test Cases

| Test Case | Input | Expected Output | Notes |
|---|---|---|---|
| Simple text message | `messages:[{role:"user",content:"Hi"}]` | `messages:[{role:"user",content:"Hi"}]` | String content passes through unchanged. |
| System string | `system:"Be helpful"` | `messages[0]:{role:"system",content:"Be helpful"}` | System prepended as first message. |
| System array | `system:[{type:"text",text:"Part1"},{type:"text",text:"Part2"}]` | `messages[0]:{role:"system",content:"Part1Part2"}` | Text parts concatenated. |
| Image base64 | `content:[{type:"image",source:{type:"base64",media_type:"image/png",data:"abc..."}}]` | `content:[{type:"image_url",image_url:{url:"data:image/png;base64,abc..."}}]` | Construct data URI from `media_type`+`data`. |
| Image URL | `source:{type:"url",url:"https://..."}` | `image_url:{url:"https://..."}` | URL passes through. |
| `tool_choice:"any"` | `tool_choice:"any"` | `tool_choice:"required"` | `"any"` → `"required"`. |
| `tool_choice` object | `{type:"tool",name:"calc"}` | `{type:"function",function:{name:"calc"}}` | Wrap in function object. |
| `top_k` present | `top_k:40` | *(field absent)* | Drop silently. |
| `thinking` enabled | `thinking:{type:"enabled",budget_tokens:2000}` | *(field absent)* | Drop. No equivalent. |
| `metadata.user_id` | `metadata:{user_id:"u1"}` | `user:"u1"` | Map nested to top-level. |
| `stop_sequences` | `stop_sequences:["STOP","END"]` | `stop:["STOP","END"]` | Array maps directly. |
| Tool result array content | `content:[{type:"tool_result",content:[{type:"text",text:"ok"}]}]` | `{role:"tool",content:"ok"}` | Extract text from array content. |
| Multiple `tool_use` blocks | `content:[{type:"tool_use",id:"t1",...},{type:"tool_use",id:"t2",...}]` | `tool_calls:[{id:"t1",...},{id:"t2",...}]` | All tool calls in one assistant message. |
| Mixed text+tool_use | `content:[{type:"text",text:"ok"},{type:"tool_use",...}]` | `{content:"ok",tool_calls:[...]}` | Text in `content`, tools in `tool_calls`. |

#### 3.4.2 Response Test Cases

| Test Case | Input | Expected Output | Notes |
|---|---|---|---|
| Text response | `choices[0].message.content:"Hello"` | `content:[{type:"text",text:"Hello"}]` | Wrap string in block array. |
| Null content | `choices[0].message.content:null` | `content:[]` | Empty content array. |
| `finish_reason:"stop"` | `finish_reason:"stop"` | `stop_reason:"end_turn"` | Map `stop` → `end_turn`. |
| `finish_reason:"length"` | `finish_reason:"length"` | `stop_reason:"max_tokens"` | Map `length` → `max_tokens`. |
| `finish_reason:"tool_calls"` | `finish_reason:"tool_calls"` | `stop_reason:"tool_use"` | Map `tool_calls` → `tool_use`. |
| Tool call arg parsing | `function.arguments:"{\"x\":1}"` | `input:{x:1}` | `JSON.parse` the arguments string. |
| Malformed tool args | `function.arguments:"not json"` | `input:{}` (or raw string) | Catch parse error; use empty object or raw string fallback. |
| Usage rename | `usage:{prompt_tokens:10,completion_tokens:5}` | `usage:{input_tokens:10,output_tokens:5}` | Rename fields; drop `total_tokens`. |
| `n>1` choices | `choices:[{...},{...}]` | content from `choices[0]` | Only first choice used; Anthropic has no multi-choice. |
| `content_filter` finish | `finish_reason:"content_filter"` | `stop_reason:"end_turn"` | Best approximation. |

---

## 4. Chat Completions → Anthropic

Convert OpenAI Chat Completions requests/responses to Anthropic Messages format.

### 4.1 Request Conversion

#### 4.1.1 Field Mapping

| Source Field | Target Field | Conversion Notes |
|---|---|---|
| `model` | `model` | Pass through or remap. |
| `max_completion_tokens` / `max_tokens` | `max_tokens` | Try `max_completion_tokens` first; fall back to `max_tokens`. Required by Anthropic. |
| `messages` (system role) | `system` | Extract system message(s) to top-level `system` field. |
| `messages` (user/assistant/tool) | `messages` | See §4.1.2. |
| `temperature` | `temperature` | Direct. OAI allows 0-2; clamp to 0-1 for Anthropic. |
| `top_p` | `top_p` | Direct. |
| `n` | *(drop, warn if >1)* | Anthropic does not support `n>1`. Use `n=1` only. |
| `stop` (string) | `stop_sequences:[stop]` | Wrap single string in array. |
| `stop` (array) | `stop_sequences` | Direct array mapping. |
| `stream` | `stream` | Direct. |
| `tools` | `tools` | See §4.1.3. |
| `tool_choice:"auto"` | `tool_choice:"auto"` | Direct. |
| `tool_choice:"none"` | *(drop tools from request)* | Anthropic has no `"none"`; remove tools instead. |
| `tool_choice:"required"` | `tool_choice:"any"` | `"required"` → `"any"`. |
| `tool_choice:{type:"function",function:{name}}` | `tool_choice:{type:"tool",name}` | Restructure. |
| `response_format` | *(drop)* | No Anthropic equivalent; can use prompt engineering. |
| `frequency_penalty` | *(drop)* | No equivalent. |
| `presence_penalty` | *(drop)* | No equivalent. |
| `logprobs` | *(drop)* | No equivalent. |
| `seed` | *(drop)* | No equivalent. |
| `user` | `metadata.user_id` | Map to metadata. |
| `stream_options` | *(drop)* | Not applicable in Anthropic. |

> ℹ️ **Note:** `max_tokens` is required in Anthropic. If neither `max_tokens` nor `max_completion_tokens` is present, you must inject a default (e.g. `4096`) or return an error.

#### 4.1.2 Message Conversion

| Source | Target | Notes |
|---|---|---|
| `role:"system"` | Extract to top-level `system` field | Multiple system messages: join with `\n\n`. |
| `role:"user", content:string` | `{role:"user",content:string}` | Direct. |
| `role:"user", content:[{type:"text",text}]` | `{role:"user",content:[{type:"text",text}]}` | Direct array form. |
| `role:"user", content:[{type:"image_url",image_url:{url:"data:image/png;base64,..."}}]` | `{type:"image",source:{type:"base64",media_type:"image/png",data:"..."}}` | Parse data URI into source object. |
| `role:"user", content:[{type:"image_url",image_url:{url:"https://..."}}]` | `{type:"image",source:{type:"url",url:"..."}}` | URL image source. |
| `role:"assistant", content:string` | `{role:"assistant",content:string}` | Direct. |
| `role:"assistant", tool_calls:[...]` | `{role:"assistant",content:[{type:"tool_use",...}]}` | See tool_calls conversion below. |
| `role:"assistant", content+tool_calls` | `{role:"assistant",content:[{type:"text",...},{type:"tool_use",...}]}` | Combine text and tool_use blocks. |
| `role:"tool", tool_call_id, content` | `{role:"user",content:[{type:"tool_result",tool_use_id,content}]}` | Tool results become user role with `tool_result` block. |
| Multiple tool messages | Combine into one user message with multiple `tool_result` blocks | Anthropic allows multiple `tool_result` in one user message. |

#### 4.1.3 Tool Conversion

| Source Field | Target Field | Notes |
|---|---|---|
| `tools[].function.name` | `tools[].name` | Direct. |
| `tools[].function.description` | `tools[].description` | Direct. |
| `tools[].function.parameters` | `tools[].input_schema` | Direct (both JSON Schema). |
| `tools[].type` (always `"function"`) | *(drop)* | Anthropic tools have no `type` field at top level. |

#### 4.1.4 Request Example

```json
// CHAT COMPLETIONS REQUEST
{
  "model": "gpt-4o",
  "max_completion_tokens": 512,
  "temperature": 1.8,
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "Search for cats." },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_abc",
        "type": "function",
        "function": { "name": "search", "arguments": "{\"query\":\"cats\"}" }
      }]
    },
    { "role": "tool", "tool_call_id": "call_abc", "content": "Found 100 results." }
  ],
  "tools": [{ "type": "function", "function": {
    "name": "search", "description": "Web search.",
    "parameters": { "type":"object","properties":{"query":{"type":"string"}},"required":["query"] }
  }}],
  "tool_choice": "required",
  "stop": "DONE",
  "user": "user_xyz"
}
```

```json
// CONVERTED ANTHROPIC REQUEST
{
  "model": "claude-3-5-sonnet-20241022",
  "max_tokens": 512,
  "system": "You are a helpful assistant.",
  "temperature": 1.0,
  "messages": [
    { "role": "user", "content": "Search for cats." },
    {
      "role": "assistant",
      "content": [{
        "type": "tool_use",
        "id": "call_abc",
        "name": "search",
        "input": { "query": "cats" }
      }]
    },
    {
      "role": "user",
      "content": [{
        "type": "tool_result",
        "tool_use_id": "call_abc",
        "content": "Found 100 results."
      }]
    }
  ],
  "tools": [{
    "name": "search",
    "description": "Web search.",
    "input_schema": { "type":"object","properties":{"query":{"type":"string"}},"required":["query"] }
  }],
  "tool_choice": { "type": "any" },
  "stop_sequences": ["DONE"],
  "metadata": { "user_id": "user_xyz" }
}
```

> ⚠️ **Note:** Temperature clamping: OAI temperature of `1.8` is clamped to `1.0` for Anthropic. Log this adjustment.

---

### 4.2 Response Conversion

#### 4.2.1 Field Mapping

| Source Field | Target Field | Conversion Notes |
|---|---|---|
| `id` (`msg_...`) | `id` | Pass through. |
| `type:"message"` | `object:"chat.completion"` | Set `object` field. |
| *(absent)* | `created` | Generate Unix timestamp. |
| `model` | `model` | Direct. |
| `role:"assistant"` | `choices[0].message.role:"assistant"` | Wrap in `choices` array. |
| `content:[{type:"text",text}]` | `choices[0].message.content:text` | Extract text string. |
| `content:[{type:"tool_use",...}]` | `choices[0].message.content:null, tool_calls:[...]` | Convert to `tool_calls`. |
| `content:[{type:"tool_use",id,name,input}]` | `tool_calls[].{id,type:"function",function:{name,arguments:JSON.stringify(input)}}` | Serialize input to arguments string. |
| `stop_reason:"end_turn"` | `finish_reason:"stop"` | Map stop reason. |
| `stop_reason:"max_tokens"` | `finish_reason:"length"` | Map stop reason. |
| `stop_reason:"tool_use"` | `finish_reason:"tool_calls"` | Map stop reason. |
| `stop_reason:"stop_sequence"` | `finish_reason:"stop"` | Map to stop. |
| `usage.input_tokens` | `usage.prompt_tokens` | Rename. |
| `usage.output_tokens` | `usage.completion_tokens` | Rename. |
| *(absent)* | `usage.total_tokens` | Compute as `input + output`. |
| *(absent)* | `choices[0].index:0` | Always 0. |
| *(absent)* | `object:"chat.completion"` | Set object type. |

---

### 4.3 Streaming Conversion (Chat Completions → Anthropic)

#### 4.3.1 Event Mapping

| Source Event | Target | Notes |
|---|---|---|
| First chunk with `delta.role` | `message_start` + `content_block_start` | Emit `message_start` then `content_block_start`. |
| `delta.content` text chunk | `content_block_delta` with `text_delta` | Wrap text in Anthropic delta format. |
| `delta.tool_calls[i]` first chunk (has id, name) | `content_block_start` with `tool_use` block | Start `tool_use` block with type/id/name. |
| `delta.tool_calls[i]` arg chunk | `content_block_delta` with `input_json_delta` | Wrap partial JSON. |
| `finish_reason` non-null chunk | `content_block_stop`, `message_delta`, `message_stop` | Close block, emit metadata, emit stop. |
| `data: [DONE]` | *(already emitted `message_stop`)* | No additional event needed. |
| `usage` in stream chunk | `message_delta` usage field | Map usage fields when present. |

---

### 4.4 Test Cases

| Test Case | Input | Expected Output | Notes |
|---|---|---|---|
| Temperature clamping | `temperature:2.0` | `temperature:1.0` | OAI max 2.0 → Anthropic max 1.0. |
| `tool_choice:"none"` | `tool_choice:"none", tools:[...]` | tools omitted from request | Remove tools entirely to prevent use. |
| `tool_choice:"required"` | `tool_choice:"required"` | `tool_choice:{type:"any"}` | `"required"` → `"any"`. |
| No max tokens | *(absent)* | `max_tokens:4096` (default) | Inject default; Anthropic requires it. |
| Multiple system msgs | `[{role:"system",content:"A"},{role:"system",content:"B"}]` | `system:"A\n\nB"` | Join with double newline. |
| Tool result string | `{role:"tool",content:"result"}` | `{type:"tool_result",content:"result"}` | Wrap in `tool_result` block. |
| Tool result null | `{role:"tool",content:null}` | `{type:"tool_result",content:""}` | Use empty string. |
| Image data URI | `image_url:{url:"data:image/jpeg;base64,abc"}` | `{type:"image",source:{type:"base64",media_type:"image/jpeg",data:"abc"}}` | Parse data URI. |
| Image https URL | `image_url:{url:"https://img.example.com/x.png"}` | `{type:"image",source:{type:"url",url:"..."}}` | Direct URL mapping. |
| `n>1` request | `n:3` | `n` ignored, single response | Log warning; Anthropic only returns 1. |
| `response_format` json | `response_format:{type:"json_object"}` | *(drop field; add instruction to system)* | Optionally add "Respond in JSON" to system prompt. |
| `frequency_penalty` | `frequency_penalty:0.5` | *(field dropped)* | No equivalent. |
| Consecutive tool msgs | `[{role:"tool",id:"t1",...},{role:"tool",id:"t2",...}]` | One user message with two `tool_result` blocks | Batch consecutive tool messages. |

---

## 5. Responses → Anthropic

Convert OpenAI Responses API requests/responses to Anthropic Messages format.

### 5.1 Request Conversion

#### 5.1.1 Field Mapping

| Source Field | Target Field | Conversion Notes |
|---|---|---|
| `model` | `model` | Pass through or remap. |
| `input` (string) | `messages:[{role:"user",content:string}]` | Single string becomes one user message. |
| `input` (array) | `messages` | See §5.1.2. |
| `instructions` | `system` | Direct mapping. |
| `max_output_tokens` | `max_tokens` | Rename. Required by Anthropic. |
| `temperature` | `temperature` | Direct. |
| `top_p` | `top_p` | Direct. |
| `stream` | `stream` | Direct. |
| `tools` (function type) | `tools` | See §5.1.3. |
| `tools` (file_search) | *(drop or custom tool)* | No native Anthropic equivalent. |
| `tools` (web_search_preview) | *(drop or custom tool)* | No native Anthropic equivalent. |
| `tools` (computer_use_preview) | tools with `computer_use` type (beta) | Anthropic has `computer_use` beta. |
| `tool_choice:"auto"` | `tool_choice:"auto"` | Direct. |
| `tool_choice:"none"` | *(drop tools)* | Remove tools from request. |
| `tool_choice:"required"` | `tool_choice:"any"` | Map. |
| `tool_choice:{type:"function",name}` | `tool_choice:{type:"tool",name}` | Restructure. |
| `previous_response_id` | *(cannot map directly)* | No Anthropic equivalent. Must resolve history from storage. |
| `truncation` | *(drop)* | No equivalent. |
| `store` | *(drop)* | No equivalent. |
| `metadata` | `metadata` | Direct. |
| `reasoning` | *(drop or map to thinking)* | Map `effort:"high"` to `thinking.budget_tokens=16000` etc. |

> ⚠️ **Note:** `previous_response_id` is a stateful feature unique to the Responses API. When converting, the converter must resolve the full conversation history from storage, then pass it as `messages` to Anthropic.

#### 5.1.2 Input Item Conversion

| Source | Target | Notes |
|---|---|---|
| `input[].type:"message", role:"user"` | `{role:"user",content:...}` | Map to Anthropic user message. |
| `input[].type:"message", role:"assistant"` | `{role:"assistant",content:...}` | Map to Anthropic assistant message. |
| `input[].type:"message", role:"system"` | *(extract to `system` field)* | Pull out as system prompt. |
| `content` (string) | `content:string` | Direct. |
| `content[].type:"input_text"` | `{type:"text",text}` | Map to Anthropic text block. |
| `content[].type:"input_image", url` | `{type:"image",source:{type:"url",url}}` | URL image. |
| `content[].type:"input_image", file_id` | *(must resolve to URL or base64)* | File must be fetched separately. |
| `content[].type:"input_file"` | *(drop or encode)* | Anthropic does not natively support file uploads. |
| `output[].type:"function_call"` | `{role:"assistant",content:[{type:"tool_use",...}]}` | Convert to Anthropic `tool_use`. |
| `output[].type:"function_call_output"` | `{role:"user",content:[{type:"tool_result",...}]}` | Convert to Anthropic `tool_result`. |

#### 5.1.3 Tool Conversion

| Source Field | Target Field | Notes |
|---|---|---|
| `tools[].type:"function"` | *(convert to Anthropic tool)* | Same as Chat Completions tool conversion. |
| `tools[].name` | `tools[].name` | Direct. |
| `tools[].description` | `tools[].description` | Direct. |
| `tools[].parameters` | `tools[].input_schema` | Rename. Both are JSON Schema. |

#### 5.1.4 Request Example

```json
// RESPONSES API REQUEST
{
  "model": "gpt-4o",
  "instructions": "You are a helpful assistant.",
  "input": [
    { "type": "message", "role": "user", "content": "What is the weather in Paris?" },
    {
      "type": "message",
      "role": "assistant",
      "content": [{ "type": "output_text", "text": "Let me check the weather for you." }]
    },
    {
      "type": "function_call",
      "call_id": "fc_001",
      "name": "get_weather",
      "arguments": "{\"city\":\"Paris\"}"
    },
    {
      "type": "function_call_output",
      "call_id": "fc_001",
      "output": "Sunny, 22°C"
    }
  ],
  "tools": [{
    "type": "function",
    "name": "get_weather",
    "description": "Get weather for a city.",
    "parameters": { "type":"object","properties":{"city":{"type":"string"}},"required":["city"] }
  }],
  "max_output_tokens": 800
}
```

```json
// CONVERTED ANTHROPIC REQUEST
{
  "model": "claude-3-5-sonnet-20241022",
  "max_tokens": 800,
  "system": "You are a helpful assistant.",
  "messages": [
    { "role": "user", "content": "What is the weather in Paris?" },
    {
      "role": "assistant",
      "content": [
        { "type": "text", "text": "Let me check the weather for you." },
        {
          "type": "tool_use",
          "id": "fc_001",
          "name": "get_weather",
          "input": { "city": "Paris" }
        }
      ]
    },
    {
      "role": "user",
      "content": [{
        "type": "tool_result",
        "tool_use_id": "fc_001",
        "content": "Sunny, 22°C"
      }]
    }
  ],
  "tools": [{
    "name": "get_weather",
    "description": "Get weather for a city.",
    "input_schema": { "type":"object","properties":{"city":{"type":"string"}},"required":["city"] }
  }]
}
```

---

### 5.2 Response Conversion

#### 5.2.1 Field Mapping

| Source Field | Target Field | Conversion Notes |
|---|---|---|
| `id` (`resp_...`) | `id` | Pass through or remap. |
| `object:"response"` | `type:"message"` | Set Anthropic type. |
| `created_at` | *(drop)* | Anthropic has no `created` field. |
| `model` | `model` | Direct. |
| `status:"completed"` | `stop_reason:"end_turn"` (or `"tool_use"`) | Infer from output content. |
| `status:"incomplete"` | `stop_reason:"max_tokens"` | Map incomplete to `max_tokens`. |
| `output[].type:"message",content[].type:"output_text"` | `content:[{type:"text",text}]` | Extract text. |
| `output[].type:"message",content[].type:"refusal"` | `content:[{type:"text",text:refusal}]` | Treat refusal as text. |
| `output[].type:"function_call"` | `content:[{type:"tool_use",id:call_id,name,input:JSON.parse(arguments)}]` | Convert function call. |
| `output[].type:"file_search_call"` | *(drop or synthetic tool_use)* | No direct Anthropic equivalent. |
| `usage.input_tokens` | `usage.input_tokens` | Direct. |
| `usage.output_tokens` | `usage.output_tokens` | Direct. |
| `usage.total_tokens` | *(drop)* | Anthropic does not include. |
| `error` | Stop with error information | Propagate error details. |

---

### 5.3 Streaming Conversion (Responses → Anthropic)

#### 5.3.1 Event Mapping

| Source Event | Target | Notes |
|---|---|---|
| `response.created` | `message_start` | Extract metadata; emit `message_start`. |
| `response.output_item.added` (message) | `content_block_start` (text) | Start new text block. |
| `response.output_text.delta` | `content_block_delta` (`text_delta`) | Forward text delta. |
| `response.output_text.done` | `content_block_stop` | Close text block. |
| `response.output_item.added` (function_call) | `content_block_start` (`tool_use`) | Start `tool_use` block. |
| `response.function_call_arguments.delta` | `content_block_delta` (`input_json_delta`) | Forward partial JSON. |
| `response.function_call_arguments.done` | `content_block_stop` | Close `tool_use` block. |
| `response.completed` | `message_delta` + `message_stop` | Emit final metadata and stop. |
| `response.failed` | `error` event | Emit Anthropic error event. |
| `response.incomplete` | `message_delta` (`stop_reason:max_tokens`) + `message_stop` | Map incomplete. |

---

### 5.4 Test Cases

| Test Case | Input | Expected Output | Notes |
|---|---|---|---|
| String input | `input:"Hello"` | `messages:[{role:"user",content:"Hello"}]` | Wrap string. |
| `previous_response_id` present | `previous_response_id:"resp_old"` | Full history resolved from store | Converter must fetch history. |
| `instructions` field | `instructions:"Be helpful"` | `system:"Be helpful"` | Direct rename. |
| `output_text` content | `output[].content[].type:"output_text",text:"Hi"` | `content:[{type:"text",text:"Hi"}]` | Extract text. |
| Refusal content | `content[].type:"refusal",refusal:"Cannot help"` | `content:[{type:"text",text:"Cannot help"}]` | Treat as text. |
| `function_call` output item | `output[].type:"function_call",call_id:"fc1",name:"fn",arguments:"{}"` | `content:[{type:"tool_use",id:"fc1",name:"fn",input:{}}]` | Convert `function_call`. |
| `status:"incomplete"` | `status:"incomplete"` | `stop_reason:"max_tokens"` | Map incomplete. |
| `file_search_call` | `output[].type:"file_search_call"` | *(dropped with warning)* | No Anthropic equivalent. |
| `reasoning:{effort:"high"}` | `reasoning:{effort:"high"}` | `thinking:{type:"enabled",budget_tokens:16000}` | Map reasoning effort to thinking budget. |
| No `max_output_tokens` | *(absent)* | `max_tokens:4096` (default) | Inject default for required Anthropic field. |

---

## 6. Anthropic → Responses

Convert Anthropic Messages API requests/responses to OpenAI Responses API format.

### 6.1 Request Conversion

#### 6.1.1 Field Mapping

| Source Field | Target Field | Conversion Notes |
|---|---|---|
| `model` | `model` | Pass through or remap. |
| `max_tokens` | `max_output_tokens` | Rename. |
| `messages` | `input` | See §6.1.2. |
| `system` (string) | `instructions` | Direct mapping. |
| `system` (array) | `instructions` (joined text) | Join text blocks. |
| `tools` | `tools` | See §6.1.3. |
| `tool_choice:"auto"` | `tool_choice:"auto"` | Direct. |
| `tool_choice:"any"` | `tool_choice:"required"` | `"any"` → `"required"`. |
| `tool_choice:{type:"tool",name}` | `tool_choice:{type:"function",name}` | Restructure. |
| `temperature` | `temperature` | Direct. |
| `top_p` | `top_p` | Direct. |
| `top_k` | *(drop)* | No equivalent. |
| `stop_sequences` | *(drop)* | No `stop` field in Responses API. |
| `stream` | `stream` | Direct. |
| `metadata.user_id` | `metadata.user_id` | Direct. |
| `thinking` | `reasoning:{effort:"high"}` | Map thinking to reasoning. Budget→effort heuristic. |

> ⚠️ **Note:** The Responses API does not have a `stop_sequences` field. These must be implemented via post-processing or prompt instructions.

#### 6.1.2 Message Conversion

| Source | Target | Notes |
|---|---|---|
| `role:"user", content:string` | `{type:"message",role:"user",content:string}` | Wrap in message item. |
| `role:"user", content:[{type:"text",text}]` | `{type:"message",role:"user",content:[{type:"input_text",text}]}` | Map text block type. |
| `role:"user", content:[{type:"image",source:{type:"base64",media_type,data}}]` | `{type:"message",content:[{type:"input_image",image_url:"data:media_type;base64,data"}]}` | Reconstruct data URI. |
| `role:"user", content:[{type:"image",source:{type:"url",url}}]` | `{type:"message",content:[{type:"input_image",image_url:url}]}` | Direct URL. |
| `role:"user", content:[{type:"tool_result",tool_use_id,content}]` | `{type:"function_call_output",call_id:tool_use_id,output:content}` | Convert to `function_call_output`. |
| `role:"assistant", content:[{type:"text",text}]` | `{type:"message",role:"assistant",content:[{type:"output_text",text}]}` | Map text block type. |
| `role:"assistant", content:[{type:"tool_use",id,name,input}]` | `{type:"function_call",call_id:id,name,arguments:JSON.stringify(input)}` | Convert to `function_call` item. |
| `role:"assistant", content:[{type:"thinking",thinking}]` | *(drop)* | No Responses equivalent. |

#### 6.1.3 Tool Conversion

| Source Field | Target Field | Notes |
|---|---|---|
| `tools[].name` | `tools[].name` | Direct. |
| `tools[].description` | `tools[].description` | Direct. |
| `tools[].input_schema` | `tools[].parameters` | Rename. Both are JSON Schema. |
| *(absent type)* | `tools[].type:"function"` | Add `type:"function"`. |

#### 6.1.4 Request Example

```json
// ANTHROPIC REQUEST
{
  "model": "claude-3-5-sonnet-20241022",
  "max_tokens": 1024,
  "system": "You are a coding assistant.",
  "messages": [
    { "role": "user", "content": "Write hello world in Python." },
    { "role": "assistant", "content": [{ "type": "text", "text": "Here it is." }] },
    { "role": "user", "content": [{ "type": "tool_result", "tool_use_id": "toolu_99", "content": "success" }] }
  ],
  "tools": [{
    "name": "run_code",
    "description": "Runs Python code.",
    "input_schema": { "type":"object","properties":{"code":{"type":"string"}},"required":["code"] }
  }],
  "tool_choice": "any",
  "temperature": 0.5
}
```

```json
// CONVERTED RESPONSES API REQUEST
{
  "model": "gpt-4o",
  "max_output_tokens": 1024,
  "instructions": "You are a coding assistant.",
  "input": [
    { "type": "message", "role": "user", "content": "Write hello world in Python." },
    { "type": "message", "role": "assistant", "content": [{ "type": "output_text", "text": "Here it is." }] },
    { "type": "function_call_output", "call_id": "toolu_99", "output": "success" }
  ],
  "tools": [{
    "type": "function",
    "name": "run_code",
    "description": "Runs Python code.",
    "parameters": { "type":"object","properties":{"code":{"type":"string"}},"required":["code"] }
  }],
  "tool_choice": "required",
  "temperature": 0.5
}
```

---

### 6.2 Response Conversion

#### 6.2.1 Field Mapping

| Source Field | Target Field | Conversion Notes |
|---|---|---|
| `id` (`resp_...`) | `id` | Pass through. |
| `object:"response"` | `type:"message"` | Set Anthropic type. |
| `created_at` | *(drop)* | Anthropic has no `created` field. |
| `model` | `model` | Direct. |
| `output[].type:"message", content[].type:"output_text"` | `content:[{type:"text",text}]` | Extract text block. |
| `output[].type:"function_call"` | `content:[{type:"tool_use",...}]` | Convert `function_call` to `tool_use`. |
| `output[].type:"function_call".call_id` | `content[].id` | Map `call_id` to `id`. |
| `output[].type:"function_call".arguments` | `content[].input = JSON.parse(arguments)` | Parse JSON string. |
| `status:"completed"` with `function_call` | `stop_reason:"tool_use"` | Infer tool_use stop reason. |
| `status:"completed"` without `function_call` | `stop_reason:"end_turn"` | Infer end_turn. |
| `status:"incomplete"` | `stop_reason:"max_tokens"` | Map incomplete. |
| `usage.input_tokens` | `usage.input_tokens` | Direct. |
| `usage.output_tokens` | `usage.output_tokens` | Direct. |

---

### 6.3 Streaming Conversion (Anthropic → Responses)

#### 6.3.1 Event Mapping

| Source Event | Target | Notes |
|---|---|---|
| `message_start` | `response.created` + `response.in_progress` | Emit two events with response metadata. |
| `content_block_start` (text) | `response.output_item.added` + `response.content_part.added` | Start output item and content part. |
| `content_block_delta` (`text_delta`) | `response.output_text.delta` | Forward delta text. |
| `content_block_stop` (text) | `response.output_text.done` + `response.output_item.done` | Close text block and output item. |
| `content_block_start` (`tool_use`) | `response.output_item.added` (`function_call`) | Start function call item. |
| `content_block_delta` (`input_json_delta`) | `response.function_call_arguments.delta` | Forward partial JSON. |
| `content_block_stop` (`tool_use`) | `response.function_call_arguments.done` + `response.output_item.done` | Close function call. |
| `message_delta` | *(buffer until `message_stop`)* | Hold `stop_reason` for final event. |
| `message_stop` | `response.completed` | Emit completed with full synthesized response. |

---

### 6.4 Test Cases

| Test Case | Input | Expected Output | Notes |
|---|---|---|---|
| System array input | `system:[{type:"text",text:"A"},{type:"text",text:"B"}]` | `instructions:"AB"` | Concatenate text. |
| `tool_choice:"any"` | `tool_choice:"any"` | `tool_choice:"required"` | Direct map. |
| Thinking block in message | `content:[{type:"thinking",thinking:"..."}]` | *(dropped)* | No Responses equivalent. |
| `stop_sequences` | `stop_sequences:["END"]` | *(dropped, warn)* | Not supported in Responses API. |
| Tool result in user msg | `content:[{type:"tool_result",tool_use_id:"t1",content:"ok"}]` | `{type:"function_call_output",call_id:"t1",output:"ok"}` | Convert to `function_call_output`. |
| Multiple `tool_use` blocks | `content:[{type:"tool_use",id:"a",...},{type:"tool_use",id:"b",...}]` | Two `function_call` items in input | Each becomes separate item. |
| Image base64 in user | `source:{type:"base64",media_type:"image/png",data:"abc"}` | `content:[{type:"input_image",image_url:"data:image/png;base64,abc"}]` | Construct data URI. |
| `metadata.user_id` | `metadata:{user_id:"u1"}` | `metadata:{user_id:"u1"}` | Direct pass-through. |

---

## 7. Chat Completions → Responses

Convert OpenAI Chat Completions API requests/responses to OpenAI Responses API format.

### 7.1 Request Conversion

#### 7.1.1 Field Mapping

| Source Field | Target Field | Conversion Notes |
|---|---|---|
| `model` | `model` | Direct. |
| `max_completion_tokens` / `max_tokens` | `max_output_tokens` | Prefer `max_completion_tokens`; fall back to `max_tokens`. |
| `messages` | `input` + `instructions` | Extract system → `instructions`; rest → `input` array. |
| `temperature` | `temperature` | Direct. |
| `top_p` | `top_p` | Direct. |
| `n` | *(drop)* | Responses API does not support `n>1`. |
| `stop` | *(drop)* | No `stop` field in Responses API. |
| `stream` | `stream` | Direct. |
| `tools` (function) | `tools` (function) | See §7.1.2. |
| `tool_choice:"auto"` | `tool_choice:"auto"` | Direct. |
| `tool_choice:"none"` | `tool_choice:"none"` | Direct. |
| `tool_choice:"required"` | `tool_choice:"required"` | Direct. |
| `tool_choice:{type:"function",function:{name}}` | `tool_choice:{type:"function",name}` | Remove nested `function` object. |
| `response_format:{type:"json_object"}` | *(drop)* | No equivalent; prompt engineer. |
| `frequency_penalty` / `presence_penalty` / `logprobs` / `seed` | *(drop)* | Not supported in Responses API. |
| `user` | `metadata.user_id` | Nest in metadata. |

#### 7.1.2 Message Conversion

| Source | Target | Notes |
|---|---|---|
| `role:"system"` | *(extract to `instructions` field)* | Multiple system messages: join with `\n\n`. |
| `role:"user", content:string` | `{type:"message",role:"user",content:string}` | Wrap in message item. |
| `role:"user", content:[{type:"text",text}]` | `{type:"message",role:"user",content:[{type:"input_text",text}]}` | Retype content parts. |
| `role:"user", content:[{type:"image_url",image_url:{url}}]` | `{type:"message",content:[{type:"input_image",image_url:url}]}` | Rename type and flatten. |
| `role:"assistant", content:string` | `{type:"message",role:"assistant",content:[{type:"output_text",text:content}]}` | Wrap in `output_text` part. |
| `role:"assistant", tool_calls:[...]` | `{type:"function_call",...}` | Convert each tool call to `function_call` item. |
| `role:"assistant", content+tool_calls` | Message item + `function_call` items | Separate into distinct items. |
| `role:"tool", tool_call_id, content` | `{type:"function_call_output",call_id:tool_call_id,output:content}` | Convert to `function_call_output`. |

#### 7.1.3 Tool Conversion

| Source Field | Target Field | Notes |
|---|---|---|
| `tools[].function.name` | `tools[].name` | Direct. |
| `tools[].function.description` | `tools[].description` | Direct. |
| `tools[].function.parameters` | `tools[].parameters` | Direct. |
| `tools[].type:"function"` | `tools[].type:"function"` | Direct. |

#### 7.1.4 Request Example

```json
// CHAT COMPLETIONS REQUEST
{
  "model": "gpt-4o",
  "messages": [
    { "role": "system", "content": "Be concise." },
    { "role": "user", "content": "List capitals of Europe." },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_001",
        "type": "function",
        "function": { "name": "search_capitals", "arguments": "{\"region\":\"Europe\"}" }
      }]
    },
    { "role": "tool", "tool_call_id": "call_001", "content": "Paris, Berlin, Rome..." }
  ],
  "tools": [{ "type": "function", "function": {
    "name": "search_capitals", "description": "Search for capitals.",
    "parameters": { "type":"object","properties":{"region":{"type":"string"}} }
  }}],
  "tool_choice": { "type": "function", "function": { "name": "search_capitals" } },
  "max_completion_tokens": 256
}
```

```json
// CONVERTED RESPONSES API REQUEST
{
  "model": "gpt-4o",
  "instructions": "Be concise.",
  "input": [
    { "type": "message", "role": "user", "content": "List capitals of Europe." },
    {
      "type": "function_call",
      "call_id": "call_001",
      "name": "search_capitals",
      "arguments": "{\"region\":\"Europe\"}"
    },
    {
      "type": "function_call_output",
      "call_id": "call_001",
      "output": "Paris, Berlin, Rome..."
    }
  ],
  "tools": [{ "type": "function",
    "name": "search_capitals", "description": "Search for capitals.",
    "parameters": { "type":"object","properties":{"region":{"type":"string"}} }
  }],
  "tool_choice": { "type": "function", "name": "search_capitals" },
  "max_output_tokens": 256
}
```

---

### 7.2 Response Conversion

#### 7.2.1 Field Mapping

| Source Field | Target Field | Conversion Notes |
|---|---|---|
| `id` (`resp_...`) | `id` | Pass through. |
| `object:"response"` | `object:"chat.completion"` | Set Chat Completions object type. |
| `created_at` | `created` | Direct (both Unix timestamps). |
| `model` | `model` | Direct. |
| `output[].type:"message",content[].type:"output_text"` | `choices[0].message.content:text` | Extract text. |
| `output[].type:"message",content[].type:"refusal"` | `choices[0].message.content:null`, refusal field | Map refusal. |
| `output[].type:"function_call"` | `choices[0].message.tool_calls[...]` | Convert to `tool_calls`. |
| `output[].type:"function_call".call_id` | `tool_calls[].id` | Direct. |
| `output[].type:"function_call".arguments` | `tool_calls[].function.arguments` | Direct (already a JSON string). |
| `status:"completed"` | `finish_reason:"stop"` or `"tool_calls"` | Infer from output contents. |
| `status:"incomplete"` | `finish_reason:"length"` | Map incomplete. |
| `status:"failed"` | `finish_reason:"content_filter"` (or error) | Best approximation. |
| `usage.input_tokens` | `usage.prompt_tokens` | Rename. |
| `usage.output_tokens` | `usage.completion_tokens` | Rename. |
| `usage.total_tokens` | `usage.total_tokens` | Direct. |

---

### 7.3 Streaming Conversion (Chat Completions → Responses)

#### 7.3.1 Event Mapping

| Source Event | Target | Notes |
|---|---|---|
| First delta (`role:"assistant"`) | `response.created` + `response.in_progress` + `response.output_item.added` + `response.content_part.added` | Emit 4 setup events. |
| `delta.content` text chunk | `response.output_text.delta` | Forward text. |
| `delta.tool_calls` first chunk (id, name) | `response.output_item.added` (`function_call`) | Start new function call item. |
| `delta.tool_calls` argument chunk | `response.function_call_arguments.delta` | Forward partial args. |
| `finish_reason:"stop"` | `response.output_text.done` + `response.output_item.done` + `response.completed` | Close and complete. |
| `finish_reason:"tool_calls"` | `response.function_call_arguments.done` + `response.output_item.done` + `response.completed` | Close function call and complete. |
| `finish_reason:"length"` | `response.incomplete` | Map to incomplete. |
| `data: [DONE]` | *(already emitted `response.completed`)* | No additional event. |

---

### 7.4 Test Cases

| Test Case | Input | Expected Output | Notes |
|---|---|---|---|
| Assistant text+tool | `{content:"ok",tool_calls:[...]}` | message item + `function_call` item | Split into separate input items. |
| Multiple tool results | `[{role:"tool",...},{role:"tool",...}]` | Two `function_call_output` items | Each becomes separate item. |
| `tool_choice` nested | `{type:"function",function:{name:"fn"}}` | `{type:"function",name:"fn"}` | Flatten structure. |
| `response_format` json | `response_format:{type:"json_object"}` | *(dropped)* | No equivalent. |
| Stop sequences | `stop:["\n"]` | *(dropped)* | Not supported. |
| `user` field | `user:"u1"` | `metadata:{user_id:"u1"}` | Nest in metadata. |
| `n>1` | `n:2` | *(ignored)* | Responses does not support `n>1`. |
| Status incomplete | `status:"incomplete"` | `finish_reason:"length"` | Map incomplete. |
| Refusal content | `content[].type:"refusal"` | `message.content:null` + refusal | Map refusal correctly. |

---

## 8. Responses → Chat Completions

Convert OpenAI Responses API requests/responses to OpenAI Chat Completions format.

### 8.1 Request Conversion

#### 8.1.1 Field Mapping

| Source Field | Target Field | Conversion Notes |
|---|---|---|
| `model` | `model` | Direct. |
| `max_output_tokens` | `max_completion_tokens` | Rename. |
| `input` (string) | `messages:[{role:"user",content:string}]` | Wrap in user message. |
| `input` (array) | `messages` | See §8.1.2. |
| `instructions` | `messages` (system role, prepended) | Prepend as system message. |
| `temperature` | `temperature` | Direct. |
| `top_p` | `top_p` | Direct. |
| `stream` | `stream` | Direct. |
| `tools` (function) | `tools` | See §8.1.3. |
| `tools` (file_search / web_search) | *(drop)* | No Chat Completions equivalent. |
| `tool_choice:"auto"` | `tool_choice:"auto"` | Direct. |
| `tool_choice:"none"` | `tool_choice:"none"` | Direct. |
| `tool_choice:"required"` | `tool_choice:"required"` | Direct. |
| `tool_choice:{type:"function",name}` | `tool_choice:{type:"function",function:{name}}` | Wrap name in `function` object. |
| `previous_response_id` | *(resolve history)* | Must expand to full `messages` array. |
| `truncation` | *(drop)* | No equivalent. |
| `store` | *(drop)* | No equivalent. |
| `metadata.user_id` | `user` | Extract and promote to top-level. |
| `reasoning:{effort}` | *(drop or model param)* | o-series models accept reasoning in model param. |

#### 8.1.2 Input Item Conversion

| Source | Target | Notes |
|---|---|---|
| `type:"message",role:"user"` | `{role:"user",content:...}` | Direct role mapping. |
| `type:"message",role:"assistant"` | `{role:"assistant",content:...}` | Direct role mapping. |
| `type:"message",role:"system"` | *(merge with `instructions`)* | Combine with instructions if present. |
| `content` (string) | `content:string` | Direct. |
| `content[].type:"input_text"` | `{type:"text",text}` | Rename type. |
| `content[].type:"output_text"` | `content:text` (string) | Extract text to string for assistant. |
| `content[].type:"input_image",image_url` | `{type:"image_url",image_url:{url:image_url}}` | Wrap in `image_url` object. |
| `type:"function_call"` | `{role:"assistant",content:null,tool_calls:[...]}` | Convert to assistant+`tool_calls`. |
| `type:"function_call".call_id` | `tool_calls[].id` | Direct. |
| `type:"function_call".arguments` | `tool_calls[].function.arguments` | Direct (already JSON string). |
| `type:"function_call_output"` | `{role:"tool",tool_call_id:call_id,content:output}` | Convert to `tool` role message. |

#### 8.1.3 Tool Conversion

| Source Field | Target Field | Notes |
|---|---|---|
| `tools[].name` | `tools[].function.name` | Nest in `function` object. |
| `tools[].description` | `tools[].function.description` | Nest in `function` object. |
| `tools[].parameters` | `tools[].function.parameters` | Nest in `function` object. |
| `tools[].type:"function"` | `tools[].type:"function"` | Direct. |

#### 8.1.4 Request Example

```json
// RESPONSES API REQUEST
{
  "model": "gpt-4o",
  "instructions": "You are a travel assistant.",
  "input": [
    { "type": "message", "role": "user", "content": "Best time to visit Tokyo?" },
    {
      "type": "function_call",
      "call_id": "fc_tokyo",
      "name": "lookup_travel",
      "arguments": "{\"city\":\"Tokyo\"}"
    },
    {
      "type": "function_call_output",
      "call_id": "fc_tokyo",
      "output": "Spring (March-May) and Autumn (Sep-Nov) are best."
    }
  ],
  "tools": [{ "type": "function",
    "name": "lookup_travel", "description": "Lookup travel info.",
    "parameters": { "type":"object","properties":{"city":{"type":"string"}} }
  }],
  "tool_choice": { "type": "function", "name": "lookup_travel" },
  "max_output_tokens": 512
}
```

```json
// CONVERTED CHAT COMPLETIONS REQUEST
{
  "model": "gpt-4o",
  "messages": [
    { "role": "system", "content": "You are a travel assistant." },
    { "role": "user", "content": "Best time to visit Tokyo?" },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "fc_tokyo",
        "type": "function",
        "function": { "name": "lookup_travel", "arguments": "{\"city\":\"Tokyo\"}" }
      }]
    },
    {
      "role": "tool",
      "tool_call_id": "fc_tokyo",
      "content": "Spring (March-May) and Autumn (Sep-Nov) are best."
    }
  ],
  "tools": [{ "type": "function", "function": {
    "name": "lookup_travel", "description": "Lookup travel info.",
    "parameters": { "type":"object","properties":{"city":{"type":"string"}} }
  }}],
  "tool_choice": { "type": "function", "function": { "name": "lookup_travel" } },
  "max_completion_tokens": 512
}
```

---

### 8.2 Response Conversion

#### 8.2.1 Field Mapping

| Source Field | Target Field | Conversion Notes |
|---|---|---|
| `id` (`resp_...`) | `id` | Pass through. |
| `object:"response"` | `object:"chat.completion"` | Set Chat Completions type. |
| `created_at` | `created` | Direct. |
| `model` | `model` | Direct. |
| `output[].type:"message",content[].type:"output_text"` | `choices[0].message.content:text` | Extract text string. |
| `output[].type:"function_call"` | `choices[0].message.tool_calls[...]` | Convert each. |
| `output[].call_id` | `tool_calls[].id` | Direct. |
| `output[].name` | `tool_calls[].function.name` | Nest in `function`. |
| `output[].arguments` | `tool_calls[].function.arguments` | Direct (JSON string). |
| `status:"completed"` (text only) | `finish_reason:"stop"` | Text completion. |
| `status:"completed"` (with `function_call`) | `finish_reason:"tool_calls"` | Tool call completion. |
| `status:"incomplete"` | `finish_reason:"length"` | Map incomplete. |
| `status:"failed"` | `finish_reason:"content_filter"` | Best approximation. |
| `usage.input_tokens` | `usage.prompt_tokens` | Rename. |
| `usage.output_tokens` | `usage.completion_tokens` | Rename. |
| `usage.total_tokens` | `usage.total_tokens` | Direct. |

---

### 8.3 Streaming Conversion (Responses → Chat Completions)

#### 8.3.1 Event Mapping

| Source Event | Target | Notes |
|---|---|---|
| `response.created` | First delta chunk: `delta.role="assistant",content=""` | Bootstrap stream. |
| `response.output_text.delta` | `chunk: delta.content=delta` | Forward text delta. |
| `response.output_item.added` (`function_call`) | `chunk: delta.tool_calls=[{index,id:call_id,type:"function",function:{name,arguments:""}}]` | Bootstrap tool call. |
| `response.function_call_arguments.delta` | `chunk: delta.tool_calls=[{index,function:{arguments:delta}}]` | Forward partial args. |
| `response.output_text.done` | *(buffer)* | No extra chunk needed. |
| `response.function_call_arguments.done` | *(buffer)* | No extra chunk needed. |
| `response.completed` (text) | Final chunk: `finish_reason:"stop"` + `data:[DONE]` | Close stream. |
| `response.completed` (`function_call`) | Final chunk: `finish_reason:"tool_calls"` + `data:[DONE]` | Close stream. |
| `response.incomplete` | Final chunk: `finish_reason:"length"` + `data:[DONE]` | Map incomplete. |
| `response.failed` | `data:{error:{...}}` + `data:[DONE]` | Error chunk. |

---

### 8.4 Test Cases

| Test Case | Input | Expected Output | Notes |
|---|---|---|---|
| String input | `input:"Hello"` | `messages:[{role:"user",content:"Hello"}]` | Wrap string. |
| `instructions` present | `instructions:"Be brief"` | `messages[0]:{role:"system",content:"Be brief"}` | Prepend system. |
| `function_call` item | `{type:"function_call",call_id:"c1",name:"fn",arguments:"{}"}` | `{role:"assistant",content:null,tool_calls:[{id:"c1",type:"function",function:{name:"fn",arguments:"{}"}}]}` | Convert to assistant message. |
| `function_call_output` item | `{type:"function_call_output",call_id:"c1",output:"result"}` | `{role:"tool",tool_call_id:"c1",content:"result"}` | Convert to tool role. |
| `tool_choice` name only | `{type:"function",name:"fn"}` | `{type:"function",function:{name:"fn"}}` | Add `function` wrapper. |
| `file_search` tool | `tools:[{type:"file_search"}]` | *(dropped)* | No Chat Completions equivalent. |
| `web_search` tool | `tools:[{type:"web_search_preview"}]` | *(dropped)* | No Chat Completions equivalent. |
| `metadata.user_id` | `metadata:{user_id:"u1"}` | `user:"u1"` | Promote to top-level. |
| `output_text` + refusal | `content:[{type:"output_text",...},{type:"refusal",...}]` | `message.content:text` (refusal in separate field) | Handle mixed content. |
| `previous_response_id` | `previous_response_id:"resp_old"` | Full messages history | Must resolve from storage. |
| `status:"failed"` | `status:"failed"` | `finish_reason:"content_filter"` | Best approximation. |
| Multiple `function_calls` | `output:[{type:"function_call",call_id:"a",...},{type:"function_call",call_id:"b",...}]` | `tool_calls:[{id:"a",...},{id:"b",...}]` | Both in one assistant message. |

---

## 9. Cross-Protocol Corner Cases

This section documents corner cases that apply to multiple conversion directions and require special handling in any production converter.

### 9.1 Message Ordering & Role Alternation

> ⚠️ **Anthropic requires strict alternation of user/assistant roles.** If consecutive messages of the same role appear after conversion, inject empty filler messages.

- **Consecutive user messages:** merge into one user message with multiple content blocks, or inject an empty assistant message between them.
- **Consecutive assistant messages:** merge content, or inject an empty user message. This is rare but can occur when converting from Responses API.
- **Starting with assistant message:** prepend an empty user message if Anthropic requires the first message to be user role.

---

### 9.2 Empty Content

| Scenario | Protocol | Handling |
|---|---|---|
| `content:""` | Various | Anthropic: reject empty text blocks. OAI: pass through. Convert empty strings to `null` or omit. |
| `content:null` | Chat Completions assistant message | Means `tool_calls` present. Convert to empty `content:[]`. |
| `content:[]` | Anthropic empty content array | Valid for tool-use-only messages. Keep as-is. |
| `tool_result` with `null` content | Anthropic `tool_result` | Use empty string `""` as content. |
| `tool_result` with array content | Anthropic `tool_result` | Flatten array of text blocks to joined string for Chat Completions/Responses. |

---

### 9.3 Tool Call ID Consistency

- IDs must be consistent across assistant `tool_use` and the matching user `tool_result` messages.
- Anthropic uses `"toolu_..."` format; Chat Completions uses `"call_..."`; Responses uses arbitrary strings.
- When converting, **preserve the original IDs** rather than regenerating them to maintain reference integrity.
- If an ID is missing (e.g. Responses API `function_call_output` without `call_id`), attempt lookup from prior `function_call` items in the conversation.

---

### 9.4 Streaming Interruption & Reconnection

- If a stream is interrupted mid-way, the partial content may be invalid JSON (for tool arguments) or incomplete text.
- **Buffer all `content_block_delta` / `input_json_delta` events** before attempting `JSON.parse`.
- On reconnection, the client must restart from the beginning of the message (stateless protocols) or use `previous_response_id` (Responses API).
- Always emit a final `[DONE]` or `message_stop` even if the upstream stream was interrupted, to prevent client hangs.

---

### 9.5 Token Counting Discrepancies

- Anthropic counts tokens differently from OpenAI tokenizers. `usage` fields will not match exactly when proxying.
- When routing Anthropic→OAI, the usage returned to the client should reflect the actual upstream usage from whichever API was called.
- Cache tokens (`cache_read_input_tokens`, `cache_creation_input_tokens`) are Anthropic-specific and have no OAI equivalent; **drop them** when converting to OAI.

---

### 9.6 Multi-Modal Content Ordering

- When a user message has both text and images, Anthropic and Chat Completions both support array content. Order should be preserved.
- Chat Completions does not support text that follows an image in the same message in some older model versions; **always put text first** if unsure.
- Responses API: use `input_text` and `input_image` content part types; order is preserved.

---

### 9.7 Tool Result Content Types

- Anthropic `tool_result.content` can be: a string, an array of text blocks, or an array of mixed text/image blocks.
- Chat Completions tool message `content` must be a string. **Flatten array content** to a single string.
- Responses API `function_call_output.output` must be a string. Same flattening applies.
- If `tool_result` contains an image block, it cannot be faithfully represented in Chat Completions or Responses tool output. **Log a warning** and include the text portions only.

---

### 9.8 Thinking / Extended Reasoning

- Anthropic `thinking` blocks (`{type:"thinking",thinking:"..."}`) have **no equivalent** in Chat Completions or the standard Responses API.
- **Drop thinking blocks** when converting away from Anthropic. They cannot be injected back into subsequent Anthropic requests as content blocks (use the `thinking` field instead).
- When converting Responses `reasoning:{effort}` to Anthropic `thinking`: map `low`→4000, `medium`→8000, `high`→16000 `budget_tokens` as a heuristic.

---

### 9.9 Stop Sequence Handling

- Anthropic `stop_sequences` is an array of strings.
- Chat Completions `stop` accepts a string or array (max 4 elements).
- **Responses API has no `stop` field.** Implement via post-processing: check if the last token matches a stop sequence and truncate.
- When converting Anthropic→Responses, log a warning if `stop_sequences` were present; they will not be honoured.

---

## 10. Implementation Reference

### 10.1 Converter Architecture

A production-grade protocol converter should be structured as:

1. **Request interceptor:** detect source protocol from headers/URL/body shape.
2. **Request transformer:** convert request payload to target protocol.
3. **Upstream caller:** send converted request to target API.
4. **Response transformer:** convert response back to source protocol format.
5. **Streaming transformer** (if `stream=true`): convert SSE stream events on-the-fly.

---

### 10.2 Protocol Detection Heuristics

| Protocol | Primary Signal | Secondary Signal |
|---|---|---|
| Anthropic | Header `anthropic-version` present | Or `model` starts with `"claude-"`. |
| Chat Completions | URL path ends in `/chat/completions` | Or body has `"messages"` array with OAI roles. |
| Responses | URL path ends in `/responses` | Or body has `"input"` field and no `"messages"`. |

---

### 10.3 Required Utility Functions

```javascript
// 1. Parse data URI
function parseDataUri(uri) {
  const match = uri.match(/^data:([^;]+);base64,(.+)$/);
  if (!match) throw new Error('Invalid data URI');
  return { mediaType: match[1], data: match[2] };
}

// 2. Build data URI
function buildDataUri(mediaType, data) {
  return `data:${mediaType};base64,${data}`;
}

// 3. Safe JSON parse (for tool arguments)
function safeJsonParse(str, fallback = {}) {
  try { return JSON.parse(str); }
  catch { return fallback; }
}

// 4. Clamp temperature
function clampTemperature(t, max = 1.0) {
  return Math.min(Math.max(t ?? 1.0, 0), max);
}

// 5. Map stop reason
const stopReasonMap = {
  'end_turn': 'stop',
  'max_tokens': 'length',
  'tool_use': 'tool_calls',
  'stop_sequence': 'stop'
};
const finishReasonMap = Object.fromEntries(
  Object.entries(stopReasonMap).map(([k, v]) => [v, k])
);

// 6. Generate synthetic ID
function syntheticId(prefix = 'msg') {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

// 7. Flatten tool_result content
function flattenToolContent(content) {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content.filter(b => b.type === 'text').map(b => b.text).join('\n');
  }
  return String(content ?? '');
}

// 8. Ensure alternating roles (Anthropic requirement)
function ensureAlternatingRoles(messages) {
  const result = [];
  for (const msg of messages) {
    if (result.length > 0 && result[result.length - 1].role === msg.role) {
      const filler = msg.role === 'user'
        ? { role: 'assistant', content: [{ type: 'text', text: '...' }] }
        : { role: 'user', content: [{ type: 'text', text: '...' }] };
      result.push(filler);
    }
    result.push(msg);
  }
  return result;
}
```

---

### 10.4 Streaming Transformer Pattern

```javascript
// Generic SSE stream transformer (Node.js / Web Streams)
async function* transformAnthropicToOAIStream(anthropicStream) {
  let msgId, model, inputTokens;

  for await (const event of anthropicStream) {
    const { type, ...data } = event;

    switch (type) {
      case 'message_start':
        msgId = data.message.id;
        model = data.message.model;
        inputTokens = data.message.usage.input_tokens;
        yield makeChunk(msgId, model, { role: 'assistant', content: '' });
        break;

      case 'content_block_start':
        if (data.content_block.type === 'tool_use') {
          yield makeChunk(msgId, model, {
            tool_calls: [{
              index: data.index,
              id: data.content_block.id,
              type: 'function',
              function: { name: data.content_block.name, arguments: '' }
            }]
          });
        }
        break;

      case 'content_block_delta':
        if (data.delta.type === 'text_delta') {
          yield makeChunk(msgId, model, { content: data.delta.text });
        } else if (data.delta.type === 'input_json_delta') {
          yield makeChunk(msgId, model, {
            tool_calls: [{
              index: data.index,
              function: { arguments: data.delta.partial_json }
            }]
          });
        }
        break;

      case 'message_delta': {
        const finishReason = stopReasonMap[data.delta.stop_reason] || 'stop';
        yield makeChunk(msgId, model, {}, finishReason, {
          prompt_tokens: inputTokens,
          completion_tokens: data.usage.output_tokens,
          total_tokens: inputTokens + data.usage.output_tokens
        });
        break;
      }

      case 'message_stop':
        yield '[DONE]';
        break;
    }
  }
}
```

---

### 10.5 Error Handling Reference

| Error Condition | Anthropic | Chat Completions | Responses |
|---|---|---|---|
| Rate limit | `{"type":"error","error":{"type":"rate_limit_error",...}}` | `{"error":{"type":"requests","code":"rate_limit_exceeded",...}}` | `{"error":{"code":"rate_limit_exceeded",...}}` |
| Invalid request | `error.type:"invalid_request_error"` | `error.type:"invalid_request_error"` | `error.code:"invalid_request"` |
| Auth failure | `error.type:"authentication_error"` | `error.code:"invalid_api_key"` | `error.code:"invalid_api_key"` |
| Overload | `error.type:"overloaded_error"` | HTTP 429 with `Retry-After` | HTTP 429 with `Retry-After` |
| Context too long | `error.type:"invalid_request_error"` (context window message) | `error.code:"context_length_exceeded"` | `error.code:"context_length_exceeded"` |

---

*End of LLM API Protocol Conversion Reference Guide*
