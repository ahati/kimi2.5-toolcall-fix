// Package types provides shared type definitions for OpenAI and Anthropic streaming formats.
// This package centralizes all type definitions to avoid duplication across the codebase.
package types

// StreamChunk represents a single chunk in an OpenAI streaming response.
// It contains the chunk identifier, model information, and an array of choices
// with delta content that gets streamed to the client.
//
// Fields:
//   - ID: Unique identifier for the streaming response
//   - Object: Type of object (typically "chat.completion.chunk")
//   - Created: Unix timestamp of creation
//   - Model: Model identifier used for the completion
//   - Choices: Array of streaming choice deltas
//   - Usage: Token usage statistics (present in final chunk)
type StreamChunk struct {
	ID      string         `json:"id,omitempty"`
	Object  string         `json:"object,omitempty"`
	Created int64          `json:"created,omitempty"`
	Model   string         `json:"model,omitempty"`
	Choices []StreamChoice `json:"choices"`
	Usage   *StreamUsage   `json:"usage,omitempty"`
}

// StreamChoice represents a single choice in an OpenAI streaming response chunk.
// Each choice contains a delta with incremental content and an optional finish reason.
//
// Fields:
//   - Index: Zero-based index of this choice in the choices array
//   - Delta: Incremental content for this streaming chunk
//   - FinishReason: Reason for completion (nil until final chunk)
type StreamChoice struct {
	Index        int         `json:"index"`
	Delta        StreamDelta `json:"delta"`
	FinishReason *string     `json:"finish_reason,omitempty"`
}

// StreamDelta represents incremental content in an OpenAI streaming response.
// It contains the role, content text, reasoning content, and any tool calls.
//
// Fields:
//   - Role: Role of the message author (e.g., "assistant")
//   - Content: Incremental text content for this chunk
//   - Reasoning: Reasoning content (extended thinking)
//   - ReasoningContent: Alternative reasoning content field
//   - ToolCalls: Array of tool call deltas
//   - FinishReason: Reason for completion (deprecated in delta)
type StreamDelta struct {
	Role             string           `json:"role,omitempty"`
	Content          string           `json:"content,omitempty"`
	Reasoning        string           `json:"reasoning,omitempty"`
	ReasoningContent string           `json:"reasoning_content,omitempty"`
	ToolCalls        []StreamToolCall `json:"tool_calls,omitempty"`
	FinishReason     *string          `json:"finish_reason,omitempty"`
}

// StreamToolCall represents a tool call delta in an OpenAI streaming response.
// It contains the tool identifier, type, index, and function call details.
//
// Fields:
//   - ID: Unique identifier for this tool call (may be empty in early deltas)
//   - Type: Type of tool call (typically "function")
//   - Index: Index of this tool call in the accumulated tool calls array
//   - Function: Function call details with name and arguments
type StreamToolCall struct {
	ID       string         `json:"id,omitempty"`
	Type     string         `json:"type,omitempty"`
	Index    int            `json:"index"`
	Function StreamFunction `json:"function"`
}

// StreamFunction represents function call details in an OpenAI streaming response.
// It contains the function name and incremental arguments as JSON string.
//
// Fields:
//   - Name: Name of the function being called
//   - Arguments: Incremental JSON arguments string (accumulated across chunks)
type StreamFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// StreamUsage represents token usage statistics in an OpenAI streaming response.
// It appears in the final chunk of a streaming response.
//
// Fields:
//   - PromptTokens: Number of tokens in the prompt
//   - CompletionTokens: Number of tokens in the completion
//   - TotalTokens: Total tokens used (optional, can be computed)
type StreamUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens,omitempty"`
}
