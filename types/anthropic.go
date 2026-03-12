// Package types provides shared type definitions for OpenAI and Anthropic streaming formats.
// This package centralizes all type definitions to avoid duplication across the codebase.
package types

import "encoding/json"

// Event represents a Server-Sent Event in an Anthropic streaming response.
// Each event has a type that determines which other fields are populated.
//
// Fields:
//   - Type: Event type (e.g., "message_start", "content_block_start", "content_block_delta")
//   - Index: Index of the content block this event relates to
//   - Delta: Incremental content change (RawMessage for flexible parsing)
//   - ContentBlock: Full content block definition (for "content_block_start" events)
//   - Message: Full message object (for "message_start" events)
//   - MessageUsage: Token usage statistics (for "message_delta" events)
//   - StopReason: Reason for completion (for "message_delta" events)
//   - StopSequence: Stop sequence that triggered completion (if applicable)
type Event struct {
	Type         string          `json:"type"`
	Index        *int            `json:"index,omitempty"`
	Delta        json.RawMessage `json:"delta,omitempty"`
	ContentBlock json.RawMessage `json:"content_block,omitempty"`
	Message      *Message        `json:"message,omitempty"`
	MessageUsage *Usage          `json:"usage,omitempty"`
	StopReason   string          `json:"stop_reason,omitempty"`
	StopSequence *string         `json:"stop_sequence,omitempty"`
}

// Message represents the complete message in an Anthropic response.
// It appears in the "message_start" event at the beginning of a streaming response.
//
// Fields:
//   - ID: Unique identifier for the message
//   - Type: Type of object (typically "message")
//   - Role: Role of the message author (typically "assistant")
//   - Content: Array of content blocks (text, tool use, etc.)
//   - Model: Model identifier used for the response
//   - Usage: Token usage statistics
type Message struct {
	ID      string         `json:"id"`
	Type    string         `json:"type"`
	Role    string         `json:"role"`
	Content []ContentBlock `json:"content"`
	Model   string         `json:"model"`
	Usage   *Usage         `json:"usage,omitempty"`
}

// ContentBlock represents a single content block in an Anthropic message.
// Content blocks can be text, tool use, or thinking blocks.
//
// Fields:
//   - Type: Block type ("text", "tool_use", "thinking")
//   - Text: Text content (for text blocks)
//   - ID: Unique identifier (for tool_use blocks)
//   - Name: Function name (for tool_use blocks)
//   - Input: Function input as JSON (for tool_use blocks)
//   - Thinking: Thinking content (for thinking blocks)
type ContentBlock struct {
	Type     string          `json:"type"`
	Text     string          `json:"text,omitempty"`
	ID       string          `json:"id,omitempty"`
	Name     string          `json:"name,omitempty"`
	Input    json.RawMessage `json:"input,omitempty"`
	Thinking string          `json:"thinking,omitempty"`
}

// Usage represents token usage statistics in an Anthropic response.
// It tracks input and output token counts for billing and monitoring.
//
// Fields:
//   - InputTokens: Number of tokens in the input prompt
//   - OutputTokens: Number of tokens in the generated response
type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// TextDelta represents an incremental text change in an Anthropic streaming response.
// It appears as the delta in "content_block_delta" events for text content.
//
// Fields:
//   - Type: Delta type (always "text_delta")
//   - Text: Incremental text content
type TextDelta struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// ThinkingDelta represents an incremental thinking content change in an Anthropic response.
// It appears as the delta in "content_block_delta" events for thinking blocks.
//
// Fields:
//   - Type: Delta type (always "thinking_delta")
//   - Thinking: Incremental thinking content
type ThinkingDelta struct {
	Type     string `json:"type"`
	Thinking string `json:"thinking"`
}

// InputJSONDelta represents an incremental JSON input change in an Anthropic response.
// It appears as the delta in "content_block_delta" events for tool use blocks.
//
// Fields:
//   - Type: Delta type (always "input_json_delta")
//   - PartialJSON: Incremental JSON string for tool input
type InputJSONDelta struct {
	Type        string `json:"type"`
	PartialJSON string `json:"partial_json"`
}

// Error represents an error response from the Anthropic API.
// It contains the error type and detailed error information.
//
// Fields:
//   - Type: Error type identifier
//   - Error: Detailed error information
type Error struct {
	Type  string      `json:"type"`
	Error ErrorDetail `json:"error"`
}

// ErrorDetail represents detailed error information in an Anthropic error response.
// It provides the specific error type and human-readable message.
//
// Fields:
//   - Type: Specific error type (e.g., "invalid_request_error", "authentication_error")
//   - Message: Human-readable error message
type ErrorDetail struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}
