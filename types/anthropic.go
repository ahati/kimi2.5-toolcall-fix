// Package types defines data structures for OpenAI and Anthropic API formats.
// This file contains types specific to the Anthropic Messages API format.
package types

import "encoding/json"

// MessageRequest represents an Anthropic Messages API request.
// This is the primary request structure for the /v1/messages endpoint.
type MessageRequest struct {
	// Model is the model identifier to use.
	// Valid values: "claude-3-opus-20240229", "claude-3-sonnet-20240229", etc.
	Model string `json:"model"`
	// Messages is the conversation history.
	// Each message has a role and content.
	Messages []MessageInput `json:"messages"`
	// MaxTokens is the maximum number of tokens to generate.
	// Required for Anthropic API; unlike OpenAI, this is mandatory.
	MaxTokens int `json:"max_tokens"`
	// Stream enables streaming responses when true.
	// Default: false. Set to true for SSE streaming responses.
	Stream bool `json:"stream,omitempty"`

	// Sampling parameters
	// Temperature controls randomness in output generation.
	// Range: 0.0 to 1.0. Higher values produce more random output.
	Temperature float64 `json:"temperature,omitempty"`
	// TopP controls diversity via nucleus sampling.
	// Range: 0.0 to 1.0. Alternative to temperature.
	TopP float64 `json:"top_p,omitempty"`
	// TopK limits sampling to the K most likely tokens.
	// Range: 0 to infinity. 0 means no limit.
	TopK int `json:"top_k,omitempty"`

	// Stop sequences
	// StopSequences is a list of sequences where the API will stop generating.
	StopSequences []string `json:"stop_sequences,omitempty"`

	// System prompt
	// System provides system-level instructions.
	// Can be a string or structured content blocks.
	System interface{} `json:"system,omitempty"`

	// Tool calling
	// Tools is a list of tools the model may call.
	// Optional; each tool defines a function the model can invoke.
	Tools []ToolDef `json:"tools,omitempty"`
	// ToolChoice specifies how the model should choose which tool to use.
	// Optional; defaults to auto if not specified.
	// Values: {"type": "auto"}, {"type": "any"}, {"type": "tool", "name": "..."}
	ToolChoice *ToolChoice `json:"tool_choice,omitempty"`

	// Extended thinking
	// Thinking enables extended thinking mode for supported models.
	Thinking *ThinkingConfig `json:"thinking,omitempty"`

	// Metadata contains arbitrary metadata for the request.
	// Used for tracking and logging purposes.
	Metadata json.RawMessage `json:"metadata,omitempty"`
}

// MessageInput represents a single message input in an Anthropic conversation.
// Each message has a role and content, which can be text or structured blocks.
type MessageInput struct {
	// Role identifies the speaker: "user" or "assistant".
	// Required for each message in the conversation.
	Role string `json:"role"`
	// Content is the message content.
	// Can be a string or array of content blocks for multi-modal input.
	Content interface{} `json:"content"`
}

// ToolDef represents a tool definition in the Anthropic API format.
// Defines a tool that the model can choose to call during generation.
type ToolDef struct {
	// Name is the tool name.
	// Must be unique within the tools list; used in tool_use content blocks.
	Name string `json:"name"`
	// Description explains what the tool does.
	// Used by the model to decide when to call the tool.
	Description string `json:"description,omitempty"`
	// InputSchema is a JSON Schema object defining the tool's parameters.
	// Describes the expected structure of the input object.
	InputSchema json.RawMessage `json:"input_schema"`
}

// Event represents a streaming event from the Anthropic API.
// Each event has a type indicating what kind of data it contains.
type Event struct {
	// Type indicates the event type.
	// Values: "message_start", "content_block_start", "content_block_delta",
	// "content_block_stop", "message_delta", "message_stop", "ping", "error".
	Type string `json:"type"`
	// Index indicates the content block index for content events.
	// Used to correlate deltas with their content blocks.
	Index *int `json:"index,omitempty"`
	// Delta contains incremental content for content_block_delta events.
	// Raw JSON that must be parsed based on the delta type.
	Delta json.RawMessage `json:"delta,omitempty"`
	// ContentBlock contains full content block data for content_block_start events.
	// Describes a text or tool_use block.
	ContentBlock json.RawMessage `json:"content_block,omitempty"`
	// Message contains message metadata for message_start events.
	// Provides ID, model, and usage information.
	Message *MessageInfo    `json:"message,omitempty"`
	Usage   *AnthropicUsage `json:"usage,omitempty"`
	// StopReason indicates why generation stopped.
	// Values: "end_turn", "max_tokens", "stop_sequence", "tool_use".
	StopReason string `json:"stop_reason,omitempty"`
	// StopSequence is the custom stop sequence that triggered stop, if any.
	// Only present when stop_reason is "stop_sequence".
	StopSequence *string `json:"stop_sequence,omitempty"`
}

// MessageInfo contains metadata about a message from the Anthropic API.
// Provides context about the generated response.
type MessageInfo struct {
	// ID is a unique identifier for the message.
	// Format: "msg_xxx". Used for logging and debugging.
	ID string `json:"id"`
	// Type identifies the object type.
	// Always "message" for message responses.
	Type string `json:"type"`
	// Role is always "assistant" for responses.
	// Indicates this is a model-generated message.
	Role string `json:"role"`
	// Content is an array of content blocks in the message.
	// Can include text and tool_use blocks.
	Content []ContentBlock `json:"content"`
	// Model is the model used for generation.
	// May differ from requested model for aliases.
	Model string          `json:"model"`
	Usage *AnthropicUsage `json:"usage,omitempty"`
}

// ContentBlock represents a block of content in an Anthropic message.
// Each block is either text, tool use, tool result, thinking, or image content.
type ContentBlock struct {
	// Type identifies the content block type.
	// Values: "text", "tool_use", "tool_result", "thinking", "image".
	Type string `json:"type"`
	// Text contains the text content for text blocks.
	// Only present when Type is "text".
	Text string `json:"text,omitempty"`
	// ID is a unique identifier for tool_use blocks.
	// Used to correlate tool results with their calls.
	ID string `json:"id,omitempty"`
	// Name is the tool name for tool_use blocks.
	// Only present when Type is "tool_use".
	Name string `json:"name,omitempty"`
	// Input is the tool input for tool_use blocks.
	// JSON object containing the tool arguments.
	Input json.RawMessage `json:"input,omitempty"`
	// Thinking contains the model's reasoning process.
	// Only present when Type is "thinking".
	Thinking string `json:"thinking,omitempty"`
	// ToolUseID references the tool call for tool_result blocks.
	// Only present when Type is "tool_result".
	ToolUseID string `json:"tool_use_id,omitempty"`
	// Content is the result content for tool_result blocks.
	// Can be a string or array of content blocks.
	Content interface{} `json:"content,omitempty"`
	// Source contains image source data for image blocks.
	// Only present when Type is "image".
	Source *ImageSource `json:"source,omitempty"`
	// IsError indicates if the tool result is an error.
	// Only present when Type is "tool_result".
	IsError bool `json:"is_error,omitempty"`
}

// ImageSource represents the source of an image in Anthropic format.
type ImageSource struct {
	// Type is the source type: "base64" or "url".
	Type string `json:"type"`
	// MediaType is the MIME type of the image (e.g., "image/png").
	MediaType string `json:"media_type,omitempty"`
	// Data is the base64-encoded image data (for base64 type).
	Data string `json:"data,omitempty"`
	// URL is the image URL (for url type).
	URL string `json:"url,omitempty"`
}

// TextDelta represents a text content delta in a streaming response.
// Used for incremental text updates during streaming.
type TextDelta struct {
	// Type is always "text_delta" for text content updates.
	Type string `json:"type"`
	// Text is the incremental text to append.
	// Accumulated across deltas to form complete text.
	Text string `json:"text"`
}

// ThinkingDelta represents a thinking content delta in a streaming response.
// Used for incremental reasoning updates during streaming.
type ThinkingDelta struct {
	// Type is always "thinking_delta" for thinking content updates.
	Type string `json:"type"`
	// Thinking is the incremental thinking content to append.
	// Accumulated across deltas to form complete reasoning.
	Thinking string `json:"thinking"`
}

// InputJSONDelta represents a partial JSON input delta for tool use in streaming.
// Tool inputs are streamed incrementally as partial JSON strings.
type InputJSONDelta struct {
	// Type is always "input_json_delta" for tool input updates.
	Type string `json:"type"`
	// PartialJSON is a fragment of the JSON input string.
	// Concatenated across deltas to form the complete JSON.
	PartialJSON string `json:"partial_json"`
}

// AnthropicUsage represents token usage statistics from the Anthropic API.
// Uses snake_case field names matching Anthropic's API format.
type AnthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	// Cache-related tokens
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
}

// ToolChoice specifies how the model should choose which tool to use.
type ToolChoice struct {
	// Type is the choice type: "auto", "any", or "tool"
	Type string `json:"type"`
	// Name is the tool name when Type is "tool"
	Name string `json:"name,omitempty"`
}

// AnthropicErrorResponse represents an error response from the Anthropic API.
type AnthropicErrorResponse struct {
	// Type identifies the error response type.
	// Typically "error" for error responses.
	Type string `json:"type"`
	// Error contains the error details.
	// Always present in error responses.
	Error AnthropicErrorDetail `json:"error"`
}

// AnthropicErrorDetail contains the details of an Anthropic API error.
// Provides information about what went wrong.
type AnthropicErrorDetail struct {
	// Type categorizes the error.
	// Values: "invalid_request_error", "authentication_error", "not_found_error", etc.
	Type string `json:"type"`
	// Message is a human-readable error description.
	// Provides context about the error.
	Message string `json:"message"`
}

// MessageCountTokensRequest represents a request to the /v1/messages/count_tokens endpoint.
// This request structure matches the Anthropic Messages API count_tokens format.
type MessageCountTokensRequest struct {
	// Model is the model identifier to use for token counting.
	// Required field - different models may have different tokenization.
	Model string `json:"model"`
	// Messages is the conversation history to count tokens for.
	// Required field - array of message objects.
	Messages []MessageInput `json:"messages"`
	// System provides system-level instructions.
	// Optional; can be a string or structured content blocks.
	System interface{} `json:"system,omitempty"`
	// Tools is a list of tools that may be called.
	// Optional; tool definitions affect token count.
	Tools []ToolDef `json:"tools,omitempty"`
	// Thinking enables extended thinking mode.
	// Optional; thinking budget affects token count.
	Thinking *ThinkingConfig `json:"thinking,omitempty"`
}

// ThinkingConfig represents extended thinking configuration.
// Used to enable and configure thinking mode for supported models.
type ThinkingConfig struct {
	// Type enables thinking mode.
	// Value: "enabled" to activate thinking.
	Type string `json:"type"`
	// BudgetTokens is the maximum tokens for thinking.
	// Required when thinking is enabled.
	BudgetTokens int `json:"budget_tokens"`
}

// MessageCountTokensResponse represents the response from the count_tokens endpoint.
// Returns the estimated input token count for the provided messages.
type MessageCountTokensResponse struct {
	// InputTokens is the estimated number of tokens in the input.
	// This is an estimate; actual token count may differ slightly.
	InputTokens int `json:"input_tokens"`
}
