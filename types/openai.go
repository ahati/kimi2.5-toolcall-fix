// Package types defines data structures for OpenAI and Anthropic API formats.
// This file contains types specific to the OpenAI Chat Completions API format.
package types

import "encoding/json"

// ChatCompletionRequest represents an OpenAI chat completion API request.
// This is the primary request structure for the /v1/chat/completions endpoint.
type ChatCompletionRequest struct {
	// Model is the model identifier to use for completion.
	// Valid values: "gpt-4", "gpt-3.5-turbo", or any OpenAI-compatible model ID.
	Model string `json:"model"`
	// Messages is the conversation history as an array of message objects.
	// Must contain at least one message with role "user".
	Messages []Message `json:"messages"`
	// MaxTokens is the maximum number of tokens to generate.
	// Optional; if not set, the model's default limit is used.
	MaxTokens int `json:"max_tokens,omitempty"`
	// Stream enables streaming responses when true.
	// Default: false. Set to true for SSE streaming responses.
	Stream bool `json:"stream,omitempty"`
	// StreamOptions configures streaming behavior.
	// Set include_usage: true to receive usage statistics in the final chunk.
	StreamOptions *StreamOptions `json:"stream_options,omitempty"`

	// Sampling parameters
	// Temperature controls randomness in output generation.
	// Range: 0.0 to 2.0. Higher values produce more random output.
	Temperature float64 `json:"temperature,omitempty"`
	// TopP controls diversity via nucleus sampling.
	// Range: 0.0 to 1.0. Alternative to temperature.
	TopP float64 `json:"top_p,omitempty"`
	// TopK limits sampling to the K most likely tokens.
	// Not supported by all providers.
	TopK int `json:"top_k,omitempty"`

	// Stop sequences
	// Stop sequences where the API will stop generating further tokens.
	// Can be a string or array of strings.
	Stop interface{} `json:"stop,omitempty"`

	// Penalties
	// PresencePenalty penalizes new tokens based on whether they appear in the text so far.
	// Range: -2.0 to 2.0.
	PresencePenalty float64 `json:"presence_penalty,omitempty"`
	// FrequencyPenalty penalizes new tokens based on their frequency in the text so far.
	// Range: -2.0 to 2.0.
	FrequencyPenalty float64 `json:"frequency_penalty,omitempty"`

	// Output control
	// N is the number of chat completion choices to generate.
	N int `json:"n,omitempty"`
	// ResponseFormat specifies the format of the response.
	// Use for JSON mode or structured outputs.
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`

	// Tool calling
	// Tools is a list of tools the model may call.
	// Optional; each tool defines a function the model can invoke.
	Tools []Tool `json:"tools,omitempty"`
	// ToolChoice controls which tool is called by the model.
	// Can be "none", "auto", "required", or a specific tool object.
	ToolChoice interface{} `json:"tool_choice,omitempty"`
	// ParallelToolCalls enables parallel tool calling when true.
	ParallelToolCalls *bool `json:"parallel_tool_calls,omitempty"`

	// Advanced parameters
	// LogitBias modifies the likelihood of specified tokens appearing in the completion.
	LogitBias map[int]float64 `json:"logit_bias,omitempty"`
	// LogProbs whether to return log probabilities of the output tokens.
	LogProbs *bool `json:"logprobs,omitempty"`
	// TopLogProbs specifies how many top log probabilities to return (0-20).
	TopLogProbs int `json:"top_logprobs,omitempty"`
	// Seed enables deterministic sampling for reproducible outputs.
	Seed *int `json:"seed,omitempty"`

	// User identifier for abuse monitoring.
	User string `json:"user,omitempty"`

	// Service tier for the request.
	ServiceTier string `json:"service_tier,omitempty"`

	// Deprecated: System field is non-standard. Use a system message in Messages array instead.
	// Kept for backwards compatibility with existing clients.
	System string `json:"system,omitempty"`
}

// StreamOptions configures streaming behavior for chat completion requests.
type StreamOptions struct {
	// IncludeUsage enables usage statistics in the final chunk when true.
	IncludeUsage bool `json:"include_usage,omitempty"`
}

// ResponseFormat specifies the format of the response output.
// Used for JSON mode and structured outputs.
type ResponseFormat struct {
	// Type specifies the response format type.
	// Values: "text" (default), "json_object", "json_schema".
	Type string `json:"type"`
	// JSONSchema specifies the JSON schema for structured outputs.
	// Required when Type is "json_schema".
	JSONSchema *JSONSchemaConfig `json:"json_schema,omitempty"`
}

// JSONSchemaConfig configures JSON schema for structured outputs.
type JSONSchemaConfig struct {
	// Name is a descriptive name for the schema.
	Name string `json:"name"`
	// Description is an optional description of the schema.
	Description string `json:"description,omitempty"`
	// Schema is the JSON Schema definition.
	Schema json.RawMessage `json:"schema"`
	// Strict enables strict schema validation when true.
	Strict bool `json:"strict,omitempty"`
}

// Message represents a single message in a chat conversation.
// Each message has a role and content, optionally including tool calls.
type Message struct {
	// Role identifies the speaker: "system", "user", "assistant", or "tool".
	// Required for each message in the conversation.
	Role string `json:"role"`
	// Content is the message content.
	// Can be a string or array of content parts for multi-modal messages.
	Content interface{} `json:"content,omitempty"`
	// ToolCalls contains tool calls made by the assistant.
	// Only present in assistant messages that call tools.
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	// ToolCallID references the tool call being responded to.
	// Only present in tool response messages.
	ToolCallID string `json:"tool_call_id,omitempty"`
}

// ToolCall represents a function call made by the model.
// When the model decides to call a tool, it generates a ToolCall object.
type ToolCall struct {
	// ID is a unique identifier for this tool call.
	// Used to correlate tool responses with their calls.
	ID string `json:"id"`
	// Type is the type of tool call.
	// Currently only "function" is supported.
	Type string `json:"type"`
	// Index is the position of this tool call in the streaming response.
	// Used for incremental updates during streaming.
	Index int `json:"index"`
	// Function contains the function name and arguments to call.
	Function Function `json:"function"`
}

// Function represents the function details within a tool call.
// Contains the function name and JSON-encoded arguments.
type Function struct {
	// Name is the name of the function to call.
	// Must match a function name in the tools list.
	Name string `json:"name"`
	// Arguments is a JSON-encoded string of function arguments.
	// Must be parsed by the caller to get actual argument values.
	Arguments string `json:"arguments"`
}

// Tool represents a tool that can be called by the model.
// Tools extend the model's capabilities by allowing it to call external functions.
type Tool struct {
	// Type is the tool type.
	// Currently only "function" is supported.
	Type string `json:"type"`
	// Function describes the function that can be called.
	Function ToolFunction `json:"function"`
}

// ToolFunction describes a function that can be used as a tool.
// Defines the function signature including name, description, and parameters.
type ToolFunction struct {
	// Name is the function name to call.
	// Must be a valid identifier; used in tool calls.
	Name string `json:"name"`
	// Description explains what the function does.
	// Used by the model to decide when to call the function.
	Description string `json:"description,omitempty"`
	// Parameters is a JSON Schema object defining the function's parameters.
	// Describes the expected structure of the arguments object.
	Parameters json.RawMessage `json:"parameters,omitempty"`
}

// Chunk represents a streaming response chunk from the OpenAI API.
// Each chunk contains a partial response during SSE streaming.
type Chunk struct {
	// ID is a unique identifier for the completion.
	// Consistent across all chunks in a single streaming response.
	ID string `json:"id"`
	// Object identifies the object type.
	// Typically "chat.completion.chunk" for streaming responses.
	Object string `json:"object"`
	// Created is the Unix timestamp of chunk creation.
	// Seconds since epoch.
	Created int64 `json:"created"`
	// Model is the model used for generation.
	// May differ from requested model for aliases.
	Model string `json:"model"`
	// Choices is an array of completion choices.
	// Usually contains one choice for chat completions.
	Choices []Choice `json:"choices"`
	// Usage contains token usage statistics.
	// Only present in the final chunk when stream_options.include_usage is true.
	Usage *Usage `json:"usage,omitempty"`
}

// Choice represents a single choice within a streaming chunk.
// Contains the incremental content delta and completion status.
type Choice struct {
	// Index is the position of this choice in the choices array.
	// Zero for single-choice completions.
	Index int `json:"index"`
	// Delta contains the incremental content for this chunk.
	// Content is accumulated across all chunks.
	Delta Delta `json:"delta"`
	// FinishReason indicates why generation stopped.
	// Values: "stop", "length", "tool_calls", "content_filter", or nil if ongoing.
	FinishReason *string `json:"finish_reason,omitempty"`
}

// Delta represents the incremental content in a streaming response.
// Each chunk adds to the accumulated response through its delta.
type Delta struct {
	// Role is set in the first chunk to indicate the assistant role.
	// Only present in the initial chunk.
	Role string `json:"role,omitempty"`
	// Content is the text content for this chunk.
	// Accumulated across chunks to form the complete response.
	Content string `json:"content,omitempty"`
	// Reasoning contains the model's reasoning process (if supported).
	// Used for models that expose chain-of-thought.
	Reasoning string `json:"reasoning,omitempty"`
	// ReasoningContent is an alternative field for reasoning content.
	// Some models use this instead of Reasoning.
	ReasoningContent string `json:"reasoning_content,omitempty"`
	// ToolCalls contains incremental tool call information.
	// Tool calls are streamed incrementally across multiple chunks.
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	// FinishReason is duplicated here in some API versions.
	// Prefer Choice.FinishReason for consistency.
	FinishReason *string `json:"finish_reason,omitempty"`
}

// Usage represents token usage statistics for a completion.
// Provides insight into the computational cost of the request.
type Usage struct {
	// PromptTokens is the number of tokens in the prompt.
	// Includes system prompt, messages, and tool definitions.
	PromptTokens int `json:"prompt_tokens"`
	// CompletionTokens is the number of tokens in the generated response.
	// Does not include streaming overhead.
	CompletionTokens int `json:"completion_tokens"`
	// TotalTokens is the sum of prompt and completion tokens.
	// Useful for cost estimation.
	TotalTokens int `json:"total_tokens"`
	// PromptTokensDetails contains detailed prompt token counts.
	PromptTokensDetails *PromptTokensDetails `json:"prompt_tokens_details,omitempty"`
	// CompletionTokensDetails contains detailed completion token counts.
	CompletionTokensDetails *CompletionTokensDetails `json:"completion_tokens_details,omitempty"`
}

// PromptTokensDetails contains detailed prompt token information.
type PromptTokensDetails struct {
	// CachedTokens is the number of tokens read from cache.
	CachedTokens int `json:"cached_tokens,omitempty"`
}

// CompletionTokensDetails contains detailed completion token information.
type CompletionTokensDetails struct {
	// ReasoningTokens is the number of tokens used for reasoning.
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

// ErrorResponse represents an error response from the OpenAI API.
// Returned when the API cannot process the request.
type ErrorResponse struct {
	// Error contains the error details.
	// Always present in error responses.
	Error ErrorDetail `json:"error"`
}

// ErrorDetail contains the details of an API error.
// Provides information about what went wrong.
type ErrorDetail struct {
	// Type categorizes the error.
	// Common values: "invalid_request_error", "authentication_error", etc.
	Type string `json:"type"`
	// Message is a human-readable error description.
	// Provides context about the error.
	Message string `json:"message"`
	// Code is an optional error code for programmatic handling.
	// Not always present; depends on error type.
	Code string `json:"code,omitempty"`
}
