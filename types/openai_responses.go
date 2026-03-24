// Package types defines data structures for OpenAI and Anthropic API formats.
// This file contains types specific to the OpenAI Responses API format.
package types

import "encoding/json"

// ResponsesRequest represents an OpenAI Responses API request.
// This is the primary request structure for the /v1/responses endpoint.
type ResponsesRequest struct {
	// Model is the model identifier to use for the response.
	Model string `json:"model"`
	// Input is the input to the model.
	// Can be a string, array of input items, or a structured input object.
	Input interface{} `json:"input"`
	// Instructions provides system-level instructions (replaces system message).
	Instructions string `json:"instructions,omitempty"`
	// MaxOutputTokens is the maximum number of tokens to generate.
	MaxOutputTokens int `json:"max_output_tokens,omitempty"`
	// Stream enables streaming responses when true.
	// Use pointer to distinguish between "not specified" (nil) and "explicitly false".
	Stream *bool `json:"stream,omitempty"`
	// Tools is a list of tools the model may call.
	Tools []ResponsesTool `json:"tools,omitempty"`
	// ToolChoice specifies which tool the model should use.
	// Values: "none", "auto", "required", or {"type": "function", "function": {"name": "..."}}
	ToolChoice interface{} `json:"tool_choice,omitempty"`
	// Temperature controls randomness in output generation.
	Temperature float64 `json:"temperature,omitempty"`
	// TopP controls diversity via nucleus sampling.
	TopP float64 `json:"top_p,omitempty"`
	// PreviousResponseId is the ID of the previous response for multi-turn conversations.
	PreviousResponseID string `json:"previous_response_id,omitempty"`
	// Reasoning enables reasoning mode for supported models.
	Reasoning *ReasoningConfig `json:"reasoning,omitempty"`
	// ParallelToolCalls enables parallel tool calling.
	ParallelToolCalls bool `json:"parallel_tool_calls,omitempty"`
	// ResponseFormat specifies the format of the response.
	// Use for JSON mode or structured outputs.
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
	// Metadata contains arbitrary metadata for the request.
	// Used for tracking and logging purposes, including user_id.
	Metadata map[string]interface{} `json:"metadata,omitempty"`
	// Store controls whether the response should be stored server-side.
	// Default is true. When false, enables ZDR (Zero Data Retention) mode.
	// Use pointer to distinguish between "not specified" (nil, defaults to true) and "explicitly false".
	Store *bool `json:"store,omitempty"`
}

// ReasoningConfig represents reasoning configuration for supported models.
type ReasoningConfig struct {
	// Effort controls the model's reasoning effort level.
	// Values: "low", "medium", "high"
	Effort string `json:"effort,omitempty"`
	// Summary determines whether to return reasoning summary.
	// Values: "concise", "detailed", or omitted for no summary.
	Summary string `json:"summary,omitempty"`
}

// ResponsesTool represents a tool in the OpenAI Responses API format.
type ResponsesTool struct {
	// Type is the tool type.
	// Values: "function", "file_search", "web_search", "computer_use_preview"
	Type string `json:"type"`
	// Name is the tool name (for flat format function tools, or computer_use_preview).
	Name string `json:"name,omitempty"`
	// Description explains what the tool does (for flat format function tools).
	Description string `json:"description,omitempty"`
	// Parameters is JSON Schema for the tool parameters (for flat format function tools).
	Parameters json.RawMessage `json:"parameters,omitempty"`
	// Strict enables strict mode (for flat format function tools).
	Strict bool `json:"strict,omitempty"`
	// DisplayWidth is the display width for computer_use_preview.
	DisplayWidth int `json:"display_width,omitempty"`
	// DisplayHeight is the display height for computer_use_preview.
	DisplayHeight int `json:"display_height,omitempty"`
	// Environment is the environment for computer_use_preview.
	Environment string `json:"environment,omitempty"`
	// Function contains function definition for function type tools (nested format).
	Function *ResponsesToolFunction `json:"function,omitempty"`
}

// ResponsesToolFunction describes a function tool for the Responses API.
type ResponsesToolFunction struct {
	// Name is the function name.
	Name string `json:"name"`
	// Description explains what the function does.
	Description string `json:"description,omitempty"`
	// Parameters is a JSON Schema object defining the function's parameters.
	Parameters json.RawMessage `json:"parameters,omitempty"`
	// Strict enables strict mode for the function.
	Strict bool `json:"strict,omitempty"`
}

// InputItem represents an input item in the conversation.
type InputItem struct {
	// Type identifies the input item type.
	// Values: "message", "function_call", "function_call_output"
	Type string `json:"type"`
	// Role identifies the speaker for message type.
	// Values: "user", "assistant", "system", "developer"
	Role string `json:"role,omitempty"`
	// Content is the message content.
	// Can be a string or array of content parts.
	Content interface{} `json:"content,omitempty"`
	// ID is the unique identifier for function_call type.
	ID string `json:"id,omitempty"`
	// CallID is the call ID for function_call and function_call_output types.
	CallID string `json:"call_id,omitempty"`
	// ToolCallID is an alternative field name for call_id (used by some clients).
	ToolCallID string `json:"tool_call_id,omitempty"`
	// Name is the function name for function_call type.
	Name string `json:"name,omitempty"`
	// Arguments is the JSON-encoded arguments for function_call type.
	Arguments string `json:"arguments,omitempty"`
	// Output is the function result for function_call_output type.
	Output string `json:"output,omitempty"`
}

// ContentPart represents a content part in a message.
type ContentPart struct {
	// Type identifies the content part type.
	// Values: "input_text", "input_image", "input_file", "output_text"
	Type string `json:"type"`
	// Text is the text content for text types.
	Text string `json:"text,omitempty"`
	// ImageURL is the URL for image content.
	ImageURL string `json:"image_url,omitempty"`
	// FileData contains file data for file types.
	FileData *FileData `json:"file_data,omitempty"`
	// Annotations contains annotations for output text.
	Annotations []Annotation `json:"annotations,omitempty"`
}

// FileData represents file data in a content part.
type FileData struct {
	// Filename is the name of the file.
	Filename string `json:"filename"`
	// FileData is the base64-encoded file content.
	FileData string `json:"file_data"`
}

// Annotation represents an annotation in output text.
type Annotation struct {
	// Type identifies the annotation type.
	// Values: "url_citation", "file_citation", "file_path"
	Type string `json:"type"`
	// URL is the URL for url_citation type.
	URL string `json:"url,omitempty"`
	// Title is the title for url_citation type.
	Title string `json:"title,omitempty"`
	// FileID is the file ID for file_citation type.
	FileID string `json:"file_id,omitempty"`
}

// ResponsesResponse represents a response from the OpenAI Responses API.
type ResponsesResponse struct {
	// ID is a unique identifier for the response.
	ID string `json:"id"`
	// Object identifies the object type.
	// Always "response".
	Object string `json:"object"`
	// CreatedAt is the Unix timestamp of response creation.
	CreatedAt int64 `json:"created_at"`
	// Status indicates the response status.
	// Values: "in_progress", "completed", "incomplete"
	Status string `json:"status"`
	// Error contains error information if the request failed.
	Error *ResponsesError `json:"error,omitempty"`
	// IncompleteDetails contains details if the response is incomplete.
	IncompleteDetails *IncompleteDetails `json:"incomplete_details,omitempty"`
	// Instructions that were used for this response.
	Instructions string `json:"instructions,omitempty"`
	// MaxOutputTokens that were used.
	MaxOutputTokens int `json:"max_output_tokens,omitempty"`
	// Model used for generation.
	Model string `json:"model"`
	// Output contains the generated output items.
	Output []OutputItem `json:"output"`
	// ParallelToolCalls setting used.
	ParallelToolCalls bool `json:"parallel_tool_calls,omitempty"`
	// PreviousResponseID for multi-turn conversations.
	PreviousResponseID string `json:"previous_response_id,omitempty"`
	// Reasoning summary if requested.
	Reasoning *ReasoningSummary `json:"reasoning,omitempty"`
	// Usage contains token usage statistics.
	Usage *ResponsesUsage `json:"usage,omitempty"`
	// User identifier.
	User string `json:"user,omitempty"`
	// Metadata associated with the response.
	Metadata map[string]string `json:"metadata,omitempty"`
}

// ReasoningSummary contains reasoning summary if requested.
type ReasoningSummary struct {
	// Summary contains the reasoning text.
	Summary string `json:"summary,omitempty"`
}

// ResponsesError represents an error in the response.
type ResponsesError struct {
	// Code is the error code.
	Code string `json:"code"`
	// Message is the error message.
	Message string `json:"message"`
}

// IncompleteDetails contains details about incomplete responses.
type IncompleteDetails struct {
	// Reason indicates why the response is incomplete.
	// Values: "max_output_tokens", "content_filter"
	Reason string `json:"reason"`
}

// OutputItem represents an output item in the response.
type OutputItem struct {
	// Type identifies the output type.
	// Values: "message", "file_search_call", "web_search_call", "computer_use_call", "function_call", "reasoning"
	Type string `json:"type"`
	// ID is the unique identifier for this output item.
	ID string `json:"id,omitempty"`
	// Status of the output item.
	// Values: "in_progress", "completed", "incomplete"
	Status string `json:"status,omitempty"`
	// Role is always "assistant" for message type.
	Role string `json:"role,omitempty"`
	// Content contains the output content for message type.
	Content []OutputContent `json:"content,omitempty"`
	// CallID for tool calls.
	CallID string `json:"call_id,omitempty"`
	// Name is the function name for function_call type.
	Name string `json:"name,omitempty"`
	// Arguments are the function arguments for function_call type.
	Arguments string `json:"arguments,omitempty"`
	// Summary contains reasoning summary for reasoning type.
	Summary string `json:"summary,omitempty"`
	// Action for computer_use_call.
	Action interface{} `json:"action,omitempty"`
	// PendingSafetyChecks for computer_use_call.
	PendingSafetyChecks []SafetyCheck `json:"pending_safety_checks,omitempty"`
}

// SafetyCheck represents a safety check for computer use.
type SafetyCheck struct {
	// Code identifies the safety check.
	Code string `json:"code"`
	// Message describes the safety check.
	Message string `json:"message"`
}

// OutputContent represents content within an output item.
type OutputContent struct {
	// Type identifies the content type.
	// Values: "output_text", "reasoning", "function_call", "refusal"
	Type string `json:"type"`
	// Text is the text content for output_text type.
	Text string `json:"text,omitempty"`
	// Annotations for output_text type.
	Annotations []Annotation `json:"annotations,omitempty"`
	// ID is the call ID for function_call type.
	ID string `json:"id,omitempty"`
	// CallID is the call ID for function_call type.
	CallID string `json:"call_id,omitempty"`
	// Name is the function name for function_call type.
	Name string `json:"name,omitempty"`
	// Arguments are the function arguments for function_call type.
	Arguments string `json:"arguments,omitempty"`
	// Summary is the reasoning summary for reasoning type.
	Summary string `json:"summary,omitempty"`
}

// ResponsesUsage represents token usage for the response.
type ResponsesUsage struct {
	// InputTokens is the number of input tokens.
	InputTokens int `json:"input_tokens"`
	// OutputTokens is the number of output tokens.
	OutputTokens int `json:"output_tokens"`
	// TotalTokens is the total number of tokens.
	TotalTokens int `json:"total_tokens"`
	// InputTokensDetails contains detailed input token counts.
	InputTokensDetails *InputTokensDetails `json:"input_tokens_details,omitempty"`
	// OutputTokensDetails contains detailed output token counts.
	OutputTokensDetails *OutputTokensDetails `json:"output_tokens_details,omitempty"`
}

// InputTokensDetails contains detailed input token information.
type InputTokensDetails struct {
	// CachedTokens is the number of cached tokens.
	CachedTokens int `json:"cached_tokens,omitempty"`
}

// OutputTokensDetails contains detailed output token information.
type OutputTokensDetails struct {
	// ReasoningTokens is the number of reasoning tokens.
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

// ResponsesEventType enumerates all Responses API SSE event names.
type ResponsesEventType string

const (
	EventResponseCreated                    ResponsesEventType = "response.created"
	EventResponseInProgress                 ResponsesEventType = "response.in_progress"
	EventResponseOutputItemAdded            ResponsesEventType = "response.output_item.added"
	EventResponseContentPartAdded           ResponsesEventType = "response.content_part.added"
	EventResponseOutputTextDelta            ResponsesEventType = "response.output_text.delta"
	EventResponseOutputTextDone             ResponsesEventType = "response.output_text.done"
	EventResponseFunctionCallArgumentsDelta ResponsesEventType = "response.function_call_arguments.delta"
	EventResponseFunctionCallArgumentsDone  ResponsesEventType = "response.function_call_arguments.done"
	EventResponseOutputItemDone             ResponsesEventType = "response.output_item.done"
	EventResponseCompleted                  ResponsesEventType = "response.completed"
	EventResponseFailed                     ResponsesEventType = "response.failed"
	EventResponseIncomplete                 ResponsesEventType = "response.incomplete"
)

// ResponsesStreamEvent represents a streaming event in the Responses API.
type ResponsesStreamEvent struct {
	// Type identifies the event type.
	// Values: "response.created", "response.in_progress", "response.output_item.added",
	// "response.content_part.added", "response.output_text.delta", "response.function_call_arguments.delta",
	// "response.content_part.done", "response.output_item.done", "response.completed",
	// "response.incomplete", "error"
	Type string `json:"type"`
	// Response contains the full response for certain events.
	Response *ResponsesResponse `json:"response,omitempty"`
	// OutputItem for output item events.
	OutputItem *OutputItem `json:"item,omitempty"`
	// ContentIndex for content events.
	ContentIndex int `json:"content_index,omitempty"`
	// Delta for delta events.
	Delta string `json:"delta,omitempty"`
	// OutputIndex maps Responses output_index to block index.
	OutputIndex int `json:"output_index,omitempty"`
	// ItemID for item-related events.
	ItemID string `json:"item_id,omitempty"`
	// Arguments for function_call_arguments.done events.
	Arguments string `json:"arguments,omitempty"`
	// Text content for text events.
	Text string `json:"text,omitempty"`
	// Error information.
	Error *ResponsesError `json:"error,omitempty"`
}

// ReasoningOutputItem represents a reasoning output item in the Responses API.
// Reasoning items contain the model's internal reasoning process and appear
// before other output items in the response.
type ReasoningOutputItem struct {
	// Type is always "reasoning".
	Type string `json:"type"`
	// ID is the unique identifier for this reasoning item.
	// Format: "rs_xxx" (derived from message ID).
	ID string `json:"id"`
	// Summary contains the reasoning summary text segments.
	Summary []SummaryTextItem `json:"summary"`
}

// SummaryTextItem represents a text segment within a reasoning summary.
type SummaryTextItem struct {
	// Type is always "summary_text".
	Type string `json:"type"`
	// Text is the reasoning summary content.
	Text string `json:"text"`
}

// FunctionCallOutputItem represents a function_call output item in the Responses API.
// Function call items are separate output items, not nested within message content.
type FunctionCallOutputItem struct {
	// Type is always "function_call".
	Type string `json:"type"`
	// ID is the unique identifier for this function call.
	// This matches the tool call ID from Anthropic (e.g., "toolu_xxx").
	ID string `json:"id"`
	// CallID is the call ID used to reference this function call.
	// Same as ID for consistency.
	CallID string `json:"call_id"`
	// Name is the name of the function being called.
	Name string `json:"name"`
	// Arguments is the JSON-encoded arguments for the function call.
	Arguments string `json:"arguments"`
}

// MessageOutputItem represents a message output item in the Responses API.
// Message items contain the assistant's text response.
type MessageOutputItem struct {
	// Type is always "message".
	Type string `json:"type"`
	// ID is the unique identifier for this message.
	// Format: "msg_xxx" or the response ID.
	ID string `json:"id"`
	// Status indicates the message status.
	// Values: "in_progress", "completed".
	Status string `json:"status"`
	// Role is always "assistant".
	Role string `json:"role"`
	// Content contains the message content parts.
	Content []map[string]interface{} `json:"content"`
}

// NewReasoningOutputItem creates a new reasoning output item.
// The ID should follow the "rs_xxx" format convention.
func NewReasoningOutputItem(id, summaryText string) *ReasoningOutputItem {
	return &ReasoningOutputItem{
		Type: "reasoning",
		ID:   id,
		Summary: []SummaryTextItem{
			{Type: "summary_text", Text: summaryText},
		},
	}
}

// NewFunctionCallOutputItem creates a new function_call output item.
// The ID should be the tool call ID from Anthropic (e.g., "toolu_xxx").
func NewFunctionCallOutputItem(id, name, arguments string) *FunctionCallOutputItem {
	return &FunctionCallOutputItem{
		Type:      "function_call",
		ID:        id,
		CallID:    id,
		Name:      name,
		Arguments: arguments,
	}
}

// NewMessageOutputItem creates a new message output item.
// The content should be an array of content parts (e.g., output_text).
func NewMessageOutputItem(id string, content []map[string]interface{}) *MessageOutputItem {
	return &MessageOutputItem{
		Type:    "message",
		ID:      id,
		Status:  "completed",
		Role:    "assistant",
		Content: content,
	}
}
