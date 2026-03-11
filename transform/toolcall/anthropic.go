package toolcall

import (
	"encoding/json"
	"fmt"

	"ai-proxy/types"
)

// AnthropicFormatter formats parsed events as Anthropic streaming message events.
// It implements the OutputFormatter interface for Anthropic-compatible streaming output.
//
// @brief Formatter for Anthropic streaming message format.
//
// @note Anthropic streaming format uses event types with typed delta updates.
//
//	Events include: content_block_start, content_block_delta, content_block_stop, message_stop.
//
// @note Tool calls are represented as content blocks with type "tool_use".
//
//	Each tool call gets a unique block index that must be consistent across events.
//
// @note This formatter is NOT thread-safe. Create a new instance for each request.
//
// @pre Must be initialized via NewAnthropicFormatter before use.
// @post Produces valid Anthropic streaming events for all event types.
//
// Output Format:
// - Content: event: content_block_delta\ndata: {"type":"content_block_delta","index":N,"delta":{"type":"text_delta","text":"..."}}
// - ToolStart: event: content_block_start\ndata: {"type":"content_block_start","index":N,"content_block":{"type":"tool_use","id":"...","name":"..."}}
// - ToolArgs: event: content_block_delta\ndata: {"type":"content_block_delta","index":N,"delta":{"type":"input_json_delta","partial_json":"..."}}
// - ToolEnd: event: content_block_stop\ndata: {"type":"content_block_stop","index":N}
type AnthropicFormatter struct {
	// messageID is the unique identifier for the message response.
	// Included in message_start event (not emitted by this formatter).
	messageID string

	// model is the model name for the response.
	// Included in message_start event (not emitted by this formatter).
	model string

	// blockIndex tracks the current content block index.
	// Incremented for each new content block (text or tool use).
	// Starts at -1 and is incremented before first use.
	// Each tool call increments the index via FormatToolStart.
	blockIndex int

	// toolsEmitted indicates whether any tool call events have been formatted.
	// Used to determine if content blocks should include tool use markers.
	toolsEmitted bool
}

// NewAnthropicFormatter creates a formatter for Anthropic streaming responses.
//
// @brief Creates a new AnthropicFormatter initialized with message ID and model.
//
// @param messageID The unique identifier for the message.
//
//	Format: typically "msg_xxx" or similar.
//	May be empty (Anthropic format may not require it).
//
// @param model The model name to include in the response.
//
//	Should match the model requested in the API call.
//	Examples: "claude-3-opus-20240229", "kimi-k2.5".
//	May be empty.
//
// @return *AnthropicFormatter A new formatter ready for use.
//
// @pre None (messageID and model may be empty).
// @post Formatter is ready to format events.
// @post blockIndex is initialized to -1 (will be incremented to 0 on first use).
//
// @note The blockIndex starts at -1 because it's incremented before first use.
func NewAnthropicFormatter(messageID, model string) *AnthropicFormatter {
	return &AnthropicFormatter{
		messageID: messageID,
		model:     model,
		// Start at -1 so first increment yields 0.
		blockIndex: -1,
	}
}

// FormatContent formats text content as a content_block_delta event.
//
// @brief Formats a text content event into Anthropic streaming event format.
//
// @param content The text content to format.
//
//	Must be valid UTF-8 for correct JSON encoding.
//	Empty content results in empty output.
//
// @return []byte Formatted SSE data with event type header.
//
//	Returns nil or empty slice if content is empty.
//	Format: "event: content_block_delta\ndata: {...}\n\n"
//	Caller must not modify the returned slice.
//
// @pre content should be valid UTF-8.
// @pre blockIndex should be >= 0 (content block should exist).
// @post Returned data is valid Anthropic streaming event.
//
// @note Content is formatted as text_delta within content_block_delta.
//
//	The index field refers to the current content block.
func (f *AnthropicFormatter) FormatContent(content string) []byte {
	return f.formatEvent("content_block_delta", f.blockIndex, types.TextDelta{
		Type: "text_delta",
		Text: content,
	})
}

// FormatToolStart formats the beginning of a tool call as a content_block_start event.
//
// @brief Formats a tool call start event into Anthropic streaming event format.
//
// @param id The unique identifier for this tool call.
//
//	Format: typically a UUID or similar unique string.
//	Must be unique within the response.
//
// @param name The name of the function being called.
//
//	Must match a defined tool/function name.
//
// @param index The zero-based index of this tool call.
//
//	Note: This parameter is ignored; blockIndex is used instead.
//	The parser's index is tracked separately from Anthropic's block index.
//
// @return []byte Formatted SSE data with event type header.
//
//	Format: "event: content_block_start\ndata: {...}\n\n"
//	Caller must not modify the returned slice.
//
// @pre id must be unique within the response.
// @pre name must be a valid function name.
// @post toolsEmitted is set to true.
// @post blockIndex is incremented.
// @post Returned data is valid Anthropic streaming event.
//
// @note Tool calls are represented as content blocks with type "tool_use".
//
//	The input field is initialized as empty JSON object.
func (f *AnthropicFormatter) FormatToolStart(id, name string, index int) []byte {
	// Mark that tool events have been emitted.
	f.toolsEmitted = true
	// Increment block index for this new content block.
	f.blockIndex++
	return f.formatEvent("content_block_start", f.blockIndex, types.ContentBlock{
		Type: "tool_use",
		ID:   id,
		Name: name,
		// Initialize input as empty JSON object.
		// The actual input will be streamed via FormatToolArgs.
		Input: json.RawMessage("{}"),
	})
}

// FormatToolArgs formats tool call arguments as a content_block_delta with partial JSON.
//
// @brief Formats tool call argument fragment into Anthropic streaming event format.
//
// @param args The argument data fragment.
//
//	Must be valid JSON fragment for correct client parsing.
//	May be empty (returns formatted event with empty partial_json).
//
// @param index The zero-based index of this tool call from the parser.
//
//	Note: This parameter is ignored; blockIndex is used instead.
//	Maintained for interface compatibility.
//
// @return []byte Formatted SSE data with event type header.
//
//	Format: "event: content_block_delta\ndata: {...}\n\n"
//	Caller must not modify the returned slice.
//
// @pre args should be valid JSON fragment.
// @pre blockIndex must match a previous FormatToolStart call.
// @post Returned data is valid Anthropic streaming event.
//
// @note Anthropic uses "input_json_delta" with "partial_json" field for streaming.
//
//	The client concatenates all partial_json values to form complete input.
func (f *AnthropicFormatter) FormatToolArgs(args string, index int) []byte {
	return f.formatEvent("content_block_delta", f.blockIndex, types.InputJSONDelta{
		Type:        "input_json_delta",
		PartialJSON: args,
	})
}

// FormatToolEnd formats the end of a tool call as a content_block_stop event.
//
// @brief Formats a tool call end event into Anthropic streaming event format.
//
// @param index The zero-based index of this tool call from the parser.
//
//	Note: This parameter is ignored; blockIndex is used instead.
//	Maintained for interface compatibility.
//
// @return []byte Formatted SSE data with event type header.
//
//	Format: "event: content_block_stop\ndata: {...}\n\n"
//	Caller must not modify the returned slice.
//
// @pre blockIndex must match a previous FormatToolStart call.
// @pre All argument fragments must have been sent via FormatToolArgs.
// @post Returned data is valid Anthropic streaming event.
//
// @note Anthropic requires explicit content_block_stop events to close each block.
//
//	This is different from OpenAI which has no explicit end marker.
func (f *AnthropicFormatter) FormatToolEnd(index int) []byte {
	return f.formatEvent("content_block_stop", f.blockIndex, nil)
}

// FormatSectionEnd returns nil as Anthropic format does not require section end markers.
//
// @brief Formats a section end event into Anthropic streaming event format.
//
// @return []byte Always returns nil (Anthropic format has no section end marker).
//
// @pre None (method is a no-op for Anthropic format).
// @post No output is produced.
//
// @note Anthropic streaming format does not have explicit section end markers.
func (f *AnthropicFormatter) FormatSectionEnd() []byte {
	// Anthropic format does not require section end markers.
	return nil
}

// FormatDone returns the message_stop event for Anthropic streaming responses.
//
// @brief Formats the stream termination event for Anthropic format.
//
// @return []byte The message_stop event: "event: message_stop\ndata: {}\n\n"
//
//	Caller must not modify the returned slice.
//
// @pre All content blocks must have been closed with content_block_stop.
// @post Stream should be terminated after sending this event.
//
// @note This is the final event in an Anthropic streaming response.
//
//	It signals to the client that no more events will be sent.
func (f *AnthropicFormatter) FormatDone() []byte {
	return []byte("event: message_stop\ndata: {}\n\n")
}

// formatEvent creates a formatted SSE event with the specified type and data.
//
// @brief Internal method to format an Anthropic SSE event.
//
// @param eventType The Anthropic event type name.
//
//	Must be a valid Anthropic event type.
//	Examples: "content_block_start", "content_block_delta", "content_block_stop".
//
// @param index The content block index.
//
//	Must be >= 0 for events that require an index.
//	Use -1 for events without an index.
//
// @param data The event payload.
//
//	Must be JSON-encodable or nil.
//	For nil, the delta field is omitted from output.
//
// @return []byte Formatted SSE event string.
//
//	Format: "event: <type>\ndata: {...}\n\n"
//
// @pre eventType must be a valid Anthropic event type.
// @pre data must be JSON-encodable or nil.
// @post Returned data is valid SSE format.
func (f *AnthropicFormatter) formatEvent(eventType string, index int, data interface{}) []byte {
	var deltaBytes json.RawMessage
	if data != nil {
		b, _ := json.Marshal(data)
		deltaBytes = b
	}
	event := types.Event{
		Type:  eventType,
		Delta: deltaBytes,
	}
	// Only include index if it's non-negative.
	// Negative index means this event doesn't have an index field.
	if index >= 0 {
		event.Index = intPtr(index)
	}
	return f.marshalEvent(event)
}

// marshalEvent serializes an event to SSE format.
//
// @brief Internal method to serialize an event to SSE data format.
//
// @param event The event to serialize.
//
//	Must have valid fields for JSON encoding.
//
// @return []byte Formatted SSE event string.
//
//	Format: "event: <type>\ndata: {...}\n\n"
//	Panics if JSON encoding fails (should never happen).
//
// @pre event must be JSON-encodable.
// @post Returned data is valid SSE format.
//
// @note Uses json.Marshal which cannot fail for the Event type.
func (f *AnthropicFormatter) marshalEvent(event types.Event) []byte {
	data, _ := json.Marshal(event)
	// Format as SSE with event type header and data body.
	// Double newline is required by SSE specification.
	return []byte(fmt.Sprintf("event: %s\ndata: %s\n\n", event.Type, string(data)))
}

// intPtr returns a pointer to an int value.
// This is a helper function for creating optional int fields.
//
// @brief Creates a pointer to an int value.
//
// @param i The int value to point to.
//
// @return *int Pointer to the provided int value.
//
// @pre None.
// @post Returned pointer is valid and non-nil.
//
// @note Used for optional JSON fields that should be omitted when not set.
func intPtr(i int) *int {
	return &i
}

// SetMessageID updates the message ID used in output events.
//
// @brief Updates the message ID for subsequent output events.
//
// @param id The new message ID.
//
//	Format: typically "msg_xxx" or similar.
//	Must be valid UTF-8 for JSON encoding.
//
// @return None (method returns no value).
//
// @pre id should be a valid message ID format.
// @post All subsequent events will use the new message ID.
func (f *AnthropicFormatter) SetMessageID(id string) {
	f.messageID = id
}

// SetModel updates the model name used in output events.
//
// @brief Updates the model name for subsequent output events.
//
// @param model The new model name.
//
//	Should match the model requested by the client.
//	Must be valid UTF-8 for JSON encoding.
//
// @return None (method returns no value).
//
// @pre model should be a valid model identifier.
// @post All subsequent events will use the new model name.
func (f *AnthropicFormatter) SetModel(model string) {
	f.model = model
}

// ToolsEmitted reports whether any tool call events have been formatted.
//
// @brief Returns whether tool call events have been emitted.
//
// @return bool True if at least one FormatToolStart was called.
//
//	False if no tool calls have been formatted.
//
// @pre None.
// @post No state is modified.
//
// @note Used to determine response structure for message formatting.
//
//	If tools were emitted, the response should indicate tool use.
func (f *AnthropicFormatter) ToolsEmitted() bool {
	return f.toolsEmitted
}

// BlockIndex returns the current content block index.
//
// @brief Returns the current content block index counter.
//
// @return int The current block index.
//
//	Returns -1 if no blocks have been created.
//
// @pre None.
// @post No state is modified.
//
// @note Used for testing and debugging block index management.
func (f *AnthropicFormatter) BlockIndex() int {
	return f.blockIndex
}

// IncrementBlockIndex advances the content block index.
//
// @brief Increments the block index by one.
//
// @return None (method returns no value).
//
// @pre None.
// @post blockIndex is incremented by 1.
//
// @note Used when text content precedes tool calls.
//
//	Text content blocks must be accounted for in the index.
func (f *AnthropicFormatter) IncrementBlockIndex() {
	f.blockIndex++
}

// SetBlockIndex sets the content block index directly.
//
// @brief Sets the block index to a specific value.
//
// @param index The new block index value.
//
//	Should be >= -1 (typical range).
//
// @return None (method returns no value).
//
// @pre index should be a valid block index.
// @post blockIndex is set to the provided value.
//
// @note Used when synchronizing block index with external state.
//
//	For example, when pre-pending text content blocks.
func (f *AnthropicFormatter) SetBlockIndex(index int) {
	f.blockIndex = index
}
