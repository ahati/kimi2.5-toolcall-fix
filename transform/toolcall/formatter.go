package toolcall

// OutputFormatter defines methods for formatting parsed events into API-specific output.
// Implementations produce SSE data in OpenAI or Anthropic streaming format.
//
// @brief Interface for formatting parsed tool call events to API-specific SSE format.
//
// @note Each API (OpenAI, Anthropic) has different streaming formats for tool calls.
//
//	Implementations must correctly format events according to the target API spec.
//
// @note Implementations must be safe for concurrent use if shared across requests.
//
//	Each request should typically use its own formatter instance.
//
// @pre Formatter must be properly initialized before use.
// @post Output is written to the configured destination.
//
// Thread Safety:
// - Implementations that maintain state (e.g., block indices) must synchronize access.
// - Stateless implementations are inherently thread-safe.
type OutputFormatter interface {
	// FormatContent formats regular text content.
	//
	// @brief Formats a text content event into API-specific SSE format.
	//
	// @param content The text content to format.
	//               Must be valid UTF-8.
	//               May be empty (implementation may return empty output).
	//
	// @return []byte Formatted SSE data for the content event.
	//                Returns nil or empty slice if content is empty.
	//                Caller must not modify the returned slice.
	//
	// @pre content should be valid UTF-8 for correct JSON encoding.
	// @post Returned data is ready to write to SSE stream.
	//
	// @note For OpenAI, this produces a chat.completion.chunk with delta content.
	// @note For Anthropic, this produces a content_block_delta event.
	FormatContent(content string) []byte

	// FormatToolStart formats the beginning of a tool call with ID and name.
	//
	// @brief Formats a tool call start event into API-specific SSE format.
	//
	// @param id The unique identifier for this tool call.
	//           Format varies by API (OpenAI uses "call_xxx", Anthropic uses UUIDs).
	//           Must be unique within the response.
	//
	// @param name The name of the function being called.
	//             Must match a defined tool/function name.
	//             Must be valid UTF-8.
	//
	// @param index The zero-based index of this tool call in the response.
	//              Must be >= 0 and consistent across related events.
	//              Used to correlate start, args, and end events.
	//
	// @return []byte Formatted SSE data for the tool start event.
	//                Returns nil or empty slice for APIs that don't require explicit start.
	//                Caller must not modify the returned slice.
	//
	// @pre id must be unique within the response.
	// @pre name must be a valid function name.
	// @pre index must be >= 0 and consistent.
	// @post Returned data is ready to write to SSE stream.
	//
	// @note For OpenAI, this produces a chunk with delta containing tool_calls array.
	// @note For Anthropic, this produces a content_block_start event with tool_use type.
	FormatToolStart(id, name string, index int) []byte

	// FormatToolArgs formats tool call argument data.
	//
	// @brief Formats tool call argument fragment into API-specific SSE format.
	//
	// @param args The argument data fragment.
	//             Must be valid JSON fragment for correct client parsing.
	//             May be empty (implementation may return empty output).
	//             Multiple calls may stream a complete JSON object.
	//
	// @param index The zero-based index of this tool call in the response.
	//              Must match the index used in FormatToolStart.
	//              Must be >= 0.
	//
	// @return []byte Formatted SSE data for the argument fragment.
	//                Returns nil or empty slice if args is empty.
	//                Caller must not modify the returned slice.
	//
	// @pre args should be valid JSON fragment.
	// @pre index must match a previous FormatToolStart call.
	// @post Returned data is ready to write to SSE stream.
	//
	// @note For OpenAI, this produces a chunk with delta containing tool_calls with function.arguments.
	// @note For Anthropic, this produces a content_block_delta with input_json_delta.
	FormatToolArgs(args string, index int) []byte

	// FormatToolEnd formats the end of a tool call.
	//
	// @brief Formats a tool call end event into API-specific SSE format.
	//
	// @param index The zero-based index of this tool call in the response.
	//              Must match the index used in FormatToolStart.
	//              Must be >= 0.
	//
	// @return []byte Formatted SSE data for the tool end event.
	//                Returns nil or empty slice for APIs that don't require explicit end.
	//                Caller must not modify the returned slice.
	//
	// @pre index must match a previous FormatToolStart call.
	// @pre All argument fragments must have been sent via FormatToolArgs.
	// @post Returned data is ready to write to SSE stream.
	//
	// @note For OpenAI, this typically returns nil (no explicit end marker needed).
	// @note For Anthropic, this produces a content_block_stop event.
	FormatToolEnd(index int) []byte

	// FormatSectionEnd formats the end of the tool calls section.
	//
	// @brief Formats a section end event into API-specific SSE format.
	//
	// @return []byte Formatted SSE data for the section end event.
	//                Returns nil or empty slice for APIs that don't require section end.
	//                Caller must not modify the returned slice.
	//
	// @pre All tool calls in the section must have been fully formatted.
	// @post Returned data is ready to write to SSE stream.
	//
	// @note Most APIs do not require explicit section end markers.
	// @note This is included for future extensibility and completeness.
	FormatSectionEnd() []byte

	// SetMessageID updates the message ID used in output events.
	//
	// @brief Sets the message ID for subsequent output events.
	//
	// @param id The message ID to use.
	//
	//	Must be valid UTF-8 for JSON encoding.
	//	Format varies by API (OpenAI: "chatcmpl-xxx", Anthropic: "msg_xxx").
	//
	// @pre None.
	// @post Subsequent events will use this ID.
	//
	// @note Typically called when ID is extracted from first upstream chunk.
	SetMessageID(id string)

	// SetModel updates the model name used in output events.
	//
	// @brief Sets the model name for subsequent output events.
	//
	// @param model The model name to use.
	//
	//	Must be valid UTF-8 for JSON encoding.
	//	Examples: "gpt-4", "moonshotai/Kimi-K2.5-TEE", "kimi-k2.5".
	//
	// @pre None.
	// @post Subsequent events will use this model name.
	//
	// @note Typically called when model is extracted from first upstream chunk.
	SetModel(model string)
}
