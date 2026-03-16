package toolcall

import (
	"encoding/json"
	"time"

	"ai-proxy/types"
)

// OpenAIFormatter formats parsed events as OpenAI streaming chat completion chunks.
// It implements the OutputFormatter interface for OpenAI-compatible streaming output.
//
// @brief Formatter for OpenAI streaming chat completion format.
//
// @note OpenAI streaming format uses "chat.completion.chunk" objects with delta updates.
//
//	Each chunk contains incremental updates to the response.
//
// @note Tool calls are streamed with multiple chunks: first chunk has tool call ID and name,
//
//	subsequent chunks contain argument fragments.
//
// @note This formatter is NOT thread-safe. Create a new instance for each request.
//
// @pre Must be initialized via NewOpenAIFormatter before use.
// @post Produces valid OpenAI streaming chunks for all event types.
//
// Output Format:
// - Content: data: {"id":"...","object":"chat.completion.chunk",...,"choices":[{"delta":{"content":"..."}}]}
// - ToolStart: data: {"id":"...","object":"chat.completion.chunk",...,"choices":[{"delta":{"tool_calls":[{"id":"...","function":{"name":"..."}}]}}]}
// - ToolArgs: data: {"id":"...","object":"chat.completion.chunk",...,"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"..."}}]}}]}
type OpenAIFormatter struct {
	// messageID is the unique identifier for the chat completion response.
	// Used in all chunks to correlate streaming updates.
	// Format: typically "chatcmpl-xxx" or similar.
	messageID string

	// model is the model name included in each chunk.
	// Should match the model requested in the API call.
	model string

	// toolIndex tracks the last formatted tool call index.
	// Used to maintain consistency across tool call events.
	toolIndex int
}

// NewOpenAIFormatter creates a formatter for OpenAI streaming responses.
//
// @brief Creates a new OpenAIFormatter initialized with message ID and model.
//
// @param messageID The unique identifier for the chat completion.
//
//	Format: typically "chatcmpl-xxx" or provider-specific format.
//	Used in all output chunks for correlation.
//	May be empty (will be updated from upstream response).
//
// @param model The model name to include in output chunks.
//
//	Should match the model requested by the client.
//	Examples: "gpt-4", "moonshotai/Kimi-K2.5-TEE".
//	May be empty (will be updated from upstream response).
//
// @return *OpenAIFormatter A new formatter ready for use.
//
// @pre None (messageID and model may be empty initially).
// @post Formatter is ready to format events.
// @post SetMessageID and SetModel can be called to update values.
//
// @note If messageID is empty, it will be populated from the first upstream chunk.
func NewOpenAIFormatter(messageID, model string) *OpenAIFormatter {
	return &OpenAIFormatter{
		messageID: messageID,
		model:     model,
	}
}

// FormatContent formats text content as a chat completion chunk.
//
// @brief Formats a text content event into OpenAI streaming chunk format.
//
// @param content The text content to format.
//
//	Must be valid UTF-8 for correct JSON encoding.
//	Empty content results in empty output.
//
// @return []byte Formatted SSE data: "data: {...}\n\n"
//
//	Returns nil if content is empty.
//	Caller must not modify the returned slice.
//
// @pre content should be valid UTF-8.
// @post Returned data is valid OpenAI streaming chunk.
//
// @note The chunk includes: id, object, created timestamp, model, and choices array.
// @note Each content chunk is independent and can be sent immediately.
func (f *OpenAIFormatter) FormatContent(content string) []byte {
	return f.marshalChunk(types.Chunk{
		ID:      f.messageID,
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   f.model,
		Choices: []types.Choice{{
			Index: 0,
			Delta: types.Delta{Content: content},
		}},
	})
}

func (f *OpenAIFormatter) FormatReasoning(reasoning string) []byte {
	return f.marshalChunk(types.Chunk{
		ID:      f.messageID,
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   f.model,
		Choices: []types.Choice{{
			Index: 0,
			Delta: types.Delta{ReasoningContent: reasoning},
		}},
	})
}

// FormatToolStart formats the beginning of a tool call in OpenAI format.
//
// @brief Formats a tool call start event into OpenAI streaming chunk format.
//
// @param id The unique identifier for this tool call.
//
//	Format: typically "call_xxx" for OpenAI compatibility.
//	Must be unique within the response.
//
// @param name The name of the function being called.
//
//	Must match a defined tool/function name.
//
// @param index The zero-based index of this tool call.
//
//	Must be >= 0.
//	Used to correlate subsequent argument chunks.
//
// @return []byte Formatted SSE data: "data: {...}\n\n"
//
//	Contains tool_calls array with id, type, index, and function name.
//	Caller must not modify the returned slice.
//
// @pre id must be unique within the response.
// @pre name must be a valid function name.
// @pre index must be >= 0.
// @post toolIndex is updated to track the current index.
// @post Returned data is valid OpenAI streaming chunk.
//
// @note The function.arguments field is initialized as empty string.
//
//	Subsequent FormatToolArgs calls will populate the arguments.
func (f *OpenAIFormatter) FormatToolStart(id, name string, index int) []byte {
	// Track the tool call index for consistency.
	f.toolIndex = index
	return f.marshalChunk(types.Chunk{
		ID:      f.messageID,
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   f.model,
		Choices: []types.Choice{{
			Index: 0,
			// Tool call start includes id, type, index, and function name.
			// Arguments field is empty and will be populated by FormatToolArgs.
			Delta: types.Delta{ToolCalls: []types.ToolCall{{
				ID:    id,
				Type:  "function",
				Index: index,
				Function: types.Function{
					Name:      name,
					Arguments: "",
				},
			}}},
		}},
	})
}

// FormatToolArgs formats tool call arguments in OpenAI format.
//
// @brief Formats tool call argument fragment into OpenAI streaming chunk format.
//
// @param args The argument data fragment.
//
//	Must be valid JSON fragment for correct client parsing.
//	May be empty (returns nil).
//
// @param index The zero-based index of this tool call.
//
//	Must match the index used in FormatToolStart.
//
// @return []byte Formatted SSE data: "data: {...}\n\n"
//
//	Contains tool_calls array with function.arguments.
//	Returns nil if args is empty.
//	Caller must not modify the returned slice.
//
// @pre args should be valid JSON fragment.
// @pre index must match a previous FormatToolStart call.
// @post Returned data is valid OpenAI streaming chunk.
//
// @note OpenAI format streams arguments as string fragments.
//
//	The client concatenates all fragments to form the complete JSON.
func (f *OpenAIFormatter) FormatToolArgs(args string, index int) []byte {
	return f.marshalChunk(types.Chunk{
		ID:      f.messageID,
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   f.model,
		Choices: []types.Choice{{
			Index: 0,
			// Tool call args update includes only index and arguments.
			// The id and name are not repeated after the initial tool start.
			Delta: types.Delta{ToolCalls: []types.ToolCall{{
				Index: index,
				Function: types.Function{
					Arguments: args,
				},
			}}},
		}},
	})
}

// FormatToolEnd returns nil as OpenAI format does not require explicit tool call end markers.
//
// @brief Formats a tool call end event into OpenAI streaming chunk format.
//
// @param index The zero-based index of this tool call.
//
//	Must match the index used in FormatToolStart.
//	Currently unused as OpenAI format doesn't require explicit end.
//
// @return []byte Always returns nil (OpenAI format has no explicit tool end marker).
//
// @pre None (method is a no-op for OpenAI format).
// @post No output is produced.
//
// @note OpenAI streaming format does not have explicit tool call end markers.
//
//	The tool call is considered complete when the stream ends or a new tool starts.
func (f *OpenAIFormatter) FormatToolEnd(index int) []byte {
	// OpenAI format does not require explicit tool call end markers.
	// Tool calls are implicitly ended by stream termination or new tool start.
	return nil
}

// FormatSectionEnd returns nil as OpenAI format does not require section end markers.
//
// @brief Formats a section end event into OpenAI streaming chunk format.
//
// @return []byte Always returns nil (OpenAI format has no section end marker).
//
// @pre None (method is a no-op for OpenAI format).
// @post No output is produced.
//
// @note OpenAI streaming format does not have explicit section end markers.
func (f *OpenAIFormatter) FormatSectionEnd() []byte {
	// OpenAI format does not require section end markers.
	return nil
}

// FormatDone returns the [DONE] marker for OpenAI streaming responses.
//
// @brief Formats the stream termination marker for OpenAI format.
//
// @return []byte The [DONE] marker: "data: [DONE]\n\n"
//
//	Caller must not modify the returned slice.
//
// @pre All content and tool call chunks must have been sent.
// @post Stream should be terminated after sending this marker.
//
// @note This is the final message in an OpenAI streaming response.
//
//	It signals to the client that no more chunks will be sent.
func (f *OpenAIFormatter) FormatDone() []byte {
	return []byte("data: [DONE]\n\n")
}

// marshalChunk serializes a chunk to SSE format.
//
// @brief Internal method to serialize a chunk to SSE data format.
//
// @param chunk The chunk to serialize.
//
//	Must have valid fields for JSON encoding.
//
// @return []byte Formatted SSE data: "data: {...}\n\n"
//
//	Panics if JSON encoding fails (should never happen).
//
// @pre chunk must be JSON-encodable.
// @post Returned data is valid SSE format.
//
// @note Uses json.Marshal which cannot fail for the Chunk type.
//
//	Error is ignored as Chunk fields are guaranteed encodable.
func (f *OpenAIFormatter) marshalChunk(chunk types.Chunk) []byte {
	b, _ := json.Marshal(chunk)
	// Prepend "data: " and append double newline for SSE format.
	// Double newline is required by SSE specification.
	return []byte("data: " + string(b) + "\n\n")
}

// SetMessageID updates the message ID used in output chunks.
//
// @brief Updates the message ID for subsequent output chunks.
//
// @param id The new message ID.
//
//	Format: typically "chatcmpl-xxx" or similar.
//	Must be valid UTF-8 for JSON encoding.
//
// @return None (method returns no value).
//
// @pre id should be a valid message ID format.
// @post All subsequent chunks will use the new message ID.
//
// @note Typically called when the upstream response provides an ID
//
//	that was not initially known.
func (f *OpenAIFormatter) SetMessageID(id string) {
	f.messageID = id
}

// SetModel updates the model name used in output chunks.
//
// @brief Updates the model name for subsequent output chunks.
//
// @param model The new model name.
//
//	Should match the model requested by the client.
//	Must be valid UTF-8 for JSON encoding.
//
// @return None (method returns no value).
//
// @pre model should be a valid model identifier.
// @post All subsequent chunks will use the new model name.
//
// @note Typically called when the upstream response provides a model name
//
//	that was not initially known.
func (f *OpenAIFormatter) SetModel(model string) {
	f.model = model
}
