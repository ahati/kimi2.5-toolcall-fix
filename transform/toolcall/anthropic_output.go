// Package toolcall provides tool call transformation functionality for streaming responses.
// It implements a state machine parser that extracts tool calls from special delimiter tokens
// and format-specific output formatters that emit properly formatted deltas.
package toolcall

import (
	"encoding/json"
	"fmt"
	"io"

	"ai-proxy/types"
)

// OutputContext specifies the content block context for Anthropic output.
// This determines whether text content is emitted as thinking or regular text.
type OutputContext int

const (
	// ContextThinking indicates text should be emitted as thinking content blocks.
	// Used when the model is outputting reasoning/thinking content.
	ContextThinking OutputContext = iota

	// ContextText indicates text should be emitted as regular text content blocks.
	// Used for standard assistant response content.
	ContextText
)

// AnthropicOutput implements EventHandler for Anthropic-compatible streaming output format.
// It generates SSE events in the Anthropic Messages API streaming format.
type AnthropicOutput struct {
	// writer is the destination for SSE output.
	writer io.Writer
	// context determines whether text is emitted as thinking or regular content.
	context OutputContext
	// blockIndex tracks the current content block index for the next block.
	blockIndex int
	// currentIndex holds the index of the currently active content block.
	currentIndex int
	// toolsEmitted indicates whether any tool calls have been emitted.
	toolsEmitted bool
	// blockOpen indicates whether a content block is currently open.
	blockOpen bool
}

// NewAnthropicOutput creates a new Anthropic output formatter.
//
// @brief    Initializes a new Anthropic format output handler.
// @param    writer            The io.Writer to send SSE events to.
// @param    context           The output context (thinking or text).
// @param    initialBlockIndex The starting index for content blocks.
// @return   Pointer to a newly allocated AnthropicOutput instance.
//
// @pre      writer must not be nil.
// @pre      initialBlockIndex should be 0 unless resuming a stream.
// @post     AnthropicOutput is ready to receive events.
func NewAnthropicOutput(writer io.Writer, context OutputContext, initialBlockIndex int) *AnthropicOutput {
	return &AnthropicOutput{
		writer:     writer,
		context:    context,
		blockIndex: initialBlockIndex,
	}
}

// SetBlockOpen sets whether a content block is currently open.
//
// @brief    Updates the block open state for managing content block transitions.
// @param    open True if a content block is open, false otherwise.
//
// @note     Used to coordinate with external content block management.
//
// @post     blockOpen is updated to the provided value.
func (o *AnthropicOutput) SetBlockOpen(open bool) {
	o.blockOpen = open
}

// ToolsEmitted reports whether any tool calls have been emitted.
//
// @brief    Returns whether tool calls have been output.
// @return   True if at least one tool call has been emitted.
//
// @note     Used to determine if tool use content blocks exist in the output.
func (o *AnthropicOutput) ToolsEmitted() bool {
	return o.toolsEmitted
}

// BlockIndex returns the current content block index.
//
// @brief    Returns the index for the next content block to be created.
// @return   The zero-based content block index.
//
// @note     This value increments as content blocks are opened.
func (o *AnthropicOutput) BlockIndex() int {
	return o.blockIndex
}

// OnText handles regular text content by emitting it as a content block delta.
//
// @brief    Emits text content in Anthropic streaming format.
// @param    text The text content to emit.
//
// @note     Empty text is ignored without producing output.
// @note     In ContextThinking, text is emitted as thinking_delta.
// @note     In ContextText, text is emitted as text_delta.
//
// @pre      AnthropicOutput must be initialized.
func (o *AnthropicOutput) OnText(text string) {
	if text == "" {
		return
	}

	if o.context == ContextThinking {
		delta := types.ThinkingDelta{
			Type:     "thinking_delta",
			Thinking: text,
		}
		o.writeContentBlockDelta(o.currentIndex, delta)
	} else {
		delta := types.TextDelta{
			Type: "text_delta",
			Text: text,
		}
		o.writeContentBlockDelta(o.currentIndex, delta)
	}
}

// OnToolCallStart handles the beginning of a tool call by emitting content_block_start.
//
// @brief    Emits the start of a tool use content block in Anthropic format.
// @param    id    The normalized tool call identifier.
// @param    name  The function name to call.
// @param    index The zero-based index of this tool call (unused, index derived from blockIndex).
//
// @note     Closes any open content block before starting the new one.
// @note     Emits content_block_start event with tool_use type.
// @note     Input is initialized to empty JSON object.
//
// @pre      AnthropicOutput must be initialized.
// @post     toolsEmitted is set to true.
// @post     currentIndex is set to the current blockIndex.
// @post     blockOpen is set to true.
func (o *AnthropicOutput) OnToolCallStart(id, name string, index int) {
	if o.blockOpen {
		o.writeContentBlockStop(o.currentIndex)
		o.blockIndex++
	}

	o.toolsEmitted = true
	o.currentIndex = o.blockIndex

	block := types.ContentBlock{
		Type:  "tool_use",
		ID:    id,
		Name:  name,
		Input: json.RawMessage("{}"),
	}
	o.writeContentBlockStart(o.currentIndex, block)
	o.blockOpen = true
}

// OnToolCallArgs handles tool call arguments by emitting input_json_delta.
//
// @brief    Emits tool call arguments in Anthropic streaming format.
// @param    args  The JSON arguments string (may be partial).
// @param    index The zero-based index of the tool call (unused).
//
// @note     Arguments are streamed incrementally as partial JSON.
// @note     Uses input_json_delta event type with partial_json field.
//
// @pre      AnthropicOutput must be initialized.
// @pre      A tool call must have been started with OnToolCallStart.
func (o *AnthropicOutput) OnToolCallArgs(args string, index int) {
	delta := types.InputJSONDelta{
		Type:        "input_json_delta",
		PartialJSON: args,
	}
	o.writeContentBlockDelta(o.currentIndex, delta)
}

// OnToolCallEnd handles the end of a tool call by emitting content_block_stop.
//
// @brief    Emits the end of a tool use content block in Anthropic format.
// @param    index The zero-based index of the completed tool call (unused).
//
// @note     Emits content_block_stop event to close the tool use block.
//
// @pre      AnthropicOutput must be initialized.
// @pre      A tool call must have been started with OnToolCallStart.
// @post     blockIndex is incremented.
// @post     blockOpen is set to false.
func (o *AnthropicOutput) OnToolCallEnd(index int) {
	o.writeContentBlockStop(o.currentIndex)
	o.blockIndex++
	o.blockOpen = false
}

// writeContentBlockStart emits a content_block_start SSE event.
//
// @brief    Writes a content_block_start event for a new content block.
// @param    index The content block index.
// @param    block The content block to emit.
//
// @note     The content block is serialized to JSON for the content_block field.
// @note     Event type is "content_block_start".
//
// @pre      writer must be initialized and writable.
func (o *AnthropicOutput) writeContentBlockStart(index int, block types.ContentBlock) {
	blockJSON, _ := json.Marshal(block)
	event := types.Event{
		Type:         "content_block_start",
		Index:        intPtr(index),
		ContentBlock: blockJSON,
	}
	o.writeSSE(event)
}

// writeContentBlockDelta emits a content_block_delta SSE event.
//
// @brief    Writes a content_block_delta event for streaming content.
// @param    index The content block index.
// @param    delta The delta content (TextDelta, ThinkingDelta, or InputJSONDelta).
//
// @note     The delta is serialized to JSON for the delta field.
// @note     Event type is "content_block_delta".
//
// @pre      writer must be initialized and writable.
func (o *AnthropicOutput) writeContentBlockDelta(index int, delta interface{}) {
	deltaJSON, _ := json.Marshal(delta)
	event := types.Event{
		Type:  "content_block_delta",
		Index: intPtr(index),
		Delta: deltaJSON,
	}
	o.writeSSE(event)
}

// writeContentBlockStop emits a content_block_stop SSE event.
//
// @brief    Writes a content_block_stop event to close a content block.
// @param    index The content block index to close.
//
// @note     Event type is "content_block_stop".
// @note     This event has no data beyond the type and index.
//
// @pre      writer must be initialized and writable.
func (o *AnthropicOutput) writeContentBlockStop(index int) {
	event := types.Event{
		Type:  "content_block_stop",
		Index: intPtr(index),
	}
	o.writeSSE(event)
}

// writeSSE writes an event as a Server-Sent Events formatted message.
//
// @brief    Serializes and writes an event in SSE format.
// @param    event The event to write.
//
// @note     Output format is "event: <type>\ndata: <json>\n\n".
// @note     JSON marshal errors are silently ignored.
//
// @pre      writer must be initialized and writable.
func (o *AnthropicOutput) writeSSE(event types.Event) {
	data, _ := json.Marshal(event)
	line := fmt.Sprintf("event: %s\ndata: %s\n\n", event.Type, string(data))
	o.writer.Write([]byte(line))
}

// intPtr returns a pointer to an int value.
//
// @brief    Creates a pointer to an int for use in JSON serialization.
// @param    i The int value to point to.
// @return   Pointer to the provided int value.
//
// @note     Used for nullable int fields in API response types.
func intPtr(i int) *int {
	return &i
}
