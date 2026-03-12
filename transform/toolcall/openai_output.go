// Package toolcall provides tool call transformation functionality for streaming responses.
// It implements a state machine parser that extracts tool calls from special delimiter tokens
// and format-specific output formatters that emit properly formatted deltas.
package toolcall

import (
	"encoding/json"
	"io"

	"ai-proxy/types"
)

// OpenAIOutput implements EventHandler for OpenAI-compatible streaming output format.
// It generates SSE events in the OpenAI chat completions streaming format.
type OpenAIOutput struct {
	// writer is the destination for SSE output.
	writer io.Writer
	// base provides the template for output chunks, containing model info and metadata.
	base types.StreamChunk
	// current holds the ID of the tool call currently being processed.
	current string
}

// NewOpenAIOutput creates a new OpenAI output formatter.
//
// @brief    Initializes a new OpenAI format output handler.
// @param    writer The io.Writer to send SSE events to.
// @param    base   The base chunk template containing model and ID information.
// @return   Pointer to a newly allocated OpenAIOutput instance.
//
// @pre      writer must not be nil.
// @pre      base should contain valid ID and model fields for the response.
// @post     OpenAIOutput is ready to receive events.
func NewOpenAIOutput(writer io.Writer, base types.StreamChunk) *OpenAIOutput {
	return &OpenAIOutput{
		writer: writer,
		base:   base,
	}
}

// OnText handles regular text content by emitting it as a content delta.
//
// @brief    Emits text content in OpenAI streaming format.
// @param    text The text content to emit.
//
// @note     Text is wrapped in a StreamDelta with Content field set.
// @note     Each call produces one SSE data event.
//
// @pre      OpenAIOutput must be initialized.
func (o *OpenAIOutput) OnText(text string) {
	chunk := o.shallowCopy()
	chunk.Choices[0].Delta = types.StreamDelta{Content: text}
	o.writeChunk(chunk)
}

// OnToolCallStart handles the beginning of a tool call by emitting the initial delta.
//
// @brief    Emits the start of a tool call in OpenAI streaming format.
// @param    id    The normalized tool call identifier.
// @param    name  The function name to call.
// @param    index The zero-based index of this tool call in the sequence.
//
// @note     Emits a StreamDelta containing ToolCalls with ID, Type, Index, and Function.Name.
// @note     The tool call type is always set to "function".
//
// @pre      OpenAIOutput must be initialized.
// @post     current is set to the provided id for tracking.
func (o *OpenAIOutput) OnToolCallStart(id, name string, index int) {
	o.current = id
	chunk := o.shallowCopy()
	chunk.Choices[0].Delta = types.StreamDelta{
		ToolCalls: []types.StreamToolCall{{
			ID:       id,
			Type:     "function",
			Index:    index,
			Function: types.StreamFunction{Name: name},
		}},
	}
	o.writeChunk(chunk)
}

// OnToolCallArgs handles tool call arguments by emitting them as a function arguments delta.
//
// @brief    Emits tool call arguments in OpenAI streaming format.
// @param    args  The JSON arguments string (may be partial).
// @param    index The zero-based index of the tool call.
//
// @note     Arguments are streamed incrementally and may arrive in multiple calls.
// @note     Emits a StreamDelta containing ToolCalls with Index and Function.Arguments.
//
// @pre      OpenAIOutput must be initialized.
func (o *OpenAIOutput) OnToolCallArgs(args string, index int) {
	chunk := o.shallowCopy()
	chunk.Choices[0].Delta = types.StreamDelta{
		ToolCalls: []types.StreamToolCall{{
			Index:    index,
			Function: types.StreamFunction{Arguments: args},
		}},
	}
	o.writeChunk(chunk)
}

// OnToolCallEnd handles the end of a tool call (no-op for OpenAI format).
//
// @brief    Signals the end of a tool call in OpenAI streaming format.
// @param    index The zero-based index of the completed tool call.
//
// @note     OpenAI format does not require explicit end markers for tool calls.
//
//	Arguments simply stop arriving when the call is complete.
//
// @pre      OpenAIOutput must be initialized.
func (o *OpenAIOutput) OnToolCallEnd(index int) {
}

// shallowCopy creates a copy of the base chunk with a new choices slice.
//
// @brief    Creates a minimal copy of the base chunk for output.
// @return   A new StreamChunk with copied base fields and fresh choices.
//
// @note     Preserves ID, Object, Model, and Created fields from base.
// @note     Creates a new single-element Choices slice with preserved Index and FinishReason.
//
// @pre      base must be initialized.
func (o *OpenAIOutput) shallowCopy() types.StreamChunk {
	cp := o.base
	cp.Choices = []types.StreamChoice{{Index: 0}}
	if len(o.base.Choices) > 0 {
		cp.Choices[0].Index = o.base.Choices[0].Index
		cp.Choices[0].FinishReason = o.base.Choices[0].FinishReason
	}
	return cp
}

// writeChunk serializes and writes a chunk as an SSE data event.
//
// @brief    Writes a StreamChunk as an SSE data event to the output writer.
// @param    chunk The chunk to serialize and write.
//
// @note     Output format is "data: <json>\n\n" for SSE compliance.
// @note     JSON marshal errors are silently ignored.
//
// @pre      writer must be initialized and writable.
func (o *OpenAIOutput) writeChunk(chunk types.StreamChunk) {
	b, err := json.Marshal(chunk)
	if err != nil {
		return
	}
	o.writer.Write([]byte("data: "))
	o.writer.Write(b)
	o.writer.Write([]byte("\n\n"))
}
