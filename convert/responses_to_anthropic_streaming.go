// Package convert provides converters between different API formats.
// This file implements OpenAI Responses API to Anthropic Messages streaming conversion.
package convert

import (
	"encoding/json"
	"io"

	"ai-proxy/logging"
	"ai-proxy/transform"
	"ai-proxy/types"

	"github.com/tmaxmax/go-sse"
)

// ResponsesToAnthropicStreamingTransformer converts OpenAI Responses API SSE events
// to Anthropic Messages API SSE format.
// It implements the transform.SSETransformer interface.
type ResponsesToAnthropicStreamingTransformer struct {
	w io.Writer

	// State tracked during streaming
	responseID string
	model      string

	// Block index tracking
	blockIndex         int
	outputIndexToBlock map[int]int // Maps output_index from Responses to block index in Anthropic

	// Token usage tracking
	inputTokens  int
	outputTokens int
}

// NewResponsesToAnthropicStreamingTransformer creates a new streaming transformer
// that converts Responses API SSE events to Anthropic Messages format.
func NewResponsesToAnthropicStreamingTransformer(w io.Writer) *ResponsesToAnthropicStreamingTransformer {
	return &ResponsesToAnthropicStreamingTransformer{
		w:                  w,
		outputIndexToBlock: make(map[int]int),
	}
}

// Transform processes a Responses API SSE event and converts it to Anthropic format.
func (t *ResponsesToAnthropicStreamingTransformer) Transform(event *sse.Event) error {
	if event.Data == "" || event.Data == "[DONE]" {
		return nil
	}

	var respEvent types.ResponsesStreamEvent
	if err := json.Unmarshal([]byte(event.Data), &respEvent); err != nil {
		// Log and skip unparseable events
		logging.DebugMsg("Failed to parse Responses event: %v", err)
		return nil
	}

	return t.handleEvent(&respEvent)
}

// handleEvent processes a single Responses API event and emits corresponding Anthropic events.
func (t *ResponsesToAnthropicStreamingTransformer) handleEvent(event *types.ResponsesStreamEvent) error {
	switch event.Type {
	case "response.created":
		return t.handleResponseCreated(event)
	case "response.output_item.added":
		return t.handleOutputItemAdded(event)
	case "response.content_part.added":
		return t.handleContentPartAdded(event)
	case "response.output_text.delta":
		return t.handleOutputTextDelta(event)
	case "response.function_call_arguments.delta":
		return t.handleFunctionCallArgsDelta(event)
	case "response.output_text.done":
		return t.handleOutputTextDone(event)
	case "response.function_call_arguments.done":
		return t.handleFunctionCallArgsDone(event)
	case "response.output_item.done":
		return t.handleOutputItemDone(event)
	case "response.completed":
		return t.handleResponseCompleted(event)
	case "response.incomplete":
		return t.handleResponseIncomplete(event)
	case "response.failed":
		return t.handleResponseFailed(event)
	default:
		// Skip unknown events
		return nil
	}
}

// handleResponseCreated handles response.created event → emits message_start.
func (t *ResponsesToAnthropicStreamingTransformer) handleResponseCreated(event *types.ResponsesStreamEvent) error {
	if event.Response == nil {
		return nil
	}

	t.responseID = event.Response.ID
	t.model = event.Response.Model

	// Emit message_start event
	msgStart := map[string]interface{}{
		"type": "message_start",
		"message": map[string]interface{}{
			"id":    t.responseID,
			"type":  "message",
			"role":  "assistant",
			"model": t.model,
			"usage": map[string]int{
				"input_tokens": t.inputTokens,
			},
		},
	}
	return t.emitEvent("message_start", msgStart)
}

// handleOutputItemAdded handles response.output_item.added event.
// For function_call type, emits content_block_start with tool_use.
func (t *ResponsesToAnthropicStreamingTransformer) handleOutputItemAdded(event *types.ResponsesStreamEvent) error {
	if event.OutputItem == nil {
		return nil
	}

	// Track the mapping from output_index to block_index
	t.outputIndexToBlock[event.ContentIndex] = t.blockIndex

	if event.OutputItem.Type == "function_call" {
		// Emit content_block_start for tool_use
		blockStart := map[string]interface{}{
			"type":  "content_block_start",
			"index": t.blockIndex,
			"content_block": map[string]interface{}{
				"type": "tool_use",
				"id":   event.OutputItem.CallID,
				"name": event.OutputItem.Name,
			},
		}
		if err := t.emitEvent("content_block_start", blockStart); err != nil {
			return err
		}
		t.blockIndex++
	}

	return nil
}

// handleContentPartAdded handles response.content_part.added event.
// For output_text type, emits content_block_start with text.
func (t *ResponsesToAnthropicStreamingTransformer) handleContentPartAdded(event *types.ResponsesStreamEvent) error {
	if event.OutputItem == nil {
		return nil
	}

	// Check if this is an output_text content part being added to a message
	// The OutputItem here represents the content part
	if event.OutputItem.Type == "output_text" || (event.OutputItem.Content != nil && len(event.OutputItem.Content) > 0) {
		// For content_part.added with output_text, emit content_block_start for text
		// Track the mapping
		t.outputIndexToBlock[event.ContentIndex] = t.blockIndex

		blockStart := map[string]interface{}{
			"type":  "content_block_start",
			"index": t.blockIndex,
			"content_block": map[string]interface{}{
				"type": "text",
				"text": "",
			},
		}
		if err := t.emitEvent("content_block_start", blockStart); err != nil {
			return err
		}
		t.blockIndex++
	}

	return nil
}

// handleOutputTextDelta handles response.output_text.delta event → emits content_block_delta with text_delta.
func (t *ResponsesToAnthropicStreamingTransformer) handleOutputTextDelta(event *types.ResponsesStreamEvent) error {
	if event.Delta == "" {
		return nil
	}

	idx := t.outputIndexToBlock[event.ContentIndex]

	delta := map[string]interface{}{
		"type":  "content_block_delta",
		"index": idx,
		"delta": map[string]interface{}{
			"type": "text_delta",
			"text": event.Delta,
		},
	}
	return t.emitEvent("content_block_delta", delta)
}

// handleFunctionCallArgsDelta handles response.function_call_arguments.delta event
// → emits content_block_delta with input_json_delta.
func (t *ResponsesToAnthropicStreamingTransformer) handleFunctionCallArgsDelta(event *types.ResponsesStreamEvent) error {
	if event.Delta == "" {
		return nil
	}

	idx := t.outputIndexToBlock[event.ContentIndex]

	delta := map[string]interface{}{
		"type":  "content_block_delta",
		"index": idx,
		"delta": map[string]interface{}{
			"type":         "input_json_delta",
			"partial_json": event.Delta,
		},
	}
	return t.emitEvent("content_block_delta", delta)
}

// handleOutputTextDone handles response.output_text.done event → emits content_block_stop.
func (t *ResponsesToAnthropicStreamingTransformer) handleOutputTextDone(event *types.ResponsesStreamEvent) error {
	idx := t.outputIndexToBlock[event.ContentIndex]

	blockStop := map[string]interface{}{
		"type":  "content_block_stop",
		"index": idx,
	}
	return t.emitEvent("content_block_stop", blockStop)
}

// handleFunctionCallArgsDone handles response.function_call_arguments.done event → emits content_block_stop.
func (t *ResponsesToAnthropicStreamingTransformer) handleFunctionCallArgsDone(event *types.ResponsesStreamEvent) error {
	idx := t.outputIndexToBlock[event.ContentIndex]

	blockStop := map[string]interface{}{
		"type":  "content_block_stop",
		"index": idx,
	}
	return t.emitEvent("content_block_stop", blockStop)
}

// handleOutputItemDone handles response.output_item.done event.
// Currently no specific action needed as content_block_stop is emitted on text/function done events.
func (t *ResponsesToAnthropicStreamingTransformer) handleOutputItemDone(event *types.ResponsesStreamEvent) error {
	// No action needed - content_block_stop already emitted
	return nil
}

// handleResponseCompleted handles response.completed event → emits message_delta + message_stop.
func (t *ResponsesToAnthropicStreamingTransformer) handleResponseCompleted(event *types.ResponsesStreamEvent) error {
	if event.Response == nil {
		return nil
	}

	// Determine stop_reason based on output items
	stopReason := "end_turn"
	for _, item := range event.Response.Output {
		if item.Type == "function_call" {
			stopReason = "tool_use"
			break
		}
	}

	// Update usage if available
	if event.Response.Usage != nil {
		t.inputTokens = event.Response.Usage.InputTokens
		t.outputTokens = event.Response.Usage.OutputTokens
	}

	// Emit message_delta
	msgDelta := map[string]interface{}{
		"type": "message_delta",
		"delta": map[string]interface{}{
			"stop_reason": stopReason,
		},
		"usage": map[string]int{
			"output_tokens": t.outputTokens,
		},
	}
	if err := t.emitEvent("message_delta", msgDelta); err != nil {
		return err
	}

	// Emit message_stop
	return t.emitEvent("message_stop", map[string]string{"type": "message_stop"})
}

// handleResponseIncomplete handles response.incomplete event → emits message_delta with max_tokens + message_stop.
func (t *ResponsesToAnthropicStreamingTransformer) handleResponseIncomplete(event *types.ResponsesStreamEvent) error {
	// Emit message_delta with max_tokens stop reason
	msgDelta := map[string]interface{}{
		"type": "message_delta",
		"delta": map[string]interface{}{
			"stop_reason": "max_tokens",
		},
		"usage": map[string]int{
			"output_tokens": t.outputTokens,
		},
	}
	if err := t.emitEvent("message_delta", msgDelta); err != nil {
		return err
	}

	// Emit message_stop
	return t.emitEvent("message_stop", map[string]string{"type": "message_stop"})
}

// handleResponseFailed handles response.failed event → emits error event.
func (t *ResponsesToAnthropicStreamingTransformer) handleResponseFailed(event *types.ResponsesStreamEvent) error {
	errEvent := map[string]interface{}{
		"type": "error",
		"error": map[string]interface{}{
			"type":    "api_error",
			"message": "upstream response failed",
		},
	}
	return t.emitEvent("error", errEvent)
}

// emitEvent writes an Anthropic SSE event with the given name and data.
func (t *ResponsesToAnthropicStreamingTransformer) emitEvent(name string, data interface{}) error {
	dataBytes, err := json.Marshal(data)
	if err != nil {
		return err
	}

	// Write event name
	if _, err := t.w.Write([]byte("event: " + name + "\n")); err != nil {
		return err
	}

	// Write data
	if _, err := t.w.Write([]byte("data: " + string(dataBytes) + "\n\n")); err != nil {
		return err
	}

	return nil
}

// Flush writes any buffered data.
func (t *ResponsesToAnthropicStreamingTransformer) Flush() error {
	return nil
}

// Close flushes and releases resources.
func (t *ResponsesToAnthropicStreamingTransformer) Close() error {
	return t.Flush()
}

// Verify interface compliance
var _ transform.SSETransformer = (*ResponsesToAnthropicStreamingTransformer)(nil)
