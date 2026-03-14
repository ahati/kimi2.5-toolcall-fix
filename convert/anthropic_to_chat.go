// Package convert provides converters between different API formats.
// This file implements Anthropic Messages API to OpenAI Chat Completions conversion.
package convert

import (
	"encoding/json"
	"io"
	"strings"
	"time"

	"ai-proxy/types"

	"github.com/tmaxmax/go-sse"
)

// AnthropicToChatTransformer converts Anthropic SSE responses to OpenAI Chat format.
// It implements the SSETransformer interface for streaming response conversion.
//
// Key events handled:
// - message_start: emits initial chunk with ID/model
// - content_block_start: tracks block state (text/tool_use)
// - content_block_delta: emits content/tool_calls
// - content_block_stop: finalizes block
// - message_delta: maps stop_reason to finish_reason
// - message_stop: emits [DONE]
type AnthropicToChatTransformer struct {
	w io.Writer

	// Response state
	messageID string
	model     string
	created   int64

	// Block tracking
	blockIndex int
	blocks     map[int]*blockState

	// Tool call tracking
	toolCallIndex int

	// Finish reason
	finishReason string

	// Usage tracking
	usage *types.Usage
}

// blockState tracks the state of a content block.
type blockState struct {
	blockType string // "text" or "tool_use"
	id        string // tool_use ID
	name      string // tool name
	args      strings.Builder
}

// NewAnthropicToChatTransformer creates a transformer for Anthropic to OpenAI Chat SSE conversion.
func NewAnthropicToChatTransformer(w io.Writer) *AnthropicToChatTransformer {
	return &AnthropicToChatTransformer{
		w:       w,
		created: time.Now().Unix(),
		blocks:  make(map[int]*blockState),
	}
}

// Transform processes an Anthropic SSE event and converts it to OpenAI Chat format.
func (t *AnthropicToChatTransformer) Transform(event *sse.Event) error {
	if event.Data == "" {
		return nil
	}

	if event.Data == "[DONE]" {
		return t.writeDone()
	}

	var anthropicEvent types.Event
	if err := json.Unmarshal([]byte(event.Data), &anthropicEvent); err != nil {
		// Pass through unparseable events
		return t.writeData([]byte(event.Data))
	}

	return t.handleEvent(&anthropicEvent)
}

// handleEvent processes a single Anthropic event.
func (t *AnthropicToChatTransformer) handleEvent(event *types.Event) error {
	switch event.Type {
	case "message_start":
		return t.handleMessageStart(event)
	case "content_block_start":
		return t.handleContentBlockStart(event)
	case "content_block_delta":
		return t.handleContentBlockDelta(event)
	case "content_block_stop":
		return t.handleContentBlockStop(event)
	case "message_delta":
		return t.handleMessageDelta(event)
	case "message_stop":
		return t.handleMessageStop()
	case "ping":
		return nil // Ignore ping events
	default:
		// Pass through unknown events
		return nil
	}
}

// handleMessageStart extracts message metadata from message_start event.
func (t *AnthropicToChatTransformer) handleMessageStart(event *types.Event) error {
	if event.Message != nil {
		t.messageID = event.Message.ID
		t.model = event.Message.Model
	}
	return nil
}

// handleContentBlockStart handles content_block_start event.
func (t *AnthropicToChatTransformer) handleContentBlockStart(event *types.Event) error {
	if event.Index == nil || event.ContentBlock == nil {
		return nil
	}

	idx := *event.Index
	var block types.ContentBlock
	if err := json.Unmarshal(event.ContentBlock, &block); err != nil {
		return nil
	}

	t.blocks[idx] = &blockState{
		blockType: block.Type,
		id:        block.ID,
		name:      block.Name,
	}

	// For tool_use, emit the initial tool call chunk with ID and name
	if block.Type == "tool_use" {
		t.toolCallIndex++
		chunk := t.createChunk()
		chunk.Choices = []types.Choice{{
			Index: 0,
			Delta: types.Delta{
				ToolCalls: []types.ToolCall{{
					ID:    block.ID,
					Type:  "function",
					Index: t.toolCallIndex - 1,
					Function: types.Function{
						Name: block.Name,
					},
				}},
			},
		}}
		return t.writeChunk(chunk)
	}

	return nil
}

// handleContentBlockDelta handles content_block_delta event.
func (t *AnthropicToChatTransformer) handleContentBlockDelta(event *types.Event) error {
	if event.Index == nil || event.Delta == nil {
		return nil
	}

	idx := *event.Index
	block, exists := t.blocks[idx]
	if !exists {
		return nil
	}

	// Try to parse as text_delta
	var textDelta types.TextDelta
	if err := json.Unmarshal(event.Delta, &textDelta); err == nil && textDelta.Type == "text_delta" {
		return t.emitTextDelta(textDelta.Text)
	}

	// Try to parse as thinking_delta (treat as text for OpenAI format)
	var thinkingDelta types.ThinkingDelta
	if err := json.Unmarshal(event.Delta, &thinkingDelta); err == nil && thinkingDelta.Type == "thinking_delta" {
		// OpenAI Chat format doesn't have a native thinking type,
		// so we emit it as regular content
		return t.emitTextDelta(thinkingDelta.Thinking)
	}

	// Try to parse as input_json_delta (tool call arguments)
	var inputDelta types.InputJSONDelta
	if err := json.Unmarshal(event.Delta, &inputDelta); err == nil && inputDelta.Type == "input_json_delta" {
		if block.blockType == "tool_use" {
			block.args.WriteString(inputDelta.PartialJSON)
			return t.emitToolCallArgsDelta(block.id, inputDelta.PartialJSON)
		}
	}

	return nil
}

// handleContentBlockStop handles content_block_stop event.
func (t *AnthropicToChatTransformer) handleContentBlockStop(_ *types.Event) error {
	// Block finalization is handled by message_delta for finish_reason
	return nil
}

// handleMessageDelta handles message_delta event with stop_reason.
func (t *AnthropicToChatTransformer) handleMessageDelta(event *types.Event) error {
	// Capture usage
	if event.Usage != nil {
		t.usage = &types.Usage{
			PromptTokens:     event.Usage.InputTokens,
			CompletionTokens: event.Usage.OutputTokens,
			TotalTokens:      event.Usage.InputTokens + event.Usage.OutputTokens,
		}
	}

	// Map stop_reason to finish_reason
	if event.StopReason != "" {
		t.finishReason = t.mapStopReason(event.StopReason)
	}

	return nil
}

// handleMessageStop emits the final chunk with finish_reason and [DONE].
func (t *AnthropicToChatTransformer) handleMessageStop() error {
	// Emit final chunk with finish_reason
	chunk := t.createChunk()
	chunk.Choices = []types.Choice{{
		Index:        0,
		Delta:        types.Delta{},
		FinishReason: &t.finishReason,
	}}
	if t.usage != nil {
		chunk.Usage = t.usage
	}

	if err := t.writeChunk(chunk); err != nil {
		return err
	}

	return t.writeDone()
}

// mapStopReason maps Anthropic stop_reason to OpenAI finish_reason.
func (t *AnthropicToChatTransformer) mapStopReason(stopReason string) string {
	switch stopReason {
	case "end_turn":
		return "stop"
	case "max_tokens":
		return "length"
	case "tool_use":
		return "tool_calls"
	case "stop_sequence":
		return "stop"
	default:
		return "stop"
	}
}

// emitTextDelta emits a text content chunk.
func (t *AnthropicToChatTransformer) emitTextDelta(text string) error {
	if text == "" {
		return nil
	}

	chunk := t.createChunk()
	chunk.Choices = []types.Choice{{
		Index: 0,
		Delta: types.Delta{
			Content: text,
		},
	}}

	return t.writeChunk(chunk)
}

// emitToolCallArgsDelta emits a tool call arguments delta chunk.
func (t *AnthropicToChatTransformer) emitToolCallArgsDelta(id, args string) error {
	chunk := t.createChunk()
	chunk.Choices = []types.Choice{{
		Index: 0,
		Delta: types.Delta{
			ToolCalls: []types.ToolCall{{
				ID:    id,
				Index: t.toolCallIndex - 1,
				Function: types.Function{
					Arguments: args,
				},
			}},
		},
	}}

	return t.writeChunk(chunk)
}

// createChunk creates a base chunk with common fields.
func (t *AnthropicToChatTransformer) createChunk() *types.Chunk {
	return &types.Chunk{
		ID:      t.messageID,
		Object:  "chat.completion.chunk",
		Created: t.created,
		Model:   t.model,
	}
}

// writeChunk writes a chunk as SSE data.
func (t *AnthropicToChatTransformer) writeChunk(chunk *types.Chunk) error {
	data, err := json.Marshal(chunk)
	if err != nil {
		return err
	}
	return t.writeData(data)
}

// writeData writes raw data as SSE event.
func (t *AnthropicToChatTransformer) writeData(data []byte) error {
	if len(data) == 0 {
		return nil
	}
	_, err := t.w.Write([]byte("data: "))
	if err != nil {
		return err
	}
	_, err = t.w.Write(data)
	if err != nil {
		return err
	}
	_, err = t.w.Write([]byte("\n\n"))
	return err
}

// writeDone writes the [DONE] marker.
func (t *AnthropicToChatTransformer) writeDone() error {
	_, err := t.w.Write([]byte("data: [DONE]\n\n"))
	return err
}

// Flush writes any buffered data.
func (t *AnthropicToChatTransformer) Flush() error {
	return nil
}

// Close flushes and releases resources.
func (t *AnthropicToChatTransformer) Close() error {
	return t.Flush()
}