// Package convert provides conversion between OpenAI Chat and Anthropic Messages API formats.
package convert

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"ai-proxy/types"

	"github.com/tmaxmax/go-sse"
)

// ChatToAnthropicConverter converts OpenAI ChatCompletionRequest to Anthropic MessageRequest.
// It implements the RequestConverter interface for the OpenAI Chat to Anthropic conversion.
type ChatToAnthropicConverter struct{}

// NewChatToAnthropicConverter creates a new converter for OpenAI Chat to Anthropic format.
func NewChatToAnthropicConverter() *ChatToAnthropicConverter {
	return &ChatToAnthropicConverter{}
}

// Convert transforms an OpenAI ChatCompletionRequest body to Anthropic MessageRequest format.
// It handles message conversion, tool conversion, and parameter mapping.
func (c *ChatToAnthropicConverter) Convert(body []byte) ([]byte, error) {
	var openReq types.ChatCompletionRequest
	if err := json.Unmarshal(body, &openReq); err != nil {
		return nil, fmt.Errorf("failed to parse OpenAI request: %w", err)
	}

	// Build Anthropic request
	anthReq := types.MessageRequest{
		Model: openReq.Model,
	}

	// Extract system message and convert messages
	anthReq.System, anthReq.Messages = c.extractSystemAndMessages(openReq.Messages, openReq.System)

	// Set max_tokens - Anthropic requires this field
	anthReq.MaxTokens = openReq.MaxTokens
	if anthReq.MaxTokens == 0 {
		anthReq.MaxTokens = 4096 // Default max tokens for Anthropic API
	}

	// Copy optional parameters
	anthReq.Stream = openReq.Stream
	anthReq.Temperature = openReq.Temperature
	anthReq.TopP = openReq.TopP

	// Convert tools
	if len(openReq.Tools) > 0 {
		anthReq.Tools = ConvertOpenAIToolsToAnthropic(openReq.Tools)
	}

	return json.Marshal(anthReq)
}

// extractSystemAndMessages extracts the system message and converts remaining messages.
// System messages can come from either the system field or messages with role "system".
func (c *ChatToAnthropicConverter) extractSystemAndMessages(messages []types.Message, systemField string) (interface{}, []types.MessageInput) {
	var systemParts []string
	var nonSystemMessages []types.Message

	// Start with system field if present
	if systemField != "" {
		systemParts = append(systemParts, systemField)
	}

	// Extract system messages from messages array
	for _, msg := range messages {
		if msg.Role == "system" {
			text := ExtractTextFromContent(msg.Content)
			if text != "" {
				systemParts = append(systemParts, text)
			}
		} else {
			nonSystemMessages = append(nonSystemMessages, msg)
		}
	}

	// Convert non-system messages
	anthMessages := ConvertOpenAIMessagesToAnthropic(nonSystemMessages)

	// Return system as string if present
	var system interface{}
	if len(systemParts) > 0 {
		system = strings.Join(systemParts, "\n\n")
	}

	return system, anthMessages
}

// ChatToAnthropicTransformer converts OpenAI SSE responses to Anthropic format.
// It implements the SSETransformer interface for streaming response conversion.
type ChatToAnthropicTransformer struct {
	w io.Writer

	// State for tracking message info
	messageID  string
	model      string
	started    bool
	blockIndex int

	// Tool call tracking
	toolCalls     map[int]*chatToolCallState // index -> state
	currentToolID int
}

// chatToolCallState tracks the state of an in-progress tool call.
type chatToolCallState struct {
	id   string
	name string
	args strings.Builder
}

// NewChatToAnthropicTransformer creates a transformer for OpenAI to Anthropic SSE conversion.
func NewChatToAnthropicTransformer(w io.Writer) *ChatToAnthropicTransformer {
	return &ChatToAnthropicTransformer{
		w:         w,
		toolCalls: make(map[int]*chatToolCallState),
	}
}

// Transform processes an OpenAI SSE event and converts it to Anthropic format.
func (t *ChatToAnthropicTransformer) Transform(event *sse.Event) error {
	if event.Data == "" {
		return nil
	}

	// Handle [DONE] marker
	if event.Data == "[DONE]" {
		return t.writeEvent("message_stop", nil)
	}

	// Parse OpenAI chunk
	var chunk types.Chunk
	if err := json.Unmarshal([]byte(event.Data), &chunk); err != nil {
		// Pass through unparseable data
		return t.writeData([]byte(event.Data))
	}

	return t.handleChunk(chunk)
}

// handleChunk processes an OpenAI Chunk and emits appropriate Anthropic events.
func (t *ChatToAnthropicTransformer) handleChunk(chunk types.Chunk) error {
	// Capture message ID and model from first chunk
	if !t.started && chunk.ID != "" {
		t.messageID = chunk.ID
		t.model = chunk.Model
		t.started = true

		// Emit message_start event
		msg := map[string]interface{}{
			"id":    t.messageID,
			"type":  "message",
			"role":  "assistant",
			"model": t.model,
			"content": []interface{}{},
			"status": "in_progress",
		}
		if err := t.writeEvent("message_start", map[string]interface{}{"message": msg}); err != nil {
			return err
		}
	}

	// Handle choices
	if len(chunk.Choices) == 0 {
		// Handle usage if present (final chunk)
		if chunk.Usage != nil {
			return t.emitUsage(chunk.Usage)
		}
		return nil
	}

	choice := chunk.Choices[0]
	delta := choice.Delta

	// Handle finish reason
	if choice.FinishReason != nil && *choice.FinishReason != "" {
		return t.handleFinishReason(*choice.FinishReason, chunk.Usage)
	}

	// Handle role assignment in first delta
	if delta.Role != "" {
		// Role is already set in message_start, nothing to do
		return nil
	}

	// Handle text content
	if delta.Content != "" {
		return t.emitTextDelta(delta.Content)
	}

	// Handle tool calls
	if len(delta.ToolCalls) > 0 {
		return t.handleToolCalls(delta.ToolCalls)
	}

	// Handle reasoning content (if present)
	reasoning := delta.Reasoning
	if reasoning == "" {
		reasoning = delta.ReasoningContent
	}
	if reasoning != "" {
		// Emit as thinking content
		return t.emitThinkingDelta(reasoning)
	}

	return nil
}

// handleToolCalls processes tool call deltas from OpenAI format.
func (t *ChatToAnthropicTransformer) handleToolCalls(toolCalls []types.ToolCall) error {
	for _, tc := range toolCalls {
		// Check if this is a new tool call
		state, exists := t.toolCalls[tc.Index]
		if !exists {
			// New tool call - emit content_block_start
			state = &chatToolCallState{
				id:   tc.ID,
				name: tc.Function.Name,
			}
			t.toolCalls[tc.Index] = state
			t.currentToolID++

			// Emit content_block_start for tool_use
			if err := t.emitToolUseStart(tc.Index, tc.ID, tc.Function.Name); err != nil {
				return err
			}
		}

		// Emit arguments delta
		if tc.Function.Arguments != "" {
			state.args.WriteString(tc.Function.Arguments)
			if err := t.emitInputJSONDelta(tc.Index, tc.Function.Arguments); err != nil {
				return err
			}
		}
	}
	return nil
}

// handleFinishReason processes the finish reason and emits appropriate events.
func (t *ChatToAnthropicTransformer) handleFinishReason(reason string, usage *types.Usage) error {
	// Map OpenAI finish reason to Anthropic stop_reason
	var stopReason string
	switch reason {
	case "stop":
		stopReason = "end_turn"
	case "length":
		stopReason = "max_tokens"
	case "tool_calls":
		stopReason = "tool_use"
	case "content_filter":
		stopReason = "end_turn" // No direct equivalent
	default:
		stopReason = "end_turn"
	}

	// Close any open tool call blocks
	for index := range t.toolCalls {
		if err := t.writeEvent("content_block_stop", map[string]interface{}{
			"index": index,
		}); err != nil {
			return err
		}
	}

	// Build message_delta event
	eventData := map[string]interface{}{
		"delta": map[string]interface{}{
			"stop_reason": stopReason,
		},
		"usage": map[string]interface{}{},
	}
	if usage != nil {
		eventData["delta"].(map[string]interface{})["usage"] = map[string]interface{}{
			"output_tokens": usage.CompletionTokens,
		}
		eventData["usage"] = map[string]interface{}{
			"output_tokens": usage.CompletionTokens,
		}
	}

	if err := t.writeEvent("message_delta", eventData); err != nil {
		return err
	}

	return nil
}

// emitUsage emits a message_delta with usage information.
func (t *ChatToAnthropicTransformer) emitUsage(usage *types.Usage) error {
	if usage == nil {
		return nil
	}
	return t.writeEvent("message_delta", map[string]interface{}{
		"usage": map[string]interface{}{
			"output_tokens": usage.CompletionTokens,
		},
	})
}

// emitTextDelta emits a content_block_delta event with text_delta.
func (t *ChatToAnthropicTransformer) emitTextDelta(text string) error {
	// Start text block if not started
	if t.blockIndex == 0 || t.hasToolCallsOnly() {
		// Check if we need to start a new text block
		if !t.hasTextBlock() {
			if err := t.emitTextStart(t.blockIndex); err != nil {
				return err
			}
			t.blockIndex++
		}
	}

	return t.writeEvent("content_block_delta", map[string]interface{}{
		"index": 0, // Text is always at index 0
		"delta": map[string]interface{}{
			"type": "text_delta",
			"text": text,
		},
	})
}

// hasToolCallsOnly returns true if we've only seen tool calls so far.
func (t *ChatToAnthropicTransformer) hasToolCallsOnly() bool {
	return len(t.toolCalls) > 0 && t.blockIndex == 0
}

// hasTextBlock returns true if a text block has been started.
func (t *ChatToAnthropicTransformer) hasTextBlock() bool {
	return t.blockIndex > 0 && len(t.toolCalls) == 0
}

// emitTextStart emits a content_block_start event for text.
func (t *ChatToAnthropicTransformer) emitTextStart(index int) error {
	return t.writeEvent("content_block_start", map[string]interface{}{
		"index": index,
		"content_block": map[string]interface{}{
			"type": "text",
			"text": "",
		},
	})
}

// emitThinkingDelta emits a content_block_delta event with thinking_delta.
func (t *ChatToAnthropicTransformer) emitThinkingDelta(thinking string) error {
	// Start thinking block if not started
	if t.blockIndex == 0 {
		if err := t.emitThinkingStart(t.blockIndex); err != nil {
			return err
		}
		t.blockIndex++
	}

	return t.writeEvent("content_block_delta", map[string]interface{}{
		"index": 0,
		"delta": map[string]interface{}{
			"type":     "thinking_delta",
			"thinking": thinking,
		},
	})
}

// emitThinkingStart emits a content_block_start event for thinking.
func (t *ChatToAnthropicTransformer) emitThinkingStart(index int) error {
	return t.writeEvent("content_block_start", map[string]interface{}{
		"index": index,
		"content_block": map[string]interface{}{
			"type":     "thinking",
			"thinking": "",
		},
	})
}

// emitToolUseStart emits a content_block_start event for tool_use.
func (t *ChatToAnthropicTransformer) emitToolUseStart(index int, id, name string) error {
	// Adjust index: tool blocks come after any text/thinking blocks
	blockIdx := t.blockIndex + index

	return t.writeEvent("content_block_start", map[string]interface{}{
		"index": blockIdx,
		"content_block": map[string]interface{}{
			"type":  "tool_use",
			"id":    id,
			"name":  name,
			"input": map[string]interface{}{},
		},
	})
}

// emitInputJSONDelta emits a content_block_delta event with input_json_delta.
func (t *ChatToAnthropicTransformer) emitInputJSONDelta(index int, partialJSON string) error {
	// Adjust index: tool blocks come after any text/thinking blocks
	blockIdx := t.blockIndex + index

	return t.writeEvent("content_block_delta", map[string]interface{}{
		"index": blockIdx,
		"delta": map[string]interface{}{
			"type":        "input_json_delta",
			"partial_json": partialJSON,
		},
	})
}

// writeEvent writes an Anthropic SSE event.
func (t *ChatToAnthropicTransformer) writeEvent(eventType string, data map[string]interface{}) error {
	if data == nil {
		data = make(map[string]interface{})
	}
	data["type"] = eventType

	jsonData, err := json.Marshal(data)
	if err != nil {
		return err
	}

	return t.writeData(jsonData)
}

// writeData writes SSE data to the output writer.
func (t *ChatToAnthropicTransformer) writeData(data []byte) error {
	if len(data) == 0 {
		return nil
	}
	_, err := t.w.Write([]byte("data: " + string(data) + "\n\n"))
	return err
}

// Flush writes any buffered data.
func (t *ChatToAnthropicTransformer) Flush() error {
	return nil
}

// Close flushes and emits final events.
func (t *ChatToAnthropicTransformer) Close() error {
	// Emit message_stop if we started
	if t.started {
		return t.writeEvent("message_stop", nil)
	}
	return nil
}