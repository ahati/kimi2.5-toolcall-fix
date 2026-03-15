// Package convert provides conversion between OpenAI Chat and Anthropic Messages API formats.
package convert

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"ai-proxy/logging"
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
	// Force streaming mode - this proxy only supports SSE streaming
	anthReq.Stream = openReq.Stream || true
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
	messageID       string
	model           string
	started         bool
	blockIndex      int
	contentOpen     bool  // Track if a content block (thinking/text) is open
	contentType     string // Track the type of current content block: "thinking" or "text"
	deltaSent       bool // Track if message_delta was already sent
	messageStopSent bool // Track if message_stop was already sent

	// Tool call tracking
	toolCalls     map[int]*chatToolCallState // index -> state
	currentToolID int

	// Usage tracking - captured from final upstream chunk
	promptTokens     int
	completionTokens int

	// Finish reason tracking - delay message_delta until we have usage
	finishReason string
}

// chatToolCallState tracks the state of an in-progress tool call.
type chatToolCallState struct {
	id       string
	name     string
	args     strings.Builder
	blockIdx int // The actual block index for this tool call
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

	// Handle [DONE] marker - trigger Close() which handles all cleanup
	if event.Data == "[DONE]" {
		logging.DebugMsg("Transform: received [DONE], calling Close()")
		return t.Close()
	}

	// Parse OpenAI chunk
	var chunk types.Chunk
	if err := json.Unmarshal([]byte(event.Data), &chunk); err != nil {
		// Pass through unparseable data as raw SSE
		_, err := fmt.Fprintf(t.w, "data: %s\n\n", event.Data)
		return err
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
		logging.DebugMsg("handleChunk: first chunk, ID=%s, model=%s", t.messageID, t.model)

		// Extract usage if available in first chunk (some providers include it)
		inputTokens := 0
		if chunk.Usage != nil && chunk.Usage.PromptTokens > 0 {
			inputTokens = chunk.Usage.PromptTokens
			t.promptTokens = chunk.Usage.PromptTokens
		}

		// Emit message_start event per Anthropic spec
		// Must include usage field - SDKs expect this to exist
		msg := map[string]interface{}{
			"id":            t.messageID,
			"type":          "message",
			"role":          "assistant",
			"model":         t.model,
			"content":       []interface{}{},
			"stop_reason":   nil,
			"stop_sequence": nil,
			"usage": map[string]interface{}{
				"input_tokens":  inputTokens,
				"output_tokens": 1,
			},
		}
		if err := t.writeEvent("message_start", map[string]interface{}{"message": msg}); err != nil {
			return err
		}
	}

	// Handle choices
	if len(chunk.Choices) == 0 {
		// Handle usage if present (final chunk from upstream)
		// Upstream sends usage in the last chunk with empty choices array
		if chunk.Usage != nil {
			t.promptTokens = chunk.Usage.PromptTokens
			t.completionTokens = chunk.Usage.CompletionTokens
			logging.DebugMsg("handleChunk: usage chunk, prompt=%d, completion=%d, finishReason=%s", t.promptTokens, t.completionTokens, t.finishReason)
			// Now emit message_delta with the stored finish_reason and usage
			if t.finishReason != "" {
				t.handleFinishReason(t.finishReason, chunk.Usage)
				t.finishReason = ""
			}
		}
		return nil
	}

	choice := chunk.Choices[0]
	delta := choice.Delta

	// Handle finish reason - store it for later emission with usage
	if choice.FinishReason != nil && *choice.FinishReason != "" {
		t.finishReason = *choice.FinishReason
		logging.DebugMsg("handleChunk: finish reason=%s, waiting for usage", t.finishReason)
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
			logging.DebugMsg("handleToolCalls: new tool idx=%d, id=%s, name=%s", tc.Index, tc.ID, tc.Function.Name)
			// Close any open content block (thinking/text) before starting tool_use
			if t.contentOpen {
				logging.DebugMsg("handleToolCalls: closing content block idx=%d", t.blockIndex-1)
				if err := t.writeEvent("content_block_stop", map[string]interface{}{
					"index": t.blockIndex - 1,
				}); err != nil {
					return err
				}
				t.contentOpen = false
			}

			// New tool call - calculate block index and emit content_block_start
			blockIdx := t.blockIndex + len(t.toolCalls)
			state = &chatToolCallState{
				id:       tc.ID,
				name:     tc.Function.Name,
				blockIdx: blockIdx,
			}
			t.toolCalls[tc.Index] = state

			// Emit content_block_start for tool_use
			if err := t.emitToolUseStart(blockIdx, tc.ID, tc.Function.Name); err != nil {
				return err
			}
		}

		// Emit arguments delta
		if tc.Function.Arguments != "" {
			state.args.WriteString(tc.Function.Arguments)
			if err := t.emitInputJSONDelta(state.blockIdx, tc.Function.Arguments); err != nil {
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

	logging.DebugMsg("handleFinishReason: reason=%s -> stopReason=%s, contentOpen=%v, toolCalls=%d, usage=%d/%d",
		reason, stopReason, t.contentOpen, len(t.toolCalls), t.promptTokens, t.completionTokens)

	// Close open content block (thinking/text) first
	if t.contentOpen {
		contentBlockIndex := t.blockIndex - 1
		logging.DebugMsg("handleFinishReason: closing content block idx=%d", contentBlockIndex)
		if err := t.writeEvent("content_block_stop", map[string]interface{}{
			"index": contentBlockIndex,
		}); err != nil {
			return err
		}
		t.contentOpen = false
	}

	// Close tool call blocks with correct block indices
	for _, state := range t.toolCalls {
		logging.DebugMsg("handleFinishReason: closing tool_use block idx=%d", state.blockIdx)
		if err := t.writeEvent("content_block_stop", map[string]interface{}{
			"index": state.blockIdx,
		}); err != nil {
			return err
		}
	}

	// Build message_delta event
	eventData := map[string]interface{}{
		"delta": map[string]interface{}{
			"stop_reason":   stopReason,
			"stop_sequence": nil,
		},
	}
	// Include usage with both tokens for compatibility
	// Some SDKs expect input_tokens in message_delta even though spec says output_tokens only
	eventData["usage"] = map[string]interface{}{
		"input_tokens":  t.promptTokens,
		"output_tokens": t.completionTokens,
	}

	logging.DebugMsg("handleFinishReason: emitting message_delta")
	if err := t.writeEvent("message_delta", eventData); err != nil {
		return err
	}

	// Mark delta as sent and clear tool calls to prevent duplicate emission
	t.deltaSent = true
	t.toolCalls = make(map[int]*chatToolCallState)

	return nil
}

// emitUsage is deprecated - usage is now captured in handleChunk and included in message_delta
func (t *ChatToAnthropicTransformer) emitUsage(usage *types.Usage) error {
	return nil
}

// emitTextDelta emits a content_block_delta event with text_delta.
func (t *ChatToAnthropicTransformer) emitTextDelta(text string) error {
	// If we have a thinking block open, close it and start a text block
	if t.contentOpen && t.contentType == "thinking" {
		logging.DebugMsg("emitTextDelta: closing thinking block at idx=%d, starting text block", t.blockIndex-1)
		if err := t.writeEvent("content_block_stop", map[string]interface{}{
			"index": t.blockIndex - 1,
		}); err != nil {
			return err
		}
		t.contentOpen = false
		// Start a new text block at the current blockIndex
		if err := t.emitTextStart(t.blockIndex); err != nil {
			return err
		}
		t.blockIndex++
	} else if !t.contentOpen {
		// No block open, start a text block
		if err := t.emitTextStart(t.blockIndex); err != nil {
			return err
		}
		t.blockIndex++
	}
	// else: text block already open, just emit the delta

	// Text block is always at the index where it was started (blockIndex - 1)
	textBlockIndex := t.blockIndex - 1
	return t.writeEvent("content_block_delta", map[string]interface{}{
		"index": textBlockIndex,
		"delta": map[string]interface{}{
			"type": "text_delta",
			"text": text,
		},
	})
}

// emitTextStart emits a content_block_start event for text.
func (t *ChatToAnthropicTransformer) emitTextStart(index int) error {
	t.contentOpen = true
	t.contentType = "text"
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
	// If we have a text block open, close it and start a thinking block
	if t.contentOpen && t.contentType == "text" {
		logging.DebugMsg("emitThinkingDelta: closing text block at idx=%d, starting thinking block", t.blockIndex-1)
		if err := t.writeEvent("content_block_stop", map[string]interface{}{
			"index": t.blockIndex - 1,
		}); err != nil {
			return err
		}
		t.contentOpen = false
		// Start a new thinking block at the current blockIndex
		if err := t.emitThinkingStart(t.blockIndex); err != nil {
			return err
		}
		t.blockIndex++
	} else if !t.contentOpen {
		// No block open, start a thinking block
		if err := t.emitThinkingStart(t.blockIndex); err != nil {
			return err
		}
		t.blockIndex++
	}
	// else: thinking block already open, just emit the delta

	// Thinking block is always at the index where it was started (blockIndex - 1)
	thinkingBlockIndex := t.blockIndex - 1
	return t.writeEvent("content_block_delta", map[string]interface{}{
		"index": thinkingBlockIndex,
		"delta": map[string]interface{}{
			"type":     "thinking_delta",
			"thinking": thinking,
		},
	})
}

// emitThinkingStart emits a content_block_start event for thinking.
func (t *ChatToAnthropicTransformer) emitThinkingStart(index int) error {
	t.contentOpen = true
	t.contentType = "thinking"
	return t.writeEvent("content_block_start", map[string]interface{}{
		"index": index,
		"content_block": map[string]interface{}{
			"type":     "thinking",
			"thinking": "",
		},
	})
}

// emitToolUseStart emits a content_block_start event for tool_use.
func (t *ChatToAnthropicTransformer) emitToolUseStart(blockIdx int, id, name string) error {
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
func (t *ChatToAnthropicTransformer) emitInputJSONDelta(blockIdx int, partialJSON string) error {
	return t.writeEvent("content_block_delta", map[string]interface{}{
		"index": blockIdx,
		"delta": map[string]interface{}{
			"type":         "input_json_delta",
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

	return t.writeSSE(eventType, jsonData)
}

// writeSSE writes a complete SSE event with event type and data.
// Format: event: <type>\ndata: <json>\n\n
func (t *ChatToAnthropicTransformer) writeSSE(eventType string, data []byte) error {
	if len(data) == 0 {
		return nil
	}
	_, err := fmt.Fprintf(t.w, "event: %s\ndata: %s\n\n", eventType, string(data))
	return err
}

// Flush writes any buffered data.
func (t *ChatToAnthropicTransformer) Flush() error {
	return nil
}

// Close flushes and emits final events.
// This handles graceful shutdown when stream is cut off mid-stream.
func (t *ChatToAnthropicTransformer) Close() error {
	logging.DebugMsg("Close: started=%v, messageStopSent=%v, deltaSent=%v, contentOpen=%v, blockIndex=%d, toolCalls=%d, finishReason=%s",
		t.started, t.messageStopSent, t.deltaSent, t.contentOpen, t.blockIndex, len(t.toolCalls), t.finishReason)

	if !t.started || t.messageStopSent {
		logging.DebugMsg("Close: early return - started=%v, messageStopSent=%v", t.started, t.messageStopSent)
		return nil
	}

	// Only close blocks and emit message_delta if not already done by handleFinishReason
	if !t.deltaSent {
		logging.DebugMsg("Close: delta not sent yet, closing blocks and emitting message_delta")
		// Close any open content block (thinking/text)
		if t.contentOpen {
			logging.DebugMsg("Close: closing content block idx=%d", t.blockIndex-1)
			if err := t.writeEvent("content_block_stop", map[string]interface{}{
				"index": t.blockIndex - 1,
			}); err != nil {
				return err
			}
			t.contentOpen = false
		}

		// Close all tool call blocks
		for _, state := range t.toolCalls {
			logging.DebugMsg("Close: closing tool_use block idx=%d", state.blockIdx)
			if err := t.writeEvent("content_block_stop", map[string]interface{}{
				"index": state.blockIdx,
			}); err != nil {
				return err
			}
		}

		// Emit message_delta if we have any content
		if t.blockIndex > 0 || len(t.toolCalls) > 0 {
			stopReason := "end_turn"
			if len(t.toolCalls) > 0 {
				stopReason = "tool_use"
			}
			logging.DebugMsg("Close: emitting message_delta with stop_reason=%s, input=%d, output=%d", stopReason, t.promptTokens, t.completionTokens)
			eventData := map[string]interface{}{
				"delta": map[string]interface{}{
					"stop_reason":   stopReason,
					"stop_sequence": nil,
				},
				"usage": map[string]interface{}{
					"input_tokens":  t.promptTokens,
					"output_tokens": t.completionTokens,
				},
			}
			if err := t.writeEvent("message_delta", eventData); err != nil {
				return err
			}
		}
	}

	logging.DebugMsg("Close: emitting message_stop")
	t.messageStopSent = true
	return t.writeEvent("message_stop", nil)
}
