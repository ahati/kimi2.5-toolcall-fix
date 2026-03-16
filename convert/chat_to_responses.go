package convert

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"

	"ai-proxy/conversation"
	"ai-proxy/logging"
	"ai-proxy/types"

	"github.com/tmaxmax/go-sse"
)

type ChatToResponsesTransformer struct {
	w io.Writer

	responseID string
	model      string
	created    int64

	contentIndex    int
	toolCallIndex   int
	currentToolCall *chatToRespToolCallState

	contentBuilder   strings.Builder
	reasoningBuilder strings.Builder

	finishReason string
	usage        *types.Usage

	inReasoning          bool
	sequenceNumber       int
	reasoningID          string
	reasoningOutputIndex int
	summaryIndex         int

	messageStarted bool // track if message item has been emitted
	completed      bool // track if response.completed has been emitted
	toolCalls      []map[string]interface{}

	// Input items for conversation storage
	inputItems []types.InputItem
}

type chatToRespToolCallState struct {
	id        string
	name      string
	arguments strings.Builder
}

func NewChatToResponsesTransformer(w io.Writer) *ChatToResponsesTransformer {
	return &ChatToResponsesTransformer{
		w:              w,
		created:        time.Now().Unix(),
		sequenceNumber: 0,
	}
}

// SetInputItems sets the input items for conversation storage.
// This should be called before streaming starts to capture the original request input.
func (t *ChatToResponsesTransformer) SetInputItems(items []types.InputItem) {
	t.inputItems = items
}

func (t *ChatToResponsesTransformer) nextSeq() int {
	t.sequenceNumber++
	return t.sequenceNumber
}

func (t *ChatToResponsesTransformer) Transform(event *sse.Event) error {
	if event.Data == "" {
		return nil
	}

	if event.Data == "[DONE]" {
		return t.writeDone()
	}

	var chunk types.Chunk
	if err := json.Unmarshal([]byte(event.Data), &chunk); err != nil {
		return t.writeData([]byte(event.Data))
	}

	return t.handleChunk(&chunk)
}

func (t *ChatToResponsesTransformer) handleChunk(chunk *types.Chunk) error {
	if t.responseID == "" && chunk.ID != "" {
		t.responseID = "resp_" + chunk.ID
		t.model = chunk.Model
		if err := t.emitResponseCreated(); err != nil {
			return err
		}
	}

	// Ensure response.created is emitted before any content
	if t.responseID == "" {
		t.responseID = fmt.Sprintf("resp_%d", t.created)
		t.model = chunk.Model
		if err := t.emitResponseCreated(); err != nil {
			return err
		}
	}

	// Capture usage when available (may come after finish_reason in separate chunk)
	if chunk.Usage != nil {
		t.usage = chunk.Usage
	}

	// Handle finish reason - store it, emit response.completed after usage arrives or at [DONE]
	if len(chunk.Choices) > 0 {
		choice := chunk.Choices[0]
		if choice.FinishReason != nil && *choice.FinishReason != "" {
			t.finishReason = *choice.FinishReason
			return nil // Don't emit response.completed yet, wait for usage or [DONE]
		}
	}

	// If no choices, check if we have finishReason waiting and usage just arrived
	if len(chunk.Choices) == 0 {
		// If we have a pending finish reason and just got usage, emit response.completed
		if t.finishReason != "" && chunk.Usage != nil {
			return t.handleFinish()
		}
		return nil
	}

	choice := chunk.Choices[0]
	delta := choice.Delta

	// Process content BEFORE role check - chunks may have both role AND content
	if delta.Content != "" {
		return t.emitTextDelta(delta.Content)
	}

	if delta.Reasoning != "" || delta.ReasoningContent != "" {
		text := delta.Reasoning
		if text == "" {
			text = delta.ReasoningContent
		}
		return t.emitReasoningDelta(text)
	}

	// Skip role-only deltas (only if no content, no reasoning, and no tool calls)
	if delta.Role != "" && len(delta.ToolCalls) == 0 {
		return nil
	}

	if len(delta.ToolCalls) > 0 {
		return t.handleToolCalls(delta.ToolCalls)
	}

	return nil
}

func (t *ChatToResponsesTransformer) emitResponseCreated() error {
	event := map[string]interface{}{
		"type":            "response.created",
		"sequence_number": t.nextSeq(),
		"response": map[string]interface{}{
			"id":         t.responseID,
			"object":     "response",
			"created_at": t.created,
			"model":      t.model,
			"status":     "in_progress",
			"output":     []interface{}{},
		},
	}
	return t.writeEvent(event)
}

func (t *ChatToResponsesTransformer) emitTextDelta(text string) error {
	// Finalize reasoning before emitting text (transition from reasoning to text)
	if t.inReasoning {
		if err := t.finalizeReasoning(); err != nil {
			return err
		}
	}

	if !t.messageStarted {
		if err := t.emitMessageItemAdded(); err != nil {
			return err
		}
		if err := t.emitContentPartAdded(); err != nil {
			return err
		}
		t.messageStarted = true
	}

	t.contentBuilder.WriteString(text)

	event := map[string]interface{}{
		"type":            "response.output_text.delta",
		"sequence_number": t.nextSeq(),
		"delta":           text,
	}
	return t.writeEvent(event)
}

// finalizeReasoning emits the done events for reasoning when transitioning to text or finishing.
func (t *ChatToResponsesTransformer) finalizeReasoning() error {
	if !t.inReasoning {
		return nil
	}

	reasoningText := t.reasoningBuilder.String()

	// Emit response.reasoning_summary_text.done
	textDoneEvent := map[string]interface{}{
		"type":            "response.reasoning_summary_text.done",
		"sequence_number": t.nextSeq(),
		"item_id":         t.reasoningID,
		"text":            reasoningText,
		"output_index":    t.reasoningOutputIndex,
		"summary_index":   t.summaryIndex,
	}
	if err := t.writeEvent(textDoneEvent); err != nil {
		return err
	}

	// Emit response.reasoning_summary_part.done
	partDoneEvent := map[string]interface{}{
		"type":            "response.reasoning_summary_part.done",
		"sequence_number": t.nextSeq(),
		"item_id":         t.reasoningID,
		"output_index":    t.reasoningOutputIndex,
		"summary_index":   t.summaryIndex,
		"part": map[string]interface{}{
			"type": "summary_text",
			"text": reasoningText,
		},
	}
	if err := t.writeEvent(partDoneEvent); err != nil {
		return err
	}

	// Emit response.output_item.done with full summary
	itemDoneEvent := map[string]interface{}{
		"type":            "response.output_item.done",
		"sequence_number": t.nextSeq(),
		"item_id":         t.reasoningID,
		"output_index":    t.reasoningOutputIndex,
		"item": map[string]interface{}{
			"type": "reasoning",
			"id":   t.reasoningID,
			"summary": []map[string]interface{}{
				{"type": "summary_text", "text": reasoningText},
			},
		},
	}
	if err := t.writeEvent(itemDoneEvent); err != nil {
		return err
	}

	// Mark reasoning as finalized
	t.inReasoning = false

	return nil
}

func (t *ChatToResponsesTransformer) emitReasoningDelta(text string) error {
	if !t.inReasoning {
		t.inReasoning = true
		t.reasoningOutputIndex = t.contentIndex
		t.contentIndex++
		t.summaryIndex = 0
		t.reasoningID = fmt.Sprintf("rs_%s", t.responseID[5:])

		// Emit response.output_item.added for reasoning
		event := map[string]interface{}{
			"type":            "response.output_item.added",
			"sequence_number": t.nextSeq(),
			"output_index":    t.reasoningOutputIndex,
			"item": map[string]interface{}{
				"type":    "reasoning",
				"id":      t.reasoningID,
				"summary": []interface{}{},
			},
		}
		if err := t.writeEvent(event); err != nil {
			return err
		}

		// Emit response.reasoning_summary_part.added
		partAddedEvent := map[string]interface{}{
			"type":            "response.reasoning_summary_part.added",
			"sequence_number": t.nextSeq(),
			"item_id":         t.reasoningID,
			"output_index":    t.reasoningOutputIndex,
			"summary_index":   t.summaryIndex,
			"part": map[string]interface{}{
				"type": "summary_text",
				"text": "",
			},
		}
		if err := t.writeEvent(partAddedEvent); err != nil {
			return err
		}
	}

	t.reasoningBuilder.WriteString(text)

	event := map[string]interface{}{
		"type":            "response.reasoning_summary_text.delta",
		"sequence_number": t.nextSeq(),
		"item_id":         t.reasoningID,
		"delta":           text,
		"output_index":    t.reasoningOutputIndex,
		"summary_index":   t.summaryIndex,
	}
	return t.writeEvent(event)
}

func (t *ChatToResponsesTransformer) emitMessageItemAdded() error {
	messageID := fmt.Sprintf("msg_%s", t.responseID[5:])
	outputIndex := 0
	// If reasoning was emitted, message comes after it
	if t.reasoningID != "" {
		outputIndex = 1
	}
	t.contentIndex = outputIndex

	event := map[string]interface{}{
		"type":            "response.output_item.added",
		"sequence_number": t.nextSeq(),
		"output_index":    outputIndex,
		"item": map[string]interface{}{
			"type":    "message",
			"id":      messageID,
			"status":  "in_progress",
			"role":    "assistant",
			"content": []interface{}{},
		},
	}
	return t.writeEvent(event)
}

func (t *ChatToResponsesTransformer) emitContentPartAdded() error {
	event := map[string]interface{}{
		"type":            "response.content_part.added",
		"sequence_number": t.nextSeq(),
		"output_index":    t.contentIndex,
		"part": map[string]interface{}{
			"type": "output_text",
			"text": "",
		},
	}
	return t.writeEvent(event)
}

func (t *ChatToResponsesTransformer) handleToolCalls(toolCalls []types.ToolCall) error {
	// Finalize reasoning before handling tool calls (transition from reasoning to tools)
	if t.inReasoning {
		if err := t.finalizeReasoning(); err != nil {
			return err
		}
	}

	for _, tc := range toolCalls {
		// In OpenAI streaming format, subsequent tool call chunks have empty ID.
		// Only create a new tool call when we have a non-empty ID.
		if tc.ID != "" && (t.currentToolCall == nil || t.currentToolCall.id != tc.ID) {
			// Close previous tool call if exists
			if t.currentToolCall != nil {
				if err := t.emitToolCallDone(); err != nil {
					return err
				}
			}

			outputIndex := t.contentIndex
			t.toolCallIndex = outputIndex
			t.contentIndex = outputIndex + 1

			t.currentToolCall = &chatToRespToolCallState{
				id:   tc.ID,
				name: tc.Function.Name,
			}

			event := map[string]interface{}{
				"type":            "response.output_item.added",
				"sequence_number": t.nextSeq(),
				"output_index":    outputIndex,
				"item": map[string]interface{}{
					"type":      "function_call",
					"id":        tc.ID,
					"call_id":   tc.ID,
					"name":      tc.Function.Name,
					"arguments": "",
				},
			}
			if err := t.writeEvent(event); err != nil {
				return err
			}
		}

		// Append arguments to current tool call (works for both initial and subsequent chunks)
		if tc.Function.Arguments != "" && t.currentToolCall != nil {
			t.currentToolCall.arguments.WriteString(tc.Function.Arguments)

			event := map[string]interface{}{
				"type":            "response.function_call_arguments.delta",
				"sequence_number": t.nextSeq(),
				"item_id":         t.currentToolCall.id,
				"call_id":         t.currentToolCall.id,
				"output_index":    t.toolCallIndex,
				"delta":           tc.Function.Arguments,
			}
			if err := t.writeEvent(event); err != nil {
				return err
			}
		}
	}
	return nil
}

func (t *ChatToResponsesTransformer) emitToolCallDone() error {
	if t.currentToolCall == nil {
		return nil
	}

	toolCallItem := map[string]interface{}{
		"type":      "function_call",
		"id":        t.currentToolCall.id,
		"call_id":   t.currentToolCall.id,
		"name":      t.currentToolCall.name,
		"arguments": t.currentToolCall.arguments.String(),
	}

	event := map[string]interface{}{
		"type":            "response.output_item.done",
		"sequence_number": t.nextSeq(),
		"output_index":    t.toolCallIndex,
		"item":            toolCallItem,
	}

	t.toolCalls = append(t.toolCalls, toolCallItem)
	t.currentToolCall = nil
	return t.writeEvent(event)
}

func (t *ChatToResponsesTransformer) handleFinish() error {
	if t.completed {
		return nil // Already emitted response.completed
	}

	// Finalize reasoning if still in progress
	if t.inReasoning {
		if err := t.finalizeReasoning(); err != nil {
			return err
		}
	}

	if t.currentToolCall != nil {
		if err := t.emitToolCallDone(); err != nil {
			return err
		}
	}

	var outputItems []map[string]interface{}

	if t.reasoningID != "" && t.reasoningBuilder.Len() > 0 {
		outputItems = append(outputItems, map[string]interface{}{
			"type": "reasoning",
			"id":   t.reasoningID,
			"summary": []map[string]interface{}{
				{"type": "summary_text", "text": t.reasoningBuilder.String()},
			},
		})
	}

	// Message comes before tool calls (matches streaming output_index)
	if t.contentBuilder.Len() > 0 || t.messageStarted {
		messageID := fmt.Sprintf("msg_%s", t.responseID[5:])
		finalText := t.contentBuilder.String()

		// Emit done events for message
		if t.messageStarted {
			if err := t.writeEvent(map[string]interface{}{
				"type":            "response.output_text.done",
				"sequence_number": t.nextSeq(),
				"text":            finalText,
			}); err != nil {
				return err
			}

			if err := t.writeEvent(map[string]interface{}{
				"type":            "response.content_part.done",
				"sequence_number": t.nextSeq(),
				"output_index":    t.contentIndex,
				"part":            map[string]interface{}{"type": "output_text", "text": finalText},
			}); err != nil {
				return err
			}

			if err := t.writeEvent(map[string]interface{}{
				"type":            "response.output_item.done",
				"sequence_number": t.nextSeq(),
				"output_index":    t.contentIndex,
				"item": map[string]interface{}{
					"type":    "message",
					"id":      messageID,
					"status":  "completed",
					"role":    "assistant",
					"content": []map[string]interface{}{{"type": "output_text", "text": finalText}},
				},
			}); err != nil {
				return err
			}
		}

		outputItems = append(outputItems, map[string]interface{}{
			"type":    "message",
			"id":      messageID,
			"status":  "completed",
			"role":    "assistant",
			"content": []map[string]interface{}{{"type": "output_text", "text": finalText}},
		})
	}

	// Tool calls come after message (matches streaming output_index)
	for _, tc := range t.toolCalls {
		outputItems = append(outputItems, tc)
	}

	response := map[string]interface{}{
		"id":     t.responseID,
		"object": "response",
		"model":  t.model,
		"status": "completed",
		"output": outputItems,
	}

	if t.usage != nil {
		usageData := map[string]interface{}{
			"input_tokens":  t.usage.PromptTokens,
			"output_tokens": t.usage.CompletionTokens,
			"total_tokens":  t.usage.TotalTokens,
		}
		// Include cache tokens if available
		if t.usage.PromptTokensDetails != nil && t.usage.PromptTokensDetails.CachedTokens > 0 {
			usageData["input_tokens_details"] = map[string]interface{}{
				"cached_tokens": t.usage.PromptTokensDetails.CachedTokens,
			}
		}
		if t.usage.CompletionTokensDetails != nil && t.usage.CompletionTokensDetails.ReasoningTokens > 0 {
			usageData["output_tokens_details"] = map[string]interface{}{
				"reasoning_tokens": t.usage.CompletionTokensDetails.ReasoningTokens,
			}
		}
		response["usage"] = usageData
	}

	event := map[string]interface{}{
		"type":            "response.completed",
		"sequence_number": t.nextSeq(),
		"response":        response,
	}
	if err := t.writeEvent(event); err != nil {
		return err
	}
	t.completed = true

	// Store conversation for previous_response_id support
	t.storeConversation(outputItems)

	return nil
}

// storeConversation saves the conversation to the default store for previous_response_id support.
// This enables multi-turn conversations without re-sending the entire history.
func (t *ChatToResponsesTransformer) storeConversation(outputItems []map[string]interface{}) {
	// Only store if we have a response ID and the store is initialized
	if t.responseID == "" {
		return
	}

	// Convert outputItems to types.OutputItem slice
	outputs := make([]types.OutputItem, 0, len(outputItems))
	for _, item := range outputItems {
		outputItem := convertChatToOutputItem(item)
		if outputItem != nil {
			outputs = append(outputs, *outputItem)
		}
	}

	// Store the conversation
	conv := &conversation.Conversation{
		ID:     t.responseID,
		Input:  t.inputItems,
		Output: outputs,
	}
	conversation.StoreInDefault(conv)
	logging.DebugMsg("[%s] Stored conversation with %d input items and %d output items",
		t.responseID, len(t.inputItems), len(outputs))
}

// convertChatToOutputItem converts a map[string]interface{} to types.OutputItem.
func convertChatToOutputItem(item map[string]interface{}) *types.OutputItem {
	if item == nil {
		return nil
	}

	itemType, _ := item["type"].(string)
	if itemType == "" {
		return nil
	}

	output := &types.OutputItem{
		Type: itemType,
	}

	if id, ok := item["id"].(string); ok {
		output.ID = id
	}
	if status, ok := item["status"].(string); ok {
		output.Status = status
	}
	if role, ok := item["role"].(string); ok {
		output.Role = role
	}
	if callID, ok := item["call_id"].(string); ok {
		output.CallID = callID
	}
	if name, ok := item["name"].(string); ok {
		output.Name = name
	}
	if args, ok := item["arguments"].(string); ok {
		output.Arguments = args
	}

	// Handle content array for message type
	if itemType == "message" {
		if content, ok := item["content"].([]map[string]interface{}); ok {
			output.Content = make([]types.OutputContent, 0, len(content))
			for _, part := range content {
				contentItem := types.OutputContent{}
				if partType, ok := part["type"].(string); ok {
					contentItem.Type = partType
				}
				if text, ok := part["text"].(string); ok {
					contentItem.Text = text
				}
				output.Content = append(output.Content, contentItem)
			}
		}
	}

	return output
}

func (t *ChatToResponsesTransformer) writeEvent(event map[string]interface{}) error {
	data, err := json.Marshal(event)
	if err != nil {
		return err
	}
	return t.writeData(data)
}

func (t *ChatToResponsesTransformer) writeData(data []byte) error {
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

func (t *ChatToResponsesTransformer) writeDone() error {
	// Finalize any pending tool call before writing [DONE]
	if t.currentToolCall != nil {
		if err := t.emitToolCallDone(); err != nil {
			return err
		}
	}

	// If we have a pending finish reason, emit response.completed now
	// (this happens when usage chunk never arrived or came with empty choices)
	if t.finishReason != "" {
		if err := t.handleFinish(); err != nil {
			return err
		}
	}

	_, err := t.w.Write([]byte("data: [DONE]\n\n"))
	return err
}

func (t *ChatToResponsesTransformer) Flush() error {
	return nil
}

func (t *ChatToResponsesTransformer) Close() error {
	// Finalize any pending tool call
	if t.currentToolCall != nil {
		if err := t.emitToolCallDone(); err != nil {
			return err
		}
	}

	// If we have pending content or finish reason, emit response.completed
	if t.finishReason != "" || t.messageStarted || t.contentBuilder.Len() > 0 {
		if err := t.handleFinish(); err != nil {
			return err
		}
	}

	return t.Flush()
}
