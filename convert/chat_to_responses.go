package convert

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"

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

	inReasoning    bool
	reasoningIndex int
	sequenceNumber int

	messageStarted bool // track if message item has been emitted
	completed      bool // track if response.completed has been emitted
	toolCalls []map[string]interface{}
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

func (t *ChatToResponsesTransformer) emitReasoningDelta(text string) error {
	if !t.inReasoning {
		t.inReasoning = true
		t.reasoningIndex = t.contentIndex
		t.contentIndex++

		reasoningID := fmt.Sprintf("rs_%s", t.responseID[5:])
		event := map[string]interface{}{
			"type":            "response.output_item.added",
			"sequence_number": t.nextSeq(),
			"output_index":    t.reasoningIndex,
			"item": map[string]interface{}{
				"type":    "reasoning",
				"id":      reasoningID,
				"summary": []interface{}{},
			},
		}
		if err := t.writeEvent(event); err != nil {
			return err
		}
	}

	t.reasoningBuilder.WriteString(text)

	event := map[string]interface{}{
		"type":            "response.reasoning_summary_text.delta",
		"sequence_number": t.nextSeq(),
		"delta":           text,
	}
	return t.writeEvent(event)
}

func (t *ChatToResponsesTransformer) emitMessageItemAdded() error {
	messageID := fmt.Sprintf("msg_%s", t.responseID[5:])
	outputIndex := 0
	if t.inReasoning {
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
			if t.inReasoning {
				outputIndex++
			}
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

	if t.currentToolCall != nil {
		if err := t.emitToolCallDone(); err != nil {
			return err
		}
	}

	var outputItems []map[string]interface{}

	if t.inReasoning {
		reasoningID := fmt.Sprintf("rs_%s", t.responseID[5:])
		outputItems = append(outputItems, map[string]interface{}{
			"type": "reasoning",
			"id":   reasoningID,
			"summary": []map[string]interface{}{
				{"type": "summary_text", "text": t.reasoningBuilder.String()},
			},
		})
	}

	for _, tc := range t.toolCalls {
		outputItems = append(outputItems, tc)
	}

	messageID := fmt.Sprintf("msg_%s", t.responseID[5:])
	outputItems = append(outputItems, map[string]interface{}{
		"type":    "message",
		"id":      messageID,
		"status":  "completed",
		"role":    "assistant",
		"content": []map[string]interface{}{{"type": "output_text", "text": t.contentBuilder.String()}},
	})

	response := map[string]interface{}{
		"id":     t.responseID,
		"object": "response",
		"model":  t.model,
		"status": "completed",
		"output": outputItems,
	}

	if t.usage != nil {
		response["usage"] = map[string]interface{}{
			"input_tokens":  t.usage.PromptTokens,
			"output_tokens": t.usage.CompletionTokens,
			"total_tokens":  t.usage.TotalTokens,
		}
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
	return nil
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
	return t.Flush()
}
