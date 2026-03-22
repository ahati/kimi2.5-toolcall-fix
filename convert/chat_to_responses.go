package convert

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"

	"ai-proxy/conversation"
	"ai-proxy/logging"
	"ai-proxy/transform/toolcall"
	"ai-proxy/types"

	"github.com/tmaxmax/go-sse"
)

// ─────────────────────────────────────────────────────────────────────────────
// Chat Completions → Responses — Request
// ─────────────────────────────────────────────────────────────────────────────

// ChatToResponsesRequest converts a Chat Completions Request into a Responses Request.
//
// Dropped: n, stop, response_format, frequency_penalty, presence_penalty,
// logprobs, seed — none have Responses API equivalents.
func ChatToResponsesRequest(req *types.ChatCompletionRequest) (*types.ResponsesRequest, error) {
	out := &types.ResponsesRequest{
		Model:       req.Model,
		Stream:      req.Stream,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		ToolChoice:  ChatToolChoiceToResponses(marshalToolChoiceFromRequest(req.ToolChoice)),
	}

	if req.MaxTokens > 0 {
		out.MaxOutputTokens = req.MaxTokens
	}

	if req.User != "" {
		out.Metadata = map[string]interface{}{"user_id": req.User}
	}

	for _, t := range req.Tools {
		out.Tools = append(out.Tools, types.ResponsesTool{
			Type:        "function",
			Name:        t.Function.Name,
			Description: t.Function.Description,
			Parameters:  t.Function.Parameters,
		})
	}

	instructions, input, err := chatMessagesToResponsesInput(req.Messages)
	if err != nil {
		return nil, fmt.Errorf("converting messages: %w", err)
	}
	out.Instructions = instructions
	out.Input = input
	return out, nil
}

func marshalToolChoiceFromRequest(tc interface{}) json.RawMessage {
	if tc == nil {
		return nil
	}
	b, _ := json.Marshal(tc)
	return b
}

func chatMessagesToResponsesInput(msgs []types.Message) (instructions string, input interface{}, err error) {
	var sysParts []string
	var items []types.InputItem

	for _, msg := range msgs {
		switch msg.Role {
		case "system":
			if str, ok := msg.Content.(string); ok {
				sysParts = append(sysParts, str)
			}
		case "user":
			item, convErr := chatUserMessageToResponsesItem(msg)
			if convErr != nil {
				return "", nil, convErr
			}
			items = append(items, item)
		case "assistant":
			convItems, convErr := chatAssistantMessageToResponsesItems(msg)
			if convErr != nil {
				return "", nil, convErr
			}
			items = append(items, convItems...)
		case "tool":
			output := extractContentString(msg.Content)
			items = append(items, types.InputItem{
				Type:   "function_call_output",
				CallID: msg.ToolCallID,
				Output: output,
			})
		}
	}

	for i, s := range sysParts {
		if i > 0 {
			instructions += "\n\n"
		}
		instructions += s
	}

	// If only a single user message, return as string for simplicity
	if len(items) == 1 && items[0].Type == "message" && items[0].Role == "user" {
		if str, ok := items[0].Content.(string); ok {
			return instructions, str, nil
		}
	}

	return instructions, items, nil
}

func chatUserMessageToResponsesItem(msg types.Message) (types.InputItem, error) {
	item := types.InputItem{Type: "message", Role: "user"}

	switch c := msg.Content.(type) {
	case string:
		item.Content = c
		return item, nil
	case []interface{}:
		var parts []types.ContentPart
		for _, p := range c {
			if partMap, ok := p.(map[string]interface{}); ok {
				partType, _ := partMap["type"].(string)
				switch partType {
				case "text":
					text, _ := partMap["text"].(string)
					parts = append(parts, types.ContentPart{Type: "input_text", Text: text})
				case "image_url":
					if imageURL, ok := partMap["image_url"].(map[string]interface{}); ok {
						if url, ok := imageURL["url"].(string); ok {
							parts = append(parts, types.ContentPart{Type: "input_image", ImageURL: url})
						}
					}
				}
			}
		}
		item.Content = parts
	default:
		if str := extractContentString(msg.Content); str != "" {
			item.Content = str
		}
	}
	return item, nil
}

func chatAssistantMessageToResponsesItems(msg types.Message) ([]types.InputItem, error) {
	var out []types.InputItem

	// Handle text content
	if msg.Content != nil {
		if str := extractContentString(msg.Content); str != "" {
			out = append(out, types.InputItem{
				Type: "message",
				Role: "assistant",
				Content: []types.ContentPart{{
					Type: "output_text",
					Text: str,
				}},
			})
		}
	}

	// Handle tool calls
	for _, tc := range msg.ToolCalls {
		out = append(out, types.InputItem{
			Type:      "function_call",
			CallID:    tc.ID,
			Name:      tc.Function.Name,
			Arguments: tc.Function.Arguments,
		})
	}

	return out, nil
}

func extractContentString(content interface{}) string {
	if content == nil {
		return ""
	}
	switch c := content.(type) {
	case string:
		return c
	case []interface{}:
		var parts []string
		for _, p := range c {
			if partMap, ok := p.(map[string]interface{}); ok {
				if partType, _ := partMap["type"].(string); partType == "text" {
					if text, ok := partMap["text"].(string); ok {
						parts = append(parts, text)
					}
				}
			}
		}
		return strings.Join(parts, "")
	default:
		b, _ := json.Marshal(content)
		return string(b)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Chat Completions → Responses — Streaming
// ─────────────────────────────────────────────────────────────────────────────

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

	// Tool call extraction from reasoning content (for Kimi-style markup)
	toolCallTransform bool             // enabled by config
	parser            *toolcall.Parser // parser for tool call markup
	extractedToolArgs strings.Builder  // args for extracted tool calls
	extractedToolID   string           // current extracted tool ID
	extractedToolName string           // current extracted tool name
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
		parser:         toolcall.NewParser(toolcall.DefaultTokens),
	}
}

// SetInputItems sets the input items for conversation storage.
// This should be called before streaming starts to capture the original request input.
func (t *ChatToResponsesTransformer) SetInputItems(items []types.InputItem) {
	t.inputItems = items
}

// SetToolCallTransform enables or disables tool call extraction from reasoning content.
// When enabled, the transformer will parse Kimi-style tool call markup in reasoning text
// and emit proper function_call output items.
func (t *ChatToResponsesTransformer) SetToolCallTransform(enabled bool) {
	t.toolCallTransform = enabled
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

	// Flush any remaining parser state (in case tool calls were being parsed)
	if t.toolCallTransform {
		for {
			events := t.parser.Parse("")
			if len(events) == 0 {
				break
			}
			for _, e := range events {
				if err := t.writeToolCallParserEvent(e); err != nil {
					return err
				}
			}
		}
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
	// Check if tool call extraction is enabled and content contains tool call markup
	if t.toolCallTransform && (!t.parser.IsIdle() || strings.Contains(text, "<|tool_call")) {
		return t.processReasoningWithToolCalls(text)
	}

	// Mark that we've started emitting content
	if !t.messageStarted {
		t.messageStarted = true
	}

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

// processReasoningWithToolCalls handles reasoning content that contains tool call markup.
// It extracts tool calls and emits appropriate Responses API events.
func (t *ChatToResponsesTransformer) processReasoningWithToolCalls(text string) error {
	logging.InfoMsg("[%s] Tool call markup detected in reasoning content, extracting tool calls", t.responseID)

	events := t.parser.Parse(text)
	for _, e := range events {
		if err := t.writeToolCallParserEvent(e); err != nil {
			return err
		}
	}
	return nil
}

// writeToolCallParserEvent converts a parser Event to Responses API format.
func (t *ChatToResponsesTransformer) writeToolCallParserEvent(e toolcall.Event) error {
	switch e.Type {
	case toolcall.EventContent:
		// Regular reasoning content - emit as reasoning summary delta
		if e.Text != "" {
			t.reasoningBuilder.WriteString(e.Text)
			// Emit reasoning delta event
			event := map[string]interface{}{
				"type":            "response.reasoning_summary_text.delta",
				"sequence_number": t.nextSeq(),
				"item_id":         t.reasoningID,
				"delta":           e.Text,
				"output_index":    t.reasoningOutputIndex,
				"summary_index":   t.summaryIndex,
			}
			return t.writeEvent(event)
		}
	case toolcall.EventToolStart:
		// Start a new function_call output item
		t.extractedToolArgs.Reset()
		return t.emitExtractedToolCallStart(e.ID, e.Name)
	case toolcall.EventToolArgs:
		// Accumulate and emit function call arguments delta
		t.extractedToolArgs.WriteString(e.Args)
		event := map[string]interface{}{
			"type":            "response.function_call_arguments.delta",
			"sequence_number": t.nextSeq(),
			"item_id":         t.extractedToolID,
			"call_id":         t.extractedToolID,
			"output_index":    t.toolCallIndex,
			"delta":           e.Args,
		}
		return t.writeEvent(event)
	case toolcall.EventToolEnd:
		// End the function_call output item
		args := t.extractedToolArgs.String()
		return t.emitExtractedToolCallEnd(args)
	case toolcall.EventSectionEnd:
		// Tool calls section ended - continue with reasoning if there's more content
	}
	return nil
}

// emitExtractedToolCallStart emits events to start a function_call output item from extracted markup.
func (t *ChatToResponsesTransformer) emitExtractedToolCallStart(id, name string) error {
	// Finalize reasoning before emitting tool call
	if t.inReasoning {
		if err := t.finalizeReasoning(); err != nil {
			return err
		}
	}

	outputIndex := t.contentIndex
	t.toolCallIndex = outputIndex
	t.contentIndex = outputIndex + 1
	t.extractedToolID = id
	t.extractedToolName = name

	event := map[string]interface{}{
		"type":            "response.output_item.added",
		"sequence_number": t.nextSeq(),
		"output_index":    outputIndex,
		"item": map[string]interface{}{
			"type":      "function_call",
			"id":        id,
			"call_id":   id,
			"name":      name,
			"arguments": "",
		},
	}
	return t.writeEvent(event)
}

// emitExtractedToolCallEnd emits events to end a function_call output item from extracted markup.
func (t *ChatToResponsesTransformer) emitExtractedToolCallEnd(args string) error {
	toolCallItem := map[string]interface{}{
		"type":      "function_call",
		"id":        t.extractedToolID,
		"call_id":   t.extractedToolID,
		"name":      t.extractedToolName,
		"arguments": args,
	}

	event := map[string]interface{}{
		"type":            "response.output_item.done",
		"sequence_number": t.nextSeq(),
		"output_index":    t.toolCallIndex,
		"item":            toolCallItem,
	}

	t.toolCalls = append(t.toolCalls, toolCallItem)
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
		if t.usage.PromptTokensDetails != nil && (t.usage.PromptTokensDetails.CachedTokens > 0 || t.usage.PromptTokensDetails.CacheCreationInputTokens > 0) {
			inputTokensDetails := map[string]interface{}{}
			if t.usage.PromptTokensDetails.CachedTokens > 0 {
				inputTokensDetails["cached_tokens"] = t.usage.PromptTokensDetails.CachedTokens
			}
			if t.usage.PromptTokensDetails.CacheCreationInputTokens > 0 {
				inputTokensDetails["cache_creation_input_tokens"] = t.usage.PromptTokensDetails.CacheCreationInputTokens
			}
			usageData["input_tokens_details"] = inputTokensDetails
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
