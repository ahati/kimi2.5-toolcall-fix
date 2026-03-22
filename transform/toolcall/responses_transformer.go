package toolcall

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"

	"ai-proxy/conversation"
	"ai-proxy/logging"
	"ai-proxy/transform"
	"ai-proxy/types"

	"github.com/tmaxmax/go-sse"
)

// Usage tracks token usage from Anthropic API.
type Usage struct {
	InputTokens              int
	OutputTokens             int
	CacheReadInputTokens     int
	CacheCreationInputTokens int
}

// ResponsesTransformer converts Anthropic SSE events to OpenAI Responses API format.
type ResponsesTransformer struct {
	sseWriter  *transform.SSEWriter
	formatter  *ResponsesFormatter
	parser     *Parser
	messageID  string
	model      string
	blockIndex int
	currentID  string
	responseID string

	// State flags
	inToolCall  bool
	inText      bool
	inReasoning bool

	// Content builders
	textContent       strings.Builder
	toolArgs          strings.Builder
	reasoningContent  strings.Builder
	extractedToolArgs strings.Builder // Args for tool calls extracted from thinking content

	// Output tracking
	outputIndex     int    // Current output index counter (0-indexed)
	reasoningID     string // Cached reasoning item ID
	currentToolName string // Current tool name for function_call output item

	// Sequence tracking for streaming events
	sequenceNumber int // Global sequence number counter

	// Reasoning summary tracking
	summaryIndex int // Index of current summary part within reasoning

	// Current reasoning output index (cached when reasoning starts)
	reasoningOutputIndex int

	// Current tool call output index (cached when tool_use starts)
	toolCallOutputIndex int

	// Message output index (cached when message is emitted)
	messageOutputIndex int

	// Output items for final response
	outputItems []map[string]interface{}
	currentItem map[string]interface{}
	itemAdded   bool

	// Token usage tracking
	usage *Usage

	// Message item emitted flag
	messageItemEmitted bool

	// Input items for conversation storage
	inputItems []types.InputItem

	// Tool call extraction from thinking content (for Kimi-style markup)
	toolCallTransform bool // enabled by config
}

// ResponsesFormatter formats events in OpenAI Responses API format.
type ResponsesFormatter struct {
	responseID string
	model      string
}

// NewResponsesFormatter creates a new formatter for OpenAI Responses API.
func NewResponsesFormatter(responseID, model string) *ResponsesFormatter {
	return &ResponsesFormatter{
		responseID: responseID,
		model:      model,
	}
}

// SetResponseID sets the response ID.
func (f *ResponsesFormatter) SetResponseID(id string) {
	f.responseID = id
}

// SetModel sets the model name.
func (f *ResponsesFormatter) SetModel(model string) {
	f.model = model
}

// getReasoningID generates a consistent reasoning item ID from the message ID.
// Converts "msg_xxx" to "rs_xxx" format per OpenAI Responses API convention.
func (t *ResponsesTransformer) getReasoningID() string {
	if t.reasoningID == "" && t.messageID != "" {
		if len(t.messageID) > 4 {
			t.reasoningID = "rs_" + t.messageID[4:]
		} else {
			t.reasoningID = "rs_" + t.messageID
		}
	}
	return t.reasoningID
}

// emitMessageItemAdded emits the response.output_item.added event for the message.
// This is called when the first text content is encountered, after all reasoning
// and tool calls have already emitted their output_item.added events.
// Returns nil if the message was already emitted or there's no message.
func (t *ResponsesTransformer) emitMessageItemAdded() error {
	if t.messageItemEmitted || t.currentItem == nil {
		return nil
	}

	// Calculate output index for message (after all reasoning and tool calls)
	outputIdx := t.outputIndex
	t.messageOutputIndex = outputIdx

	// Get the message item ID from currentItem
	messageItemID, _ := t.currentItem["id"].(string)
	if messageItemID == "" {
		messageItemID = t.messageID
	}

	// Emit response.output_item.added for the message
	seqNum := t.nextSequenceNumber()
	if err := t.write(t.formatter.FormatOutputItemAdded(messageItemID, outputIdx, seqNum)); err != nil {
		return err
	}

	t.messageItemEmitted = true
	t.outputIndex++ // Increment output index after emitting message item

	return nil
}

// FormatResponseCreated formats a response.created event.
func (f *ResponsesFormatter) FormatResponseCreated(sequenceNumber int) []byte {
	event := map[string]interface{}{
		"type":            "response.created",
		"sequence_number": sequenceNumber,
		"response": map[string]interface{}{
			"id":         f.responseID,
			"object":     "response",
			"created_at": 0,
			"model":      f.model,
			"status":     "in_progress",
			"output":     []interface{}{},
		},
	}
	data, _ := json.Marshal(event)
	return []byte("data: " + string(data) + "\n\n")
}

// FormatOutputItemAdded formats a response.output_item.added event for message.
// The outputIndex is the 0-indexed position in the output array.
func (f *ResponsesFormatter) FormatOutputItemAdded(itemID string, outputIndex int, sequenceNumber int) []byte {
	event := map[string]interface{}{
		"type":            "response.output_item.added",
		"item_id":         itemID,
		"output_index":    outputIndex,
		"sequence_number": sequenceNumber,
		"item": map[string]interface{}{
			"type":    "message",
			"id":      itemID,
			"status":  "in_progress",
			"role":    "assistant",
			"content": []interface{}{},
		},
	}
	data, _ := json.Marshal(event)
	return []byte("data: " + string(data) + "\n\n")
}

// FormatContentPartAdded formats a response.content_part.added event for text.
func (f *ResponsesFormatter) FormatContentPartAdded(itemID string, contentIndex int, partType string, outputIndex int, sequenceNumber int) []byte {
	event := map[string]interface{}{
		"type":            "response.content_part.added",
		"item_id":         itemID,
		"content_index":   contentIndex,
		"output_index":    outputIndex,
		"sequence_number": sequenceNumber,
		"part": map[string]interface{}{
			"type": partType,
		},
	}
	if partType == "output_text" {
		event["part"] = map[string]interface{}{
			"type": "output_text",
			"text": "",
		}
	}
	data, _ := json.Marshal(event)
	return []byte("data: " + string(data) + "\n\n")
}

// FormatOutputTextDelta formats a response.output_text.delta event.
func (f *ResponsesFormatter) FormatOutputTextDelta(itemID string, contentIndex int, delta string, outputIndex int, sequenceNumber int) []byte {
	event := map[string]interface{}{
		"type":            "response.output_text.delta",
		"item_id":         itemID,
		"content_index":   contentIndex,
		"output_index":    outputIndex,
		"sequence_number": sequenceNumber,
		"delta":           delta,
	}
	data, _ := json.Marshal(event)
	return []byte("data: " + string(data) + "\n\n")
}

// FormatFunctionCallItemAdded emits a response.output_item.added event for a function_call item.
// Function calls are separate output items in the Responses API, not content parts.
// The outputIndex is the 0-indexed position in the output array.
func (f *ResponsesFormatter) FormatFunctionCallItemAdded(itemID, name string, outputIndex int, sequenceNumber int) []byte {
	event := map[string]interface{}{
		"type":            "response.output_item.added",
		"item_id":         itemID,
		"output_index":    outputIndex,
		"sequence_number": sequenceNumber,
		"item": map[string]interface{}{
			"type":      "function_call",
			"id":        itemID,
			"call_id":   itemID,
			"name":      name,
			"arguments": "",
		},
	}
	data, _ := json.Marshal(event)
	return []byte("data: " + string(data) + "\n\n")
}

// FormatFunctionCallItemDone emits a response.output_item.done event for a function_call item.
// This signals the completion of a function call output item with the full arguments.
func (f *ResponsesFormatter) FormatFunctionCallItemDone(itemID, name, arguments string, outputIndex int, sequenceNumber int) []byte {
	event := map[string]interface{}{
		"type":            "response.output_item.done",
		"item_id":         itemID,
		"output_index":    outputIndex,
		"sequence_number": sequenceNumber,
		"item": map[string]interface{}{
			"type":      "function_call",
			"id":        itemID,
			"call_id":   itemID,
			"name":      name,
			"arguments": arguments,
		},
	}
	data, _ := json.Marshal(event)
	return []byte("data: " + string(data) + "\n\n")
}

// FormatFunctionCallArgsDelta emits a response.function_call_arguments.delta event.
// The itemID is the function_call item ID, and callID is used for the call_id field.
func (f *ResponsesFormatter) FormatFunctionCallArgsDelta(itemID, callID, delta string, outputIndex int, sequenceNumber int) []byte {
	event := map[string]interface{}{
		"type":            "response.function_call_arguments.delta",
		"item_id":         itemID,
		"call_id":         callID,
		"output_index":    outputIndex,
		"sequence_number": sequenceNumber,
		"delta":           delta,
	}
	data, _ := json.Marshal(event)
	return []byte("data: " + string(data) + "\n\n")
}

// FormatContentPartDone formats a response.content_part.done event.
func (f *ResponsesFormatter) FormatContentPartDone(itemID string, contentIndex int, partType string, content string, outputIndex int, sequenceNumber int) []byte {
	part := map[string]interface{}{
		"type": partType,
	}
	if partType == "output_text" {
		part["text"] = content
	}
	event := map[string]interface{}{
		"type":            "response.content_part.done",
		"item_id":         itemID,
		"content_index":   contentIndex,
		"output_index":    outputIndex,
		"sequence_number": sequenceNumber,
		"part":            part,
	}
	data, _ := json.Marshal(event)
	return []byte("data: " + string(data) + "\n\n")
}

// FormatOutputItemDone formats a response.output_item.done event.
// The outputIndex is the 0-indexed position in the output array.
func (f *ResponsesFormatter) FormatOutputItemDone(itemID string, item map[string]interface{}, outputIndex int, sequenceNumber int) []byte {
	event := map[string]interface{}{
		"type":            "response.output_item.done",
		"item_id":         itemID,
		"output_index":    outputIndex,
		"sequence_number": sequenceNumber,
		"item":            item,
	}
	data, _ := json.Marshal(event)
	return []byte("data: " + string(data) + "\n\n")
}

// FormatResponseCompleted formats a response.completed event.
func (f *ResponsesFormatter) FormatResponseCompleted(outputItems []map[string]interface{}, usage *Usage, sequenceNumber int) []byte {
	output := []interface{}{}
	if len(outputItems) > 0 {
		for _, item := range outputItems {
			output = append(output, item)
		}
	}
	response := map[string]interface{}{
		"id":     f.responseID,
		"object": "response",
		"model":  f.model,
		"status": "completed",
		"output": output,
	}

	// Add usage if available
	//
	// IMPORTANT: Anthropic and OpenAI have different semantics for cache tokens:
	// - Anthropic: input_tokens is fresh tokens only, cache_read/cache_creation are additive
	// - OpenAI: input_tokens is total (includes cached), cached_tokens is a subset
	//
	// Conversion: OpenAI input_tokens = Anthropic input_tokens + cache_read + cache_creation
	if usage != nil {
		totalInputTokens := usage.InputTokens + usage.CacheReadInputTokens + usage.CacheCreationInputTokens
		usageData := map[string]interface{}{
			"input_tokens":  totalInputTokens,
			"output_tokens": usage.OutputTokens,
			"total_tokens":  totalInputTokens + usage.OutputTokens,
		}
		// Include cache token details if available (these are metadata, not additional tokens in OpenAI)
		if usage.CacheReadInputTokens > 0 || usage.CacheCreationInputTokens > 0 {
			details := map[string]interface{}{}
			if usage.CacheReadInputTokens > 0 {
				details["cached_tokens"] = usage.CacheReadInputTokens
			}
			if usage.CacheCreationInputTokens > 0 {
				details["cache_creation_input_tokens"] = usage.CacheCreationInputTokens
			}
			usageData["input_tokens_details"] = details
		}
		response["usage"] = usageData
	}

	event := map[string]interface{}{
		"type":            "response.completed",
		"sequence_number": sequenceNumber,
		"response":        response,
	}
	data, _ := json.Marshal(event)
	return []byte("data: " + string(data) + "\n\n")
}

// FormatReasoningItemAdded emits a response.output_item.added event for a reasoning item.
// This signals the start of a reasoning output item in the streaming response.
// The outputIndex is the 0-indexed position in the output array (always 0 for reasoning).
func (f *ResponsesFormatter) FormatReasoningItemAdded(itemID string, outputIndex int, sequenceNumber int) []byte {
	event := map[string]interface{}{
		"type":            "response.output_item.added",
		"item_id":         itemID,
		"output_index":    outputIndex,
		"sequence_number": sequenceNumber,
		"item": map[string]interface{}{
			"type":    "reasoning",
			"id":      itemID,
			"summary": []interface{}{},
		},
	}
	data, _ := json.Marshal(event)
	return []byte("data: " + string(data) + "\n\n")
}

// FormatReasoningSummaryPartAdded emits a response.reasoning_summary_part.added event.
// This signals the start of a new summary part within the reasoning item.
func (f *ResponsesFormatter) FormatReasoningSummaryPartAdded(itemID string, outputIndex int, summaryIndex int, sequenceNumber int) []byte {
	event := map[string]interface{}{
		"type":            "response.reasoning_summary_part.added",
		"item_id":         itemID,
		"output_index":    outputIndex,
		"summary_index":   summaryIndex,
		"sequence_number": sequenceNumber,
		"part": map[string]interface{}{
			"type": "summary_text",
			"text": "",
		},
	}
	data, _ := json.Marshal(event)
	return []byte("data: " + string(data) + "\n\n")
}

// FormatReasoningSummaryDelta emits a response.reasoning_summary_text.delta event.
// This streams incremental reasoning summary text to the client.
func (f *ResponsesFormatter) FormatReasoningSummaryDelta(itemID string, delta string, outputIndex int, summaryIndex int, sequenceNumber int) []byte {
	event := map[string]interface{}{
		"type":            "response.reasoning_summary_text.delta",
		"item_id":         itemID,
		"delta":           delta,
		"output_index":    outputIndex,
		"summary_index":   summaryIndex,
		"sequence_number": sequenceNumber,
	}
	data, _ := json.Marshal(event)
	return []byte("data: " + string(data) + "\n\n")
}

// FormatReasoningSummaryTextDone emits a response.reasoning_summary_text.done event.
// This signals the completion of a reasoning summary text.
func (f *ResponsesFormatter) FormatReasoningSummaryTextDone(itemID string, text string, outputIndex int, summaryIndex int, sequenceNumber int) []byte {
	event := map[string]interface{}{
		"type":            "response.reasoning_summary_text.done",
		"item_id":         itemID,
		"text":            text,
		"output_index":    outputIndex,
		"summary_index":   summaryIndex,
		"sequence_number": sequenceNumber,
	}
	data, _ := json.Marshal(event)
	return []byte("data: " + string(data) + "\n\n")
}

// FormatReasoningSummaryPartDone emits a response.reasoning_summary_part.done event.
// This signals the completion of a reasoning summary part.
func (f *ResponsesFormatter) FormatReasoningSummaryPartDone(itemID string, text string, outputIndex int, summaryIndex int, sequenceNumber int) []byte {
	event := map[string]interface{}{
		"type":            "response.reasoning_summary_part.done",
		"item_id":         itemID,
		"output_index":    outputIndex,
		"summary_index":   summaryIndex,
		"sequence_number": sequenceNumber,
		"part": map[string]interface{}{
			"type": "summary_text",
			"text": text,
		},
	}
	data, _ := json.Marshal(event)
	return []byte("data: " + string(data) + "\n\n")
}

// FormatReasoningItemDone emits a response.output_item.done event with the full summary.
// This signals the completion of a reasoning output item.
// The outputIndex is the 0-indexed position in the output array.
func (f *ResponsesFormatter) FormatReasoningItemDone(itemID, summaryText string, outputIndex int, sequenceNumber int) []byte {
	event := map[string]interface{}{
		"type":            "response.output_item.done",
		"item_id":         itemID,
		"output_index":    outputIndex,
		"sequence_number": sequenceNumber,
		"item": map[string]interface{}{
			"type": "reasoning",
			"id":   itemID,
			"summary": []map[string]interface{}{
				{"type": "summary_text", "text": summaryText},
			},
		},
	}
	data, _ := json.Marshal(event)
	return []byte("data: " + string(data) + "\n\n")
}

// NewResponsesTransformer creates a new transformer for OpenAI Responses API.
func NewResponsesTransformer(output io.Writer) *ResponsesTransformer {
	return &ResponsesTransformer{
		sseWriter:      transform.NewSSEWriter(output),
		formatter:      NewResponsesFormatter("", ""),
		parser:         NewParser(DefaultTokens),
		outputItems:    make([]map[string]interface{}, 0),
		sequenceNumber: 0,
		summaryIndex:   0,
	}
}

// SetInputItems sets the input items for conversation storage.
// This should be called before streaming starts to capture the original request input.
func (t *ResponsesTransformer) SetInputItems(items []types.InputItem) {
	t.inputItems = items
}

// SetToolCallTransform enables or disables tool call extraction from thinking content.
// When enabled, the transformer will parse Kimi-style tool call markup in thinking text
// and emit proper function_call output items.
func (t *ResponsesTransformer) SetToolCallTransform(enabled bool) {
	t.toolCallTransform = enabled
}

func (t *ResponsesTransformer) nextSequenceNumber() int {
	t.sequenceNumber++
	return t.sequenceNumber
}

// Transform processes an SSE event and converts it to OpenAI Responses format.
func (t *ResponsesTransformer) Transform(event *sse.Event) error {
	if event.Data == "" {
		return nil
	}

	if event.Data == "[DONE]" {
		return t.write([]byte("data: [DONE]\n\n"))
	}

	var anthropicEvent types.Event
	if err := json.Unmarshal([]byte(event.Data), &anthropicEvent); err != nil {
		return t.write([]byte("data: " + event.Data + "\n\n"))
	}

	return t.handleEvent(anthropicEvent)
}

// handleEvent processes an Anthropic event and converts it to Responses format.
func (t *ResponsesTransformer) handleEvent(event types.Event) error {
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
		return t.handleMessageStop(event)
	case "ping":
		// Pass through ping events
		return nil
	default:
		// Pass through unknown events
		return nil
	}
}

func (t *ResponsesTransformer) handleMessageStart(event types.Event) error {
	if event.Message != nil && event.Message.ID != "" {
		t.messageID = event.Message.ID
		t.responseID = "resp_" + event.Message.ID[4:] // Convert msg_xxx to resp_xxx
		t.model = event.Message.Model
		t.formatter.SetResponseID(t.responseID)
		t.formatter.SetModel(t.model)
		// Capture usage from message_start event (includes cache tokens)
		if event.Message.Usage != nil {
			t.usage = &Usage{
				InputTokens:              event.Message.Usage.InputTokens,
				OutputTokens:             event.Message.Usage.OutputTokens,
				CacheReadInputTokens:     event.Message.Usage.CacheReadInputTokens,
				CacheCreationInputTokens: event.Message.Usage.CacheCreationInputTokens,
			}
		}
	}

	// Send response.created event
	seqNum := t.nextSequenceNumber()
	if err := t.write(t.formatter.FormatResponseCreated(seqNum)); err != nil {
		return err
	}

	// Create the message item but don't emit it yet - we need to know the output_index
	// which depends on how many reasoning and tool_call items come before it.
	t.itemAdded = true
	// Use messageID (msg_xxx) for the message item, NOT responseID (resp_xxx)
	// to avoid ID collision with the response itself
	messageItemID := t.messageID
	if messageItemID == "" {
		// Safe fallback: derive from response ID or generate unique ID
		if len(t.responseID) > 5 {
			messageItemID = "msg_" + t.responseID[5:] // derive from response ID
		} else {
			messageItemID = fmt.Sprintf("msg_%d_%d", time.Now().UnixMilli(), t.sequenceNumber)
		}
	}
	t.currentItem = map[string]interface{}{
		"type":    "message",
		"id":      messageItemID,
		"status":  "in_progress",
		"role":    "assistant",
		"content": []map[string]interface{}{},
	}
	return nil
}

func (t *ResponsesTransformer) handleContentBlockStart(event types.Event) error {
	if event.Index == nil {
		return nil
	}

	if event.ContentBlock != nil {
		var block types.ContentBlock
		if err := json.Unmarshal(event.ContentBlock, &block); err == nil {
			t.blockIndex = *event.Index

			switch block.Type {
			case "text":
				t.inText = true
				t.textContent.Reset()
				if err := t.emitMessageItemAdded(); err != nil {
					return err
				}
				seqNum := t.nextSequenceNumber()
				messageItemID, _ := t.currentItem["id"].(string)
				if messageItemID == "" {
					messageItemID = t.messageID
				}
				return t.write(t.formatter.FormatContentPartAdded(messageItemID, 0, "output_text", t.messageOutputIndex, seqNum))
			case "thinking":
				t.inReasoning = true
				t.reasoningContent.Reset()
				reasoningID := t.getReasoningID()
				outputIdx := t.outputIndex
				t.reasoningOutputIndex = outputIdx
				t.outputIndex++
				t.summaryIndex = 0

				// Emit response.output_item.added for reasoning
				seqNum := t.nextSequenceNumber()
				if err := t.write(t.formatter.FormatReasoningItemAdded(reasoningID, outputIdx, seqNum)); err != nil {
					return err
				}

				// Emit response.reasoning_summary_part.added
				seqNum = t.nextSequenceNumber()
				return t.write(t.formatter.FormatReasoningSummaryPartAdded(reasoningID, outputIdx, t.summaryIndex, seqNum))
			case "tool_use":
				t.inToolCall = true
				t.currentID = block.ID
				t.currentToolName = block.Name
				t.toolArgs.Reset()
				outputIdx := t.outputIndex
				t.toolCallOutputIndex = outputIdx
				t.outputIndex++
				seqNum := t.nextSequenceNumber()
				return t.write(t.formatter.FormatFunctionCallItemAdded(block.ID, block.Name, outputIdx, seqNum))
			}
		}
	}

	return nil
}

func (t *ResponsesTransformer) handleContentBlockDelta(event types.Event) error {
	if event.Index == nil {
		return nil
	}

	if event.Delta != nil {
		// Try to parse as text_delta
		var textDelta types.TextDelta
		if err := json.Unmarshal(event.Delta, &textDelta); err == nil && textDelta.Type == "text_delta" {
			if t.inText {
				t.textContent.WriteString(textDelta.Text)
				seqNum := t.nextSequenceNumber()
				messageItemID, _ := t.currentItem["id"].(string)
				if messageItemID == "" {
					messageItemID = t.messageID
				}
				return t.write(t.formatter.FormatOutputTextDelta(messageItemID, 0, textDelta.Text, t.messageOutputIndex, seqNum))
			}
		}

		// Try to parse as thinking_delta
		var thinkingDelta types.ThinkingDelta
		if err := json.Unmarshal(event.Delta, &thinkingDelta); err == nil && thinkingDelta.Type == "thinking_delta" {
			if t.inReasoning {
				// Check if content contains tool call markup (only when toolCallTransform is enabled)
				if t.toolCallTransform {
					wasIdle := t.parser.IsIdle()
					hasMarkup := t.parser.tokens.ContainsAny(thinkingDelta.Thinking)
					if !wasIdle || hasMarkup {
						// Only log when starting to parse tool calls (transition from idle to parsing)
						if wasIdle && hasMarkup {
							logging.InfoMsg("[%s] Tool call markup detected in thinking content, extracting tool calls", t.messageID)
						}
						return t.processThinkingWithToolCalls(thinkingDelta.Thinking)
					}
				}
				// Normal thinking content - pass through
				t.reasoningContent.WriteString(thinkingDelta.Thinking)
				reasoningID := t.getReasoningID()
				seqNum := t.nextSequenceNumber()
				return t.write(t.formatter.FormatReasoningSummaryDelta(reasoningID, thinkingDelta.Thinking, t.reasoningOutputIndex, t.summaryIndex, seqNum))
			}
		}

		// Try to parse as input_json_delta
		var inputDelta types.InputJSONDelta
		if err := json.Unmarshal(event.Delta, &inputDelta); err == nil && inputDelta.Type == "input_json_delta" {
			if t.inToolCall {
				t.toolArgs.WriteString(inputDelta.PartialJSON)
				seqNum := t.nextSequenceNumber()
				return t.write(t.formatter.FormatFunctionCallArgsDelta(t.currentID, t.currentID, inputDelta.PartialJSON, t.toolCallOutputIndex, seqNum))
			}
		}
	}

	return nil
}

// processThinkingWithToolCalls handles thinking content that contains tool call markup.
// It extracts tool calls and emits appropriate Responses API events.
func (t *ResponsesTransformer) processThinkingWithToolCalls(text string) error {
	events := t.parser.Parse(text)
	for _, e := range events {
		if err := t.writeParserEvent(e); err != nil {
			return err
		}
	}
	return nil
}

// writeParserEvent converts a parser Event to Responses API format.
func (t *ResponsesTransformer) writeParserEvent(e Event) error {
	switch e.Type {
	case EventContent:
		// Regular thinking content - emit as reasoning summary delta
		if e.Text != "" {
			t.reasoningContent.WriteString(e.Text)
			reasoningID := t.getReasoningID()
			seqNum := t.nextSequenceNumber()
			return t.write(t.formatter.FormatReasoningSummaryDelta(reasoningID, e.Text, t.reasoningOutputIndex, t.summaryIndex, seqNum))
		}
	case EventToolStart:
		// Start a new function_call output item
		logging.InfoMsg("[%s] Tool call extracted: id=%s, name=%s", t.messageID, e.ID, e.Name)
		t.extractedToolArgs.Reset() // Reset args builder for new tool call
		return t.emitToolCallStart(e.ID, e.Name)
	case EventToolArgs:
		// Accumulate and emit function call arguments delta
		t.extractedToolArgs.WriteString(e.Args)
		seqNum := t.nextSequenceNumber()
		return t.write(t.formatter.FormatFunctionCallArgsDelta(t.currentID, t.currentID, e.Args, t.toolCallOutputIndex, seqNum))
	case EventToolEnd:
		// End the function_call output item
		args := t.extractedToolArgs.String()
		logging.InfoMsg("[%s] Tool call complete: id=%s, name=%s", t.messageID, t.currentID, t.currentToolName)
		return t.emitToolCallEnd(t.currentID, t.currentToolName, args)
	case EventSectionEnd:
		// Tool calls section ended - continue with reasoning if there's more content
	}
	return nil
}

func (t *ResponsesTransformer) emitToolCallStart(id, name string) error {
	// If we're in reasoning and have accumulated content, keep the reasoning item.
	// If we're in reasoning but have no content, we can close it.

	// Start new function_call output item
	outputIdx := t.outputIndex
	t.toolCallOutputIndex = outputIdx
	t.outputIndex++
	t.currentID = id
	t.currentToolName = name
	t.inToolCall = true

	seqNum := t.nextSequenceNumber()
	return t.write(t.formatter.FormatFunctionCallItemAdded(id, name, outputIdx, seqNum))
}

// emitToolCallEnd emits the necessary events to end a function_call output item.
func (t *ResponsesTransformer) emitToolCallEnd(id, name, args string) error {
	t.inToolCall = false

	// Track tool call in output items
	toolItem := map[string]interface{}{
		"type":      "function_call",
		"id":        id,
		"call_id":   id,
		"name":      name,
		"arguments": args,
	}
	t.outputItems = append(t.outputItems, toolItem)

	seqNum := t.nextSequenceNumber()
	return t.write(t.formatter.FormatFunctionCallItemDone(id, name, args, t.toolCallOutputIndex, seqNum))
}

func (t *ResponsesTransformer) handleContentBlockStop(event types.Event) error {
	if event.Index == nil {
		return nil
	}

	if t.inText {
		t.inText = false
		content := t.textContent.String()
		// Track content for output item - ensure content array exists
		if t.currentItem != nil {
			contents, ok := t.currentItem["content"].([]map[string]interface{})
			if !ok {
				// Initialize content array if it doesn't exist or has wrong type
				contents = []map[string]interface{}{}
			}
			t.currentItem["content"] = append(contents, map[string]interface{}{
				"type": "output_text",
				"text": content,
			})
		}
		seqNum := t.nextSequenceNumber()
		messageItemID, _ := t.currentItem["id"].(string)
		if messageItemID == "" {
			messageItemID = t.messageID
		}
		return t.write(t.formatter.FormatContentPartDone(messageItemID, 0, "output_text", content, t.messageOutputIndex, seqNum))
	}

	if t.inReasoning {
		t.inReasoning = false

		// Flush any remaining parser state (only when toolCallTransform is enabled)
		if t.toolCallTransform {
			for {
				events := t.parser.Parse("")
				if len(events) == 0 {
					break
				}
				for _, e := range events {
					if err := t.writeParserEvent(e); err != nil {
						return err
					}
				}
			}
		}

		summary := t.reasoningContent.String()
		reasoningID := t.getReasoningID()
		outputIdx := t.reasoningOutputIndex

		// Emit response.reasoning_summary_text.done
		seqNum := t.nextSequenceNumber()
		if err := t.write(t.formatter.FormatReasoningSummaryTextDone(reasoningID, summary, outputIdx, t.summaryIndex, seqNum)); err != nil {
			return err
		}

		// Emit response.reasoning_summary_part.done
		seqNum = t.nextSequenceNumber()
		if err := t.write(t.formatter.FormatReasoningSummaryPartDone(reasoningID, summary, outputIdx, t.summaryIndex, seqNum)); err != nil {
			return err
		}

		reasoningItem := map[string]interface{}{
			"type": "reasoning",
			"id":   reasoningID,
			"summary": []map[string]interface{}{
				{"type": "summary_text", "text": summary},
			},
		}
		t.outputItems = append(t.outputItems, reasoningItem)

		// Emit response.output_item.done
		seqNum = t.nextSequenceNumber()
		return t.write(t.formatter.FormatReasoningItemDone(reasoningID, summary, outputIdx, seqNum))
	}

	if t.inToolCall {
		t.inToolCall = false
		args := t.toolArgs.String()

		toolItem := map[string]interface{}{
			"type":      "function_call",
			"id":        t.currentID,
			"call_id":   t.currentID,
			"name":      t.currentToolName,
			"arguments": args,
		}
		t.outputItems = append(t.outputItems, toolItem)

		outputIdx := t.toolCallOutputIndex
		seqNum := t.nextSequenceNumber()
		return t.write(t.formatter.FormatFunctionCallItemDone(t.currentID, t.currentToolName, args, outputIdx, seqNum))
	}

	return nil
}

func (t *ResponsesTransformer) handleMessageDelta(event types.Event) error {
	// Capture/merge usage from message_delta event
	if event.Usage != nil {
		if t.usage == nil {
			t.usage = &Usage{}
		}
		// Only update tokens if the new values are non-zero
		// (message_delta may have 0 for some fields, preserve message_start values)
		if event.Usage.InputTokens > 0 {
			t.usage.InputTokens = event.Usage.InputTokens
		}
		if event.Usage.OutputTokens > 0 {
			t.usage.OutputTokens = event.Usage.OutputTokens
		}
		// Merge cache tokens (may be in message_delta or message_start)
		if event.Usage.CacheReadInputTokens > 0 {
			t.usage.CacheReadInputTokens = event.Usage.CacheReadInputTokens
		}
		if event.Usage.CacheCreationInputTokens > 0 {
			t.usage.CacheCreationInputTokens = event.Usage.CacheCreationInputTokens
		}
	}

	// Handle stop_reason - convert Anthropic stop_reason to Responses API format
	if event.StopReason != "" {
		// Map Anthropic stop_reason to appropriate Responses API output
		// Anthropic: "end_turn", "max_tokens", "stop_sequence", "tool_use"
		switch event.StopReason {
		case "tool_use":
			// Tool use is handled via content blocks, no special event needed
			logging.InfoMsg("[%s] Stop reason: tool_use", t.messageID)
		case "max_tokens":
			logging.InfoMsg("[%s] Stop reason: max_tokens", t.messageID)
		case "end_turn":
			logging.InfoMsg("[%s] Stop reason: end_turn", t.messageID)
		}
	}
	return nil
}

func (t *ResponsesTransformer) handleMessageStop(event types.Event) error {
	// Emit message item completion if there's a message item with content.
	// This includes text content OR tool calls (model can respond with just tool calls).
	hasTextContent := t.textContent.Len() > 0
	hasToolCalls := len(t.outputItems) > 0
	if t.itemAdded && t.currentItem != nil && (hasTextContent || hasToolCalls) {
		// If message item wasn't emitted but we have text, emit it now
		if !t.messageItemEmitted {
			if err := t.emitMessageItemAdded(); err != nil {
				return err
			}
		}

		// Calculate output index for message (after all reasoning and tool calls)
		outputIdx := t.messageOutputIndex

		// Get the message item ID from currentItem (set in handleMessageStart)
		// This ensures we use msg_xxx format, not resp_xxx
		messageItemID, _ := t.currentItem["id"].(string)
		if messageItemID == "" {
			messageItemID = t.messageID
		}

		// Mark message as completed and emit output_item.done
		t.currentItem["status"] = "completed"
		seqNum := t.nextSequenceNumber()
		if err := t.write(t.formatter.FormatOutputItemDone(messageItemID, t.currentItem, outputIdx, seqNum)); err != nil {
			return err
		}
		// Add to output items for response.completed
		t.outputItems = append(t.outputItems, t.currentItem)
	}

	// Send response.completed event with populated output and usage
	seqNum := t.nextSequenceNumber()
	if err := t.write(t.formatter.FormatResponseCompleted(t.outputItems, t.usage, seqNum)); err != nil {
		return err
	}

	// Store conversation for previous_response_id support
	t.storeConversation()

	return nil
}

// storeConversation saves the conversation to the default store for previous_response_id support.
// This enables multi-turn conversations without re-sending the entire history.
func (t *ResponsesTransformer) storeConversation() {
	// Only store if we have a response ID and the store is initialized
	if t.responseID == "" {
		return
	}

	// Convert outputItems to types.OutputItem slice
	outputItems := make([]types.OutputItem, 0, len(t.outputItems))
	for _, item := range t.outputItems {
		outputItem := convertToOutputItem(item)
		if outputItem != nil {
			outputItems = append(outputItems, *outputItem)
		}
	}

	// Store the conversation
	conv := &conversation.Conversation{
		ID:     t.responseID,
		Input:  t.inputItems,
		Output: outputItems,
	}
	conversation.StoreInDefault(conv)
	logging.DebugMsg("[%s] Stored conversation with %d input items and %d output items",
		t.responseID, len(t.inputItems), len(outputItems))
}

// convertToOutputItem converts a map[string]interface{} to types.OutputItem.
func convertToOutputItem(item map[string]interface{}) *types.OutputItem {
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

func (t *ResponsesTransformer) write(data []byte) error {
	if len(data) == 0 {
		return nil
	}
	_, err := t.sseWriter.WriteRaw(data)
	return err
}

// Flush writes any buffered data.
func (t *ResponsesTransformer) Flush() error {
	return nil
}

// Close flushes and releases resources.
func (t *ResponsesTransformer) Close() error {
	return t.Flush()
}

// EmitError sends a response.failed event for stream errors.
// This notifies clients when the stream terminates unexpectedly.
func (t *ResponsesTransformer) EmitError(streamErr error) error {
	if t.responseID == "" {
		return nil
	}

	event := map[string]interface{}{
		"type":            "response.failed",
		"sequence_number": t.outputIndex,
		"response": map[string]interface{}{
			"id":     t.responseID,
			"object": "response",
			"model":  t.model,
			"status": "failed",
			"error": map[string]interface{}{
				"message": streamErr.Error(),
				"type":    "stream_error",
			},
		},
	}

	data, _ := json.Marshal(event)
	return t.sseWriter.WriteData(data)
}
