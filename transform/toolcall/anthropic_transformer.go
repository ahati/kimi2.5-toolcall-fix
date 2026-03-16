package toolcall

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"ai-proxy/logging"
	"ai-proxy/transform"
	"ai-proxy/types"

	"github.com/tmaxmax/go-sse"
)

type AnthropicTransformer struct {
	sseWriter  *transform.SSEWriter
	formatter  *AnthropicFormatter
	state      anthropicState
	buf        string
	toolIndex  int
	blockIndex int
	currentID  string
	messageID  string

	inThinking       bool
	inText           bool
	thinkingIndex    int
	textIndex        int
	needThinkingStop bool
	needTextStop     bool
	toolsEmitted     bool
}

type anthropicState int

const (
	anthropicStateIdle anthropicState = iota
	anthropicStateInSection
	anthropicStateReadingID
	anthropicStateReadingArgs
	anthropicStateTrailing
)

func NewAnthropicTransformer(output io.Writer) *AnthropicTransformer {
	return &AnthropicTransformer{
		sseWriter: transform.NewSSEWriter(output),
		formatter: NewAnthropicFormatter("", ""),
		state:     anthropicStateIdle,
	}
}

func (t *AnthropicTransformer) Transform(event *sse.Event) error {
	if event.Data == "" {
		return nil
	}

	if event.Data == "[DONE]" {
		return t.writeData([]byte("[DONE]"))
	}

	var anthropicEvent types.Event
	if err := json.Unmarshal([]byte(event.Data), &anthropicEvent); err != nil {
		return t.writePassthrough("error", []byte(event.Data))
	}
	if anthropicEvent.Type == "" {
		return t.writeData([]byte(event.Data))
	}

	return t.handleEvent(anthropicEvent, []byte(event.Data))
}

func (t *AnthropicTransformer) handleEvent(event types.Event, rawJSON []byte) error {
	switch event.Type {
	case "message_start":
		return t.handleMessageStart(event, rawJSON)
	case "content_block_start":
		return t.handleContentBlockStart(event)
	case "content_block_delta":
		return t.handleContentBlockDelta(event)
	case "content_block_stop":
		return t.handleContentBlockStop(event)
	case "message_delta":
		return t.handleMessageDelta(event, rawJSON)
	case "message_stop", "ping":
		return t.writePassthrough(event.Type, rawJSON)
	default:
		return t.writePassthrough(event.Type, rawJSON)
	}
}

func (t *AnthropicTransformer) handleMessageStart(event types.Event, rawJSON []byte) error {
	if event.Message != nil && event.Message.ID != "" {
		t.messageID = event.Message.ID
		t.blockIndex = 0
		t.formatter.SetMessageID(event.Message.ID)
		t.formatter.SetModel(event.Message.Model)
	}

	// Work with raw JSON to preserve all fields and handle flexible usage formats
	var rawData map[string]interface{}
	if err := json.Unmarshal(rawJSON, &rawData); err != nil {
		return t.writePassthrough(event.Type, rawJSON)
	}

	// Normalize usage in message object
	if message, ok := rawData["message"].(map[string]interface{}); ok {
		if usage, ok := message["usage"].(map[string]interface{}); ok {
			// Normalize field names to Anthropic format
			if _, hasInputTokens := usage["input_tokens"]; !hasInputTokens {
				if promptTokens, exists := usage["prompt_tokens"]; exists {
					usage["input_tokens"] = promptTokens
					delete(usage, "prompt_tokens")
				}
			}
			if _, hasOutputTokens := usage["output_tokens"]; !hasOutputTokens {
				if completionTokens, exists := usage["completion_tokens"]; exists {
					usage["output_tokens"] = completionTokens
					delete(usage, "completion_tokens")
				}
			}
			delete(usage, "total_tokens")
		}
	}

	return t.writePassthrough(event.Type, marshalJSON(rawData))
}

func (t *AnthropicTransformer) handleContentBlockStart(event types.Event) error {
	if event.ContentBlock != nil {
		var block types.ContentBlock
		if err := json.Unmarshal(event.ContentBlock, &block); err == nil {
			if block.Type == "thinking" {
				t.inThinking = true
				if event.Index != nil {
					t.thinkingIndex = *event.Index
				}
				t.needThinkingStop = true
				return t.writePassthrough(event.Type, marshalJSON(event))
			}
			if block.Type == "text" {
				t.inText = true
				if event.Index != nil {
					t.textIndex = *event.Index
				}
				t.needTextStop = true
				return t.writePassthrough(event.Type, marshalJSON(event))
			}
		}
	}
	if event.Index != nil && *event.Index >= t.blockIndex {
		t.blockIndex = *event.Index + 1
	}
	return t.writePassthrough(event.Type, marshalJSON(event))
}

func (t *AnthropicTransformer) handleContentBlockDelta(event types.Event) error {
	if t.inThinking {
		return t.handleThinkingDelta(event)
	}
	if t.inText {
		return t.handleTextDelta(event)
	}
	return t.writePassthrough(event.Type, marshalJSON(event))
}

func (t *AnthropicTransformer) handleThinkingDelta(event types.Event) error {
	var delta types.ThinkingDelta
	if err := json.Unmarshal(event.Delta, &delta); err != nil {
		return t.writePassthrough(event.Type, marshalJSON(event))
	}
	if delta.Type != "thinking_delta" {
		return t.writePassthrough(event.Type, marshalJSON(event))
	}

	idx := 0
	if event.Index != nil {
		idx = *event.Index
	}

	chunks := t.processThinking(delta.Thinking, idx)
	for _, chunk := range chunks {
		if err := t.write(chunk); err != nil {
			return err
		}
	}
	return nil
}

func (t *AnthropicTransformer) handleTextDelta(event types.Event) error {
	var delta types.TextDelta
	if err := json.Unmarshal(event.Delta, &delta); err != nil {
		return t.writePassthrough(event.Type, marshalJSON(event))
	}
	if delta.Type != "text_delta" {
		return t.writePassthrough(event.Type, marshalJSON(event))
	}

	idx := 0
	if event.Index != nil {
		idx = *event.Index
	}

	chunks := t.processText(delta.Text, idx)
	for _, chunk := range chunks {
		if err := t.write(chunk); err != nil {
			return err
		}
	}
	return nil
}

func (t *AnthropicTransformer) handleContentBlockStop(event types.Event) error {
	if t.inThinking {
		return t.handleThinkingBlockStop(event)
	}
	if t.inText {
		return t.handleTextBlockStop(event)
	}
	return t.writePassthrough(event.Type, marshalJSON(event))
}

func (t *AnthropicTransformer) handleThinkingBlockStop(event types.Event) error {
	t.inThinking = false
	if t.buf != "" {
		idx := 0
		if event.Index != nil {
			idx = *event.Index
		}
		t.flushRemainingThinking(idx)
	}
	t.buf = ""
	t.state = anthropicStateIdle
	if t.needThinkingStop {
		t.write(t.makeThinkingBlockStop(t.thinkingIndex))
		t.needThinkingStop = false
	}
	return nil
}

func (t *AnthropicTransformer) handleTextBlockStop(event types.Event) error {
	t.inText = false
	if t.buf != "" {
		idx := 0
		if event.Index != nil {
			idx = *event.Index
		}
		t.flushRemainingText(idx)
	}
	t.buf = ""
	t.state = anthropicStateIdle
	if t.needTextStop {
		t.write(t.makeTextBlockStop(t.textIndex))
		t.needTextStop = false
	}
	return nil
}

func (t *AnthropicTransformer) handleMessageDelta(event types.Event, rawJSON []byte) error {
	// Work with raw JSON to preserve all fields and handle flexible usage formats
	var rawData map[string]interface{}
	if err := json.Unmarshal(rawJSON, &rawData); err != nil {
		return t.writePassthrough(event.Type, rawJSON)
	}

	// Normalize usage at top level
	if usage, ok := rawData["usage"].(map[string]interface{}); ok {
		// Normalize field names to Anthropic format
		if _, hasOutputTokens := usage["output_tokens"]; !hasOutputTokens {
			if completionTokens, exists := usage["completion_tokens"]; exists {
				usage["output_tokens"] = completionTokens
				delete(usage, "completion_tokens")
			}
		}
		if _, hasInputTokens := usage["input_tokens"]; !hasInputTokens {
			if promptTokens, exists := usage["prompt_tokens"]; exists {
				usage["input_tokens"] = promptTokens
				delete(usage, "prompt_tokens")
			}
		}
		delete(usage, "total_tokens")
	}

	// Handle stop_reason conversion for tool calls
	if t.toolsEmitted {
		if delta, ok := rawData["delta"].(map[string]interface{}); ok {
			if stopReason, exists := delta["stop_reason"].(string); exists && stopReason == "end_turn" {
				logging.InfoMsg("[%s] Changing stop_reason from 'end_turn' to 'tool_use' due to emitted tool calls", t.messageID)
				delta["stop_reason"] = "tool_use"
				return t.writePassthrough(event.Type, marshalJSON(rawData))
			}
		}
	}

	return t.writePassthrough(event.Type, marshalJSON(rawData))
}

func (t *AnthropicTransformer) processThinking(text string, index int) [][]byte {
	t.buf += text
	var out [][]byte

	for {
		switch t.state {
		case anthropicStateIdle:
			idx := strings.Index(t.buf, "<|tool_calls_section_begin|>")
			if idx < 0 {
				return out
			}
			logging.InfoMsg("[%s] Tool call markup detected in thinking block, transforming to tool_use events", t.messageID)
			if idx > 0 {
				out = append(out, t.makeThinkingDelta(index, t.buf[:idx]))
			}
			t.buf = t.buf[idx+len("<|tool_calls_section_begin|>"):]
			t.state = anthropicStateInSection
			if t.needThinkingStop {
				out = append(out, t.makeThinkingBlockStop(t.thinkingIndex))
				t.needThinkingStop = false
			}

		case anthropicStateInSection:
			idx := strings.Index(t.buf, "<|tool_call_begin|>")
			endIdx := strings.Index(t.buf, "<|tool_calls_section_end|>")

			if endIdx >= 0 && (idx < 0 || endIdx < idx) {
				t.buf = t.buf[endIdx+len("<|tool_calls_section_end|>"):]
				t.state = anthropicStateTrailing
				if t.buf != "" {
					t.thinkingIndex = t.blockIndex
					t.blockIndex++
					out = append(out, t.makeThinkingBlockStart(t.thinkingIndex))
					out = append(out, t.makeThinkingDelta(t.thinkingIndex, t.buf))
					t.needThinkingStop = true
					t.buf = ""
				}
				return out
			}
			if idx < 0 {
				return out
			}
			t.buf = t.buf[idx+len("<|tool_call_begin|>"):]
			t.state = anthropicStateReadingID

		case anthropicStateReadingID:
			argIdx := strings.Index(t.buf, "<|tool_call_argument_begin|>")
			if argIdx < 0 {
				return out
			}
			rawID := strings.TrimSpace(t.buf[:argIdx])
			t.currentID = parseToolCallID(rawID, t.toolIndex)
			name := parseFunctionName(rawID)
			logging.InfoMsg("[%s] Tool call parsed: name=%s, id=%s, blockIndex=%d", t.messageID, name, t.currentID, t.blockIndex+1)
			t.buf = t.buf[argIdx+len("<|tool_call_argument_begin|>"):]
			t.state = anthropicStateReadingArgs
			t.blockIndex++
			out = append(out, t.makeToolUseBlockStart(name))

		case anthropicStateReadingArgs:
			endIdx := strings.Index(t.buf, "<|tool_call_end|>")
			if endIdx < 0 {
				if t.buf != "" {
					out = append(out, t.makeInputJSONDelta(t.buf))
					t.buf = ""
				}
				return out
			}
			args := t.buf[:endIdx]
			if args != "" {
				out = append(out, t.makeInputJSONDelta(args))
			}
			out = append(out, t.makeContentBlockStop())
			t.buf = t.buf[endIdx+len("<|tool_call_end|>"):]
			t.toolIndex++
			t.state = anthropicStateInSection

		case anthropicStateTrailing:
			idx := strings.Index(t.buf, "<|tool_calls_section_begin|>")
			if idx >= 0 {
				if idx > 0 {
					out = append(out, t.makeThinkingDelta(index, t.buf[:idx]))
				}
				t.buf = t.buf[idx+len("<|tool_calls_section_begin|>"):]
				t.state = anthropicStateInSection
				continue
			}
			if t.buf != "" {
				out = append(out, t.makeThinkingDelta(index, t.buf))
			}
			return out
		}
	}
}

func (t *AnthropicTransformer) processText(text string, index int) [][]byte {
	t.buf += text
	var out [][]byte

	for {
		switch t.state {
		case anthropicStateIdle:
			idx := strings.Index(t.buf, "<|tool_calls_section_begin|>")
			if idx < 0 {
				return out
			}
			logging.InfoMsg("[%s] Tool call markup detected in text block, transforming to tool_use events", t.messageID)
			if idx > 0 {
				out = append(out, t.makeTextDelta(index, t.buf[:idx]))
			}
			t.buf = t.buf[idx+len("<|tool_calls_section_begin|>"):]
			t.state = anthropicStateInSection
			if t.needTextStop {
				out = append(out, t.makeTextBlockStop(t.textIndex))
				t.needTextStop = false
				t.blockIndex++
			}

		case anthropicStateInSection:
			idx := strings.Index(t.buf, "<|tool_call_begin|>")
			endIdx := strings.Index(t.buf, "<|tool_calls_section_end|>")

			if endIdx >= 0 && (idx < 0 || endIdx < idx) {
				t.buf = t.buf[endIdx+len("<|tool_calls_section_end|>"):]
				t.state = anthropicStateTrailing
				if t.buf != "" {
					t.textIndex = t.blockIndex
					t.blockIndex++
					out = append(out, t.makeTextBlockStart(t.textIndex))
					out = append(out, t.makeTextDelta(t.textIndex, t.buf))
					t.needTextStop = true
					t.buf = ""
				}
				return out
			}
			if idx < 0 {
				return out
			}
			t.buf = t.buf[idx+len("<|tool_call_begin|>"):]
			t.state = anthropicStateReadingID

		case anthropicStateReadingID:
			argIdx := strings.Index(t.buf, "<|tool_call_argument_begin|>")
			if argIdx < 0 {
				return out
			}
			rawID := strings.TrimSpace(t.buf[:argIdx])
			t.currentID = parseToolCallID(rawID, t.toolIndex)
			name := parseFunctionName(rawID)
			logging.InfoMsg("[%s] Tool call parsed: name=%s, id=%s, blockIndex=%d", t.messageID, name, t.currentID, t.blockIndex+1)
			t.buf = t.buf[argIdx+len("<|tool_call_argument_begin|>"):]
			t.state = anthropicStateReadingArgs
			t.blockIndex++
			out = append(out, t.makeToolUseBlockStart(name))

		case anthropicStateReadingArgs:
			endIdx := strings.Index(t.buf, "<|tool_call_end|>")
			if endIdx < 0 {
				if t.buf != "" {
					out = append(out, t.makeInputJSONDelta(t.buf))
					t.buf = ""
				}
				return out
			}
			args := t.buf[:endIdx]
			if args != "" {
				out = append(out, t.makeInputJSONDelta(args))
			}
			out = append(out, t.makeContentBlockStop())
			t.buf = t.buf[endIdx+len("<|tool_call_end|>"):]
			t.toolIndex++
			t.state = anthropicStateInSection

		case anthropicStateTrailing:
			idx := strings.Index(t.buf, "<|tool_calls_section_begin|>")
			if idx >= 0 {
				if idx > 0 {
					out = append(out, t.makeTextDelta(index, t.buf[:idx]))
				}
				t.buf = t.buf[idx+len("<|tool_calls_section_begin|>"):]
				t.state = anthropicStateInSection
				continue
			}
			if t.buf != "" {
				out = append(out, t.makeTextDelta(index, t.buf))
			}
			return out
		}
	}
}

func (t *AnthropicTransformer) makeThinkingDelta(index int, thinking string) []byte {
	event := types.Event{
		Type:  "content_block_delta",
		Index: intPtr(index),
		Delta: json.RawMessage(fmt.Sprintf(`{"type":"thinking_delta","thinking":%q}`, thinking)),
	}
	return serializeAnthropicEvent(event)
}

func (t *AnthropicTransformer) makeThinkingBlockStart(index int) []byte {
	event := types.Event{
		Type:         "content_block_start",
		Index:        intPtr(index),
		ContentBlock: json.RawMessage(`{"type":"thinking","thinking":""}`),
	}
	return serializeAnthropicEvent(event)
}

func (t *AnthropicTransformer) makeThinkingBlockStop(index int) []byte {
	event := types.Event{
		Type:  "content_block_stop",
		Index: intPtr(index),
	}
	return serializeAnthropicEvent(event)
}

func (t *AnthropicTransformer) makeTextDelta(index int, text string) []byte {
	event := types.Event{
		Type:  "content_block_delta",
		Index: intPtr(index),
		Delta: json.RawMessage(fmt.Sprintf(`{"type":"text_delta","text":%q}`, text)),
	}
	return serializeAnthropicEvent(event)
}

func (t *AnthropicTransformer) makeTextBlockStart(index int) []byte {
	event := types.Event{
		Type:         "content_block_start",
		Index:        intPtr(index),
		ContentBlock: json.RawMessage(`{"type":"text","text":""}`),
	}
	return serializeAnthropicEvent(event)
}

func (t *AnthropicTransformer) makeTextBlockStop(index int) []byte {
	event := types.Event{
		Type:  "content_block_stop",
		Index: intPtr(index),
	}
	return serializeAnthropicEvent(event)
}

func (t *AnthropicTransformer) makeToolUseBlockStart(name string) []byte {
	t.toolsEmitted = true
	event := types.Event{
		Type:         "content_block_start",
		Index:        intPtr(t.blockIndex),
		ContentBlock: json.RawMessage(fmt.Sprintf(`{"type":"tool_use","id":%q,"name":%q,"input":{}}`, t.currentID, name)),
	}
	return serializeAnthropicEvent(event)
}

func (t *AnthropicTransformer) makeInputJSONDelta(partialJSON string) []byte {
	event := types.Event{
		Type:  "content_block_delta",
		Index: intPtr(t.blockIndex),
		Delta: json.RawMessage(fmt.Sprintf(`{"type":"input_json_delta","partial_json":%q}`, partialJSON)),
	}
	return serializeAnthropicEvent(event)
}

func (t *AnthropicTransformer) makeContentBlockStop() []byte {
	event := types.Event{
		Type:  "content_block_stop",
		Index: intPtr(t.blockIndex),
	}
	return serializeAnthropicEvent(event)
}

func (t *AnthropicTransformer) flushRemainingThinking(index int) {
	event := types.Event{
		Type:  "content_block_delta",
		Index: intPtr(index),
		Delta: json.RawMessage(fmt.Sprintf(`{"type":"thinking_delta","thinking":%q}`, t.buf)),
	}
	t.write(serializeAnthropicEvent(event))
}

func (t *AnthropicTransformer) flushRemainingText(index int) {
	event := types.Event{
		Type:  "content_block_delta",
		Index: intPtr(index),
		Delta: json.RawMessage(fmt.Sprintf(`{"type":"text_delta","text":%q}`, t.buf)),
	}
	t.write(serializeAnthropicEvent(event))
}

func (t *AnthropicTransformer) write(data []byte) error {
	if len(data) == 0 {
		return nil
	}
	_, err := t.sseWriter.WriteRaw(data)
	return err
}

func (t *AnthropicTransformer) writeData(data []byte) error {
	return t.sseWriter.WriteData(data)
}

func (t *AnthropicTransformer) writePassthrough(eventType string, data []byte) error {
	if len(data) == 0 {
		return nil
	}
	return t.sseWriter.WriteEvent(eventType, data)
}

func (t *AnthropicTransformer) Flush() error {
	if t.buf == "" {
		return nil
	}

	if t.inThinking {
		t.flushRemainingThinking(t.thinkingIndex)
	} else if t.inText {
		t.flushRemainingText(t.textIndex)
	} else {
		switch t.state {
		case anthropicStateIdle, anthropicStateTrailing:
			t.thinkingIndex = t.blockIndex
			t.blockIndex++
			t.write(t.makeThinkingBlockStart(t.thinkingIndex))
			t.flushRemainingThinking(t.thinkingIndex)
			t.write(t.makeThinkingBlockStop(t.thinkingIndex))
		case anthropicStateInSection, anthropicStateReadingID, anthropicStateReadingArgs:
			t.thinkingIndex = t.blockIndex
			t.blockIndex++
			t.write(t.makeThinkingBlockStart(t.thinkingIndex))
			debugContent := fmt.Sprintf("[INCOMPLETE TOOL CALL - state=%d] %s", t.state, t.buf)
			event := types.Event{
				Type:  "content_block_delta",
				Index: intPtr(t.thinkingIndex),
				Delta: json.RawMessage(fmt.Sprintf(`{"type":"thinking_delta","thinking":%q}`, debugContent)),
			}
			t.write(serializeAnthropicEvent(event))
			t.write(t.makeThinkingBlockStop(t.thinkingIndex))
		}
	}

	t.buf = ""
	return nil
}

func (t *AnthropicTransformer) Close() error {
	return t.Flush()
}
