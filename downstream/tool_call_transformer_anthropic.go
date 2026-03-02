package downstream

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"

	"ai-proxy/logging"
	"github.com/tmaxmax/go-sse"
)

func intPtr(i int) *int {
	return &i
}

const (
	anthropicTokSectionBegin = "<|tool_calls_section_begin|>"
	anthropicTokCallBegin    = "<|tool_call_begin|>"
	anthropicTokArgBegin     = "<|tool_call_argument_begin|>"
	anthropicTokCallEnd      = "<|tool_call_end|>"
	anthropicTokSectionEnd   = "<|tool_calls_section_end|>"
)

type anthropicState int

const (
	anthropicStateIdle anthropicState = iota
	anthropicStateInSection
	anthropicStateReadingID
	anthropicStateReadingArgs
	anthropicStateTrailing
)

type AnthropicToolCallTransformer struct {
	output           io.Writer
	state            anthropicState
	buf              string
	toolIndex        int
	blockIndex       int
	messageID        string
	currentID        string
	messageSent      bool
	inThinking       bool
	inText           bool
	thinkingIndex    int
	textIndex        int
	needThinkingStop bool
	needTextStop     bool
	processAsText    bool
}

func NewAnthropicToolCallTransformer(output io.Writer) *AnthropicToolCallTransformer {
	return &AnthropicToolCallTransformer{
		output: output,
		state:  anthropicStateIdle,
	}
}

func (t *AnthropicToolCallTransformer) Transform(event *sse.Event) {
	if event.Data == "" {
		return
	}

	var anthropicEvent AnthropicEvent
	if err := json.Unmarshal([]byte(event.Data), &anthropicEvent); err != nil {
		t.writeSSE([]byte("event: error\ndata: {\"type\": \"error\", \"error\": {\"type\": \"invalid_request_error\", \"message\": \"Failed to parse event\"}}\n\n"))
		return
	}

	switch anthropicEvent.Type {
	case "message_start":
		t.handleMessageStart(&anthropicEvent)
	case "content_block_start":
		t.handleContentBlockStart(&anthropicEvent)
	case "content_block_delta":
		t.handleContentBlockDelta(&anthropicEvent)
	case "content_block_stop":
		t.handleContentBlockStop(&anthropicEvent)
	case "message_delta", "message_stop", "ping":
		t.writeEvent(&anthropicEvent)
	default:
		t.writeEvent(&anthropicEvent)
	}
}

func (t *AnthropicToolCallTransformer) handleMessageStart(event *AnthropicEvent) {
	if event.Message != nil {
		t.messageID = event.Message.ID
		t.blockIndex = 0
	}
	t.writeEvent(event)
}

func (t *AnthropicToolCallTransformer) handleContentBlockStart(event *AnthropicEvent) {
	var block AnthropicContentBlock
	if err := json.Unmarshal(event.ContentBlock, &block); err == nil {
		if block.Type == "thinking" {
			t.inThinking = true
			if event.Index != nil {
				t.thinkingIndex = *event.Index
			}
			t.needThinkingStop = true
			t.writeEvent(event)
			return
		}
		if block.Type == "text" {
			t.inText = true
			if event.Index != nil {
				t.textIndex = *event.Index
			}
			t.needTextStop = true
			t.writeEvent(event)
			return
		}
	}
	t.writeEvent(event)
	if event.Index != nil && *event.Index >= t.blockIndex {
		t.blockIndex = *event.Index + 1
	}
}

func (t *AnthropicToolCallTransformer) handleContentBlockDelta(event *AnthropicEvent) {
	if t.inThinking {
		var delta ThinkingDelta
		if err := json.Unmarshal(event.Delta, &delta); err != nil {
			t.writeEvent(event)
			return
		}
		if delta.Type == "thinking_delta" {
			t.processAsText = false
			idx := 0
			if event.Index != nil {
				idx = *event.Index
			}
			chunks, err := t.processThinking(delta.Thinking, idx)
			if err != nil {
				t.writeEvent(event)
				return
			}
			for _, chunk := range chunks {
				t.writeSSE(chunk)
			}
			return
		}
	}

	if t.inText {
		var delta TextDelta
		if err := json.Unmarshal(event.Delta, &delta); err != nil {
			t.writeEvent(event)
			return
		}
		if delta.Type == "text_delta" {
			t.processAsText = true
			idx := 0
			if event.Index != nil {
				idx = *event.Index
			}
			chunks, err := t.processText(delta.Text, idx)
			if err != nil {
				t.writeEvent(event)
				return
			}
			for _, chunk := range chunks {
				t.writeSSE(chunk)
			}
			return
		}
	}

	t.writeEvent(event)
}

func (t *AnthropicToolCallTransformer) handleContentBlockStop(event *AnthropicEvent) {
	if t.inThinking {
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
			t.writeEvent(&AnthropicEvent{
				Type:  "content_block_stop",
				Index: intPtr(t.thinkingIndex),
			})
			t.needThinkingStop = false
		}
		return
	}
	if t.inText {
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
			t.writeEvent(&AnthropicEvent{
				Type:  "content_block_stop",
				Index: intPtr(t.textIndex),
			})
			t.needTextStop = false
		}
		return
	}
	t.writeEvent(event)
}

func (t *AnthropicToolCallTransformer) processThinking(text string, index int) ([][]byte, error) {
	t.buf += text
	var out [][]byte

	for {
		switch t.state {
		case anthropicStateIdle:
			idx := strings.Index(t.buf, anthropicTokSectionBegin)
			if idx < 0 {
				return out, nil
			}
			if idx > 0 {
				out = append(out, t.makeThinkingDelta(index, t.buf[:idx]))
			}
			t.buf = t.buf[idx+len(anthropicTokSectionBegin):]
			t.state = anthropicStateInSection
			if t.needThinkingStop {
				out = append(out, t.makeThinkingBlockStop(t.thinkingIndex))
				t.needThinkingStop = false
			}

		case anthropicStateInSection:
			idx := strings.Index(t.buf, anthropicTokCallBegin)
			endIdx := strings.Index(t.buf, anthropicTokSectionEnd)

			if endIdx >= 0 && (idx < 0 || endIdx < idx) {
				t.buf = t.buf[endIdx+len(anthropicTokSectionEnd):]
				t.state = anthropicStateTrailing
				if t.buf != "" {
					t.thinkingIndex = t.blockIndex
					t.blockIndex++
					out = append(out, t.makeThinkingBlockStart(t.thinkingIndex))
					out = append(out, t.makeThinkingDelta(t.thinkingIndex, t.buf))
					t.needThinkingStop = true
					t.buf = ""
				}
				return out, nil
			}
			if idx < 0 {
				return out, nil
			}
			t.buf = t.buf[idx+len(anthropicTokCallBegin):]
			t.state = anthropicStateReadingID

		case anthropicStateReadingID:
			argIdx := strings.Index(t.buf, anthropicTokArgBegin)
			if argIdx < 0 {
				return out, nil
			}
			rawID := strings.TrimSpace(t.buf[:argIdx])
			t.currentID = t.parseToolCallID(rawID, t.toolIndex)
			name := t.parseFunctionName(rawID)
			t.buf = t.buf[argIdx+len(anthropicTokArgBegin):]
			t.state = anthropicStateReadingArgs
			out = append(out, t.makeToolUseBlockStart(name))

		case anthropicStateReadingArgs:
			endIdx := strings.Index(t.buf, anthropicTokCallEnd)
			if endIdx < 0 {
				if t.buf != "" {
					out = append(out, t.makeInputJSONDelta(t.buf))
					t.buf = ""
				}
				return out, nil
			}
			args := t.buf[:endIdx]
			if args != "" {
				out = append(out, t.makeInputJSONDelta(args))
			}
			out = append(out, t.makeContentBlockStop())
			t.buf = t.buf[endIdx+len(anthropicTokCallEnd):]
			t.toolIndex++
			t.blockIndex++
			t.state = anthropicStateInSection

		case anthropicStateTrailing:
			idx := strings.Index(t.buf, anthropicTokSectionBegin)
			if idx >= 0 {
				if idx > 0 {
					out = append(out, t.makeThinkingDelta(index, t.buf[:idx]))
				}
				t.buf = t.buf[idx+len(anthropicTokSectionBegin):]
				t.state = anthropicStateInSection
				continue
			}
			if t.buf != "" {
				out = append(out, t.makeThinkingDelta(index, t.buf))
			}
			return out, nil
		}
	}
}

func (t *AnthropicToolCallTransformer) processText(text string, index int) ([][]byte, error) {
	t.buf += text
	var out [][]byte

	for {
		switch t.state {
		case anthropicStateIdle:
			idx := strings.Index(t.buf, anthropicTokSectionBegin)
			if idx < 0 {
				return out, nil
			}
			if idx > 0 {
				out = append(out, t.makeTextDelta(index, t.buf[:idx]))
			}
			t.buf = t.buf[idx+len(anthropicTokSectionBegin):]
			t.state = anthropicStateInSection
			if t.needTextStop {
				out = append(out, t.makeTextBlockStop(t.textIndex))
				t.needTextStop = false
			}

		case anthropicStateInSection:
			idx := strings.Index(t.buf, anthropicTokCallBegin)
			endIdx := strings.Index(t.buf, anthropicTokSectionEnd)

			if endIdx >= 0 && (idx < 0 || endIdx < idx) {
				t.buf = t.buf[endIdx+len(anthropicTokSectionEnd):]
				t.state = anthropicStateTrailing
				if t.buf != "" {
					t.textIndex = t.blockIndex
					t.blockIndex++
					out = append(out, t.makeTextBlockStart(t.textIndex))
					out = append(out, t.makeTextDelta(t.textIndex, t.buf))
					t.needTextStop = true
					t.buf = ""
				}
				return out, nil
			}
			if idx < 0 {
				return out, nil
			}
			t.buf = t.buf[idx+len(anthropicTokCallBegin):]
			t.state = anthropicStateReadingID

		case anthropicStateReadingID:
			argIdx := strings.Index(t.buf, anthropicTokArgBegin)
			if argIdx < 0 {
				return out, nil
			}
			rawID := strings.TrimSpace(t.buf[:argIdx])
			t.currentID = t.parseToolCallID(rawID, t.toolIndex)
			name := t.parseFunctionName(rawID)
			t.buf = t.buf[argIdx+len(anthropicTokArgBegin):]
			t.state = anthropicStateReadingArgs
			out = append(out, t.makeToolUseBlockStart(name))

		case anthropicStateReadingArgs:
			endIdx := strings.Index(t.buf, anthropicTokCallEnd)
			if endIdx < 0 {
				if t.buf != "" {
					out = append(out, t.makeInputJSONDelta(t.buf))
					t.buf = ""
				}
				return out, nil
			}
			args := t.buf[:endIdx]
			if args != "" {
				out = append(out, t.makeInputJSONDelta(args))
			}
			out = append(out, t.makeContentBlockStop())
			t.buf = t.buf[endIdx+len(anthropicTokCallEnd):]
			t.toolIndex++
			t.blockIndex++
			t.state = anthropicStateInSection

		case anthropicStateTrailing:
			idx := strings.Index(t.buf, anthropicTokSectionBegin)
			if idx >= 0 {
				if idx > 0 {
					out = append(out, t.makeTextDelta(index, t.buf[:idx]))
				}
				t.buf = t.buf[idx+len(anthropicTokSectionBegin):]
				t.state = anthropicStateInSection
				continue
			}
			if t.buf != "" {
				out = append(out, t.makeTextDelta(index, t.buf))
			}
			return out, nil
		}
	}
}

func (t *AnthropicToolCallTransformer) makeThinkingDelta(index int, thinking string) []byte {
	delta := ThinkingDelta{
		Type:     "thinking_delta",
		Thinking: thinking,
	}
	deltaJSON, _ := json.Marshal(delta)
	event := AnthropicEvent{
		Type:  "content_block_delta",
		Index: intPtr(index),
		Delta: deltaJSON,
	}
	return t.serializeEvent(event)
}

func (t *AnthropicToolCallTransformer) makeThinkingBlockStart(index int) []byte {
	block := AnthropicContentBlock{
		Type:     "thinking",
		Thinking: "",
	}
	blockJSON, _ := json.Marshal(block)
	event := AnthropicEvent{
		Type:         "content_block_start",
		Index:        intPtr(index),
		ContentBlock: blockJSON,
	}
	return t.serializeEvent(event)
}

func (t *AnthropicToolCallTransformer) makeThinkingBlockStop(index int) []byte {
	event := AnthropicEvent{
		Type:  "content_block_stop",
		Index: intPtr(index),
	}
	return t.serializeEvent(event)
}

func (t *AnthropicToolCallTransformer) makeTextDelta(index int, text string) []byte {
	delta := TextDelta{
		Type: "text_delta",
		Text: text,
	}
	deltaJSON, _ := json.Marshal(delta)
	event := AnthropicEvent{
		Type:  "content_block_delta",
		Index: intPtr(index),
		Delta: deltaJSON,
	}
	return t.serializeEvent(event)
}

func (t *AnthropicToolCallTransformer) makeTextBlockStart(index int) []byte {
	block := AnthropicContentBlock{
		Type: "text",
		Text: "",
	}
	blockJSON, _ := json.Marshal(block)
	event := AnthropicEvent{
		Type:         "content_block_start",
		Index:        intPtr(index),
		ContentBlock: blockJSON,
	}
	return t.serializeEvent(event)
}

func (t *AnthropicToolCallTransformer) makeTextBlockStop(index int) []byte {
	event := AnthropicEvent{
		Type:  "content_block_stop",
		Index: intPtr(index),
	}
	return t.serializeEvent(event)
}

func (t *AnthropicToolCallTransformer) makeToolUseBlockStart(name string) []byte {
	toolBlock := AnthropicContentBlock{
		Type:  "tool_use",
		ID:    t.currentID,
		Name:  name,
		Input: []byte("{}"),
	}
	blockJSON, _ := json.Marshal(toolBlock)
	event := AnthropicEvent{
		Type:         "content_block_start",
		Index:        intPtr(t.blockIndex),
		ContentBlock: blockJSON,
	}
	return t.serializeEvent(event)
}

func (t *AnthropicToolCallTransformer) makeInputJSONDelta(partialJSON string) []byte {
	delta := InputJSONDelta{
		Type:        "input_json_delta",
		PartialJSON: partialJSON,
	}
	deltaJSON, _ := json.Marshal(delta)
	event := AnthropicEvent{
		Type:  "content_block_delta",
		Index: intPtr(t.blockIndex),
		Delta: deltaJSON,
	}
	return t.serializeEvent(event)
}

func (t *AnthropicToolCallTransformer) makeContentBlockStop() []byte {
	event := AnthropicEvent{
		Type:  "content_block_stop",
		Index: intPtr(t.blockIndex),
	}
	return t.serializeEvent(event)
}

func (t *AnthropicToolCallTransformer) flushRemainingThinking(index int) {
	event := AnthropicEvent{
		Type:  "content_block_delta",
		Index: intPtr(index),
		Delta: []byte(fmt.Sprintf(`{"type": "thinking_delta", "thinking": %q}`, t.buf)),
	}
	t.writeEvent(&event)
}

func (t *AnthropicToolCallTransformer) flushRemainingText(index int) {
	event := AnthropicEvent{
		Type:  "content_block_delta",
		Index: intPtr(index),
		Delta: []byte(fmt.Sprintf(`{"type": "text_delta", "text": %q}`, t.buf)),
	}
	t.writeEvent(&event)
}

func (t *AnthropicToolCallTransformer) writeEvent(event *AnthropicEvent) {
	data := t.serializeEvent(*event)
	if len(data) > 0 {
		t.writeSSE(data)
	}
}

func (t *AnthropicToolCallTransformer) serializeEvent(event AnthropicEvent) []byte {
	data, err := json.Marshal(event)
	if err != nil {
		logging.ErrorMsg("Failed to serialize event: %v", err)
		return nil
	}
	return []byte(fmt.Sprintf("event: %s\ndata: %s\n\n", event.Type, string(data)))
}

func (t *AnthropicToolCallTransformer) writeSSE(data []byte) {
	if len(data) == 0 {
		return
	}
	if _, err := t.output.Write(data); err != nil {
		logging.ErrorMsg("Failed to write to output: %v", err)
	}
}

func (t *AnthropicToolCallTransformer) parseToolCallID(raw string, index int) string {
	raw = strings.TrimSpace(raw)
	if strings.HasPrefix(raw, "toolu_") {
		return raw
	}
	return fmt.Sprintf("toolu_%d_%d", index, time.Now().UnixMilli())
}

func (t *AnthropicToolCallTransformer) parseFunctionName(raw string) string {
	raw = strings.TrimSpace(raw)
	if i := strings.Index(raw, "."); i >= 0 {
		raw = raw[i+1:]
	}
	if i := strings.LastIndex(raw, ":"); i >= 0 {
		raw = raw[:i]
	}
	return raw
}

func (t *AnthropicToolCallTransformer) Close() {}

func (t *AnthropicToolCallTransformer) Flush() {
	if t.buf != "" && t.inThinking {
		logging.ErrorMsg("Anthropic transformer: unflushed buffer on close")
	}
}
