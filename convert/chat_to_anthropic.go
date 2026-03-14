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

type ChatToAnthropicConverter struct{}

func NewChatToAnthropicConverter() *ChatToAnthropicConverter {
	return &ChatToAnthropicConverter{}
}

func (c *ChatToAnthropicConverter) Convert(body []byte) ([]byte, error) {
	var openReq types.ChatCompletionRequest
	if err := json.Unmarshal(body, &openReq); err != nil {
		return nil, err
	}

	anthReq := types.MessageRequest{
		Model:       openReq.Model,
		MaxTokens:   openReq.MaxTokens,
		Stream:      openReq.Stream,
		Temperature: openReq.Temperature,
		TopP:        openReq.TopP,
		System:      openReq.System,
		Messages:    OpenAIMessagesToAnthropic(openReq.Messages),
		Tools:       OpenAIToolsToAnthropic(openReq.Tools),
	}

	if openReq.MaxTokens == 0 {
		anthReq.MaxTokens = 4096
	}

	return json.Marshal(anthReq)
}

type toolCallState struct {
	id        string
	name      string
	arguments strings.Builder
}

type ChatToAnthropicTransformer struct {
	w              io.Writer
	messageID      string
	model          string
	contentIndex   int
	blockCount     int
	toolCallIndex  int
	toolCalls      map[int]*toolCallState
	initialized    bool
	sentStart      bool
	sentBlockStart bool
	inputTokens    int
	outputTokens   int
	inTextBlock    bool
}

func NewChatToAnthropicTransformer(w io.Writer) *ChatToAnthropicTransformer {
	return &ChatToAnthropicTransformer{
		w:         w,
		toolCalls: make(map[int]*toolCallState),
	}
}

func (t *ChatToAnthropicTransformer) Transform(event *sse.Event) error {
	if event.Data == "" || event.Data == "[DONE]" {
		return nil
	}

	var chunk types.Chunk
	if err := json.Unmarshal([]byte(event.Data), &chunk); err != nil {
		return fmt.Errorf("parse chunk: %w", err)
	}

	if !t.initialized {
		t.messageID = chunk.ID
		t.model = chunk.Model
		t.initialized = true
	}

	if chunk.Usage != nil {
		t.inputTokens = chunk.Usage.PromptTokens
		t.outputTokens = chunk.Usage.CompletionTokens
	}

	if len(chunk.Choices) == 0 {
		return nil
	}

	choice := chunk.Choices[0]

	if !t.sentStart {
		if err := t.writeMessageStart(); err != nil {
			return err
		}
		t.sentStart = true
	}

	return t.processChoice(choice)
}

func (t *ChatToAnthropicTransformer) processChoice(choice types.Choice) error {
	delta := choice.Delta

	if delta.Role != "" && len(delta.ToolCalls) == 0 && delta.Content == "" {
		return nil
	}

	if len(delta.ToolCalls) > 0 {
		return t.processToolCalls(delta.ToolCalls)
	}

	if delta.Content != "" {
		return t.processContent(delta.Content)
	}

	if choice.FinishReason != nil {
		return t.processFinish(*choice.FinishReason)
	}

	return nil
}

func (t *ChatToAnthropicTransformer) ensureTextBlockStart() error {
	if t.sentBlockStart && t.inTextBlock {
		return nil
	}
	if t.sentBlockStart {
		event := types.Event{
			Type:  "content_block_stop",
			Index: intPtr(t.contentIndex),
		}
		if err := t.writeEvent(event); err != nil {
			return err
		}
		t.contentIndex++
	}
	event := types.Event{
		Type:         "content_block_start",
		Index:        intPtr(t.contentIndex),
		ContentBlock: json.RawMessage(`{"type":"text","text":""}`),
	}
	if err := t.writeEvent(event); err != nil {
		return err
	}
	t.sentBlockStart = true
	t.inTextBlock = true
	return nil
}

func (t *ChatToAnthropicTransformer) processContent(content string) error {
	if content == "" {
		return nil
	}

	if err := t.ensureTextBlockStart(); err != nil {
		return err
	}

	event := types.Event{
		Type:  "content_block_delta",
		Index: intPtr(t.contentIndex),
		Delta: json.RawMessage(fmt.Sprintf(`{"type":"text_delta","text":%q}`, content)),
	}
	return t.writeEvent(event)
}

func (t *ChatToAnthropicTransformer) processToolCalls(toolCalls []types.ToolCall) error {
	for _, tc := range toolCalls {
		idx := tc.Index
		if idx < 0 {
			idx = t.toolCallIndex
		}

		if _, exists := t.toolCalls[idx]; !exists {
			if t.sentBlockStart {
				event := types.Event{
					Type:  "content_block_stop",
					Index: intPtr(t.contentIndex),
				}
				if err := t.writeEvent(event); err != nil {
					return err
				}
				t.contentIndex++
			}
			t.inTextBlock = false

			t.toolCalls[idx] = &toolCallState{
				id:   tc.ID,
				name: tc.Function.Name,
			}
			t.toolCallIndex++

			event := types.Event{
				Type:         "content_block_start",
				Index:        intPtr(t.contentIndex),
				ContentBlock: json.RawMessage(fmt.Sprintf(`{"type":"tool_use","id":%q,"name":%q,"input":{}}`, tc.ID, tc.Function.Name)),
			}
			if err := t.writeEvent(event); err != nil {
				return err
			}
			t.sentBlockStart = true
		}

		if tc.Function.Arguments != "" {
			t.toolCalls[idx].arguments.WriteString(tc.Function.Arguments)

			event := types.Event{
				Type:  "content_block_delta",
				Index: intPtr(t.contentIndex),
				Delta: json.RawMessage(fmt.Sprintf(`{"type":"input_json_delta","partial_json":%q}`, tc.Function.Arguments)),
			}
			if err := t.writeEvent(event); err != nil {
				return err
			}
		}
	}

	return nil
}

func (t *ChatToAnthropicTransformer) processFinish(reason string) error {
	if t.sentBlockStart {
		event := types.Event{
			Type:  "content_block_stop",
			Index: intPtr(t.contentIndex),
		}
		if err := t.writeEvent(event); err != nil {
			return err
		}
	}

	stopReason := "end_turn"
	if reason == "tool_calls" {
		stopReason = "tool_use"
	} else if reason == "length" {
		stopReason = "max_tokens"
	}

	deltaData := map[string]interface{}{
		"stop_reason": stopReason,
	}
	if t.outputTokens > 0 {
		deltaData["output_tokens"] = t.outputTokens
	}
	deltaJSON, _ := json.Marshal(deltaData)

	event := types.Event{
		Type:  "message_delta",
		Delta: deltaJSON,
	}
	if err := t.writeEvent(event); err != nil {
		return err
	}

	return t.writeMessageStop()
}

func (t *ChatToAnthropicTransformer) writeMessageStart() error {
	msgData := map[string]interface{}{
		"id":    t.messageID,
		"type":  "message",
		"role":  "assistant",
		"model": t.model,
		"content": []map[string]interface{}{
			{"type": "text", "text": ""},
		},
	}
	if t.inputTokens > 0 {
		msgData["usage"] = map[string]int{
			"input_tokens":  t.inputTokens,
			"output_tokens": 0,
		}
	}

	event := types.Event{
		Type:    "message_start",
		Message: &types.MessageInfo{},
	}

	msgJSON, _ := json.Marshal(msgData)
	event.Message = &types.MessageInfo{
		ID:      t.messageID,
		Type:    "message",
		Role:    "assistant",
		Model:   t.model,
		Content: []types.ContentBlock{{Type: "text", Text: ""}},
	}

	var rawData map[string]interface{}
	json.Unmarshal(msgJSON, &rawData)
	rawMsg, _ := json.Marshal(rawData)

	_, err := t.w.Write([]byte("event: message_start\ndata: "))
	if err != nil {
		return err
	}
	_, err = t.w.Write(rawMsg)
	if err != nil {
		return err
	}
	_, err = t.w.Write([]byte("\n\n"))
	return err
}

func (t *ChatToAnthropicTransformer) writeMessageStop() error {
	_, err := t.w.Write([]byte("event: message_stop\ndata: {}\n\n"))
	return err
}

func (t *ChatToAnthropicTransformer) writeEvent(event types.Event) error {
	data, err := json.Marshal(event)
	if err != nil {
		return err
	}

	var eventType string
	switch event.Type {
	case "content_block_start":
		eventType = "content_block_start"
	case "content_block_delta":
		eventType = "content_block_delta"
	case "content_block_stop":
		eventType = "content_block_stop"
	case "message_delta":
		eventType = "message_delta"
	default:
		eventType = event.Type
	}

	_, err = fmt.Fprintf(t.w, "event: %s\ndata: %s\n\n", eventType, string(data))
	return err
}

func (t *ChatToAnthropicTransformer) Flush() error {
	return nil
}

func (t *ChatToAnthropicTransformer) Close() error {
	return t.Flush()
}

func intPtr(i int) *int {
	return &i
}

func generateMessageID() string {
	return fmt.Sprintf("msg_%d_%d", time.Now().UnixNano()/1000000, time.Now().Nanosecond())
}

func convertToolChoiceToAnthropic(toolChoice interface{}) *types.ToolChoice {
	if toolChoice == nil {
		return nil
	}

	switch v := toolChoice.(type) {
	case string:
		switch v {
		case "none":
			return nil
		case "auto":
			return &types.ToolChoice{Type: "auto"}
		case "required":
			return &types.ToolChoice{Type: "any"}
		default:
			return &types.ToolChoice{Type: "auto"}
		}
	case map[string]interface{}:
		objType, _ := v["type"].(string)
		if objType == "function" {
			if fn, ok := v["function"].(map[string]interface{}); ok {
				if name, ok := fn["name"].(string); ok {
					return &types.ToolChoice{
						Type: "tool",
						Name: name,
					}
				}
			}
		}
		return &types.ToolChoice{Type: "auto"}
	default:
		return &types.ToolChoice{Type: "auto"}
	}
}
