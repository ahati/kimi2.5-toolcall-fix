package convert

import (
	"encoding/json"
	"io"

	"ai-proxy/types"

	"github.com/tmaxmax/go-sse"
)

// ChatToAnthropicConverter converts OpenAI Chat Completions requests to Anthropic Messages format.
type ChatToAnthropicConverter struct{}

// NewChatToAnthropicConverter creates a new converter.
func NewChatToAnthropicConverter() *ChatToAnthropicConverter {
	return &ChatToAnthropicConverter{}
}

// Convert transforms OpenAI ChatCompletionRequest to Anthropic MessageRequest.
func (c *ChatToAnthropicConverter) Convert(body []byte) ([]byte, error) {
	var openReq types.ChatCompletionRequest
	if err := json.Unmarshal(body, &openReq); err != nil {
		return nil, err
	}
	return c.convertRequest(&openReq)
}

func (c *ChatToAnthropicConverter) convertRequest(openReq *types.ChatCompletionRequest) ([]byte, error) {
	anthReq := types.MessageRequest{
		Model:       openReq.Model,
		MaxTokens:   c.getMaxTokens(openReq.MaxTokens),
		Stream:      openReq.Stream,
		Temperature: openReq.Temperature,
		TopP:        openReq.TopP,
	}
	if openReq.System != "" {
		anthReq.System = openReq.System
	}
	anthReq.Messages = c.convertMessages(openReq.Messages)
	anthReq.Tools = c.convertTools(openReq.Tools)
	return json.Marshal(anthReq)
}

func (c *ChatToAnthropicConverter) getMaxTokens(maxTokens int) int {
	if maxTokens == 0 {
		return 4096
	}
	return maxTokens
}

func (c *ChatToAnthropicConverter) convertMessages(messages []types.Message) []types.MessageInput {
	result := make([]types.MessageInput, 0, len(messages))
	for _, msg := range messages {
		converted := c.convertMessage(msg)
		result = append(result, converted...)
	}
	return result
}

func (c *ChatToAnthropicConverter) convertMessage(msg types.Message) []types.MessageInput {
	role := c.mapRole(msg.Role)
	if role == "" {
		return nil
	}
	content := c.extractContent(msg.Content)
	if msg.ToolCalls != nil {
		return c.convertToolCalls(role, content, msg.ToolCalls)
	}
	if msg.ToolCallID != "" {
		return c.convertToolResult(role, msg.ToolCallID, content)
	}
	if content == "" {
		return nil
	}
	return []types.MessageInput{{Role: role, Content: content}}
}

func (c *ChatToAnthropicConverter) mapRole(role string) string {
	switch role {
	case "user":
		return "user"
	case "assistant":
		return "assistant"
	case "system":
		return ""
	case "tool":
		return "user"
	default:
		return "user"
	}
}

func (c *ChatToAnthropicConverter) extractContent(content interface{}) string {
	if content == nil {
		return ""
	}
	if s, ok := content.(string); ok {
		return s
	}
	return ""
}

func (c *ChatToAnthropicConverter) convertToolCalls(role string, text string, toolCalls []types.ToolCall) []types.MessageInput {
	blocks := make([]map[string]interface{}, 0, len(toolCalls)+1)
	if text != "" {
		blocks = append(blocks, map[string]interface{}{"type": "text", "text": text})
	}
	for _, tc := range toolCalls {
		var inputObj map[string]interface{}
		if tc.Function.Arguments != "" {
			json.Unmarshal([]byte(tc.Function.Arguments), &inputObj)
		}
		blocks = append(blocks, map[string]interface{}{
			"type":  "tool_use",
			"id":    tc.ID,
			"name":  tc.Function.Name,
			"input": inputObj,
		})
	}
	return []types.MessageInput{{Role: role, Content: blocks}}
}

func (c *ChatToAnthropicConverter) convertToolResult(role string, toolCallID string, output string) []types.MessageInput {
	blocks := []map[string]interface{}{{
		"type":        "tool_result",
		"tool_use_id": toolCallID,
		"content":     output,
	}}
	return []types.MessageInput{{Role: role, Content: blocks}}
}

func (c *ChatToAnthropicConverter) convertTools(tools []types.Tool) []types.ToolDef {
	if len(tools) == 0 {
		return nil
	}
	result := make([]types.ToolDef, 0, len(tools))
	for _, tool := range tools {
		if tool.Type == "function" {
			result = append(result, types.ToolDef{
				Name:        tool.Function.Name,
				Description: tool.Function.Description,
				InputSchema: tool.Function.Parameters,
			})
		}
	}
	return result
}

// ChatToAnthropicTransformer converts OpenAI Chat Completions streaming responses to Anthropic format.
type ChatToAnthropicTransformer struct {
	w          io.Writer
	formatter  *anthropicFormatter
	blockIndex int
	toolIndex  int
}

type anthropicFormatter struct {
	messageID string
	model     string
}

// NewChatToAnthropicTransformer creates a new transformer.
func NewChatToAnthropicTransformer(w io.Writer) *ChatToAnthropicTransformer {
	return &ChatToAnthropicTransformer{
		w:         w,
		formatter: &anthropicFormatter{},
	}
}

// Transform converts an OpenAI SSE event to Anthropic format.
func (t *ChatToAnthropicTransformer) Transform(event *sse.Event) error {
	if event.Data == "" || event.Data == "[DONE]" {
		return nil
	}
	var chunk types.Chunk
	if err := json.Unmarshal([]byte(event.Data), &chunk); err != nil {
		return t.writePassthrough(event.Data)
	}
	if chunk.ID != "" {
		t.formatter.messageID = chunk.ID
	}
	if chunk.Model != "" {
		t.formatter.model = chunk.Model
	}
	for _, choice := range chunk.Choices {
		if err := t.transformChoice(&choice); err != nil {
			return err
		}
	}
	return nil
}

func (t *ChatToAnthropicTransformer) transformChoice(choice *types.Choice) error {
	if choice.Delta.Content != "" {
		return t.transformContent(choice.Delta.Content)
	}
	if len(choice.Delta.ToolCalls) > 0 {
		return t.transformToolCalls(choice.Delta.ToolCalls)
	}
	return nil
}

func (t *ChatToAnthropicTransformer) transformContent(content string) error {
	if t.blockIndex == 0 {
		t.blockIndex++
		if err := t.writeContentBlockStart(); err != nil {
			return err
		}
	}
	event := map[string]interface{}{
		"type":  "content_block_delta",
		"index": 0,
		"delta": map[string]interface{}{
			"type": "text_delta",
			"text": content,
		},
	}
	return t.writeEvent(event)
}

func (t *ChatToAnthropicTransformer) transformToolCalls(toolCalls []types.ToolCall) error {
	for _, tc := range toolCalls {
		if err := t.transformToolCall(&tc); err != nil {
			return err
		}
	}
	return nil
}

func (t *ChatToAnthropicTransformer) transformToolCall(tc *types.ToolCall) error {
	if tc.Function.Name != "" {
		t.blockIndex++
		if err := t.writeToolBlockStart(tc.ID, tc.Function.Name); err != nil {
			return err
		}
	}
	if tc.Function.Arguments != "" {
		if err := t.writeToolArgs(tc.Function.Arguments); err != nil {
			return err
		}
	}
	return nil
}

func (t *ChatToAnthropicTransformer) writeContentBlockStart() error {
	event := map[string]interface{}{
		"type":          "content_block_start",
		"index":         0,
		"content_block": map[string]interface{}{"type": "text", "text": ""},
	}
	return t.writeEvent(event)
}

func (t *ChatToAnthropicTransformer) writeToolBlockStart(id, name string) error {
	event := map[string]interface{}{
		"type":  "content_block_start",
		"index": t.blockIndex,
		"content_block": map[string]interface{}{
			"type":  "tool_use",
			"id":    id,
			"name":  name,
			"input": map[string]interface{}{},
		},
	}
	return t.writeEvent(event)
}

func (t *ChatToAnthropicTransformer) writeToolArgs(args string) error {
	event := map[string]interface{}{
		"type":  "content_block_delta",
		"index": t.blockIndex,
		"delta": map[string]interface{}{
			"type":         "input_json_delta",
			"partial_json": args,
		},
	}
	return t.writeEvent(event)
}

func (t *ChatToAnthropicTransformer) writeEvent(event map[string]interface{}) error {
	data, err := json.Marshal(event)
	if err != nil {
		return err
	}
	return t.writeEventData(event["type"].(string), data)
}

func (t *ChatToAnthropicTransformer) writeEventData(eventType string, data []byte) error {
	_, err := t.w.Write([]byte("event: " + eventType + "\n"))
	if err != nil {
		return err
	}
	_, err = t.w.Write([]byte("data: "))
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

func (t *ChatToAnthropicTransformer) writePassthrough(data string) error {
	_, err := t.w.Write([]byte("data: " + data + "\n\n"))
	return err
}

// Flush writes any buffered data.
func (t *ChatToAnthropicTransformer) Flush() error {
	return nil
}

// Close releases resources.
func (t *ChatToAnthropicTransformer) Close() error {
	return t.Flush()
}
