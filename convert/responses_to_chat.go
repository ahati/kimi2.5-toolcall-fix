package convert

import (
	"encoding/json"
	"io"

	"ai-proxy/types"

	"github.com/tmaxmax/go-sse"
)

type ResponsesToChatConverter struct{}

func NewResponsesToChatConverter() *ResponsesToChatConverter {
	return &ResponsesToChatConverter{}
}

func (c *ResponsesToChatConverter) Convert(body []byte) ([]byte, error) {
	var req types.ResponsesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, err
	}

	chatReq := types.ChatCompletionRequest{
		Model:       req.Model,
		MaxTokens:   req.MaxOutputTokens,
		Stream:      req.Stream,
		Temperature: req.Temperature,
		TopP:        req.TopP,
	}

	if req.Instructions != "" {
		chatReq.System = req.Instructions
	}

	chatReq.Messages = c.convertInput(req.Input)
	chatReq.Tools = c.convertTools(req.Tools)

	return json.Marshal(chatReq)
}

func (c *ResponsesToChatConverter) convertInput(input interface{}) []types.Message {
	if input == nil {
		return nil
	}

	switch v := input.(type) {
	case string:
		return []types.Message{{
			Role:    "user",
			Content: v,
		}}
	case []interface{}:
		return c.convertInputArray(v)
	default:
		return nil
	}
}

func (c *ResponsesToChatConverter) convertInputArray(items []interface{}) []types.Message {
	messages := make([]types.Message, 0, len(items))
	for _, item := range items {
		msg := c.convertInputItem(item)
		if msg != nil {
			messages = append(messages, *msg)
		}
	}
	return messages
}

func (c *ResponsesToChatConverter) convertInputItem(item interface{}) *types.Message {
	data, err := json.Marshal(item)
	if err != nil {
		return nil
	}

	var inputItem types.InputItem
	if err := json.Unmarshal(data, &inputItem); err != nil {
		return nil
	}

	if inputItem.Type != "message" {
		return nil
	}

	return &types.Message{
		Role:    inputItem.Role,
		Content: inputItem.Content,
	}
}

func (c *ResponsesToChatConverter) convertTools(tools []types.ResponsesTool) []types.Tool {
	if len(tools) == 0 {
		return nil
	}

	result := make([]types.Tool, 0, len(tools))
	for _, tool := range tools {
		if tool.Type == "function" || tool.Type == "" {
			converted := c.convertTool(tool)
			if converted != nil {
				result = append(result, *converted)
			}
		}
	}
	return result
}

func (c *ResponsesToChatConverter) convertTool(tool types.ResponsesTool) *types.Tool {
	var name, description string
	var parameters json.RawMessage

	if tool.Function != nil {
		name = tool.Function.Name
		description = tool.Function.Description
		parameters = tool.Function.Parameters
	} else {
		name = tool.Name
		description = tool.Description
		parameters = tool.Parameters
	}

	if name == "" {
		return nil
	}

	return &types.Tool{
		Type: "function",
		Function: types.ToolFunction{
			Name:        name,
			Description: description,
			Parameters:  parameters,
		},
	}
}

type ResponsesToChatTransformer struct {
	w         io.Writer
	messageID string
	model     string
	index     int
}

func NewResponsesToChatTransformer(w io.Writer) *ResponsesToChatTransformer {
	return &ResponsesToChatTransformer{w: w}
}

func (t *ResponsesToChatTransformer) Transform(event *sse.Event) error {
	if event.Data == "" {
		return nil
	}

	if event.Data == "[DONE]" {
		return t.writeData([]byte("[DONE]"))
	}

	var respEvent types.ResponsesStreamEvent
	if err := json.Unmarshal([]byte(event.Data), &respEvent); err != nil {
		return t.writeData([]byte(event.Data))
	}

	return t.handleEvent(respEvent)
}

func (t *ResponsesToChatTransformer) handleEvent(event types.ResponsesStreamEvent) error {
	switch event.Type {
	case "response.created", "response.in_progress":
		return t.handleResponseCreated(event)
	case "response.output_text.delta":
		return t.handleOutputTextDelta(event)
	case "response.function_call_arguments.delta":
		return t.handleFunctionCallDelta(event)
	case "response.completed":
		return t.handleResponseCompleted(event)
	default:
		return nil
	}
}

func (t *ResponsesToChatTransformer) handleResponseCreated(event types.ResponsesStreamEvent) error {
	if event.Response != nil {
		if event.Response.ID != "" {
			t.messageID = event.Response.ID
		}
		if event.Response.Model != "" {
			t.model = event.Response.Model
		}
	}
	return nil
}

func (t *ResponsesToChatTransformer) handleOutputTextDelta(event types.ResponsesStreamEvent) error {
	chunk := types.Chunk{
		ID:      t.messageID,
		Object:  "chat.completion.chunk",
		Created: 0,
		Model:   t.model,
		Choices: []types.Choice{{
			Index: 0,
			Delta: types.Delta{
				Role:    "assistant",
				Content: event.Delta,
			},
		}},
	}
	data, _ := json.Marshal(chunk)
	return t.writeData(data)
}

func (t *ResponsesToChatTransformer) handleFunctionCallDelta(event types.ResponsesStreamEvent) error {
	if event.OutputItem == nil {
		return nil
	}

	item := event.OutputItem
	chunk := types.Chunk{
		ID:      t.messageID,
		Object:  "chat.completion.chunk",
		Created: 0,
		Model:   t.model,
		Choices: []types.Choice{{
			Index: 0,
			Delta: types.Delta{
				ToolCalls: []types.ToolCall{{
					Index: t.index,
					Type:  "function",
					Function: types.Function{
						Name:      "",
						Arguments: event.Delta,
					},
				}},
			},
		}},
	}
	if item.ID != "" {
		chunk.Choices[0].Delta.ToolCalls[0].ID = item.ID
	}
	if item.CallID != "" {
		chunk.Choices[0].Delta.ToolCalls[0].ID = item.CallID
	}
	data, _ := json.Marshal(chunk)
	return t.writeData(data)
}

func (t *ResponsesToChatTransformer) handleResponseCompleted(event types.ResponsesStreamEvent) error {
	finishReason := "stop"
	chunk := types.Chunk{
		ID:      t.messageID,
		Object:  "chat.completion.chunk",
		Created: 0,
		Model:   t.model,
		Choices: []types.Choice{{
			Index:        0,
			Delta:        types.Delta{},
			FinishReason: &finishReason,
		}},
	}
	if event.Response != nil && event.Response.Usage != nil {
		chunk.Usage = &types.Usage{
			PromptTokens:     event.Response.Usage.InputTokens,
			CompletionTokens: event.Response.Usage.OutputTokens,
			TotalTokens:      event.Response.Usage.TotalTokens,
		}
	}
	data, _ := json.Marshal(chunk)
	return t.writeData(data)
}

func (t *ResponsesToChatTransformer) writeData(data []byte) error {
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

func (t *ResponsesToChatTransformer) Flush() error {
	if f, ok := t.w.(interface{ Flush() error }); ok {
		return f.Flush()
	}
	return nil
}

func (t *ResponsesToChatTransformer) Close() error {
	return nil
}

func (t *ResponsesToChatTransformer) write(data []byte) error {
	_, err := t.w.Write(data)
	return err
}
