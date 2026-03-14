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

type ResponsesToChatConverter struct{}

func NewResponsesToChatConverter() *ResponsesToChatConverter {
	return &ResponsesToChatConverter{}
}

func (c *ResponsesToChatConverter) Convert(body []byte) ([]byte, error) {
	var respReq types.ResponsesRequest
	if err := json.Unmarshal(body, &respReq); err != nil {
		return nil, fmt.Errorf("failed to parse Responses request: %w", err)
	}

	chatReq := types.ChatCompletionRequest{
		Model:       respReq.Model,
		MaxTokens:   respReq.MaxOutputTokens,
		Stream:      respReq.Stream,
		Temperature: respReq.Temperature,
		TopP:        respReq.TopP,
		System:      respReq.Instructions,
		Messages:    convertResponsesInputToMessages(respReq.Input),
		Tools:       convertResponsesToolsToChat(respReq.Tools),
	}

	return json.Marshal(chatReq)
}

func convertResponsesInputToMessages(input interface{}) []types.Message {
	if input == nil {
		return []types.Message{}
	}

	if s, ok := input.(string); ok {
		return []types.Message{{Role: "user", Content: s}}
	}

	if arr, ok := input.([]interface{}); ok {
		messages := make([]types.Message, 0, len(arr))
		for _, item := range arr {
			if msg, ok := item.(map[string]interface{}); ok {
				if m := convertInputItemToMessage(msg); m != nil {
					messages = append(messages, *m)
				}
			}
		}
		return messages
	}

	return []types.Message{}
}

func convertInputItemToMessage(item map[string]interface{}) *types.Message {
	itemType, _ := item["type"].(string)

	switch itemType {
	case "message":
		return convertMessageItem(item)
	case "function_call":
		return convertFunctionCallItem(item)
	case "function_call_output":
		return convertFunctionCallOutputItem(item)
	}

	return nil
}

func convertMessageItem(item map[string]interface{}) *types.Message {
	role, _ := item["role"].(string)
	switch role {
	case "developer", "system":
		return nil
	case "assistant":
		role = "assistant"
	default:
		role = "user"
	}

	content := extractResponsesContent(item["content"])
	if content == "" {
		return nil
	}

	return &types.Message{
		Role:    role,
		Content: content,
	}
}

func convertFunctionCallItem(item map[string]interface{}) *types.Message {
	callID, _ := item["call_id"].(string)
	name, _ := item["name"].(string)
	args, _ := item["arguments"].(string)

	if callID == "" {
		callID, _ = item["id"].(string)
	}

	return &types.Message{
		Role: "assistant",
		ToolCalls: []types.ToolCall{
			{
				ID:   callID,
				Type: "function",
				Function: types.Function{
					Name:      name,
					Arguments: args,
				},
			},
		},
	}
}

func convertFunctionCallOutputItem(item map[string]interface{}) *types.Message {
	callID, _ := item["call_id"].(string)
	output, _ := item["output"].(string)

	return &types.Message{
		Role:       "tool",
		Content:    output,
		ToolCallID: callID,
	}
}

func convertResponsesToolsToChat(respTools []types.ResponsesTool) []types.Tool {
	if len(respTools) == 0 {
		return nil
	}

	tools := make([]types.Tool, 0, len(respTools))
	for _, rt := range respTools {
		if rt.Type == "function" {
			tool := types.Tool{Type: "function"}
			if rt.Function != nil {
				tool.Function = types.ToolFunction{
					Name:        rt.Function.Name,
					Description: rt.Function.Description,
					Parameters:  rt.Function.Parameters,
				}
			} else {
				tool.Function = types.ToolFunction{
					Name:        rt.Name,
					Description: rt.Description,
					Parameters:  rt.Parameters,
				}
			}
			tools = append(tools, tool)
		}
	}
	return tools
}

type ResponsesToChatTransformer struct {
	w          io.Writer
	responseID string
	model      string
	created    int64
	started    bool

	toolCallIndex int
	toolCallID    string
	toolCallName  string
	toolCallArgs  strings.Builder
}

func NewResponsesToChatTransformer(w io.Writer) *ResponsesToChatTransformer {
	return &ResponsesToChatTransformer{
		w:       w,
		created: time.Now().Unix(),
	}
}

func (t *ResponsesToChatTransformer) Transform(event *sse.Event) error {
	if event.Data == "" {
		return nil
	}

	if event.Data == "[DONE]" {
		return t.writeDone()
	}

	var respEvent types.ResponsesStreamEvent
	if err := json.Unmarshal([]byte(event.Data), &respEvent); err != nil {
		return t.writeData([]byte(event.Data))
	}

	return t.handleEvent(&respEvent)
}

func (t *ResponsesToChatTransformer) handleEvent(event *types.ResponsesStreamEvent) error {
	switch event.Type {
	case "response.created":
		return t.handleResponseCreated(event)
	case "response.output_text.delta":
		return t.handleTextDelta(event)
	case "response.function_call_arguments.delta":
		return t.handleFunctionCallDelta(event)
	case "response.completed":
		return t.handleCompleted(event)
	case "response.function_call":
		return t.handleFunctionCall(event)
	}
	return nil
}

func (t *ResponsesToChatTransformer) handleResponseCreated(event *types.ResponsesStreamEvent) error {
	if event.Response != nil {
		t.responseID = event.Response.ID
		t.model = event.Response.Model
		t.started = true
	}
	return nil
}

func (t *ResponsesToChatTransformer) handleTextDelta(event *types.ResponsesStreamEvent) error {
	if event.Delta == "" {
		return nil
	}

	chunk := types.Chunk{
		ID:      t.responseID,
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   t.model,
		Choices: []types.Choice{{
			Index: 0,
			Delta: types.Delta{
				Content: event.Delta,
			},
		}},
	}

	return t.writeChunk(&chunk)
}

func (t *ResponsesToChatTransformer) handleFunctionCall(event *types.ResponsesStreamEvent) error {
	if event.OutputItem == nil {
		return nil
	}

	t.toolCallID = event.OutputItem.CallID
	if t.toolCallID == "" {
		t.toolCallID = event.OutputItem.ID
	}

	var funcName string
	if len(event.OutputItem.Content) > 0 {
		funcName = event.OutputItem.Content[0].Name
	}

	t.toolCallName = funcName
	t.toolCallArgs.Reset()

	chunk := types.Chunk{
		ID:      t.responseID,
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   t.model,
		Choices: []types.Choice{{
			Index: 0,
			Delta: types.Delta{
				ToolCalls: []types.ToolCall{{
					ID:    t.toolCallID,
					Type:  "function",
					Index: t.toolCallIndex,
					Function: types.Function{
						Name:      t.toolCallName,
						Arguments: "",
					},
				}},
			},
		}},
	}

	t.toolCallIndex++

	return t.writeChunk(&chunk)
}

func (t *ResponsesToChatTransformer) handleFunctionCallDelta(event *types.ResponsesStreamEvent) error {
	if event.Delta == "" {
		return nil
	}

	t.toolCallArgs.WriteString(event.Delta)

	chunk := types.Chunk{
		ID:      t.responseID,
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   t.model,
		Choices: []types.Choice{{
			Index: 0,
			Delta: types.Delta{
				ToolCalls: []types.ToolCall{{
					Index: t.toolCallIndex - 1,
					Function: types.Function{
						Arguments: event.Delta,
					},
				}},
			},
		}},
	}

	return t.writeChunk(&chunk)
}

func (t *ResponsesToChatTransformer) handleCompleted(event *types.ResponsesStreamEvent) error {
	finishReason := "stop"
	if t.toolCallIndex > 0 {
		finishReason = "tool_calls"
	}

	chunk := types.Chunk{
		ID:      t.responseID,
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
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

	return t.writeChunk(&chunk)
}

func (t *ResponsesToChatTransformer) writeChunk(chunk *types.Chunk) error {
	data, err := json.Marshal(chunk)
	if err != nil {
		return err
	}
	return t.writeData(data)
}

func (t *ResponsesToChatTransformer) writeData(data []byte) error {
	if len(data) == 0 {
		return nil
	}
	_, err := t.w.Write([]byte("data: " + string(data) + "\n\n"))
	return err
}

func (t *ResponsesToChatTransformer) writeDone() error {
	_, err := t.w.Write([]byte("data: [DONE]\n\n"))
	return err
}

func (t *ResponsesToChatTransformer) Flush() error {
	return nil
}

func (t *ResponsesToChatTransformer) Close() error {
	if t.started {
		return t.writeDone()
	}
	return nil
}
