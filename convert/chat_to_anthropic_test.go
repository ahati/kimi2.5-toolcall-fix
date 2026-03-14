package convert

import (
	"bytes"
	"encoding/json"
	"testing"

	"ai-proxy/types"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tmaxmax/go-sse"
)

func TestNewChatToAnthropicConverter(t *testing.T) {
	converter := NewChatToAnthropicConverter()
	assert.NotNil(t, converter)
}

func TestConvertRequest_SimpleMessage(t *testing.T) {
	converter := NewChatToAnthropicConverter()
	openReq := types.ChatCompletionRequest{
		Model:    "claude-3-opus",
		Messages: []types.Message{{Role: "user", Content: "Hello"}},
	}
	body, err := json.Marshal(openReq)
	require.NoError(t, err)
	result, err := converter.Convert(body)
	require.NoError(t, err)
	var anthReq types.MessageRequest
	err = json.Unmarshal(result, &anthReq)
	require.NoError(t, err)
	assert.Equal(t, "claude-3-opus", anthReq.Model)
	assert.Equal(t, 4096, anthReq.MaxTokens)
	assert.Len(t, anthReq.Messages, 1)
	assert.Equal(t, "user", anthReq.Messages[0].Role)
	content, _ := anthReq.Messages[0].Content.(string)
	assert.Equal(t, "Hello", content)
}

func TestConvertRequest_WithMaxTokens(t *testing.T) {
	converter := NewChatToAnthropicConverter()
	openReq := types.ChatCompletionRequest{
		Model:     "claude-3-opus",
		MaxTokens: 8192,
		Messages:  []types.Message{{Role: "user", Content: "Hello"}},
	}
	body, err := json.Marshal(openReq)
	require.NoError(t, err)
	result, err := converter.Convert(body)
	require.NoError(t, err)
	var anthReq types.MessageRequest
	err = json.Unmarshal(result, &anthReq)
	require.NoError(t, err)
	assert.Equal(t, 8192, anthReq.MaxTokens)
}

func TestConvertRequest_WithSystem(t *testing.T) {
	converter := NewChatToAnthropicConverter()
	openReq := types.ChatCompletionRequest{
		Model:    "claude-3-opus",
		System:   "You are a helpful assistant",
		Messages: []types.Message{{Role: "user", Content: "Hello"}},
	}
	body, err := json.Marshal(openReq)
	require.NoError(t, err)
	result, err := converter.Convert(body)
	require.NoError(t, err)
	var anthReq types.MessageRequest
	err = json.Unmarshal(result, &anthReq)
	require.NoError(t, err)
	system, _ := anthReq.System.(string)
	assert.Equal(t, "You are a helpful assistant", system)
}

func TestConvertRequest_WithStream(t *testing.T) {
	converter := NewChatToAnthropicConverter()
	openReq := types.ChatCompletionRequest{
		Model:    "claude-3-opus",
		Stream:   true,
		Messages: []types.Message{{Role: "user", Content: "Hello"}},
	}
	body, err := json.Marshal(openReq)
	require.NoError(t, err)
	result, err := converter.Convert(body)
	require.NoError(t, err)
	var anthReq types.MessageRequest
	err = json.Unmarshal(result, &anthReq)
	require.NoError(t, err)
	assert.True(t, anthReq.Stream)
}

func TestConvertRequest_WithTemperature(t *testing.T) {
	converter := NewChatToAnthropicConverter()
	openReq := types.ChatCompletionRequest{
		Model:       "claude-3-opus",
		Temperature: 0.7,
		Messages:    []types.Message{{Role: "user", Content: "Hello"}},
	}
	body, err := json.Marshal(openReq)
	require.NoError(t, err)
	result, err := converter.Convert(body)
	require.NoError(t, err)
	var anthReq types.MessageRequest
	err = json.Unmarshal(result, &anthReq)
	require.NoError(t, err)
	assert.Equal(t, 0.7, anthReq.Temperature)
}

func TestConvertRequest_WithTools(t *testing.T) {
	converter := NewChatToAnthropicConverter()
	openReq := types.ChatCompletionRequest{
		Model: "claude-3-opus",
		Tools: []types.Tool{{
			Type: "function",
			Function: types.ToolFunction{
				Name:        "get_weather",
				Description: "Get weather for a location",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}}}`),
			},
		}},
		Messages: []types.Message{{Role: "user", Content: "What's the weather?"}},
	}
	body, err := json.Marshal(openReq)
	require.NoError(t, err)
	result, err := converter.Convert(body)
	require.NoError(t, err)
	var anthReq types.MessageRequest
	err = json.Unmarshal(result, &anthReq)
	require.NoError(t, err)
	require.Len(t, anthReq.Tools, 1)
	assert.Equal(t, "get_weather", anthReq.Tools[0].Name)
	assert.Equal(t, "Get weather for a location", anthReq.Tools[0].Description)
}

func TestConvertRequest_MultipleMessages(t *testing.T) {
	converter := NewChatToAnthropicConverter()
	openReq := types.ChatCompletionRequest{
		Model: "claude-3-opus",
		Messages: []types.Message{
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Hi there"},
			{Role: "user", Content: "How are you?"},
		},
	}
	body, err := json.Marshal(openReq)
	require.NoError(t, err)
	result, err := converter.Convert(body)
	require.NoError(t, err)
	var anthReq types.MessageRequest
	err = json.Unmarshal(result, &anthReq)
	require.NoError(t, err)
	assert.Len(t, anthReq.Messages, 3)
	assert.Equal(t, "user", anthReq.Messages[0].Role)
	assert.Equal(t, "assistant", anthReq.Messages[1].Role)
	assert.Equal(t, "user", anthReq.Messages[2].Role)
}

func TestConvertRequest_WithToolCalls(t *testing.T) {
	converter := NewChatToAnthropicConverter()
	openReq := types.ChatCompletionRequest{
		Model: "claude-3-opus",
		Messages: []types.Message{{
			Role:    "assistant",
			Content: "I'll check the weather",
			ToolCalls: []types.ToolCall{{
				ID:   "call_123",
				Type: "function",
				Function: types.Function{
					Name:      "get_weather",
					Arguments: `{"location":"NYC"}`,
				},
			}},
		}},
	}
	body, err := json.Marshal(openReq)
	require.NoError(t, err)
	result, err := converter.Convert(body)
	require.NoError(t, err)
	var anthReq types.MessageRequest
	err = json.Unmarshal(result, &anthReq)
	require.NoError(t, err)
	require.Len(t, anthReq.Messages, 1)
	assert.Equal(t, "assistant", anthReq.Messages[0].Role)
	blocks, ok := anthReq.Messages[0].Content.([]interface{})
	require.True(t, ok)
	assert.Len(t, blocks, 2)
}

func TestConvertRequest_WithToolResult(t *testing.T) {
	converter := NewChatToAnthropicConverter()
	openReq := types.ChatCompletionRequest{
		Model: "claude-3-opus",
		Messages: []types.Message{{
			Role:       "tool",
			Content:    "72°F and sunny",
			ToolCallID: "call_123",
		}},
	}
	body, err := json.Marshal(openReq)
	require.NoError(t, err)
	result, err := converter.Convert(body)
	require.NoError(t, err)
	var anthReq types.MessageRequest
	err = json.Unmarshal(result, &anthReq)
	require.NoError(t, err)
	require.Len(t, anthReq.Messages, 1)
	assert.Equal(t, "user", anthReq.Messages[0].Role)
	blocks, ok := anthReq.Messages[0].Content.([]interface{})
	require.True(t, ok)
	require.Len(t, blocks, 1)
	block := blocks[0].(map[string]interface{})
	assert.Equal(t, "tool_result", block["type"])
	assert.Equal(t, "call_123", block["tool_use_id"])
}

func TestConvertRequest_SystemMessageSkipped(t *testing.T) {
	converter := NewChatToAnthropicConverter()
	openReq := types.ChatCompletionRequest{
		Model: "claude-3-opus",
		Messages: []types.Message{
			{Role: "system", Content: "Be helpful"},
			{Role: "user", Content: "Hello"},
		},
	}
	body, err := json.Marshal(openReq)
	require.NoError(t, err)
	result, err := converter.Convert(body)
	require.NoError(t, err)
	var anthReq types.MessageRequest
	err = json.Unmarshal(result, &anthReq)
	require.NoError(t, err)
	assert.Len(t, anthReq.Messages, 1)
	assert.Equal(t, "user", anthReq.Messages[0].Role)
}

func TestConvertRequest_InvalidJSON(t *testing.T) {
	converter := NewChatToAnthropicConverter()
	result, err := converter.Convert([]byte("invalid json"))
	assert.Error(t, err)
	assert.Nil(t, result)
}

func TestNewChatToAnthropicTransformer(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewChatToAnthropicTransformer(&buf)
	assert.NotNil(t, transformer)
}

func TestTransformContent(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewChatToAnthropicTransformer(&buf)
	chunk := types.Chunk{
		ID:      "msg_123",
		Model:   "claude-3-opus",
		Choices: []types.Choice{{Delta: types.Delta{Content: "Hello"}}},
	}
	data, _ := json.Marshal(chunk)
	event := &sse.Event{Data: string(data)}
	err := transformer.Transform(event)
	require.NoError(t, err)
	output := buf.String()
	assert.Contains(t, output, "event: content_block_start")
	assert.Contains(t, output, "event: content_block_delta")
	assert.Contains(t, output, `"text":"Hello"`)
}

func TestTransformToolCall(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewChatToAnthropicTransformer(&buf)
	chunk := types.Chunk{
		ID:    "msg_123",
		Model: "claude-3-opus",
		Choices: []types.Choice{{
			Delta: types.Delta{
				ToolCalls: []types.ToolCall{{
					ID:   "call_123",
					Type: "function",
					Function: types.Function{
						Name:      "get_weather",
						Arguments: `{"location":"NYC"}`,
					},
				}},
			},
		}},
	}
	data, _ := json.Marshal(chunk)
	event := &sse.Event{Data: string(data)}
	err := transformer.Transform(event)
	require.NoError(t, err)
	output := buf.String()
	assert.Contains(t, output, "event: content_block_start")
	assert.Contains(t, output, `"type":"tool_use"`)
	assert.Contains(t, output, `"name":"get_weather"`)
	assert.Contains(t, output, "event: content_block_delta")
	assert.Contains(t, output, `"partial_json`)
}

func TestTransformEmptyEvent(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewChatToAnthropicTransformer(&buf)
	event := &sse.Event{Data: ""}
	err := transformer.Transform(event)
	require.NoError(t, err)
	assert.Empty(t, buf.String())
}

func TestTransformDoneMarker(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewChatToAnthropicTransformer(&buf)
	event := &sse.Event{Data: "[DONE]"}
	err := transformer.Transform(event)
	require.NoError(t, err)
	assert.Empty(t, buf.String())
}

func TestTransformInvalidJSON(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewChatToAnthropicTransformer(&buf)
	event := &sse.Event{Data: "not json"}
	err := transformer.Transform(event)
	require.NoError(t, err)
	output := buf.String()
	assert.Contains(t, output, "data: not json")
}

func TestFlush(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewChatToAnthropicTransformer(&buf)
	err := transformer.Flush()
	assert.NoError(t, err)
}

func TestClose(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewChatToAnthropicTransformer(&buf)
	err := transformer.Close()
	assert.NoError(t, err)
}

func TestConvertEmptyMessages(t *testing.T) {
	converter := NewChatToAnthropicConverter()
	openReq := types.ChatCompletionRequest{
		Model:    "claude-3-opus",
		Messages: []types.Message{},
	}
	body, err := json.Marshal(openReq)
	require.NoError(t, err)
	result, err := converter.Convert(body)
	require.NoError(t, err)
	var anthReq types.MessageRequest
	err = json.Unmarshal(result, &anthReq)
	require.NoError(t, err)
	assert.Len(t, anthReq.Messages, 0)
}

func TestConvertToolOnlyMessage(t *testing.T) {
	converter := NewChatToAnthropicConverter()
	openReq := types.ChatCompletionRequest{
		Model: "claude-3-opus",
		Messages: []types.Message{{
			Role: "assistant",
			ToolCalls: []types.ToolCall{{
				ID:   "call_456",
				Type: "function",
				Function: types.Function{
					Name:      "calculate",
					Arguments: `{"x":5,"y":10}`,
				},
			}},
		}},
	}
	body, err := json.Marshal(openReq)
	require.NoError(t, err)
	result, err := converter.Convert(body)
	require.NoError(t, err)
	var anthReq types.MessageRequest
	err = json.Unmarshal(result, &anthReq)
	require.NoError(t, err)
	require.Len(t, anthReq.Messages, 1)
	blocks, ok := anthReq.Messages[0].Content.([]interface{})
	require.True(t, ok)
	require.Len(t, blocks, 1)
	block := blocks[0].(map[string]interface{})
	assert.Equal(t, "tool_use", block["type"])
}

func TestConvertNoTools(t *testing.T) {
	converter := NewChatToAnthropicConverter()
	openReq := types.ChatCompletionRequest{
		Model:    "claude-3-opus",
		Messages: []types.Message{{Role: "user", Content: "Hello"}},
	}
	body, err := json.Marshal(openReq)
	require.NoError(t, err)
	result, err := converter.Convert(body)
	require.NoError(t, err)
	var anthReq types.MessageRequest
	err = json.Unmarshal(result, &anthReq)
	require.NoError(t, err)
	assert.Nil(t, anthReq.Tools)
}

func TestConvertMultipleToolCalls(t *testing.T) {
	converter := NewChatToAnthropicConverter()
	openReq := types.ChatCompletionRequest{
		Model: "claude-3-opus",
		Messages: []types.Message{{
			Role:    "assistant",
			Content: "Let me check",
			ToolCalls: []types.ToolCall{
				{
					ID:   "call_1",
					Type: "function",
					Function: types.Function{
						Name:      "get_weather",
						Arguments: `{"city":"NYC"}`,
					},
				},
				{
					ID:   "call_2",
					Type: "function",
					Function: types.Function{
						Name:      "get_time",
						Arguments: `{"timezone":"EST"}`,
					},
				},
			},
		}},
	}
	body, err := json.Marshal(openReq)
	require.NoError(t, err)
	result, err := converter.Convert(body)
	require.NoError(t, err)
	var anthReq types.MessageRequest
	err = json.Unmarshal(result, &anthReq)
	require.NoError(t, err)
	require.Len(t, anthReq.Messages, 1)
	blocks, ok := anthReq.Messages[0].Content.([]interface{})
	require.True(t, ok)
	assert.Len(t, blocks, 3) // text + 2 tool_use blocks
}
