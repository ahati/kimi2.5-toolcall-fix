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

func TestResponsesToChatConverter_Convert_StringInput(t *testing.T) {
	converter := NewResponsesToChatConverter()

	respReq := types.ResponsesRequest{
		Model: "gpt-4",
		Input: "Hello, world!",
	}

	body, err := json.Marshal(respReq)
	require.NoError(t, err)

	result, err := converter.Convert(body)
	require.NoError(t, err)

	var chatReq types.ChatCompletionRequest
	err = json.Unmarshal(result, &chatReq)
	require.NoError(t, err)

	assert.Equal(t, "gpt-4", chatReq.Model)
	assert.Len(t, chatReq.Messages, 1)
	assert.Equal(t, "user", chatReq.Messages[0].Role)
	assert.Equal(t, "Hello, world!", chatReq.Messages[0].Content)
}

func TestResponsesToChatConverter_Convert_ArrayInput(t *testing.T) {
	converter := NewResponsesToChatConverter()

	respReq := types.ResponsesRequest{
		Model: "gpt-4",
		Input: []interface{}{
			map[string]interface{}{
				"type":    "message",
				"role":    "user",
				"content": "Hello",
			},
			map[string]interface{}{
				"type":    "message",
				"role":    "assistant",
				"content": "Hi there!",
			},
		},
	}

	body, err := json.Marshal(respReq)
	require.NoError(t, err)

	result, err := converter.Convert(body)
	require.NoError(t, err)

	var chatReq types.ChatCompletionRequest
	err = json.Unmarshal(result, &chatReq)
	require.NoError(t, err)

	assert.Len(t, chatReq.Messages, 2)
	assert.Equal(t, "user", chatReq.Messages[0].Role)
	assert.Equal(t, "Hello", chatReq.Messages[0].Content)
	assert.Equal(t, "assistant", chatReq.Messages[1].Role)
	assert.Equal(t, "Hi there!", chatReq.Messages[1].Content)
}

func TestResponsesToChatConverter_Convert_WithInstructions(t *testing.T) {
	converter := NewResponsesToChatConverter()

	respReq := types.ResponsesRequest{
		Model:        "gpt-4",
		Input:        "Hello",
		Instructions: "You are a helpful assistant.",
	}

	body, err := json.Marshal(respReq)
	require.NoError(t, err)

	result, err := converter.Convert(body)
	require.NoError(t, err)

	var chatReq types.ChatCompletionRequest
	err = json.Unmarshal(result, &chatReq)
	require.NoError(t, err)

	assert.Equal(t, "You are a helpful assistant.", chatReq.System)
}

func TestResponsesToChatConverter_Convert_WithTools(t *testing.T) {
	converter := NewResponsesToChatConverter()

	respReq := types.ResponsesRequest{
		Model: "gpt-4",
		Input: "Hello",
		Tools: []types.ResponsesTool{
			{
				Type: "function",
				Function: &types.ResponsesToolFunction{
					Name:        "bash",
					Description: "Execute bash commands",
					Parameters:  json.RawMessage(`{"type":"object"}`),
				},
			},
		},
	}

	body, err := json.Marshal(respReq)
	require.NoError(t, err)

	result, err := converter.Convert(body)
	require.NoError(t, err)

	var chatReq types.ChatCompletionRequest
	err = json.Unmarshal(result, &chatReq)
	require.NoError(t, err)

	require.Len(t, chatReq.Tools, 1)
	assert.Equal(t, "function", chatReq.Tools[0].Type)
	assert.Equal(t, "bash", chatReq.Tools[0].Function.Name)
	assert.Equal(t, "Execute bash commands", chatReq.Tools[0].Function.Description)
}

func TestResponsesToChatConverter_Convert_Streaming(t *testing.T) {
	converter := NewResponsesToChatConverter()

	respReq := types.ResponsesRequest{
		Model:  "gpt-4",
		Input:  "Hello",
		Stream: true,
	}

	body, err := json.Marshal(respReq)
	require.NoError(t, err)

	result, err := converter.Convert(body)
	require.NoError(t, err)

	var chatReq types.ChatCompletionRequest
	err = json.Unmarshal(result, &chatReq)
	require.NoError(t, err)

	assert.True(t, chatReq.Stream)
}

func TestResponsesToChatConverter_Convert_FlatFormatTools(t *testing.T) {
	converter := NewResponsesToChatConverter()

	respReq := types.ResponsesRequest{
		Model: "gpt-4",
		Input: "Hello",
		Tools: []types.ResponsesTool{
			{
				Type:        "function",
				Name:        "read",
				Description: "Read files",
				Parameters:  json.RawMessage(`{"type":"object"}`),
			},
		},
	}

	body, err := json.Marshal(respReq)
	require.NoError(t, err)

	result, err := converter.Convert(body)
	require.NoError(t, err)

	var chatReq types.ChatCompletionRequest
	err = json.Unmarshal(result, &chatReq)
	require.NoError(t, err)

	require.Len(t, chatReq.Tools, 1)
	assert.Equal(t, "read", chatReq.Tools[0].Function.Name)
}

func TestResponsesToChatConverter_Convert_SkipsNonFunctionTools(t *testing.T) {
	converter := NewResponsesToChatConverter()

	respReq := types.ResponsesRequest{
		Model: "gpt-4",
		Input: "Hello",
		Tools: []types.ResponsesTool{
			{Type: "web_search"},
			{Type: "function", Name: "bash"},
		},
	}

	body, err := json.Marshal(respReq)
	require.NoError(t, err)

	result, err := converter.Convert(body)
	require.NoError(t, err)

	var chatReq types.ChatCompletionRequest
	err = json.Unmarshal(result, &chatReq)
	require.NoError(t, err)

	require.Len(t, chatReq.Tools, 1)
	assert.Equal(t, "bash", chatReq.Tools[0].Function.Name)
}

func TestResponsesToChatConverter_Convert_FunctionCallInput(t *testing.T) {
	converter := NewResponsesToChatConverter()

	respReq := types.ResponsesRequest{
		Model: "gpt-4",
		Input: []interface{}{
			map[string]interface{}{
				"type":      "function_call",
				"call_id":   "call_123",
				"name":      "bash",
				"arguments": `{"command":"ls"}`,
			},
		},
	}

	body, err := json.Marshal(respReq)
	require.NoError(t, err)

	result, err := converter.Convert(body)
	require.NoError(t, err)

	var chatReq types.ChatCompletionRequest
	err = json.Unmarshal(result, &chatReq)
	require.NoError(t, err)

	require.Len(t, chatReq.Messages, 1)
	assert.Equal(t, "assistant", chatReq.Messages[0].Role)
	require.Len(t, chatReq.Messages[0].ToolCalls, 1)
	assert.Equal(t, "call_123", chatReq.Messages[0].ToolCalls[0].ID)
	assert.Equal(t, "bash", chatReq.Messages[0].ToolCalls[0].Function.Name)
	assert.Equal(t, `{"command":"ls"}`, chatReq.Messages[0].ToolCalls[0].Function.Arguments)
}

func TestResponsesToChatConverter_Convert_FunctionCallOutputInput(t *testing.T) {
	converter := NewResponsesToChatConverter()

	respReq := types.ResponsesRequest{
		Model: "gpt-4",
		Input: []interface{}{
			map[string]interface{}{
				"type":    "function_call_output",
				"call_id": "call_123",
				"output":  "file1.txt\nfile2.txt",
			},
		},
	}

	body, err := json.Marshal(respReq)
	require.NoError(t, err)

	result, err := converter.Convert(body)
	require.NoError(t, err)

	var chatReq types.ChatCompletionRequest
	err = json.Unmarshal(result, &chatReq)
	require.NoError(t, err)

	require.Len(t, chatReq.Messages, 1)
	assert.Equal(t, "tool", chatReq.Messages[0].Role)
	assert.Equal(t, "file1.txt\nfile2.txt", chatReq.Messages[0].Content)
	assert.Equal(t, "call_123", chatReq.Messages[0].ToolCallID)
}

func TestResponsesToChatConverter_Convert_SkipsDeveloperMessages(t *testing.T) {
	converter := NewResponsesToChatConverter()

	respReq := types.ResponsesRequest{
		Model: "gpt-4",
		Input: []interface{}{
			map[string]interface{}{
				"type":    "message",
				"role":    "developer",
				"content": "System prompt",
			},
			map[string]interface{}{
				"type":    "message",
				"role":    "user",
				"content": "Hello",
			},
		},
	}

	body, err := json.Marshal(respReq)
	require.NoError(t, err)

	result, err := converter.Convert(body)
	require.NoError(t, err)

	var chatReq types.ChatCompletionRequest
	err = json.Unmarshal(result, &chatReq)
	require.NoError(t, err)

	assert.Len(t, chatReq.Messages, 1)
	assert.Equal(t, "user", chatReq.Messages[0].Role)
}

func TestResponsesToChatConverter_Convert_NilInput(t *testing.T) {
	converter := NewResponsesToChatConverter()

	respReq := types.ResponsesRequest{
		Model: "gpt-4",
		Input: nil,
	}

	body, err := json.Marshal(respReq)
	require.NoError(t, err)

	result, err := converter.Convert(body)
	require.NoError(t, err)

	var chatReq types.ChatCompletionRequest
	err = json.Unmarshal(result, &chatReq)
	require.NoError(t, err)

	assert.Len(t, chatReq.Messages, 0)
}

func TestResponsesToChatConverter_Convert_InvalidJSON(t *testing.T) {
	converter := NewResponsesToChatConverter()

	_, err := converter.Convert([]byte("invalid json"))
	assert.Error(t, err)
}

func TestResponsesToChatTransformer_TextDelta(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesToChatTransformer(&buf)

	event := &sse.Event{
		Data: `{"type":"response.output_text.delta","delta":"Hello","content_index":0}`,
	}

	err := transformer.Transform(event)
	require.NoError(t, err)

	output := buf.String()
	assert.Contains(t, output, "data: ")
	assert.Contains(t, output, `"content":"Hello"`)
	assert.Contains(t, output, `"object":"chat.completion.chunk"`)
}

func TestResponsesToChatTransformer_ResponseCreated(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesToChatTransformer(&buf)

	event := &sse.Event{
		Data: `{"type":"response.created","response":{"id":"resp_123","model":"gpt-4"}}`,
	}

	err := transformer.Transform(event)
	require.NoError(t, err)

	assert.Equal(t, "resp_123", transformer.responseID)
	assert.Equal(t, "gpt-4", transformer.model)
	assert.True(t, transformer.started)
}

func TestResponsesToChatTransformer_FunctionCallDelta(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesToChatTransformer(&buf)
	transformer.responseID = "resp_123"
	transformer.model = "gpt-4"
	transformer.toolCallIndex = 1
	transformer.started = true

	event := &sse.Event{
		Data: `{"type":"response.function_call_arguments.delta","delta":"{\"command\":\"ls\"}"}`,
	}

	err := transformer.Transform(event)
	require.NoError(t, err)

	output := buf.String()
	assert.Contains(t, output, `"arguments":"{\"command\":\"ls\"}"`)
}

func TestResponsesToChatTransformer_Complete(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesToChatTransformer(&buf)
	transformer.responseID = "resp_123"
	transformer.model = "gpt-4"
	transformer.started = true

	event := &sse.Event{
		Data: `{"type":"response.completed","response":{"id":"resp_123","model":"gpt-4","usage":{"input_tokens":10,"output_tokens":20,"total_tokens":30}}}`,
	}

	err := transformer.Transform(event)
	require.NoError(t, err)

	output := buf.String()
	assert.Contains(t, output, `"finish_reason":"stop"`)
	assert.Contains(t, output, `"prompt_tokens":10`)
	assert.Contains(t, output, `"completion_tokens":20`)
}

func TestResponsesToChatTransformer_CompleteWithToolCalls(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesToChatTransformer(&buf)
	transformer.responseID = "resp_123"
	transformer.model = "gpt-4"
	transformer.started = true
	transformer.toolCallIndex = 1

	event := &sse.Event{
		Data: `{"type":"response.completed"}`,
	}

	err := transformer.Transform(event)
	require.NoError(t, err)

	output := buf.String()
	assert.Contains(t, output, `"finish_reason":"tool_calls"`)
}

func TestResponsesToChatTransformer_MultipleEvents(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesToChatTransformer(&buf)

	events := []*sse.Event{
		{Data: `{"type":"response.created","response":{"id":"resp_123","model":"gpt-4"}}`},
		{Data: `{"type":"response.output_text.delta","delta":"Hello"}`},
		{Data: `{"type":"response.output_text.delta","delta":" world"}`},
		{Data: `{"type":"response.completed"}`},
	}

	for _, event := range events {
		err := transformer.Transform(event)
		require.NoError(t, err)
	}

	output := buf.String()
	assert.Contains(t, output, `"content":"Hello"`)
	assert.Contains(t, output, `"content":" world"`)
	assert.Contains(t, output, `"finish_reason":"stop"`)
}

func TestResponsesToChatTransformer_FunctionCall(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesToChatTransformer(&buf)
	transformer.responseID = "resp_123"
	transformer.model = "gpt-4"
	transformer.started = true

	event := &sse.Event{
		Data: `{"type":"response.function_call","item":{"type":"function_call","id":"call_123","call_id":"call_123","content":[{"name":"bash","arguments":""}]}}`,
	}

	err := transformer.Transform(event)
	require.NoError(t, err)

	output := buf.String()
	assert.Contains(t, output, `"id":"call_123"`)
	assert.Contains(t, output, `"name":"bash"`)
	assert.Contains(t, output, `"type":"function"`)
}

func TestResponsesToChatTransformer_DoneMarker(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesToChatTransformer(&buf)

	event := &sse.Event{
		Data: "[DONE]",
	}

	err := transformer.Transform(event)
	require.NoError(t, err)

	output := buf.String()
	assert.Contains(t, output, "data: [DONE]")
}

func TestResponsesToChatTransformer_Close(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesToChatTransformer(&buf)
	transformer.started = true

	err := transformer.Close()
	require.NoError(t, err)

	output := buf.String()
	assert.Contains(t, output, "data: [DONE]")
}

func TestResponsesToChatTransformer_CloseNotStarted(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesToChatTransformer(&buf)

	err := transformer.Close()
	require.NoError(t, err)

	output := buf.String()
	assert.Empty(t, output)
}

func TestResponsesToChatTransformer_Flush(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesToChatTransformer(&buf)

	err := transformer.Flush()
	require.NoError(t, err)
}

func TestResponsesToChatTransformer_EmptyData(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesToChatTransformer(&buf)

	event := &sse.Event{
		Data: "",
	}

	err := transformer.Transform(event)
	require.NoError(t, err)

	assert.Empty(t, buf.String())
}

func TestResponsesToChatTransformer_UnknownEventType(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesToChatTransformer(&buf)

	event := &sse.Event{
		Data: `{"type":"unknown.event"}`,
	}

	err := transformer.Transform(event)
	require.NoError(t, err)

	assert.Empty(t, buf.String())
}

func TestResponsesToChatTransformer_InvalidJSON(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesToChatTransformer(&buf)

	event := &sse.Event{
		Data: "not valid json",
	}

	err := transformer.Transform(event)
	require.NoError(t, err)

	output := buf.String()
	assert.Contains(t, output, "data: not valid json")
}

func TestConvertResponsesInputToMessages(t *testing.T) {
	tests := []struct {
		name       string
		input      interface{}
		expected   int
		checkRoles []string
	}{
		{
			name:     "nil input",
			input:    nil,
			expected: 0,
		},
		{
			name:       "string input",
			input:      "Hello",
			expected:   1,
			checkRoles: []string{"user"},
		},
		{
			name: "message array",
			input: []interface{}{
				map[string]interface{}{
					"type":    "message",
					"role":    "user",
					"content": "Hello",
				},
			},
			expected:   1,
			checkRoles: []string{"user"},
		},
		{
			name: "mixed array",
			input: []interface{}{
				map[string]interface{}{
					"type":    "message",
					"role":    "user",
					"content": "Hello",
				},
				map[string]interface{}{
					"type":      "function_call",
					"call_id":   "call_123",
					"name":      "bash",
					"arguments": "{}",
				},
			},
			expected:   2,
			checkRoles: []string{"user", "assistant"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := convertResponsesInputToMessages(tt.input)
			assert.Equal(t, tt.expected, len(result))
			for i, role := range tt.checkRoles {
				if i < len(result) {
					assert.Equal(t, role, result[i].Role)
				}
			}
		})
	}
}

func TestConvertResponsesToolsToChat(t *testing.T) {
	tests := []struct {
		name         string
		tools        []types.ResponsesTool
		expected     int
		expectedName string
	}{
		{
			name:     "empty tools",
			tools:    nil,
			expected: 0,
		},
		{
			name: "nested function tool",
			tools: []types.ResponsesTool{
				{
					Type: "function",
					Function: &types.ResponsesToolFunction{
						Name: "bash",
					},
				},
			},
			expected:     1,
			expectedName: "bash",
		},
		{
			name: "flat function tool",
			tools: []types.ResponsesTool{
				{
					Type: "function",
					Name: "read",
				},
			},
			expected:     1,
			expectedName: "read",
		},
		{
			name: "skip non-function tools",
			tools: []types.ResponsesTool{
				{Type: "web_search"},
			},
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := convertResponsesToolsToChat(tt.tools)
			assert.Equal(t, tt.expected, len(result))
			if tt.expectedName != "" && len(result) > 0 {
				assert.Equal(t, tt.expectedName, result[0].Function.Name)
			}
		})
	}
}
