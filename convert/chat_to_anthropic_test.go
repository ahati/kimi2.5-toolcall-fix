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

func TestChatToAnthropicConverter_Convert_Basic(t *testing.T) {
	converter := NewChatToAnthropicConverter()

	openReq := types.ChatCompletionRequest{
		Model: "gpt-4",
		Messages: []types.Message{
			{Role: "user", Content: "Hello"},
		},
		MaxTokens: 100,
	}

	body, err := json.Marshal(openReq)
	require.NoError(t, err)

	result, err := converter.Convert(body)
	require.NoError(t, err)

	var anthReq types.MessageRequest
	err = json.Unmarshal(result, &anthReq)
	require.NoError(t, err)

	assert.Equal(t, "gpt-4", anthReq.Model)
	assert.Equal(t, 100, anthReq.MaxTokens)
	assert.Len(t, anthReq.Messages, 1)
	assert.Equal(t, "user", anthReq.Messages[0].Role)
	assert.Equal(t, "Hello", anthReq.Messages[0].Content)
}

func TestChatToAnthropicConverter_Convert_WithSystem(t *testing.T) {
	converter := NewChatToAnthropicConverter()

	openReq := types.ChatCompletionRequest{
		Model: "gpt-4",
		Messages: []types.Message{
			{Role: "user", Content: "Hello"},
		},
		System:    "You are helpful",
		MaxTokens: 100,
	}

	body, err := json.Marshal(openReq)
	require.NoError(t, err)

	result, err := converter.Convert(body)
	require.NoError(t, err)

	var anthReq types.MessageRequest
	err = json.Unmarshal(result, &anthReq)
	require.NoError(t, err)

	assert.Equal(t, "You are helpful", anthReq.System)
}

func TestChatToAnthropicConverter_Convert_WithTools(t *testing.T) {
	converter := NewChatToAnthropicConverter()

	openReq := types.ChatCompletionRequest{
		Model: "gpt-4",
		Messages: []types.Message{
			{Role: "user", Content: "List files"},
		},
		MaxTokens: 100,
		Tools: []types.Tool{
			{
				Type: "function",
				Function: types.ToolFunction{
					Name:        "bash",
					Description: "Run commands",
					Parameters:  json.RawMessage(`{"type":"object"}`),
				},
			},
		},
	}

	body, err := json.Marshal(openReq)
	require.NoError(t, err)

	result, err := converter.Convert(body)
	require.NoError(t, err)

	var anthReq types.MessageRequest
	err = json.Unmarshal(result, &anthReq)
	require.NoError(t, err)

	require.Len(t, anthReq.Tools, 1)
	assert.Equal(t, "bash", anthReq.Tools[0].Name)
	assert.Equal(t, "Run commands", anthReq.Tools[0].Description)
}

func TestChatToAnthropicConverter_Convert_WithToolCalls(t *testing.T) {
	converter := NewChatToAnthropicConverter()

	openReq := types.ChatCompletionRequest{
		Model: "gpt-4",
		Messages: []types.Message{
			{Role: "user", Content: "List files"},
			{
				Role:    "assistant",
				Content: "",
				ToolCalls: []types.ToolCall{
					{
						ID:   "call_123",
						Type: "function",
						Function: types.Function{
							Name:      "bash",
							Arguments: `{"command":"ls"}`,
						},
					},
				},
			},
		},
		MaxTokens: 100,
	}

	body, err := json.Marshal(openReq)
	require.NoError(t, err)

	result, err := converter.Convert(body)
	require.NoError(t, err)

	var anthReq types.MessageRequest
	err = json.Unmarshal(result, &anthReq)
	require.NoError(t, err)

	assert.Len(t, anthReq.Messages, 2)
	assert.Equal(t, "assistant", anthReq.Messages[1].Role)

	contentArr, ok := anthReq.Messages[1].Content.([]interface{})
	require.True(t, ok, "content should be array")
	require.Len(t, contentArr, 1)

	contentBlock, ok := contentArr[0].(map[string]interface{})
	require.True(t, ok)
	assert.Equal(t, "tool_use", contentBlock["type"])
	assert.Equal(t, "call_123", contentBlock["id"])
	assert.Equal(t, "bash", contentBlock["name"])
}

func TestChatToAnthropicConverter_Convert_WithToolResults(t *testing.T) {
	converter := NewChatToAnthropicConverter()

	openReq := types.ChatCompletionRequest{
		Model: "gpt-4",
		Messages: []types.Message{
			{Role: "user", Content: "List files"},
			{
				Role:       "tool",
				Content:    "file1.txt",
				ToolCallID: "call_123",
			},
		},
		MaxTokens: 100,
	}

	body, err := json.Marshal(openReq)
	require.NoError(t, err)

	result, err := converter.Convert(body)
	require.NoError(t, err)

	var anthReq types.MessageRequest
	err = json.Unmarshal(result, &anthReq)
	require.NoError(t, err)

	assert.Len(t, anthReq.Messages, 2)
	assert.Equal(t, "user", anthReq.Messages[1].Role)

	contentArr, ok := anthReq.Messages[1].Content.([]interface{})
	require.True(t, ok, "content should be array")
	require.Len(t, contentArr, 1)

	contentBlock, ok := contentArr[0].(map[string]interface{})
	require.True(t, ok)
	assert.Equal(t, "tool_result", contentBlock["type"])
	assert.Equal(t, "call_123", contentBlock["tool_use_id"])
}

func TestChatToAnthropicConverter_Convert_Streaming(t *testing.T) {
	converter := NewChatToAnthropicConverter()

	openReq := types.ChatCompletionRequest{
		Model: "gpt-4",
		Messages: []types.Message{
			{Role: "user", Content: "Hello"},
		},
		Stream:    true,
		MaxTokens: 100,
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

func TestChatToAnthropicConverter_Convert_DefaultMaxTokens(t *testing.T) {
	converter := NewChatToAnthropicConverter()

	openReq := types.ChatCompletionRequest{
		Model: "gpt-4",
		Messages: []types.Message{
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

	assert.Equal(t, 4096, anthReq.MaxTokens)
}

func TestChatToAnthropicTransformer_ContentDelta(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewChatToAnthropicTransformer(&buf)

	chunk := types.Chunk{
		ID:      "chatcmpl-123",
		Model:   "gpt-4",
		Choices: []types.Choice{{Delta: types.Delta{Content: "Hello"}}},
	}
	chunkJSON, _ := json.Marshal(chunk)

	event := &sse.Event{Data: string(chunkJSON)}
	err := transformer.Transform(event)
	require.NoError(t, err)

	output := buf.String()
	assert.Contains(t, output, "message_start")
	assert.Contains(t, output, "content_block_delta")
	assert.Contains(t, output, "text_delta")
	assert.Contains(t, output, "Hello")
}

func TestChatToAnthropicTransformer_ToolCall(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewChatToAnthropicTransformer(&buf)

	chunk := types.Chunk{
		ID:    "chatcmpl-123",
		Model: "gpt-4",
		Choices: []types.Choice{{
			Delta: types.Delta{
				ToolCalls: []types.ToolCall{{
					ID:    "call_123",
					Type:  "function",
					Index: 0,
					Function: types.Function{
						Name:      "bash",
						Arguments: `{"command":"ls"}`,
					},
				}},
			},
		}},
	}
	chunkJSON, _ := json.Marshal(chunk)

	event := &sse.Event{Data: string(chunkJSON)}
	err := transformer.Transform(event)
	require.NoError(t, err)

	output := buf.String()
	assert.Contains(t, output, "message_start")
	assert.Contains(t, output, "content_block_start")
	assert.Contains(t, output, "tool_use")
	assert.Contains(t, output, "call_123")
	assert.Contains(t, output, "bash")
	assert.Contains(t, output, "input_json_delta")
	assert.Contains(t, output, "command")
	assert.Contains(t, output, "ls")
}

func TestChatToAnthropicTransformer_MultipleBlocks(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewChatToAnthropicTransformer(&buf)

	chunk1 := types.Chunk{
		ID:      "chatcmpl-123",
		Model:   "gpt-4",
		Choices: []types.Choice{{Delta: types.Delta{Content: "Hello"}}},
	}
	chunk1JSON, _ := json.Marshal(chunk1)

	err := transformer.Transform(&sse.Event{Data: string(chunk1JSON)})
	require.NoError(t, err)

	chunk2 := types.Chunk{
		ID:    "chatcmpl-123",
		Model: "gpt-4",
		Choices: []types.Choice{{
			Delta: types.Delta{
				ToolCalls: []types.ToolCall{{
					ID:    "call_123",
					Type:  "function",
					Index: 0,
					Function: types.Function{
						Name:      "bash",
						Arguments: `{`,
					},
				}},
			},
		}},
	}
	chunk2JSON, _ := json.Marshal(chunk2)

	err = transformer.Transform(&sse.Event{Data: string(chunk2JSON)})
	require.NoError(t, err)

	chunk3 := types.Chunk{
		ID:    "chatcmpl-123",
		Model: "gpt-4",
		Choices: []types.Choice{{
			Delta: types.Delta{
				ToolCalls: []types.ToolCall{{
					Index: 0,
					Function: types.Function{
						Arguments: `"command":"ls"}`,
					},
				}},
			},
		}},
	}
	chunk3JSON, _ := json.Marshal(chunk3)

	err = transformer.Transform(&sse.Event{Data: string(chunk3JSON)})
	require.NoError(t, err)

	output := buf.String()
	assert.Contains(t, output, "text_delta")
	assert.Contains(t, output, "Hello")
	assert.Contains(t, output, "tool_use")
	assert.Contains(t, output, "input_json_delta")
}

func TestChatToAnthropicTransformer_FullStream(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewChatToAnthropicTransformer(&buf)

	chunk1 := types.Chunk{
		ID:      "chatcmpl-123",
		Model:   "gpt-4",
		Choices: []types.Choice{{Delta: types.Delta{Role: "assistant"}}},
	}
	chunk1JSON, _ := json.Marshal(chunk1)
	_ = transformer.Transform(&sse.Event{Data: string(chunk1JSON)})

	chunk2 := types.Chunk{
		ID:      "chatcmpl-123",
		Model:   "gpt-4",
		Choices: []types.Choice{{Delta: types.Delta{Content: "Hello"}}},
	}
	chunk2JSON, _ := json.Marshal(chunk2)
	_ = transformer.Transform(&sse.Event{Data: string(chunk2JSON)})

	chunk3 := types.Chunk{
		ID:      "chatcmpl-123",
		Model:   "gpt-4",
		Choices: []types.Choice{{Delta: types.Delta{Content: " world"}}},
	}
	chunk3JSON, _ := json.Marshal(chunk3)
	_ = transformer.Transform(&sse.Event{Data: string(chunk3JSON)})

	stopReason := "stop"
	chunk4 := types.Chunk{
		ID:      "chatcmpl-123",
		Model:   "gpt-4",
		Choices: []types.Choice{{FinishReason: &stopReason}},
	}
	chunk4JSON, _ := json.Marshal(chunk4)
	_ = transformer.Transform(&sse.Event{Data: string(chunk4JSON)})

	output := buf.String()
	assert.Contains(t, output, "message_start")
	assert.Contains(t, output, "content_block_delta")
	assert.Contains(t, output, "Hello")
	assert.Contains(t, output, "world")
	assert.Contains(t, output, "content_block_stop")
	assert.Contains(t, output, "message_delta")
	assert.Contains(t, output, "end_turn")
	assert.Contains(t, output, "message_stop")
}

func TestChatToAnthropicTransformer_ToolCallsFinish(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewChatToAnthropicTransformer(&buf)

	chunk := types.Chunk{
		ID:    "chatcmpl-123",
		Model: "gpt-4",
		Choices: []types.Choice{{
			Delta: types.Delta{
				ToolCalls: []types.ToolCall{{
					ID:    "call_123",
					Type:  "function",
					Index: 0,
					Function: types.Function{
						Name:      "bash",
						Arguments: `{"command":"ls"}`,
					},
				}},
			},
		}},
	}
	chunkJSON, _ := json.Marshal(chunk)
	_ = transformer.Transform(&sse.Event{Data: string(chunkJSON)})

	stopReason := "tool_calls"
	finishChunk := types.Chunk{
		ID:      "chatcmpl-123",
		Model:   "gpt-4",
		Choices: []types.Choice{{FinishReason: &stopReason}},
	}
	finishJSON, _ := json.Marshal(finishChunk)
	_ = transformer.Transform(&sse.Event{Data: string(finishJSON)})

	output := buf.String()
	assert.Contains(t, output, "tool_use")
}

func TestChatToAnthropicTransformer_EmptyEvent(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewChatToAnthropicTransformer(&buf)

	err := transformer.Transform(&sse.Event{Data: ""})
	require.NoError(t, err)

	err = transformer.Transform(&sse.Event{Data: "[DONE]"})
	require.NoError(t, err)

	assert.Equal(t, 0, buf.Len())
}

func TestChatToAnthropicTransformer_Usage(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewChatToAnthropicTransformer(&buf)

	chunk := types.Chunk{
		ID:    "chatcmpl-123",
		Model: "gpt-4",
		Choices: []types.Choice{{
			Delta: types.Delta{Content: "Hello"},
		}},
		Usage: &types.Usage{
			PromptTokens:     10,
			CompletionTokens: 5,
		},
	}
	chunkJSON, _ := json.Marshal(chunk)
	err := transformer.Transform(&sse.Event{Data: string(chunkJSON)})
	require.NoError(t, err)

	assert.Equal(t, 10, transformer.inputTokens)
	assert.Equal(t, 5, transformer.outputTokens)
}

func TestChatToAnthropicTransformer_Flush(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewChatToAnthropicTransformer(&buf)

	err := transformer.Flush()
	require.NoError(t, err)

	err = transformer.Close()
	require.NoError(t, err)
}

func TestConvertToolChoiceToAnthropic(t *testing.T) {
	tests := []struct {
		name     string
		input    interface{}
		expected *types.ToolChoice
	}{
		{
			name:     "nil choice",
			input:    nil,
			expected: nil,
		},
		{
			name:     "none choice",
			input:    "none",
			expected: nil,
		},
		{
			name:     "auto choice",
			input:    "auto",
			expected: &types.ToolChoice{Type: "auto"},
		},
		{
			name:     "required choice",
			input:    "required",
			expected: &types.ToolChoice{Type: "any"},
		},
		{
			name:     "unknown string defaults to auto",
			input:    "unknown",
			expected: &types.ToolChoice{Type: "auto"},
		},
		{
			name: "function object",
			input: map[string]interface{}{
				"type": "function",
				"function": map[string]interface{}{
					"name": "bash",
				},
			},
			expected: &types.ToolChoice{Type: "tool", Name: "bash"},
		},
		{
			name: "object without function name defaults to auto",
			input: map[string]interface{}{
				"type":     "function",
				"function": map[string]interface{}{},
			},
			expected: &types.ToolChoice{Type: "auto"},
		},
		{
			name: "unknown object defaults to auto",
			input: map[string]interface{}{
				"type": "unknown",
			},
			expected: &types.ToolChoice{Type: "auto"},
		},
		{
			name:     "int defaults to auto",
			input:    123,
			expected: &types.ToolChoice{Type: "auto"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := convertToolChoiceToAnthropic(tt.input)
			if tt.expected == nil {
				assert.Nil(t, result)
			} else {
				require.NotNil(t, result)
				assert.Equal(t, tt.expected.Type, result.Type)
				assert.Equal(t, tt.expected.Name, result.Name)
			}
		})
	}
}

func TestGenerateMessageID(t *testing.T) {
	id1 := generateMessageID()
	id2 := generateMessageID()

	assert.NotEmpty(t, id1)
	assert.NotEmpty(t, id2)
	assert.NotEqual(t, id1, id2)
	assert.Contains(t, id1, "msg_")
	assert.Contains(t, id2, "msg_")
}
