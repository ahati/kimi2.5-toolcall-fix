package convert

import (
	"encoding/json"
	"testing"

	"ai-proxy/types"

	"github.com/stretchr/testify/assert"
)

func TestAnthropicMessagesToOpenAI(t *testing.T) {
	tests := []struct {
		name     string
		input    []types.MessageInput
		expected []types.Message
	}{
		{
			name:     "empty messages",
			input:    []types.MessageInput{},
			expected: []types.Message{},
		},
		{
			name: "string content",
			input: []types.MessageInput{
				{Role: "user", Content: "Hello"},
			},
			expected: []types.Message{
				{Role: "user", Content: "Hello"},
			},
		},
		{
			name: "multiple messages",
			input: []types.MessageInput{
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "Hi there"},
			},
			expected: []types.Message{
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "Hi there"},
			},
		},
		{
			name: "content blocks with text",
			input: []types.MessageInput{
				{
					Role: "user",
					Content: []interface{}{
						map[string]interface{}{"type": "text", "text": "Hello"},
					},
				},
			},
			expected: []types.Message{
				{Role: "user", Content: "Hello"},
			},
		},
		{
			name: "content blocks with tool_use",
			input: []types.MessageInput{
				{
					Role: "assistant",
					Content: []interface{}{
						map[string]interface{}{
							"type":  "tool_use",
							"id":    "call_123",
							"name":  "bash",
							"input": map[string]interface{}{"command": "ls"},
						},
					},
				},
			},
			expected: []types.Message{
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
		},
		{
			name: "content blocks with tool_result",
			input: []types.MessageInput{
				{
					Role: "user",
					Content: []interface{}{
						map[string]interface{}{
							"type":        "tool_result",
							"tool_use_id": "call_123",
							"content":     "file1.txt",
						},
					},
				},
			},
			expected: []types.Message{
				{
					Role:       "user",
					Content:    "",
					ToolCallID: "call_123",
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := AnthropicMessagesToOpenAI(tt.input)
			assert.Equal(t, len(tt.expected), len(result))
			for i := range result {
				assert.Equal(t, tt.expected[i].Role, result[i].Role)
				if tt.expected[i].ToolCallID != "" {
					assert.Equal(t, tt.expected[i].ToolCallID, result[i].ToolCallID)
				}
				if len(tt.expected[i].ToolCalls) > 0 {
					assert.Equal(t, len(tt.expected[i].ToolCalls), len(result[i].ToolCalls))
					for j := range result[i].ToolCalls {
						assert.Equal(t, tt.expected[i].ToolCalls[j].ID, result[i].ToolCalls[j].ID)
						assert.Equal(t, tt.expected[i].ToolCalls[j].Function.Name, result[i].ToolCalls[j].Function.Name)
					}
				}
			}
		})
	}
}

func TestOpenAIMessagesToAnthropic(t *testing.T) {
	tests := []struct {
		name     string
		input    []types.Message
		expected []types.MessageInput
	}{
		{
			name:     "empty messages",
			input:    []types.Message{},
			expected: []types.MessageInput{},
		},
		{
			name: "string content",
			input: []types.Message{
				{Role: "user", Content: "Hello"},
			},
			expected: []types.MessageInput{
				{Role: "user", Content: "Hello"},
			},
		},
		{
			name: "tool message",
			input: []types.Message{
				{Role: "tool", Content: "result", ToolCallID: "call_123"},
			},
			expected: []types.MessageInput{
				{Role: "user"},
			},
		},
		{
			name: "message with tool calls",
			input: []types.Message{
				{
					Role:    "assistant",
					Content: "text",
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
			expected: []types.MessageInput{
				{Role: "assistant"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := OpenAIMessagesToAnthropic(tt.input)
			assert.Equal(t, len(tt.expected), len(result))
			for i := range result {
				assert.Equal(t, tt.expected[i].Role, result[i].Role)
			}
		})
	}
}

func TestAnthropicToolsToOpenAI(t *testing.T) {
	tests := []struct {
		name     string
		input    []types.ToolDef
		expected []types.Tool
	}{
		{
			name:     "empty tools",
			input:    []types.ToolDef{},
			expected: []types.Tool{},
		},
		{
			name: "single tool",
			input: []types.ToolDef{
				{
					Name:        "bash",
					Description: "Execute commands",
					InputSchema: json.RawMessage(`{"type":"object"}`),
				},
			},
			expected: []types.Tool{
				{
					Type: "function",
					Function: types.ToolFunction{
						Name:        "bash",
						Description: "Execute commands",
						Parameters:  json.RawMessage(`{"type":"object"}`),
					},
				},
			},
		},
		{
			name: "multiple tools",
			input: []types.ToolDef{
				{Name: "bash", Description: "Run bash", InputSchema: json.RawMessage(`{}`)},
				{Name: "read", Description: "Read files", InputSchema: json.RawMessage(`{}`)},
			},
			expected: []types.Tool{
				{Type: "function", Function: types.ToolFunction{Name: "bash", Description: "Run bash", Parameters: json.RawMessage(`{}`)}},
				{Type: "function", Function: types.ToolFunction{Name: "read", Description: "Read files", Parameters: json.RawMessage(`{}`)}},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := AnthropicToolsToOpenAI(tt.input)
			assert.Equal(t, len(tt.expected), len(result))
			for i := range result {
				assert.Equal(t, tt.expected[i].Type, result[i].Type)
				assert.Equal(t, tt.expected[i].Function.Name, result[i].Function.Name)
				assert.Equal(t, tt.expected[i].Function.Description, result[i].Function.Description)
			}
		})
	}
}

func TestOpenAIToolsToAnthropic(t *testing.T) {
	tests := []struct {
		name     string
		input    []types.Tool
		expected []types.ToolDef
	}{
		{
			name:     "empty tools",
			input:    []types.Tool{},
			expected: []types.ToolDef{},
		},
		{
			name: "single tool",
			input: []types.Tool{
				{
					Type: "function",
					Function: types.ToolFunction{
						Name:        "bash",
						Description: "Execute commands",
						Parameters:  json.RawMessage(`{"type":"object"}`),
					},
				},
			},
			expected: []types.ToolDef{
				{
					Name:        "bash",
					Description: "Execute commands",
					InputSchema: json.RawMessage(`{"type":"object"}`),
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := OpenAIToolsToAnthropic(tt.input)
			assert.Equal(t, len(tt.expected), len(result))
			for i := range result {
				assert.Equal(t, tt.expected[i].Name, result[i].Name)
				assert.Equal(t, tt.expected[i].Description, result[i].Description)
			}
		})
	}
}

func TestExtractTextFromContent(t *testing.T) {
	tests := []struct {
		name     string
		input    interface{}
		expected string
	}{
		{
			name:     "nil content",
			input:    nil,
			expected: "",
		},
		{
			name:     "string content",
			input:    "Hello world",
			expected: "Hello world",
		},
		{
			name: "content blocks",
			input: []interface{}{
				map[string]interface{}{"type": "text", "text": "Hello"},
				map[string]interface{}{"type": "text", "text": "World"},
			},
			expected: "Hello\nWorld",
		},
		{
			name:     "empty array",
			input:    []interface{}{},
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ExtractTextFromContent(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestConvertAnthropicContentBlocks(t *testing.T) {
	tests := []struct {
		name              string
		input             []interface{}
		expectedText      string
		expectedToolCalls int
		expectedToolID    string
	}{
		{
			name:              "empty blocks",
			input:             []interface{}{},
			expectedText:      "",
			expectedToolCalls: 0,
			expectedToolID:    "",
		},
		{
			name: "text block",
			input: []interface{}{
				map[string]interface{}{"type": "text", "text": "Hello"},
			},
			expectedText:      "Hello",
			expectedToolCalls: 0,
			expectedToolID:    "",
		},
		{
			name: "multiple text blocks",
			input: []interface{}{
				map[string]interface{}{"type": "text", "text": "Hello"},
				map[string]interface{}{"type": "text", "text": "World"},
			},
			expectedText:      "Hello\nWorld",
			expectedToolCalls: 0,
			expectedToolID:    "",
		},
		{
			name: "tool_use block",
			input: []interface{}{
				map[string]interface{}{
					"type":  "tool_use",
					"id":    "call_123",
					"name":  "bash",
					"input": map[string]interface{}{"command": "ls"},
				},
			},
			expectedText:      "",
			expectedToolCalls: 1,
			expectedToolID:    "",
		},
		{
			name: "tool_result block",
			input: []interface{}{
				map[string]interface{}{
					"type":        "tool_result",
					"tool_use_id": "call_123",
					"content":     "result",
				},
			},
			expectedText:      "",
			expectedToolCalls: 0,
			expectedToolID:    "call_123",
		},
		{
			name: "mixed blocks",
			input: []interface{}{
				map[string]interface{}{"type": "text", "text": "Running command"},
				map[string]interface{}{
					"type":  "tool_use",
					"id":    "call_123",
					"name":  "bash",
					"input": map[string]interface{}{"command": "ls"},
				},
			},
			expectedText:      "Running command",
			expectedToolCalls: 1,
			expectedToolID:    "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			text, toolCalls, toolID := ConvertAnthropicContentBlocks(tt.input)
			assert.Equal(t, tt.expectedText, text)
			assert.Equal(t, tt.expectedToolCalls, len(toolCalls))
			assert.Equal(t, tt.expectedToolID, toolID)
		})
	}
}

func TestExtractSystemMessage(t *testing.T) {
	tests := []struct {
		name     string
		input    interface{}
		expected string
	}{
		{
			name:     "nil system",
			input:    nil,
			expected: "",
		},
		{
			name:     "string system",
			input:    "You are helpful",
			expected: "You are helpful",
		},
		{
			name: "content blocks system",
			input: []interface{}{
				map[string]interface{}{"type": "text", "text": "You are helpful"},
				map[string]interface{}{"type": "text", "text": "Be concise"},
			},
			expected: "You are helpfulBe concise",
		},
		{
			name:     "empty array",
			input:    []interface{}{},
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ExtractSystemMessage(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestResponsesInputToAnthropicMessages(t *testing.T) {
	tests := []struct {
		name          string
		input         interface{}
		expectedLen   int
		expectedRoles []string
	}{
		{
			name:        "nil input",
			input:       nil,
			expectedLen: 0,
		},
		{
			name:          "string input",
			input:         "Hello",
			expectedLen:   1,
			expectedRoles: []string{"user"},
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
			expectedLen:   1,
			expectedRoles: []string{"user"},
		},
		{
			name: "skip developer message",
			input: []interface{}{
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
			expectedLen:   1,
			expectedRoles: []string{"user"},
		},
		{
			name: "function call",
			input: []interface{}{
				map[string]interface{}{
					"type":      "function_call",
					"call_id":   "call_123",
					"name":      "bash",
					"arguments": `{"command":"ls"}`,
				},
			},
			expectedLen:   1,
			expectedRoles: []string{"assistant"},
		},
		{
			name: "function call output",
			input: []interface{}{
				map[string]interface{}{
					"type":    "function_call_output",
					"call_id": "call_123",
					"output":  "file1.txt",
				},
			},
			expectedLen:   1,
			expectedRoles: []string{"user"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ResponsesInputToAnthropicMessages(tt.input)
			assert.Equal(t, tt.expectedLen, len(result))
			if tt.expectedRoles != nil {
				for i, role := range tt.expectedRoles {
					assert.Equal(t, role, result[i].Role)
				}
			}
		})
	}
}

func TestResponsesToolsToAnthropic(t *testing.T) {
	tests := []struct {
		name          string
		input         []types.ResponsesTool
		expectedLen   int
		expectedNames []string
	}{
		{
			name:        "empty tools",
			input:       []types.ResponsesTool{},
			expectedLen: 0,
		},
		{
			name:        "nil tools",
			input:       nil,
			expectedLen: 0,
		},
		{
			name: "flat format tool",
			input: []types.ResponsesTool{
				{
					Type:        "function",
					Name:        "bash",
					Description: "Execute commands",
					Parameters:  json.RawMessage(`{"type":"object"}`),
				},
			},
			expectedLen:   1,
			expectedNames: []string{"bash"},
		},
		{
			name: "nested format tool",
			input: []types.ResponsesTool{
				{
					Type: "function",
					Function: &types.ResponsesToolFunction{
						Name:        "read",
						Description: "Read files",
						Parameters:  json.RawMessage(`{"type":"object"}`),
					},
				},
			},
			expectedLen:   1,
			expectedNames: []string{"read"},
		},
		{
			name: "skip non-function tools",
			input: []types.ResponsesTool{
				{Type: "web_search"},
				{Type: "function", Name: "bash"},
			},
			expectedLen:   1,
			expectedNames: []string{"bash"},
		},
		{
			name: "multiple tools",
			input: []types.ResponsesTool{
				{Type: "function", Name: "bash", Description: "Run bash"},
				{Type: "function", Name: "read", Description: "Read files"},
			},
			expectedLen:   2,
			expectedNames: []string{"bash", "read"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ResponsesToolsToAnthropic(tt.input)
			if tt.input == nil {
				assert.Nil(t, result)
			} else {
				assert.Equal(t, tt.expectedLen, len(result))
				for i, name := range tt.expectedNames {
					assert.Equal(t, name, result[i].Name)
				}
			}
		})
	}
}

func TestResponsesToolChoiceToAnthropic(t *testing.T) {
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
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ResponsesToolChoiceToAnthropic(tt.input)
			if tt.expected == nil {
				assert.Nil(t, result)
			} else {
				assert.Equal(t, tt.expected.Type, result.Type)
				assert.Equal(t, tt.expected.Name, result.Name)
			}
		})
	}
}

func TestExtractResponsesContent(t *testing.T) {
	tests := []struct {
		name     string
		input    interface{}
		expected string
	}{
		{
			name:     "nil content",
			input:    nil,
			expected: "",
		},
		{
			name:     "string content",
			input:    "Hello world",
			expected: "Hello world",
		},
		{
			name: "input_text parts",
			input: []interface{}{
				map[string]interface{}{"type": "input_text", "text": "Hello"},
				map[string]interface{}{"type": "input_text", "text": "World"},
			},
			expected: "Hello\nWorld",
		},
		{
			name: "input_image part",
			input: []interface{}{
				map[string]interface{}{"type": "input_text", "text": "Check this"},
				map[string]interface{}{"type": "input_image", "image_url": "http://example.com/img.png"},
			},
			expected: "Check this\n[Image attached]",
		},
		{
			name: "only input_image",
			input: []interface{}{
				map[string]interface{}{"type": "input_image", "image_url": "http://example.com/img.png"},
			},
			expected: "[Image attached]",
		},
		{
			name:     "empty array",
			input:    []interface{}{},
			expected: "",
		},
		{
			name:     "unknown type",
			input:    123,
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractResponsesContent(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestExtractTextFromContent_EdgeCases(t *testing.T) {
	tests := []struct {
		name     string
		input    interface{}
		expected string
	}{
		{
			name: "non-map item in array",
			input: []interface{}{
				"string item",
				map[string]interface{}{"type": "text", "text": "Hello"},
			},
			expected: "Hello",
		},
		{
			name: "map without text field",
			input: []interface{}{
				map[string]interface{}{"type": "other", "data": "value"},
			},
			expected: "",
		},
		{
			name:     "unknown type",
			input:    12345,
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ExtractTextFromContent(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestConvertAnthropicContentBlocks_EdgeCases(t *testing.T) {
	tests := []struct {
		name              string
		input             []interface{}
		expectedText      string
		expectedToolCalls int
		expectedToolID    string
	}{
		{
			name: "non-map item",
			input: []interface{}{
				"string item",
			},
			expectedText:      "",
			expectedToolCalls: 0,
			expectedToolID:    "",
		},
		{
			name: "tool_use missing fields",
			input: []interface{}{
				map[string]interface{}{"type": "tool_use"},
			},
			expectedText:      "",
			expectedToolCalls: 0,
			expectedToolID:    "",
		},
		{
			name: "tool_use with id but no name",
			input: []interface{}{
				map[string]interface{}{"type": "tool_use", "id": "call_123"},
			},
			expectedText:      "",
			expectedToolCalls: 0,
			expectedToolID:    "",
		},
		{
			name: "unknown block type",
			input: []interface{}{
				map[string]interface{}{"type": "unknown"},
			},
			expectedText:      "",
			expectedToolCalls: 0,
			expectedToolID:    "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			text, toolCalls, toolID := ConvertAnthropicContentBlocks(tt.input)
			assert.Equal(t, tt.expectedText, text)
			assert.Equal(t, tt.expectedToolCalls, len(toolCalls))
			assert.Equal(t, tt.expectedToolID, toolID)
		})
	}
}

func TestExtractSystemMessage_EdgeCases(t *testing.T) {
	tests := []struct {
		name     string
		input    interface{}
		expected string
	}{
		{
			name: "non-map item in array",
			input: []interface{}{
				"string item",
				map[string]interface{}{"type": "text", "text": "Hello"},
			},
			expected: "Hello",
		},
		{
			name: "map without text field",
			input: []interface{}{
				map[string]interface{}{"type": "other"},
			},
			expected: "",
		},
		{
			name:     "unknown type",
			input:    12345,
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ExtractSystemMessage(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestOpenAIMessageToAnthropic_EdgeCases(t *testing.T) {
	tests := []struct {
		name         string
		input        types.Message
		expectedRole string
		checkContent bool
	}{
		{
			name: "assistant with tool calls and content",
			input: types.Message{
				Role:    "assistant",
				Content: "Thinking...",
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
			expectedRole: "assistant",
			checkContent: true,
		},
		{
			name: "assistant with tool calls empty content",
			input: types.Message{
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
			expectedRole: "assistant",
			checkContent: true,
		},
		{
			name: "non-string content without tool calls",
			input: types.Message{
				Role:    "user",
				Content: map[string]interface{}{"key": "value"},
			},
			expectedRole: "user",
			checkContent: false,
		},
		{
			name: "tool calls with invalid JSON arguments",
			input: types.Message{
				Role:    "assistant",
				Content: "",
				ToolCalls: []types.ToolCall{
					{
						ID:   "call_123",
						Type: "function",
						Function: types.Function{
							Name:      "bash",
							Arguments: `{invalid json`,
						},
					},
				},
			},
			expectedRole: "assistant",
			checkContent: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := openAIMessageToAnthropic(tt.input)
			assert.Equal(t, tt.expectedRole, result.Role)
			if tt.checkContent {
				assert.NotNil(t, result.Content)
			}
		})
	}
}

func TestResponsesInputToAnthropicMessages_EdgeCases(t *testing.T) {
	tests := []struct {
		name         string
		input        interface{}
		expectedLen  int
		expectedRole string
	}{
		{
			name: "message with array content containing input_text",
			input: []interface{}{
				map[string]interface{}{
					"type": "message",
					"role": "user",
					"content": []interface{}{
						map[string]interface{}{"type": "input_text", "text": "Hello"},
						map[string]interface{}{"type": "input_image", "image_url": "http://example.com/img.png"},
					},
				},
			},
			expectedLen:  1,
			expectedRole: "user",
		},
		{
			name: "message with empty content skipped",
			input: []interface{}{
				map[string]interface{}{
					"type":    "message",
					"role":    "user",
					"content": []interface{}{},
				},
			},
			expectedLen: 0,
		},
		{
			name: "non-map item in array",
			input: []interface{}{
				"string item",
			},
			expectedLen: 0,
		},
		{
			name: "message with default role",
			input: []interface{}{
				map[string]interface{}{
					"type":    "message",
					"content": "Hello",
				},
			},
			expectedLen:  1,
			expectedRole: "user",
		},
		{
			name: "unknown item type",
			input: []interface{}{
				map[string]interface{}{
					"type": "unknown",
				},
			},
			expectedLen: 0,
		},
		{
			name: "reasoning item skipped",
			input: []interface{}{
				map[string]interface{}{
					"type": "reasoning",
					"summary": []interface{}{
						map[string]interface{}{"type": "summary_text", "text": "thinking..."},
					},
				},
			},
			expectedLen: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ResponsesInputToAnthropicMessages(tt.input)
			assert.Equal(t, tt.expectedLen, len(result))
			if tt.expectedLen > 0 && tt.expectedRole != "" {
				assert.Equal(t, tt.expectedRole, result[0].Role)
			}
		})
	}
}
