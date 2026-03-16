package convert

import (
	"ai-proxy/types"
	"encoding/json"
	"testing"
)

// TestConvertAnthropicMessagesToOpenAI tests the conversion from Anthropic to OpenAI messages.
func TestConvertAnthropicMessagesToOpenAI(t *testing.T) {
	tests := []struct {
		name     string
		input    []types.MessageInput
		expected []types.Message
	}{
		{
			name:     "empty slice",
			input:    []types.MessageInput{},
			expected: []types.Message{},
		},
		{
			name:     "nil slice",
			input:    nil,
			expected: []types.Message{},
		},
		{
			name: "simple string content",
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
				{Role: "assistant", Content: "Hi there!"},
			},
			expected: []types.Message{
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "Hi there!"},
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
						map[string]interface{}{"type": "text", "text": "Let me help you."},
						map[string]interface{}{
							"type":  "tool_use",
							"id":    "call_123",
							"name":  "get_weather",
							"input": map[string]interface{}{"city": "Paris"},
						},
					},
				},
			},
			expected: []types.Message{
				{
					Role:    "assistant",
					Content: "Let me help you.",
					ToolCalls: []types.ToolCall{
						{
							ID:   "call_123",
							Type: "function",
							Function: types.Function{
								Name:      "get_weather",
								Arguments: `{"city":"Paris"}`,
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
							"content":     "Sunny, 25°C",
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
			result := ConvertAnthropicMessagesToOpenAI(tt.input)

			if len(result) != len(tt.expected) {
				t.Errorf("expected %d messages, got %d", len(tt.expected), len(result))
				return
			}

			for i, msg := range result {
				exp := tt.expected[i]
				if msg.Role != exp.Role {
					t.Errorf("message %d: expected role %q, got %q", i, exp.Role, msg.Role)
				}
				if ExtractTextFromContent(msg.Content) != ExtractTextFromContent(exp.Content) {
					t.Errorf("message %d: expected content %v, got %v", i, exp.Content, msg.Content)
				}
				if len(msg.ToolCalls) != len(exp.ToolCalls) {
					t.Errorf("message %d: expected %d tool calls, got %d", i, len(exp.ToolCalls), len(msg.ToolCalls))
				}
				if msg.ToolCallID != exp.ToolCallID {
					t.Errorf("message %d: expected tool_call_id %q, got %q", i, exp.ToolCallID, msg.ToolCallID)
				}
			}
		})
	}
}

// TestConvertOpenAIMessagesToAnthropic tests the conversion from OpenAI to Anthropic messages.
func TestConvertOpenAIMessagesToAnthropic(t *testing.T) {
	tests := []struct {
		name     string
		input    []types.Message
		expected []types.MessageInput
	}{
		{
			name:     "empty slice",
			input:    []types.Message{},
			expected: []types.MessageInput{},
		},
		{
			name:     "nil slice",
			input:    nil,
			expected: []types.MessageInput{},
		},
		{
			name: "simple string content",
			input: []types.Message{
				{Role: "user", Content: "Hello"},
			},
			expected: []types.MessageInput{
				{Role: "user", Content: "Hello"},
			},
		},
		{
			name: "tool response message",
			input: []types.Message{
				{
					Role:       "tool",
					Content:    "Tool result",
					ToolCallID: "call_123",
				},
			},
			expected: []types.MessageInput{
				{
					Role: "user",
					Content: []map[string]interface{}{
						{
							"type":        "tool_result",
							"tool_use_id": "call_123",
							"content":     "Tool result",
						},
					},
				},
			},
		},
		{
			name: "assistant message with tool calls",
			input: []types.Message{
				{
					Role:    "assistant",
					Content: "Let me check that.",
					ToolCalls: []types.ToolCall{
						{
							ID:   "call_456",
							Type: "function",
							Function: types.Function{
								Name:      "search",
								Arguments: `{"query":"test"}`,
							},
						},
					},
				},
			},
			expected: []types.MessageInput{
				{
					Role: "assistant",
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ConvertOpenAIMessagesToAnthropic(tt.input)

			if len(result) != len(tt.expected) {
				t.Errorf("expected %d messages, got %d", len(tt.expected), len(result))
				return
			}

			for i, msg := range result {
				exp := tt.expected[i]
				if msg.Role != exp.Role {
					t.Errorf("message %d: expected role %q, got %q", i, exp.Role, msg.Role)
				}
			}
		})
	}
}

// TestConvertAnthropicToolsToOpenAI tests the conversion from Anthropic to OpenAI tools.
func TestConvertAnthropicToolsToOpenAI(t *testing.T) {
	schema := json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`)

	tests := []struct {
		name     string
		input    []types.ToolDef
		expected []types.Tool
	}{
		{
			name:     "empty slice",
			input:    []types.ToolDef{},
			expected: []types.Tool{},
		},
		{
			name:     "nil slice",
			input:    nil,
			expected: []types.Tool{},
		},
		{
			name: "single tool",
			input: []types.ToolDef{
				{
					Name:        "get_weather",
					Description: "Get weather info",
					InputSchema: schema,
				},
			},
			expected: []types.Tool{
				{
					Type: "function",
					Function: types.ToolFunction{
						Name:        "get_weather",
						Description: "Get weather info",
						Parameters:  schema,
					},
				},
			},
		},
		{
			name: "multiple tools",
			input: []types.ToolDef{
				{
					Name:        "get_weather",
					Description: "Get weather",
					InputSchema: schema,
				},
				{
					Name:        "search",
					Description: "Search the web",
					InputSchema: schema,
				},
			},
			expected: []types.Tool{
				{
					Type: "function",
					Function: types.ToolFunction{
						Name:        "get_weather",
						Description: "Get weather",
						Parameters:  schema,
					},
				},
				{
					Type: "function",
					Function: types.ToolFunction{
						Name:        "search",
						Description: "Search the web",
						Parameters:  schema,
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ConvertAnthropicToolsToOpenAI(tt.input)

			if len(result) != len(tt.expected) {
				t.Errorf("expected %d tools, got %d", len(tt.expected), len(result))
				return
			}

			for i, tool := range result {
				exp := tt.expected[i]
				if tool.Type != exp.Type {
					t.Errorf("tool %d: expected type %q, got %q", i, exp.Type, tool.Type)
				}
				if tool.Function.Name != exp.Function.Name {
					t.Errorf("tool %d: expected name %q, got %q", i, exp.Function.Name, tool.Function.Name)
				}
				if tool.Function.Description != exp.Function.Description {
					t.Errorf("tool %d: expected description %q, got %q", i, exp.Function.Description, tool.Function.Description)
				}
			}
		})
	}
}

// TestConvertOpenAIToolsToAnthropic tests the conversion from OpenAI to Anthropic tools.
func TestConvertOpenAIToolsToAnthropic(t *testing.T) {
	schema := json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}}}`)

	tests := []struct {
		name     string
		input    []types.Tool
		expected []types.ToolDef
	}{
		{
			name:     "empty slice",
			input:    []types.Tool{},
			expected: []types.ToolDef{},
		},
		{
			name:     "nil slice",
			input:    nil,
			expected: []types.ToolDef{},
		},
		{
			name: "single function tool",
			input: []types.Tool{
				{
					Type: "function",
					Function: types.ToolFunction{
						Name:        "search",
						Description: "Search the web",
						Parameters:  schema,
					},
				},
			},
			expected: []types.ToolDef{
				{
					Name:        "search",
					Description: "Search the web",
					InputSchema: schema,
				},
			},
		},
		{
			name: "skip non-function tools",
			input: []types.Tool{
				{
					Type: "custom",
					Function: types.ToolFunction{
						Name: "custom_tool",
					},
				},
				{
					Type: "function",
					Function: types.ToolFunction{
						Name:        "search",
						Description: "Search",
						Parameters:  schema,
					},
				},
			},
			expected: []types.ToolDef{
				{
					Name:        "search",
					Description: "Search",
					InputSchema: schema,
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ConvertOpenAIToolsToAnthropic(tt.input)

			if len(result) != len(tt.expected) {
				t.Errorf("expected %d tools, got %d", len(tt.expected), len(result))
				return
			}

			for i, tool := range result {
				exp := tt.expected[i]
				if tool.Name != exp.Name {
					t.Errorf("tool %d: expected name %q, got %q", i, exp.Name, tool.Name)
				}
				if tool.Description != exp.Description {
					t.Errorf("tool %d: expected description %q, got %q", i, exp.Description, tool.Description)
				}
			}
		})
	}
}

// TestExtractTextFromContent tests text extraction from various content formats.
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
			name:     "empty string",
			input:    "",
			expected: "",
		},
		{
			name:     "simple string",
			input:    "Hello, world!",
			expected: "Hello, world!",
		},
		{
			name:     "empty array",
			input:    []interface{}{},
			expected: "",
		},
		{
			name: "single text block",
			input: []interface{}{
				map[string]interface{}{"type": "text", "text": "Hello"},
			},
			expected: "Hello",
		},
		{
			name: "multiple text blocks",
			input: []interface{}{
				map[string]interface{}{"type": "text", "text": "Hello"},
				map[string]interface{}{"type": "text", "text": "World"},
			},
			expected: "Hello\nWorld",
		},
		{
			name: "input_text block",
			input: []interface{}{
				map[string]interface{}{"type": "input_text", "text": "Input text"},
			},
			expected: "Input text",
		},
		{
			name: "mixed content blocks",
			input: []interface{}{
				map[string]interface{}{"type": "text", "text": "Text"},
				map[string]interface{}{"type": "image", "source": "data"},
				map[string]interface{}{"type": "text", "text": "More text"},
			},
			expected: "Text\nMore text",
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
			if result != tt.expected {
				t.Errorf("expected %q, got %q", tt.expected, result)
			}
		})
	}
}

// TestConvertContentBlocks tests the conversion of Anthropic content blocks.
func TestConvertContentBlocks(t *testing.T) {
	tests := []struct {
		name               string
		input              []interface{}
		expectedText       string
		expectedToolCalls  int
		expectedToolCallID string
	}{
		{
			name:               "empty slice",
			input:              []interface{}{},
			expectedText:       "",
			expectedToolCalls:  0,
			expectedToolCallID: "",
		},
		{
			name: "text blocks only",
			input: []interface{}{
				map[string]interface{}{"type": "text", "text": "Hello"},
				map[string]interface{}{"type": "text", "text": "World"},
			},
			expectedText:       "Hello\nWorld",
			expectedToolCalls:  0,
			expectedToolCallID: "",
		},
		{
			name: "tool_use block",
			input: []interface{}{
				map[string]interface{}{
					"type":  "tool_use",
					"id":    "call_123",
					"name":  "get_weather",
					"input": map[string]interface{}{"city": "Paris"},
				},
			},
			expectedText:       "",
			expectedToolCalls:  1,
			expectedToolCallID: "",
		},
		{
			name: "tool_result block",
			input: []interface{}{
				map[string]interface{}{
					"type":        "tool_result",
					"tool_use_id": "call_123",
					"content":     "Sunny",
				},
			},
			expectedText:       "",
			expectedToolCalls:  0,
			expectedToolCallID: "call_123",
		},
		{
			name: "mixed content",
			input: []interface{}{
				map[string]interface{}{"type": "text", "text": "Response:"},
				map[string]interface{}{
					"type":  "tool_use",
					"id":    "call_456",
					"name":  "search",
					"input": map[string]interface{}{"q": "test"},
				},
			},
			expectedText:       "Response:",
			expectedToolCalls:  1,
			expectedToolCallID: "",
		},
		{
			name: "non-map items are skipped",
			input: []interface{}{
				"string item",
				map[string]interface{}{"type": "text", "text": "Valid"},
			},
			expectedText:       "Valid",
			expectedToolCalls:  0,
			expectedToolCallID: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			text, toolCalls, toolCallID := ConvertContentBlocks(tt.input)

			if text != tt.expectedText {
				t.Errorf("expected text %q, got %q", tt.expectedText, text)
			}
			if len(toolCalls) != tt.expectedToolCalls {
				t.Errorf("expected %d tool calls, got %d", tt.expectedToolCalls, len(toolCalls))
			}
			if toolCallID != tt.expectedToolCallID {
				t.Errorf("expected tool_call_id %q, got %q", tt.expectedToolCallID, toolCallID)
			}

			// Verify tool call details if present
			if len(toolCalls) > 0 {
				for i, tc := range toolCalls {
					if tc.Type != "function" {
						t.Errorf("tool call %d: expected type 'function', got %q", i, tc.Type)
					}
					if tc.ID == "" {
						t.Errorf("tool call %d: ID should not be empty", i)
					}
					if tc.Function.Name == "" {
						t.Errorf("tool call %d: Function.Name should not be empty", i)
					}
				}
			}
		})
	}
}

// TestExtractSystemMessage tests system message extraction from various formats.
func TestExtractSystemMessage(t *testing.T) {
	tests := []struct {
		name     string
		input    interface{}
		expected string
	}{
		{
			name:     "nil",
			input:    nil,
			expected: "",
		},
		{
			name:     "empty string",
			input:    "",
			expected: "",
		},
		{
			name:     "simple string",
			input:    "You are a helpful assistant.",
			expected: "You are a helpful assistant.",
		},
		{
			name:     "empty array",
			input:    []interface{}{},
			expected: "",
		},
		{
			name: "content blocks array",
			input: []interface{}{
				map[string]interface{}{"type": "text", "text": "You are helpful."},
				map[string]interface{}{"type": "text", "text": "Be concise."},
			},
			expected: "You are helpful.Be concise.",
		},
		{
			name:     "unknown type",
			input:    42,
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ExtractSystemMessage(tt.input)
			if result != tt.expected {
				t.Errorf("expected %q, got %q", tt.expected, result)
			}
		})
	}
}

// TestRoundTripConversion tests that converting back and forth preserves essential data.
func TestRoundTripConversion(t *testing.T) {
	t.Run("tools round trip", func(t *testing.T) {
		schema := json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`)
		original := []types.ToolDef{
			{
				Name:        "get_weather",
				Description: "Get weather",
				InputSchema: schema,
			},
		}

		// Anthropic -> OpenAI -> Anthropic
		openaiTools := ConvertAnthropicToolsToOpenAI(original)
		anthropicTools := ConvertOpenAIToolsToAnthropic(openaiTools)

		if len(anthropicTools) != len(original) {
			t.Errorf("expected %d tools, got %d", len(original), len(anthropicTools))
			return
		}

		if anthropicTools[0].Name != original[0].Name {
			t.Errorf("expected name %q, got %q", original[0].Name, anthropicTools[0].Name)
		}
	})

	t.Run("simple messages round trip", func(t *testing.T) {
		original := []types.MessageInput{
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Hi!"},
		}

		// Anthropic -> OpenAI -> Anthropic
		openaiMsgs := ConvertAnthropicMessagesToOpenAI(original)
		anthropicMsgs := ConvertOpenAIMessagesToAnthropic(openaiMsgs)

		if len(anthropicMsgs) != len(original) {
			t.Errorf("expected %d messages, got %d", len(original), len(anthropicMsgs))
			return
		}

		for i, msg := range anthropicMsgs {
			if msg.Role != original[i].Role {
				t.Errorf("message %d: expected role %q, got %q", i, original[i].Role, msg.Role)
			}
		}
	})
}

// TestExtractMediaType tests the media type extraction from data URLs.
func TestExtractMediaType(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "PNG image",
			input:    "data:image/png;base64,abc123",
			expected: "image/png",
		},
		{
			name:     "JPEG image",
			input:    "data:image/jpeg;base64,def456",
			expected: "image/jpeg",
		},
		{
			name:     "no data prefix",
			input:    "https://example.com/image.png",
			expected: "",
		},
		{
			name:     "empty string",
			input:    "",
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractMediaType(tt.input)
			if result != tt.expected {
				t.Errorf("expected %q, got %q", tt.expected, result)
			}
		})
	}
}

// TestExtractBase64Data tests the base64 data extraction from data URLs.
func TestExtractBase64Data(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "standard data URL",
			input:    "data:image/png;base64,abc123",
			expected: "abc123",
		},
		{
			name:     "no comma",
			input:    "data:image/png;base64",
			expected: "",
		},
		{
			name:     "empty string",
			input:    "",
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractBase64Data(tt.input)
			if result != tt.expected {
				t.Errorf("expected %q, got %q", tt.expected, result)
			}
		})
	}
}
