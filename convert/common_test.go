package convert

import (
	"ai-proxy/types"
	"encoding/json"
	"reflect"
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
					Content:    "Sunny, 25°C",
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
		{
			name: "consecutive tool messages are batched",
			input: []types.Message{
				{
					Role:       "tool",
					Content:    "Result 1",
					ToolCallID: "call_1",
				},
				{
					Role:       "tool",
					Content:    "Result 2",
					ToolCallID: "call_2",
				},
			},
			expected: []types.MessageInput{
				{
					Role: "user",
					Content: []interface{}{
						map[string]interface{}{
							"type":        "tool_result",
							"tool_use_id": "call_1",
							"content":     "Result 1",
						},
						map[string]interface{}{
							"type":        "tool_result",
							"tool_use_id": "call_2",
							"content":     "Result 2",
						},
					},
				},
			},
		},
		{
			name: "tool messages separated by other messages are not batched",
			input: []types.Message{
				{
					Role:       "tool",
					Content:    "Result 1",
					ToolCallID: "call_1",
				},
				{
					Role:    "assistant",
					Content: "Let me check that.",
				},
				{
					Role:       "tool",
					Content:    "Result 2",
					ToolCallID: "call_2",
				},
			},
			expected: []types.MessageInput{
				{
					Role: "user",
				},
				{
					Role:    "assistant",
					Content: "Let me check that.",
				},
				{
					Role: "user",
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
			expectedText:       "Sunny",
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

// TestFlattenToolResultContent tests canonical tool_result flattening.
func TestFlattenToolResultContent(t *testing.T) {
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
			name:     "string",
			input:    "Sunny, 25°C",
			expected: "Sunny, 25°C",
		},
		{
			name: "mixed block array",
			input: []interface{}{
				map[string]interface{}{"type": "text", "text": "Result"},
				map[string]interface{}{"type": "image", "source": map[string]interface{}{"type": "url", "url": "https://example.com"}},
				map[string]interface{}{"type": "text", "text": "Done"},
			},
			expected: "Result\nDone",
		},
		{
			name: "thinking is ignored",
			input: []interface{}{
				map[string]interface{}{"type": "thinking", "thinking": "internal"},
				map[string]interface{}{"type": "output_text", "text": "Visible"},
			},
			expected: "Visible",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := FlattenToolResultContent(tt.input); got != tt.expected {
				t.Errorf("FlattenToolResultContent() = %q, want %q", got, tt.expected)
			}
		})
	}
}

// TestExtractSystemText tests system extraction from Anthropic system payloads.
func TestExtractSystemText(t *testing.T) {
	tests := []struct {
		name     string
		input    interface{}
		expected string
	}{
		{
			name:     "json string",
			input:    json.RawMessage(`"You are helpful."`),
			expected: "You are helpful.",
		},
		{
			name:     "raw block array",
			input:    json.RawMessage(`[{"type":"text","text":"You are helpful."},{"type":"text","text":"Be concise."}]`),
			expected: "You are helpful.Be concise.",
		},
		{
			name: "structured blocks preserve order",
			input: []interface{}{
				map[string]interface{}{"type": "text", "text": "One"},
				map[string]interface{}{"type": "text", "text": "Two"},
			},
			expected: "OneTwo",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ExtractSystemText(tt.input); got != tt.expected {
				t.Errorf("ExtractSystemText() = %q, want %q", got, tt.expected)
			}
		})
	}
}

// TestImageHelpers tests image URL and data URI helper round-trips.
func TestImageHelpers(t *testing.T) {
	dataURL := "data:image/png;base64,abc123"
	source, err := OpenAIImageURLToAnthropicSource(dataURL)
	if err != nil {
		t.Fatalf("OpenAIImageURLToAnthropicSource(data URL) returned error: %v", err)
	}
	if source.Type != "base64" || source.MediaType != "image/png" || source.Data != "abc123" {
		t.Fatalf("unexpected source from data URL: %+v", source)
	}

	if got, err := AnthropicImageSourceToURL(source); err != nil || got != dataURL {
		t.Fatalf("AnthropicImageSourceToURL(data URL) = %q, %v", got, err)
	}

	chatPart, err := BuildChatImagePartFromAnthropicSource(source)
	if err != nil {
		t.Fatalf("BuildChatImagePartFromAnthropicSource returned error: %v", err)
	}
	expectedChatPart := map[string]interface{}{
		"type": "image_url",
		"image_url": map[string]interface{}{
			"url": dataURL,
		},
	}
	if !reflect.DeepEqual(chatPart, expectedChatPart) {
		t.Fatalf("unexpected chat image part: got %#v want %#v", chatPart, expectedChatPart)
	}

	responsesPart, err := BuildResponsesImagePartFromAnthropicSource(source)
	if err != nil {
		t.Fatalf("BuildResponsesImagePartFromAnthropicSource returned error: %v", err)
	}
	if responsesPart.Type != "input_image" || responsesPart.ImageURL != dataURL {
		t.Fatalf("unexpected responses image part: %+v", responsesPart)
	}

	urlSource, err := OpenAIImageURLToAnthropicSource("https://example.com/image.png")
	if err != nil {
		t.Fatalf("OpenAIImageURLToAnthropicSource(url) returned error: %v", err)
	}
	if urlSource.Type != "url" || urlSource.URL != "https://example.com/image.png" {
		t.Fatalf("unexpected URL source: %+v", urlSource)
	}
	if got, err := AnthropicImageSourceToURL(urlSource); err != nil || got != "https://example.com/image.png" {
		t.Fatalf("AnthropicImageSourceToURL(url) = %q, %v", got, err)
	}
}

// TestToolChoiceHelpers tests tool choice normalization helpers.
func TestToolChoiceHelpers(t *testing.T) {
	anthropicCases := []struct {
		name     string
		input    interface{}
		expected *types.ToolChoice
	}{
		{
			name:     "string auto",
			input:    "auto",
			expected: &types.ToolChoice{Type: "auto"},
		},
		{
			name:     "string required",
			input:    "required",
			expected: &types.ToolChoice{Type: "any"},
		},
		{
			name:  "string none",
			input: "none",
		},
		{
			name:     "flat responses object",
			input:    map[string]interface{}{"type": "function", "name": "search"},
			expected: &types.ToolChoice{Type: "tool", Name: "search"},
		},
		{
			name:     "nested function object",
			input:    map[string]interface{}{"type": "function", "function": map[string]interface{}{"name": "search"}},
			expected: &types.ToolChoice{Type: "tool", Name: "search"},
		},
	}

	for _, tt := range anthropicCases {
		t.Run("anthropic_"+tt.name, func(t *testing.T) {
			got := ConvertToolChoiceOpenAIToAnthropic(tt.input)
			if tt.expected == nil {
				if got != nil {
					t.Fatalf("expected nil, got %+v", got)
				}
				return
			}
			if got == nil {
				t.Fatalf("expected %+v, got nil", tt.expected)
			}
			if got.Type != tt.expected.Type || got.Name != tt.expected.Name {
				t.Fatalf("got %+v, want %+v", got, tt.expected)
			}
		})
	}

	responsesCases := []struct {
		name     string
		input    interface{}
		expected interface{}
	}{
		{
			name:     "string auto",
			input:    "auto",
			expected: "auto",
		},
		{
			name:     "flat responses object",
			input:    map[string]interface{}{"type": "function", "name": "search"},
			expected: map[string]interface{}{"type": "function", "function": map[string]interface{}{"name": "search"}},
		},
		{
			name:     "nested function object",
			input:    map[string]interface{}{"type": "function", "function": map[string]interface{}{"name": "search"}},
			expected: map[string]interface{}{"type": "function", "function": map[string]interface{}{"name": "search"}},
		},
	}

	for _, tt := range responsesCases {
		t.Run("responses_"+tt.name, func(t *testing.T) {
			got := ConvertResponsesToolChoiceToOpenAI(tt.input)
			if !reflect.DeepEqual(got, tt.expected) {
				t.Fatalf("got %#v, want %#v", got, tt.expected)
			}
		})
	}
}

// TestNormalizeAnthropicMessages tests Anthropic alternation normalization.
func TestNormalizeAnthropicMessages(t *testing.T) {
	tests := []struct {
		name     string
		input    []types.MessageInput
		expected []types.MessageInput
	}{
		{
			name: "starts with assistant",
			input: []types.MessageInput{
				{Role: "assistant", Content: "Hi"},
			},
			expected: []types.MessageInput{
				{Role: "user", Content: []interface{}{}},
				{Role: "assistant", Content: "Hi"},
			},
		},
		{
			name: "consecutive same roles",
			input: []types.MessageInput{
				{Role: "user", Content: "One"},
				{Role: "user", Content: "Two"},
				{Role: "assistant", Content: "Three"},
			},
			expected: []types.MessageInput{
				{Role: "user", Content: "One"},
				{Role: "assistant", Content: []interface{}{}},
				{Role: "user", Content: "Two"},
				{Role: "assistant", Content: "Three"},
			},
		},
		{
			name: "already alternating",
			input: []types.MessageInput{
				{Role: "user", Content: "One"},
				{Role: "assistant", Content: "Two"},
			},
			expected: []types.MessageInput{
				{Role: "user", Content: "One"},
				{Role: "assistant", Content: "Two"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NormalizeAnthropicMessages(tt.input)
			if len(got) != len(tt.expected) {
				t.Fatalf("len(got) = %d, want %d", len(got), len(tt.expected))
			}
			for i := range got {
				if got[i].Role != tt.expected[i].Role {
					t.Fatalf("message %d role = %q, want %q", i, got[i].Role, tt.expected[i].Role)
				}
				if !reflect.DeepEqual(got[i].Content, tt.expected[i].Content) {
					t.Fatalf("message %d content = %#v, want %#v", i, got[i].Content, tt.expected[i].Content)
				}
			}
		})
	}
}

// TestAnthropicNormalizationHelpers tests temperature and max token helpers.
func TestAnthropicNormalizationHelpers(t *testing.T) {
	if got := ClampTemperatureToAnthropic(-1); got != 0 {
		t.Fatalf("ClampTemperatureToAnthropic(-1) = %v, want 0", got)
	}
	if got := NormalizeAnthropicTemperature(1.7); got != 1 {
		t.Fatalf("NormalizeAnthropicTemperature(1.7) = %v, want 1", got)
	}
	if got := ResolveAnthropicMaxTokens(0); got != DefaultAnthropicMaxTokens {
		t.Fatalf("ResolveAnthropicMaxTokens(0) = %d, want %d", got, DefaultAnthropicMaxTokens)
	}
	if got := ResolveAnthropicMaxTokens(2048); got != 2048 {
		t.Fatalf("ResolveAnthropicMaxTokens(2048) = %d, want 2048", got)
	}
}
