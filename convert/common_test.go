package convert

import (
	"encoding/json"
	"testing"

	"ai-proxy/types"
	"github.com/stretchr/testify/assert"
)

func TestConvertAnthropicMessagesToOpenAI(t *testing.T) {
	tests := []struct {
		name     string
		input    []types.MessageInput
		expected []types.Message
	}{
		{
			name:     "nil input",
			input:    nil,
			expected: nil,
		},
		{
			name:     "empty slice",
			input:    []types.MessageInput{},
			expected: []types.Message{},
		},
		{
			name: "single message with string content",
			input: []types.MessageInput{
				{Role: "user", Content: "hello"},
			},
			expected: []types.Message{
				{Role: "user", Content: "hello"},
			},
		},
		{
			name: "multiple messages",
			input: []types.MessageInput{
				{Role: "user", Content: "hello"},
				{Role: "assistant", Content: "hi"},
				{Role: "user", Content: "how are you?"},
			},
			expected: []types.Message{
				{Role: "user", Content: "hello"},
				{Role: "assistant", Content: "hi"},
				{Role: "user", Content: "how are you?"},
			},
		},
		{
			name: "message with content blocks",
			input: []types.MessageInput{
				{Role: "user", Content: []types.ContentBlock{{Type: "text", Text: "hello"}}},
			},
			expected: []types.Message{
				{Role: "user", Content: []types.ContentBlock{{Type: "text", Text: "hello"}}},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ConvertAnthropicMessagesToOpenAI(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestConvertOpenAIMessagesToAnthropic(t *testing.T) {
	tests := []struct {
		name     string
		input    []types.Message
		expected []types.MessageInput
	}{
		{
			name:     "nil input",
			input:    nil,
			expected: nil,
		},
		{
			name:     "empty slice",
			input:    []types.Message{},
			expected: []types.MessageInput{},
		},
		{
			name: "single message with string content",
			input: []types.Message{
				{Role: "user", Content: "hello"},
			},
			expected: []types.MessageInput{
				{Role: "user", Content: "hello"},
			},
		},
		{
			name: "multiple messages",
			input: []types.Message{
				{Role: "system", Content: "be helpful"},
				{Role: "user", Content: "hello"},
				{Role: "assistant", Content: "hi there"},
			},
			expected: []types.MessageInput{
				{Role: "system", Content: "be helpful"},
				{Role: "user", Content: "hello"},
				{Role: "assistant", Content: "hi there"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ConvertOpenAIMessagesToAnthropic(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestConvertAnthropicToolsToOpenAI(t *testing.T) {
	schema := json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}}}`)

	tests := []struct {
		name     string
		input    []types.ToolDef
		expected []types.Tool
	}{
		{
			name:     "nil input",
			input:    nil,
			expected: nil,
		},
		{
			name:     "empty slice",
			input:    []types.ToolDef{},
			expected: []types.Tool{},
		},
		{
			name: "single tool",
			input: []types.ToolDef{
				{Name: "get_weather", Description: "Get weather", InputSchema: schema},
			},
			expected: []types.Tool{
				{Type: "function", Function: types.ToolFunction{Name: "get_weather", Description: "Get weather", Parameters: schema}},
			},
		},
		{
			name: "multiple tools",
			input: []types.ToolDef{
				{Name: "get_weather", Description: "Get weather", InputSchema: schema},
				{Name: "get_time", Description: "Get time", InputSchema: schema},
			},
			expected: []types.Tool{
				{Type: "function", Function: types.ToolFunction{Name: "get_weather", Description: "Get weather", Parameters: schema}},
				{Type: "function", Function: types.ToolFunction{Name: "get_time", Description: "Get time", Parameters: schema}},
			},
		},
		{
			name: "tool without description",
			input: []types.ToolDef{
				{Name: "calc", InputSchema: schema},
			},
			expected: []types.Tool{
				{Type: "function", Function: types.ToolFunction{Name: "calc", Parameters: schema}},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ConvertAnthropicToolsToOpenAI(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestConvertOpenAIToolsToAnthropic(t *testing.T) {
	schema := json.RawMessage(`{"type":"object"}`)

	tests := []struct {
		name     string
		input    []types.Tool
		expected []types.ToolDef
	}{
		{
			name:     "nil input",
			input:    nil,
			expected: nil,
		},
		{
			name:     "empty slice",
			input:    []types.Tool{},
			expected: []types.ToolDef{},
		},
		{
			name: "single tool",
			input: []types.Tool{
				{Type: "function", Function: types.ToolFunction{Name: "get_weather", Description: "Get weather", Parameters: schema}},
			},
			expected: []types.ToolDef{
				{Name: "get_weather", Description: "Get weather", InputSchema: schema},
			},
		},
		{
			name: "multiple tools",
			input: []types.Tool{
				{Type: "function", Function: types.ToolFunction{Name: "tool_a", Description: "A", Parameters: schema}},
				{Type: "function", Function: types.ToolFunction{Name: "tool_b", Description: "B", Parameters: schema}},
				{Type: "function", Function: types.ToolFunction{Name: "tool_c", Description: "C", Parameters: schema}},
			},
			expected: []types.ToolDef{
				{Name: "tool_a", Description: "A", InputSchema: schema},
				{Name: "tool_b", Description: "B", InputSchema: schema},
				{Name: "tool_c", Description: "C", InputSchema: schema},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ConvertOpenAIToolsToAnthropic(tt.input)
			assert.Equal(t, tt.expected, result)
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
			name:     "nil",
			input:    nil,
			expected: "",
		},
		{
			name:     "string",
			input:    "hello world",
			expected: "hello world",
		},
		{
			name:     "empty string",
			input:    "",
			expected: "",
		},
		{
			name:     "integer",
			input:    42,
			expected: "",
		},
		{
			name: "content blocks with text",
			input: []types.ContentBlock{
				{Type: "text", Text: "hello"},
				{Type: "text", Text: " world"},
			},
			expected: "hello world",
		},
		{
			name: "content blocks with mixed types",
			input: []types.ContentBlock{
				{Type: "text", Text: "hello"},
				{Type: "tool_use", Name: "tool"},
				{Type: "text", Text: "there"},
			},
			expected: "hellothere",
		},
		{
			name:     "empty content blocks",
			input:    []types.ContentBlock{},
			expected: "",
		},
		{
			name: "content blocks only tool_use",
			input: []types.ContentBlock{
				{Type: "tool_use", Name: "tool"},
			},
			expected: "",
		},
		{
			name: "interface slice with text blocks",
			input: []interface{}{
				types.ContentBlock{Type: "text", Text: "hello from "},
				types.ContentBlock{Type: "text", Text: "interface"},
			},
			expected: "hello from interface",
		},
		{
			name: "interface slice with mixed types",
			input: []interface{}{
				types.ContentBlock{Type: "text", Text: "hello"},
				types.ContentBlock{Type: "tool_use", Name: "tool"},
				types.ContentBlock{Type: "text", Text: "world"},
			},
			expected: "helloworld",
		},
		{
			name:     "empty interface slice",
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
			name:     "string",
			input:    "be helpful",
			expected: "be helpful",
		},
		{
			name:     "empty string",
			input:    "",
			expected: "",
		},
		{
			name: "content blocks with text",
			input: []types.ContentBlock{
				{Type: "text", Text: "be "},
				{Type: "text", Text: "helpful"},
			},
			expected: "be helpful",
		},
		{
			name: "content blocks with non-text",
			input: []types.ContentBlock{
				{Type: "text", Text: "system: "},
				{Type: "tool_use", Name: "tool"},
			},
			expected: "system: ",
		},
		{
			name:     "integer returns empty",
			input:    123,
			expected: "",
		},
		{
			name: "interface slice with text",
			input: []interface{}{
				types.ContentBlock{Type: "text", Text: "system "},
				types.ContentBlock{Type: "text", Text: "message"},
			},
			expected: "system message",
		},
		{
			name:     "empty interface slice",
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

func TestRoundTripConversion(t *testing.T) {
	schema := json.RawMessage(`{"type":"object"}`)

	originalMsgs := []types.MessageInput{
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "hi"},
	}
	convertedToOpenAI := ConvertAnthropicMessagesToOpenAI(originalMsgs)
	convertedBack := ConvertOpenAIMessagesToAnthropic(convertedToOpenAI)
	assert.Equal(t, originalMsgs, convertedBack)

	originalTools := []types.ToolDef{
		{Name: "tool1", Description: "desc1", InputSchema: schema},
		{Name: "tool2", InputSchema: schema},
	}
	convertedToOpenAITools := ConvertAnthropicToolsToOpenAI(originalTools)
	convertedToolsBack := ConvertOpenAIToolsToAnthropic(convertedToOpenAITools)
	assert.Equal(t, originalTools, convertedToolsBack)
}
