package tokens

import (
	"testing"

	"ai-proxy/types"
)

func TestNewCounter_Success(t *testing.T) {
	tests := []struct {
		name       string
		encoding   string
		shouldWork bool
	}{
		{
			name:       "cl100k_base encoding",
			encoding:   "cl100k_base",
			shouldWork: true,
		},
		{
			name:       "p50k_base encoding",
			encoding:   "p50k_base",
			shouldWork: true,
		},
		{
			name:       "r50k_base encoding",
			encoding:   "r50k_base",
			shouldWork: true,
		},
		{
			name:       "invalid encoding",
			encoding:   "invalid_encoding",
			shouldWork: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			counter, err := NewCounter(tt.encoding)
			if tt.shouldWork {
				if err != nil {
					t.Fatalf("NewCounter(%q) should not error, got: %v", tt.encoding, err)
				}
				if counter == nil {
					t.Fatal("NewCounter() returned nil counter")
				}
				if counter.encoder == nil {
					t.Fatal("NewCounter() returned counter with nil encoder")
				}
			} else {
				if err == nil {
					t.Fatalf("NewCounter(%q) should error", tt.encoding)
				}
				if counter != nil {
					t.Fatal("NewCounter() should return nil counter on error")
				}
			}
		})
	}
}

func TestCountTokensForModel(t *testing.T) {
	counter, err := CountTokensForModel("kimi-k2.5")
	if err != nil {
		t.Fatalf("CountTokensForModel() error: %v", err)
	}
	if counter == nil {
		t.Fatal("CountTokensForModel() returned nil counter")
	}

	// Should work with any model name (defaults to cl100k_base)
	counter2, err := CountTokensForModel("claude-3-opus")
	if err != nil {
		t.Fatalf("CountTokensForModel() error: %v", err)
	}
	if counter2 == nil {
		t.Fatal("CountTokensForModel() returned nil counter")
	}
}

func TestCounter_CountMessageTokens_Basic(t *testing.T) {
	counter, err := NewCounter("cl100k_base")
	if err != nil {
		t.Fatalf("NewCounter() error: %v", err)
	}

	tests := []struct {
		name      string
		request   types.MessageCountTokensRequest
		minTokens int // Minimum expected tokens
	}{
		{
			name: "simple user message",
			request: types.MessageCountTokensRequest{
				Model: "kimi-k2.5",
				Messages: []types.MessageInput{
					{
						Role:    "user",
						Content: "Hello",
					},
				},
			},
			minTokens: 5,
		},
		{
			name: "assistant message",
			request: types.MessageCountTokensRequest{
				Model: "kimi-k2.5",
				Messages: []types.MessageInput{
					{
						Role:    "assistant",
						Content: "How can I help you?",
					},
				},
			},
			minTokens: 10,
		},
		{
			name: "multiple messages",
			request: types.MessageCountTokensRequest{
				Model: "kimi-k2.5",
				Messages: []types.MessageInput{
					{
						Role:    "user",
						Content: "What is 2+2?",
					},
					{
						Role:    "assistant",
						Content: "2+2 equals 4.",
					},
				},
			},
			minTokens: 20,
		},
		{
			name: "empty messages",
			request: types.MessageCountTokensRequest{
				Model: "kimi-k2.5",
				Messages: []types.MessageInput{
					{
						Role:    "user",
						Content: "",
					},
				},
			},
			minTokens: 5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			count, err := counter.CountMessageTokens(&tt.request)
			if err != nil {
				t.Fatalf("CountMessageTokens() error: %v", err)
			}
			if count < tt.minTokens {
				t.Errorf("Token count should be at least %d, got %d", tt.minTokens, count)
			}
		})
	}
}

func TestCounter_CountMessageTokens_WithSystem(t *testing.T) {
	counter, err := NewCounter("cl100k_base")
	if err != nil {
		t.Fatalf("NewCounter() error: %v", err)
	}

	tests := []struct {
		name    string
		system  interface{}
		content string
	}{
		{
			name:    "string system prompt",
			system:  "You are a helpful assistant.",
			content: "Hello",
		},
		{
			name:    "long system prompt",
			system:  "You are a helpful assistant that provides accurate and concise answers to user questions. Always be polite and professional.",
			content: "What is Go?",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := types.MessageCountTokensRequest{
				Model:  "kimi-k2.5",
				System: tt.system,
				Messages: []types.MessageInput{
					{
						Role:    "user",
						Content: tt.content,
					},
				},
			}

			count, err := counter.CountMessageTokens(&req)
			if err != nil {
				t.Fatalf("CountMessageTokens() error: %v", err)
			}
			if count <= 0 {
				t.Errorf("Token count should be greater than 0, got %d", count)
			}
		})
	}
}

func TestCounter_CountMessageTokens_WithTools(t *testing.T) {
	counter, err := NewCounter("cl100k_base")
	if err != nil {
		t.Fatalf("NewCounter() error: %v", err)
	}

	tests := []struct {
		name  string
		tools []types.ToolDef
	}{
		{
			name: "single tool",
			tools: []types.ToolDef{
				{
					Name:        "get_weather",
					Description: "Get weather for a location",
					InputSchema: []byte(`{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}`),
				},
			},
		},
		{
			name: "multiple tools",
			tools: []types.ToolDef{
				{
					Name:        "get_weather",
					Description: "Get weather for a location",
					InputSchema: []byte(`{"type":"object","properties":{"location":{"type":"string"}}}`),
				},
				{
					Name:        "calculator",
					Description: "Perform calculations",
					InputSchema: []byte(`{"type":"object","properties":{"expression":{"type":"string"}}}`),
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := types.MessageCountTokensRequest{
				Model: "kimi-k2.5",
				Messages: []types.MessageInput{
					{
						Role:    "user",
						Content: "What's the weather in Tokyo?",
					},
				},
				Tools: tt.tools,
			}

			count, err := counter.CountMessageTokens(&req)
			if err != nil {
				t.Fatalf("CountMessageTokens() error: %v", err)
			}
			if count <= 0 {
				t.Errorf("Token count should be greater than 0, got %d", count)
			}
		})
	}
}

func TestCountTokens_Convenience(t *testing.T) {
	req := types.MessageCountTokensRequest{
		Model: "kimi-k2.5",
		Messages: []types.MessageInput{
			{
				Role:    "user",
				Content: "Hello, world!",
			},
		},
	}

	count, err := CountTokens(&req)
	if err != nil {
		t.Fatalf("CountTokens() error: %v", err)
	}
	if count <= 0 {
		t.Errorf("Token count should be greater than 0, got %d", count)
	}
}

func TestCounter_countToolDefinition(t *testing.T) {
	counter, err := NewCounter("cl100k_base")
	if err != nil {
		t.Fatalf("NewCounter() error: %v", err)
	}

	tool := types.ToolDef{
		Name:        "test_tool",
		Description: "A test tool",
		InputSchema: []byte(`{"type":"object"}`),
	}

	count := counter.countToolDefinition(tool)
	if count <= 0 {
		t.Errorf("Tool token count should be greater than 0, got %d", count)
	}
}

func TestCounter_countFormattingOverhead(t *testing.T) {
	counter, err := NewCounter("cl100k_base")
	if err != nil {
		t.Fatalf("NewCounter() error: %v", err)
	}

	// Test with different message and tool counts
	overhead1 := counter.countFormattingOverhead(1, 0)
	overhead2 := counter.countFormattingOverhead(2, 1)
	overhead3 := counter.countFormattingOverhead(5, 3)

	if overhead1 < 20 {
		t.Errorf("Overhead for 1 message should be at least 20, got %d", overhead1)
	}
	if overhead2 <= overhead1 {
		t.Errorf("Overhead should increase with more messages/tools: %d <= %d", overhead2, overhead1)
	}
	if overhead3 <= overhead2 {
		t.Errorf("Overhead should increase with more messages/tools: %d <= %d", overhead3, overhead2)
	}
}
