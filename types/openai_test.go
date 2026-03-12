package types

import (
	"encoding/json"
	"testing"
)

func TestChatCompletionRequest(t *testing.T) {
	tests := []struct {
		name     string
		input    ChatCompletionRequest
		wantJSON string
	}{
		{
			name: "minimal request",
			input: ChatCompletionRequest{
				Model:    "gpt-4",
				Messages: []Message{{Role: "user", Content: "hello"}},
			},
			wantJSON: `{"model":"gpt-4","messages":[{"role":"user","content":"hello"}]}`,
		},
		{
			name: "full request",
			input: ChatCompletionRequest{
				Model:       "gpt-4",
				Messages:    []Message{{Role: "user", Content: "hello"}},
				MaxTokens:   100,
				Stream:      true,
				Temperature: 0.7,
				TopP:        0.9,
				System:      "You are helpful",
			},
			wantJSON: `{"model":"gpt-4","messages":[{"role":"user","content":"hello"}],"max_tokens":100,"stream":true,"temperature":0.7,"top_p":0.9,"system":"You are helpful"}`,
		},
		{
			name: "with tools",
			input: ChatCompletionRequest{
				Model:    "gpt-4",
				Messages: []Message{{Role: "user", Content: "test"}},
				Tools: []Tool{
					{
						Type: "function",
						Function: ToolFunction{
							Name:        "test_func",
							Description: "A test function",
							Parameters:  json.RawMessage(`{"type":"object"}`),
						},
					},
				},
			},
			wantJSON: `{"model":"gpt-4","messages":[{"role":"user","content":"test"}],"tools":[{"type":"function","function":{"name":"test_func","description":"A test function","parameters":{"type":"object"}}}]}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.input)
			if err != nil {
				t.Fatalf("marshal error: %v", err)
			}
			if string(data) != tt.wantJSON {
				t.Errorf("got %s, want %s", data, tt.wantJSON)
			}

			var unmarshaled ChatCompletionRequest
			if err := json.Unmarshal(data, &unmarshaled); err != nil {
				t.Fatalf("unmarshal error: %v", err)
			}
			if unmarshaled.Model != tt.input.Model {
				t.Errorf("model mismatch: got %s, want %s", unmarshaled.Model, tt.input.Model)
			}
		})
	}
}

func TestChatCompletionRequestUnmarshal(t *testing.T) {
	tests := []struct {
		name    string
		jsonStr string
		want    ChatCompletionRequest
		wantErr bool
	}{
		{
			name:    "empty object",
			jsonStr: `{}`,
			want:    ChatCompletionRequest{},
		},
		{
			name:    "with all fields",
			jsonStr: `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"max_tokens":50,"stream":true,"tools":[{"type":"function","function":{"name":"test","description":"desc","parameters":{"type":"object"}}}],"temperature":0.5,"top_p":0.8,"system":"system prompt"}`,
			want: ChatCompletionRequest{
				Model:       "gpt-4",
				Messages:    []Message{{Role: "user", Content: "hi"}},
				MaxTokens:   50,
				Stream:      true,
				Tools:       []Tool{{Type: "function", Function: ToolFunction{Name: "test", Description: "desc", Parameters: json.RawMessage(`{"type":"object"}`)}}},
				Temperature: 0.5,
				TopP:        0.8,
				System:      "system prompt",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got ChatCompletionRequest
			err := json.Unmarshal([]byte(tt.jsonStr), &got)
			if (err != nil) != tt.wantErr {
				t.Errorf("unmarshal error: %v", err)
				return
			}
			if got.Model != tt.want.Model {
				t.Errorf("model: got %s, want %s", got.Model, tt.want.Model)
			}
			if got.MaxTokens != tt.want.MaxTokens {
				t.Errorf("max_tokens: got %d, want %d", got.MaxTokens, tt.want.MaxTokens)
			}
		})
	}
}

func TestMessage(t *testing.T) {
	tests := []struct {
		name     string
		input    Message
		wantJSON string
	}{
		{
			name:     "simple message",
			input:    Message{Role: "user", Content: "hello"},
			wantJSON: `{"role":"user","content":"hello"}`,
		},
		{
			name:     "assistant with tool calls",
			input:    Message{Role: "assistant", Content: nil, ToolCalls: []ToolCall{{ID: "call_123", Type: "function", Index: 0, Function: Function{Name: "test", Arguments: `{"arg":"value"}`}}}},
			wantJSON: `{"role":"assistant","tool_calls":[{"id":"call_123","type":"function","index":0,"function":{"name":"test","arguments":"{\"arg\":\"value\"}"}}]}`,
		},
		{
			name:     "tool response",
			input:    Message{Role: "tool", ToolCallID: "call_123", Content: "result"},
			wantJSON: `{"role":"tool","content":"result","tool_call_id":"call_123"}`,
		},
		{
			name:     "empty content omitted",
			input:    Message{Role: "user"},
			wantJSON: `{"role":"user"}`,
		},
		{
			name:     "content as object",
			input:    Message{Role: "user", Content: map[string]interface{}{"type": "text", "text": "hello"}},
			wantJSON: `{"role":"user","content":{"text":"hello","type":"text"}}`,
		},
		{
			name:     "content as array",
			input:    Message{Role: "user", Content: []interface{}{map[string]interface{}{"type": "text", "text": "hello"}}},
			wantJSON: `{"role":"user","content":[{"text":"hello","type":"text"}]}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.input)
			if err != nil {
				t.Fatalf("marshal error: %v", err)
			}
			if string(data) != tt.wantJSON {
				t.Errorf("got %s, want %s", data, tt.wantJSON)
			}

			var unmarshaled Message
			if err := json.Unmarshal(data, &unmarshaled); err != nil {
				t.Fatalf("unmarshal error: %v", err)
			}
			if unmarshaled.Role != tt.input.Role {
				t.Errorf("role mismatch: got %s, want %s", unmarshaled.Role, tt.input.Role)
			}
		})
	}
}

func TestToolCall(t *testing.T) {
	tests := []struct {
		name     string
		input    ToolCall
		wantJSON string
	}{
		{
			name:     "basic tool call",
			input:    ToolCall{ID: "call_1", Type: "function", Index: 0, Function: Function{Name: "get_weather", Arguments: `{"city":"NYC"}`}},
			wantJSON: `{"id":"call_1","type":"function","index":0,"function":{"name":"get_weather","arguments":"{\"city\":\"NYC\"}"}}`,
		},
		{
			name:     "empty arguments",
			input:    ToolCall{ID: "call_2", Type: "function", Index: 1, Function: Function{Name: "noop", Arguments: ""}},
			wantJSON: `{"id":"call_2","type":"function","index":1,"function":{"name":"noop","arguments":""}}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.input)
			if err != nil {
				t.Fatalf("marshal error: %v", err)
			}
			if string(data) != tt.wantJSON {
				t.Errorf("got %s, want %s", data, tt.wantJSON)
			}

			var unmarshaled ToolCall
			if err := json.Unmarshal(data, &unmarshaled); err != nil {
				t.Fatalf("unmarshal error: %v", err)
			}
			if unmarshaled.ID != tt.input.ID {
				t.Errorf("ID mismatch: got %s, want %s", unmarshaled.ID, tt.input.ID)
			}
		})
	}
}

func TestFunction(t *testing.T) {
	tests := []struct {
		name     string
		input    Function
		wantJSON string
	}{
		{
			name:     "with arguments",
			input:    Function{Name: "search", Arguments: `{"query":"test"}`},
			wantJSON: `{"name":"search","arguments":"{\"query\":\"test\"}"}`,
		},
		{
			name:     "empty arguments",
			input:    Function{Name: "ping", Arguments: ""},
			wantJSON: `{"name":"ping","arguments":""}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.input)
			if err != nil {
				t.Fatalf("marshal error: %v", err)
			}
			if string(data) != tt.wantJSON {
				t.Errorf("got %s, want %s", data, tt.wantJSON)
			}

			var unmarshaled Function
			if err := json.Unmarshal(data, &unmarshaled); err != nil {
				t.Fatalf("unmarshal error: %v", err)
			}
			if unmarshaled.Name != tt.input.Name {
				t.Errorf("name mismatch: got %s, want %s", unmarshaled.Name, tt.input.Name)
			}
		})
	}
}

func TestTool(t *testing.T) {
	tests := []struct {
		name     string
		input    Tool
		wantJSON string
	}{
		{
			name: "full tool",
			input: Tool{
				Type: "function",
				Function: ToolFunction{
					Name:        "calculate",
					Description: "Perform calculations",
					Parameters:  json.RawMessage(`{"type":"object","properties":{"expr":{"type":"string"}}}`),
				},
			},
			wantJSON: `{"type":"function","function":{"name":"calculate","description":"Perform calculations","parameters":{"type":"object","properties":{"expr":{"type":"string"}}}}}`,
		},
		{
			name: "minimal tool",
			input: Tool{
				Type: "function",
				Function: ToolFunction{
					Name: "simple",
				},
			},
			wantJSON: `{"type":"function","function":{"name":"simple"}}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.input)
			if err != nil {
				t.Fatalf("marshal error: %v", err)
			}
			if string(data) != tt.wantJSON {
				t.Errorf("got %s, want %s", data, tt.wantJSON)
			}

			var unmarshaled Tool
			if err := json.Unmarshal(data, &unmarshaled); err != nil {
				t.Fatalf("unmarshal error: %v", err)
			}
			if unmarshaled.Type != tt.input.Type {
				t.Errorf("type mismatch: got %s, want %s", unmarshaled.Type, tt.input.Type)
			}
		})
	}
}

func TestChunk(t *testing.T) {
	finishReason := "stop"
	tests := []struct {
		name     string
		input    Chunk
		wantJSON string
	}{
		{
			name: "basic chunk",
			input: Chunk{
				ID:      "chunk_1",
				Object:  "chat.completion.chunk",
				Created: 1234567890,
				Model:   "gpt-4",
				Choices: []Choice{
					{Index: 0, Delta: Delta{Role: "assistant", Content: "Hello"}},
				},
			},
			wantJSON: `{"id":"chunk_1","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"}}]}`,
		},
		{
			name: "chunk with finish reason",
			input: Chunk{
				ID:      "chunk_2",
				Object:  "chat.completion.chunk",
				Created: 1234567890,
				Model:   "gpt-4",
				Choices: []Choice{
					{Index: 0, Delta: Delta{}, FinishReason: &finishReason},
				},
			},
			wantJSON: `{"id":"chunk_2","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		},
		{
			name: "chunk with usage",
			input: Chunk{
				ID:      "chunk_3",
				Object:  "chat.completion.chunk",
				Created: 1234567890,
				Model:   "gpt-4",
				Choices: []Choice{},
				Usage:   &Usage{PromptTokens: 10, CompletionTokens: 20, TotalTokens: 30},
			},
			wantJSON: `{"id":"chunk_3","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}`,
		},
		{
			name: "chunk with tool calls in delta",
			input: Chunk{
				ID:      "chunk_4",
				Object:  "chat.completion.chunk",
				Created: 1234567890,
				Model:   "gpt-4",
				Choices: []Choice{
					{
						Index: 0,
						Delta: Delta{
							ToolCalls: []ToolCall{
								{ID: "tc_1", Type: "function", Index: 0, Function: Function{Name: "test", Arguments: `{"a":1}`}},
							},
						},
					},
				},
			},
			wantJSON: `{"id":"chunk_4","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"tool_calls":[{"id":"tc_1","type":"function","index":0,"function":{"name":"test","arguments":"{\"a\":1}"}}]}}]}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.input)
			if err != nil {
				t.Fatalf("marshal error: %v", err)
			}
			if string(data) != tt.wantJSON {
				t.Errorf("got %s, want %s", data, tt.wantJSON)
			}

			var unmarshaled Chunk
			if err := json.Unmarshal(data, &unmarshaled); err != nil {
				t.Fatalf("unmarshal error: %v", err)
			}
			if unmarshaled.ID != tt.input.ID {
				t.Errorf("ID mismatch: got %s, want %s", unmarshaled.ID, tt.input.ID)
			}
		})
	}
}

func TestChoice(t *testing.T) {
	reason := "stop"
	tests := []struct {
		name     string
		input    Choice
		wantJSON string
	}{
		{
			name:     "basic choice",
			input:    Choice{Index: 0, Delta: Delta{Content: "test"}},
			wantJSON: `{"index":0,"delta":{"content":"test"}}`,
		},
		{
			name:     "choice with finish reason",
			input:    Choice{Index: 1, Delta: Delta{Role: "assistant"}, FinishReason: &reason},
			wantJSON: `{"index":1,"delta":{"role":"assistant"},"finish_reason":"stop"}`,
		},
		{
			name:     "nil finish reason",
			input:    Choice{Index: 0, Delta: Delta{}, FinishReason: nil},
			wantJSON: `{"index":0,"delta":{}}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.input)
			if err != nil {
				t.Fatalf("marshal error: %v", err)
			}
			if string(data) != tt.wantJSON {
				t.Errorf("got %s, want %s", data, tt.wantJSON)
			}

			var unmarshaled Choice
			if err := json.Unmarshal(data, &unmarshaled); err != nil {
				t.Fatalf("unmarshal error: %v", err)
			}
			if unmarshaled.Index != tt.input.Index {
				t.Errorf("index mismatch: got %d, want %d", unmarshaled.Index, tt.input.Index)
			}
		})
	}
}

func TestDelta(t *testing.T) {
	finishReason := "tool_calls"
	tests := []struct {
		name     string
		input    Delta
		wantJSON string
	}{
		{
			name:     "empty delta",
			input:    Delta{},
			wantJSON: `{}`,
		},
		{
			name:     "content only",
			input:    Delta{Content: "Hello world"},
			wantJSON: `{"content":"Hello world"}`,
		},
		{
			name:     "role and content",
			input:    Delta{Role: "assistant", Content: "Hi"},
			wantJSON: `{"role":"assistant","content":"Hi"}`,
		},
		{
			name:     "with reasoning",
			input:    Delta{Content: "answer", Reasoning: "thinking..."},
			wantJSON: `{"content":"answer","reasoning":"thinking..."}`,
		},
		{
			name:     "with reasoning content",
			input:    Delta{Content: "answer", ReasoningContent: "reasoning content"},
			wantJSON: `{"content":"answer","reasoning_content":"reasoning content"}`,
		},
		{
			name:     "with finish reason",
			input:    Delta{Content: "done", FinishReason: &finishReason},
			wantJSON: `{"content":"done","finish_reason":"tool_calls"}`,
		},
		{
			name: "with tool calls",
			input: Delta{
				ToolCalls: []ToolCall{
					{ID: "tc_1", Type: "function", Index: 0, Function: Function{Name: "calc"}},
				},
			},
			wantJSON: `{"tool_calls":[{"id":"tc_1","type":"function","index":0,"function":{"name":"calc","arguments":""}}]}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.input)
			if err != nil {
				t.Fatalf("marshal error: %v", err)
			}
			if string(data) != tt.wantJSON {
				t.Errorf("got %s, want %s", data, tt.wantJSON)
			}

			var unmarshaled Delta
			if err := json.Unmarshal(data, &unmarshaled); err != nil {
				t.Fatalf("unmarshal error: %v", err)
			}
			if unmarshaled.Content != tt.input.Content {
				t.Errorf("content mismatch: got %s, want %s", unmarshaled.Content, tt.input.Content)
			}
		})
	}
}

func TestUsage(t *testing.T) {
	tests := []struct {
		name     string
		input    Usage
		wantJSON string
	}{
		{
			name:     "basic usage",
			input:    Usage{PromptTokens: 100, CompletionTokens: 50, TotalTokens: 150},
			wantJSON: `{"prompt_tokens":100,"completion_tokens":50,"total_tokens":150}`,
		},
		{
			name:     "zero values",
			input:    Usage{},
			wantJSON: `{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.input)
			if err != nil {
				t.Fatalf("marshal error: %v", err)
			}
			if string(data) != tt.wantJSON {
				t.Errorf("got %s, want %s", data, tt.wantJSON)
			}

			var unmarshaled Usage
			if err := json.Unmarshal(data, &unmarshaled); err != nil {
				t.Fatalf("unmarshal error: %v", err)
			}
			if unmarshaled.TotalTokens != tt.input.TotalTokens {
				t.Errorf("total tokens mismatch: got %d, want %d", unmarshaled.TotalTokens, tt.input.TotalTokens)
			}
		})
	}
}

func TestErrorResponse(t *testing.T) {
	tests := []struct {
		name     string
		input    ErrorResponse
		wantJSON string
	}{
		{
			name:     "basic error",
			input:    ErrorResponse{Error: ErrorDetail{Type: "invalid_request_error", Message: "Invalid parameter"}},
			wantJSON: `{"error":{"type":"invalid_request_error","message":"Invalid parameter"}}`,
		},
		{
			name:     "error with code",
			input:    ErrorResponse{Error: ErrorDetail{Type: "authentication_error", Message: "Invalid API key", Code: "invalid_api_key"}},
			wantJSON: `{"error":{"type":"authentication_error","message":"Invalid API key","code":"invalid_api_key"}}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.input)
			if err != nil {
				t.Fatalf("marshal error: %v", err)
			}
			if string(data) != tt.wantJSON {
				t.Errorf("got %s, want %s", data, tt.wantJSON)
			}

			var unmarshaled ErrorResponse
			if err := json.Unmarshal(data, &unmarshaled); err != nil {
				t.Fatalf("unmarshal error: %v", err)
			}
			if unmarshaled.Error.Type != tt.input.Error.Type {
				t.Errorf("error type mismatch: got %s, want %s", unmarshaled.Error.Type, tt.input.Error.Type)
			}
		})
	}
}

func TestErrorDetail(t *testing.T) {
	tests := []struct {
		name     string
		input    ErrorDetail
		wantJSON string
	}{
		{
			name:     "without code",
			input:    ErrorDetail{Type: "error_type", Message: "error message"},
			wantJSON: `{"type":"error_type","message":"error message"}`,
		},
		{
			name:     "with code",
			input:    ErrorDetail{Type: "error_type", Message: "error message", Code: "error_code"},
			wantJSON: `{"type":"error_type","message":"error message","code":"error_code"}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.input)
			if err != nil {
				t.Fatalf("marshal error: %v", err)
			}
			if string(data) != tt.wantJSON {
				t.Errorf("got %s, want %s", data, tt.wantJSON)
			}
		})
	}
}

func TestMessageContentTypes(t *testing.T) {
	t.Run("content as string", func(t *testing.T) {
		msg := Message{Role: "user", Content: "hello"}
		data, err := json.Marshal(msg)
		if err != nil {
			t.Fatalf("marshal error: %v", err)
		}
		var unmarshaled Message
		if err := json.Unmarshal(data, &unmarshaled); err != nil {
			t.Fatalf("unmarshal error: %v", err)
		}
		if str, ok := unmarshaled.Content.(string); !ok || str != "hello" {
			t.Errorf("content mismatch: got %v", unmarshaled.Content)
		}
	})

	t.Run("content as number", func(t *testing.T) {
		jsonStr := `{"role":"user","content":42}`
		var msg Message
		if err := json.Unmarshal([]byte(jsonStr), &msg); err != nil {
			t.Fatalf("unmarshal error: %v", err)
		}
		if num, ok := msg.Content.(float64); !ok || num != 42 {
			t.Errorf("content mismatch: got %v", msg.Content)
		}
	})

	t.Run("content as null", func(t *testing.T) {
		jsonStr := `{"role":"assistant","content":null}`
		var msg Message
		if err := json.Unmarshal([]byte(jsonStr), &msg); err != nil {
			t.Fatalf("unmarshal error: %v", err)
		}
		if msg.Content != nil {
			t.Errorf("expected nil content, got %v", msg.Content)
		}
	})
}

func TestToolFunctionParameters(t *testing.T) {
	t.Run("parameters as raw message", func(t *testing.T) {
		tf := ToolFunction{
			Name:        "test",
			Description: "test func",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}}}`),
		}
		data, err := json.Marshal(tf)
		if err != nil {
			t.Fatalf("marshal error: %v", err)
		}
		var unmarshaled ToolFunction
		if err := json.Unmarshal(data, &unmarshaled); err != nil {
			t.Fatalf("unmarshal error: %v", err)
		}
		if string(unmarshaled.Parameters) != string(tf.Parameters) {
			t.Errorf("parameters mismatch: got %s, want %s", unmarshaled.Parameters, tf.Parameters)
		}
	})

	t.Run("empty parameters", func(t *testing.T) {
		tf := ToolFunction{Name: "test"}
		data, err := json.Marshal(tf)
		if err != nil {
			t.Fatalf("marshal error: %v", err)
		}
		want := `{"name":"test"}`
		if string(data) != want {
			t.Errorf("got %s, want %s", data, want)
		}
	})
}
