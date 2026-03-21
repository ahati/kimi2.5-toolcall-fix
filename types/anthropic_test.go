package types

import (
	"encoding/json"
	"testing"
)

func TestMessageRequest(t *testing.T) {
	tests := []struct {
		name     string
		input    MessageRequest
		wantJSON string
	}{
		{
			name: "minimal request",
			input: MessageRequest{
				Model:    "claude-3",
				Messages: []MessageInput{{Role: "user", Content: "hello"}},
			},
			wantJSON: `{"model":"claude-3","messages":[{"role":"user","content":"hello"}],"max_tokens":0}`,
		},
		{
			name: "full request",
			input: MessageRequest{
				Model:       "claude-3",
				Messages:    []MessageInput{{Role: "user", Content: "hello"}},
				MaxTokens:   1024,
				Stream:      true,
				Temperature: 0.7,
				TopP:        0.9,
				TopK:        40,
				System:      "You are helpful",
			},
			wantJSON: `{"model":"claude-3","messages":[{"role":"user","content":"hello"}],"max_tokens":1024,"stream":true,"temperature":0.7,"top_p":0.9,"top_k":40,"system":"You are helpful"}`,
		},
		{
			name: "with tools",
			input: MessageRequest{
				Model:    "claude-3",
				Messages: []MessageInput{{Role: "user", Content: "test"}},
				Tools: []ToolDef{
					{
						Name:        "search",
						Description: "Search the web",
						InputSchema: json.RawMessage(`{"type":"object"}`),
					},
				},
			},
			wantJSON: `{"model":"claude-3","messages":[{"role":"user","content":"test"}],"max_tokens":0,"tools":[{"name":"search","description":"Search the web","input_schema":{"type":"object"}}]}`,
		},
		{
			name: "system as array",
			input: MessageRequest{
				Model:    "claude-3",
				Messages: []MessageInput{{Role: "user", Content: "hi"}},
				System: []interface{}{
					map[string]interface{}{"type": "text", "text": "You are helpful"},
				},
			},
			wantJSON: `{"model":"claude-3","messages":[{"role":"user","content":"hi"}],"max_tokens":0,"system":[{"text":"You are helpful","type":"text"}]}`,
		},
		{
			name: "with metadata",
			input: MessageRequest{
				Model:    "claude-3",
				Messages: []MessageInput{{Role: "user", Content: "test"}},
				Metadata: &AnthropicMetadata{UserID: "123"},
			},
			wantJSON: `{"model":"claude-3","messages":[{"role":"user","content":"test"}],"max_tokens":0,"metadata":{"user_id":"123"}}`,
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

			var unmarshaled MessageRequest
			if err := json.Unmarshal(data, &unmarshaled); err != nil {
				t.Fatalf("unmarshal error: %v", err)
			}
			if unmarshaled.Model != tt.input.Model {
				t.Errorf("model mismatch: got %s, want %s", unmarshaled.Model, tt.input.Model)
			}
		})
	}
}

func TestMessageInput(t *testing.T) {
	tests := []struct {
		name     string
		input    MessageInput
		wantJSON string
	}{
		{
			name:     "string content",
			input:    MessageInput{Role: "user", Content: "hello"},
			wantJSON: `{"role":"user","content":"hello"}`,
		},
		{
			name:     "array content",
			input:    MessageInput{Role: "user", Content: []interface{}{map[string]interface{}{"type": "text", "text": "hello"}}},
			wantJSON: `{"role":"user","content":[{"text":"hello","type":"text"}]}`,
		},
		{
			name:     "object content",
			input:    MessageInput{Role: "assistant", Content: map[string]interface{}{"type": "text", "text": "response"}},
			wantJSON: `{"role":"assistant","content":{"text":"response","type":"text"}}`,
		},
		{
			name:     "nil content",
			input:    MessageInput{Role: "assistant", Content: nil},
			wantJSON: `{"role":"assistant","content":null}`,
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

			var unmarshaled MessageInput
			if err := json.Unmarshal(data, &unmarshaled); err != nil {
				t.Fatalf("unmarshal error: %v", err)
			}
			if unmarshaled.Role != tt.input.Role {
				t.Errorf("role mismatch: got %s, want %s", unmarshaled.Role, tt.input.Role)
			}
		})
	}
}

func TestToolDef(t *testing.T) {
	tests := []struct {
		name     string
		input    ToolDef
		wantJSON string
	}{
		{
			name: "full tool def",
			input: ToolDef{
				Name:        "calculate",
				Description: "Perform a calculation",
				InputSchema: json.RawMessage(`{"type":"object","properties":{"expr":{"type":"string"}}}`),
			},
			wantJSON: `{"name":"calculate","description":"Perform a calculation","input_schema":{"type":"object","properties":{"expr":{"type":"string"}}}}`,
		},
		{
			name: "minimal tool def",
			input: ToolDef{
				Name:        "noop",
				InputSchema: json.RawMessage(`{"type":"object"}`),
			},
			wantJSON: `{"name":"noop","input_schema":{"type":"object"}}`,
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

			var unmarshaled ToolDef
			if err := json.Unmarshal(data, &unmarshaled); err != nil {
				t.Fatalf("unmarshal error: %v", err)
			}
			if unmarshaled.Name != tt.input.Name {
				t.Errorf("name mismatch: got %s, want %s", unmarshaled.Name, tt.input.Name)
			}
		})
	}
}

func TestEvent(t *testing.T) {
	idx := 5
	stopSeq := "stop_here"
	tests := []struct {
		name     string
		input    Event
		wantJSON string
	}{
		{
			name:     "message_start event",
			input:    Event{Type: "message_start"},
			wantJSON: `{"type":"message_start"}`,
		},
		{
			name:     "content_block_delta with index",
			input:    Event{Type: "content_block_delta", Index: &idx, Delta: json.RawMessage(`{"type":"text_delta","text":"hello"}`)},
			wantJSON: `{"type":"content_block_delta","index":5,"delta":{"type":"text_delta","text":"hello"}}`,
		},
		{
			name:     "content_block_start with content_block",
			input:    Event{Type: "content_block_start", Index: &idx, ContentBlock: json.RawMessage(`{"type":"text","text":""}`)},
			wantJSON: `{"type":"content_block_start","index":5,"content_block":{"type":"text","text":""}}`,
		},
		{
			name:     "message_delta with usage",
			input:    Event{Type: "message_delta", Usage: &AnthropicUsage{InputTokens: 10, OutputTokens: 20}},
			wantJSON: `{"type":"message_delta","usage":{"input_tokens":10,"output_tokens":20}}`,
		},
		{
			name:     "message_stop event",
			input:    Event{Type: "message_stop", StopReason: "end_turn"},
			wantJSON: `{"type":"message_stop","stop_reason":"end_turn"}`,
		},
		{
			name:     "with stop sequence",
			input:    Event{Type: "message_stop", StopReason: "stop_sequence", StopSequence: &stopSeq},
			wantJSON: `{"type":"message_stop","stop_reason":"stop_sequence","stop_sequence":"stop_here"}`,
		},
		{
			name: "with message info",
			input: Event{
				Type: "message_start",
				Message: &MessageInfo{
					ID:    "msg_123",
					Type:  "message",
					Role:  "assistant",
					Model: "claude-3",
				},
			},
			wantJSON: `{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":null,"model":"claude-3"}}`,
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

			var unmarshaled Event
			if err := json.Unmarshal(data, &unmarshaled); err != nil {
				t.Fatalf("unmarshal error: %v", err)
			}
			if unmarshaled.Type != tt.input.Type {
				t.Errorf("type mismatch: got %s, want %s", unmarshaled.Type, tt.input.Type)
			}
		})
	}
}

func TestMessageInfo(t *testing.T) {
	tests := []struct {
		name     string
		input    MessageInfo
		wantJSON string
	}{
		{
			name: "basic message info",
			input: MessageInfo{
				ID:    "msg_123",
				Type:  "message",
				Role:  "assistant",
				Model: "claude-3",
			},
			wantJSON: `{"id":"msg_123","type":"message","role":"assistant","content":null,"model":"claude-3"}`,
		},
		{
			name: "with content blocks",
			input: MessageInfo{
				ID:    "msg_456",
				Type:  "message",
				Role:  "assistant",
				Model: "claude-3",
				Content: []ContentBlock{
					{Type: "text", Text: "Hello"},
					{Type: "tool_use", ID: "tool_1", Name: "search", Input: json.RawMessage(`{"query":"test"}`)},
				},
			},
			wantJSON: `{"id":"msg_456","type":"message","role":"assistant","content":[{"type":"text","text":"Hello"},{"type":"tool_use","id":"tool_1","name":"search","input":{"query":"test"}}],"model":"claude-3"}`,
		},
		{
			name: "with usage",
			input: MessageInfo{
				ID:    "msg_789",
				Type:  "message",
				Role:  "assistant",
				Model: "claude-3",
				Usage: &AnthropicUsage{InputTokens: 50, OutputTokens: 100},
			},
			wantJSON: `{"id":"msg_789","type":"message","role":"assistant","content":null,"model":"claude-3","usage":{"input_tokens":50,"output_tokens":100}}`,
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

			var unmarshaled MessageInfo
			if err := json.Unmarshal(data, &unmarshaled); err != nil {
				t.Fatalf("unmarshal error: %v", err)
			}
			if unmarshaled.ID != tt.input.ID {
				t.Errorf("ID mismatch: got %s, want %s", unmarshaled.ID, tt.input.ID)
			}
		})
	}
}

func TestContentBlock(t *testing.T) {
	tests := []struct {
		name     string
		input    ContentBlock
		wantJSON string
	}{
		{
			name:     "text block",
			input:    ContentBlock{Type: "text", Text: "Hello world"},
			wantJSON: `{"type":"text","text":"Hello world"}`,
		},
		{
			name:     "tool_use block",
			input:    ContentBlock{Type: "tool_use", ID: "toolu_123", Name: "search", Input: json.RawMessage(`{"q":"test"}`)},
			wantJSON: `{"type":"tool_use","id":"toolu_123","name":"search","input":{"q":"test"}}`,
		},
		{
			name:     "thinking block",
			input:    ContentBlock{Type: "thinking", Thinking: "Let me think..."},
			wantJSON: `{"type":"thinking","thinking":"Let me think..."}`,
		},
		{
			name:     "minimal block",
			input:    ContentBlock{Type: "text"},
			wantJSON: `{"type":"text"}`,
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

			var unmarshaled ContentBlock
			if err := json.Unmarshal(data, &unmarshaled); err != nil {
				t.Fatalf("unmarshal error: %v", err)
			}
			if unmarshaled.Type != tt.input.Type {
				t.Errorf("type mismatch: got %s, want %s", unmarshaled.Type, tt.input.Type)
			}
		})
	}
}

func TestTextDelta(t *testing.T) {
	tests := []struct {
		name     string
		input    TextDelta
		wantJSON string
	}{
		{
			name:     "basic text delta",
			input:    TextDelta{Type: "text_delta", Text: "Hello"},
			wantJSON: `{"type":"text_delta","text":"Hello"}`,
		},
		{
			name:     "empty text",
			input:    TextDelta{Type: "text_delta", Text: ""},
			wantJSON: `{"type":"text_delta","text":""}`,
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

			var unmarshaled TextDelta
			if err := json.Unmarshal(data, &unmarshaled); err != nil {
				t.Fatalf("unmarshal error: %v", err)
			}
			if unmarshaled.Text != tt.input.Text {
				t.Errorf("text mismatch: got %s, want %s", unmarshaled.Text, tt.input.Text)
			}
		})
	}
}

func TestThinkingDelta(t *testing.T) {
	tests := []struct {
		name     string
		input    ThinkingDelta
		wantJSON string
	}{
		{
			name:     "basic thinking delta",
			input:    ThinkingDelta{Type: "thinking_delta", Thinking: "I should consider..."},
			wantJSON: `{"type":"thinking_delta","thinking":"I should consider..."}`,
		},
		{
			name:     "empty thinking",
			input:    ThinkingDelta{Type: "thinking_delta", Thinking: ""},
			wantJSON: `{"type":"thinking_delta","thinking":""}`,
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

			var unmarshaled ThinkingDelta
			if err := json.Unmarshal(data, &unmarshaled); err != nil {
				t.Fatalf("unmarshal error: %v", err)
			}
			if unmarshaled.Thinking != tt.input.Thinking {
				t.Errorf("thinking mismatch: got %s, want %s", unmarshaled.Thinking, tt.input.Thinking)
			}
		})
	}
}

func TestInputJSONDelta(t *testing.T) {
	tests := []struct {
		name     string
		input    InputJSONDelta
		wantJSON string
	}{
		{
			name:     "partial json",
			input:    InputJSONDelta{Type: "input_json_delta", PartialJSON: `{"query":"te`},
			wantJSON: `{"type":"input_json_delta","partial_json":"{\"query\":\"te"}`,
		},
		{
			name:     "empty partial json",
			input:    InputJSONDelta{Type: "input_json_delta", PartialJSON: ""},
			wantJSON: `{"type":"input_json_delta","partial_json":""}`,
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

			var unmarshaled InputJSONDelta
			if err := json.Unmarshal(data, &unmarshaled); err != nil {
				t.Fatalf("unmarshal error: %v", err)
			}
			if unmarshaled.PartialJSON != tt.input.PartialJSON {
				t.Errorf("partial_json mismatch: got %s, want %s", unmarshaled.PartialJSON, tt.input.PartialJSON)
			}
		})
	}
}

func TestAnthropicUsage(t *testing.T) {
	tests := []struct {
		name     string
		input    AnthropicUsage
		wantJSON string
	}{
		{
			name:     "basic usage",
			input:    AnthropicUsage{InputTokens: 100, OutputTokens: 50},
			wantJSON: `{"input_tokens":100,"output_tokens":50}`,
		},
		{
			name:     "zero values",
			input:    AnthropicUsage{},
			wantJSON: `{"input_tokens":0,"output_tokens":0}`,
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

			var unmarshaled AnthropicUsage
			if err := json.Unmarshal(data, &unmarshaled); err != nil {
				t.Fatalf("unmarshal error: %v", err)
			}
			if unmarshaled.InputTokens != tt.input.InputTokens {
				t.Errorf("input_tokens mismatch: got %d, want %d", unmarshaled.InputTokens, tt.input.InputTokens)
			}
		})
	}
}

func TestAnthropicErrorResponse(t *testing.T) {
	tests := []struct {
		name     string
		input    AnthropicErrorResponse
		wantJSON string
	}{
		{
			name:     "basic error",
			input:    AnthropicErrorResponse{Type: "error", Error: AnthropicErrorDetail{Type: "invalid_request_error", Message: "Invalid parameter"}},
			wantJSON: `{"type":"error","error":{"type":"invalid_request_error","message":"Invalid parameter"}}`,
		},
		{
			name:     "api error",
			input:    AnthropicErrorResponse{Type: "error", Error: AnthropicErrorDetail{Type: "authentication_error", Message: "Invalid API key"}},
			wantJSON: `{"type":"error","error":{"type":"authentication_error","message":"Invalid API key"}}`,
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

			var unmarshaled AnthropicErrorResponse
			if err := json.Unmarshal(data, &unmarshaled); err != nil {
				t.Fatalf("unmarshal error: %v", err)
			}
			if unmarshaled.Type != tt.input.Type {
				t.Errorf("type mismatch: got %s, want %s", unmarshaled.Type, tt.input.Type)
			}
		})
	}
}

func TestAnthropicErrorDetail(t *testing.T) {
	tests := []struct {
		name     string
		input    AnthropicErrorDetail
		wantJSON string
	}{
		{
			name:     "basic detail",
			input:    AnthropicErrorDetail{Type: "not_found_error", Message: "Resource not found"},
			wantJSON: `{"type":"not_found_error","message":"Resource not found"}`,
		},
		{
			name:     "rate limit error",
			input:    AnthropicErrorDetail{Type: "rate_limit_error", Message: "Rate limit exceeded"},
			wantJSON: `{"type":"rate_limit_error","message":"Rate limit exceeded"}`,
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

func TestEventUnmarshalFromJSON(t *testing.T) {
	tests := []struct {
		name    string
		jsonStr string
		want    Event
	}{
		{
			name:    "message_start",
			jsonStr: `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-3"}}`,
			want: Event{
				Type: "message_start",
				Message: &MessageInfo{
					ID:      "msg_1",
					Type:    "message",
					Role:    "assistant",
					Content: []ContentBlock{},
					Model:   "claude-3",
				},
			},
		},
		{
			name:    "content_block_delta",
			jsonStr: `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`,
			want: Event{
				Type:  "content_block_delta",
				Index: intPtr(0),
				Delta: json.RawMessage(`{"type":"text_delta","text":"Hello"}`),
			},
		},
		{
			name:    "message_delta with usage",
			jsonStr: `{"type":"message_delta","usage":{"input_tokens":10,"output_tokens":20},"stop_reason":"end_turn"}`,
			want: Event{
				Type:       "message_delta",
				Usage:      &AnthropicUsage{InputTokens: 10, OutputTokens: 20},
				StopReason: "end_turn",
			},
		},
		{
			name:    "content_block_start",
			jsonStr: `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			want: Event{
				Type:         "content_block_start",
				Index:        intPtr(0),
				ContentBlock: json.RawMessage(`{"type":"text","text":""}`),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got Event
			if err := json.Unmarshal([]byte(tt.jsonStr), &got); err != nil {
				t.Fatalf("unmarshal error: %v", err)
			}
			if got.Type != tt.want.Type {
				t.Errorf("type: got %s, want %s", got.Type, tt.want.Type)
			}
			if tt.want.Index != nil {
				if got.Index == nil || *got.Index != *tt.want.Index {
					t.Errorf("index: got %v, want %v", got.Index, tt.want.Index)
				}
			}
			if tt.want.Usage != nil {
				if got.Usage == nil || got.Usage.InputTokens != tt.want.Usage.InputTokens {
					t.Errorf("usage: got %v, want %v", got.Usage, tt.want.Usage)
				}
			}
		})
	}
}

func TestMessageRequestSystemTypes(t *testing.T) {
	t.Run("system as string", func(t *testing.T) {
		req := MessageRequest{
			Model:    "claude-3",
			Messages: []MessageInput{{Role: "user", Content: "hi"}},
			System:   "You are helpful",
		}
		data, err := json.Marshal(req)
		if err != nil {
			t.Fatalf("marshal error: %v", err)
		}
		var unmarshaled MessageRequest
		if err := json.Unmarshal(data, &unmarshaled); err != nil {
			t.Fatalf("unmarshal error: %v", err)
		}
		if str, ok := unmarshaled.System.(string); !ok || str != "You are helpful" {
			t.Errorf("system mismatch: got %v", unmarshaled.System)
		}
	})

	t.Run("system as array", func(t *testing.T) {
		jsonStr := `{"model":"claude-3","messages":[{"role":"user","content":"hi"}],"system":[{"type":"text","text":"You are helpful"}]}`
		var req MessageRequest
		if err := json.Unmarshal([]byte(jsonStr), &req); err != nil {
			t.Fatalf("unmarshal error: %v", err)
		}
		arr, ok := req.System.([]interface{})
		if !ok || len(arr) != 1 {
			t.Errorf("expected system array, got %v", req.System)
		}
	})
}

func intPtr(i int) *int {
	return &i
}
