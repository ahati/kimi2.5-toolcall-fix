package types

import (
	"encoding/json"
	"testing"
)

func TestEvent_Unmarshal(t *testing.T) {
	tests := []struct {
		name  string
		input string
	}{
		{
			name:  "message_start",
			input: `{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","model":"claude-3"}}`,
		},
		{
			name:  "content_block_start_thinking",
			input: `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		},
		{
			name:  "content_block_start_tool",
			input: `{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_123","name":"bash"}}`,
		},
		{
			name:  "content_block_delta_thinking",
			input: `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"..."}}`,
		},
		{
			name:  "content_block_delta_text",
			input: `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`,
		},
		{
			name:  "content_block_delta_input_json",
			input: `{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{}"}}`,
		},
		{
			name:  "content_block_stop",
			input: `{"type":"content_block_stop","index":0}`,
		},
		{
			name:  "message_delta",
			input: `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":10}}`,
		},
		{
			name:  "message_stop",
			input: `{"type":"message_stop"}`,
		},
		{
			name:  "ping",
			input: `{"type":"ping"}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var event Event
			if err := json.Unmarshal([]byte(tt.input), &event); err != nil {
				t.Fatalf("Failed to unmarshal: %v", err)
			}

			if event.Type == "" {
				t.Error("Expected type to be set")
			}

			t.Logf("Event type: %s", event.Type)
		})
	}
}

func TestEvent_MessageStart(t *testing.T) {
	input := `{"type":"message_start","message":{"id":"msg_abc","type":"message","role":"assistant","model":"claude-3-5-sonnet"}}`

	var event Event
	if err := json.Unmarshal([]byte(input), &event); err != nil {
		t.Fatalf("Failed to unmarshal: %v", err)
	}

	if event.Type != "message_start" {
		t.Errorf("Expected type 'message_start', got %q", event.Type)
	}
	if event.Message == nil {
		t.Fatal("Expected message to be set")
	}
	if event.Message.ID != "msg_abc" {
		t.Errorf("Expected message ID 'msg_abc', got %q", event.Message.ID)
	}
}

func TestEvent_ContentBlockStart(t *testing.T) {
	input := `{"type":"content_block_start","index":5,"content_block":{"type":"tool_use","id":"toolu_xyz","name":"read"}}`

	var event Event
	if err := json.Unmarshal([]byte(input), &event); err != nil {
		t.Fatalf("Failed to unmarshal: %v", err)
	}

	if event.Index == nil || *event.Index != 5 {
		t.Errorf("Expected index 5, got %v", event.Index)
	}

	var block ContentBlock
	if err := json.Unmarshal(event.ContentBlock, &block); err != nil {
		t.Fatalf("Failed to unmarshal content_block: %v", err)
	}

	if block.Type != "tool_use" {
		t.Errorf("Expected block type 'tool_use', got %q", block.Type)
	}
	if block.Name != "read" {
		t.Errorf("Expected block name 'read', got %q", block.Name)
	}
}

func TestContentBlock_Types(t *testing.T) {
	tests := []struct {
		name  string
		input string
	}{
		{"thinking", `{"type":"thinking","thinking":"Let me think"}`},
		{"text", `{"type":"text","text":"Hello world"}`},
		{"tool_use", `{"type":"tool_use","id":"toolu_123","name":"bash","input":{"cmd":"ls"}}`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var block ContentBlock
			if err := json.Unmarshal([]byte(tt.input), &block); err != nil {
				t.Fatalf("Failed to unmarshal: %v", err)
			}
			if block.Type != tt.name {
				t.Errorf("Expected type %q, got %q", tt.name, block.Type)
			}
		})
	}
}
