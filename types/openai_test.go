package types

import (
	"encoding/json"
	"testing"
)

func TestStreamChunk_MarshalUnmarshal(t *testing.T) {
	tests := []struct {
		name  string
		input string
	}{
		{
			name:  "basic_chunk",
			input: `{"id":"chatcmpl-1","object":"chat.completion.chunk","created":123,"model":"test","choices":[{"index":0,"delta":{"content":"Hello"}}]}`,
		},
		{
			name:  "chunk_with_reasoning",
			input: `{"id":"chatcmpl-1","choices":[{"index":0,"delta":{"reasoning":"thinking..."}}]}`,
		},
		{
			name:  "chunk_with_tool_calls",
			input: `{"id":"chatcmpl-1","choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_1","type":"function","index":0,"function":{"name":"bash","arguments":"{}"}}]}}]}`,
		},
		{
			name:  "chunk_with_usage",
			input: `{"id":"chatcmpl-1","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`,
		},
		{
			name:  "chunk_with_finish_reason",
			input: `{"id":"chatcmpl-1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var chunk StreamChunk
			if err := json.Unmarshal([]byte(tt.input), &chunk); err != nil {
				t.Fatalf("Failed to unmarshal: %v", err)
			}

			output, err := json.Marshal(chunk)
			if err != nil {
				t.Fatalf("Failed to marshal: %v", err)
			}

			t.Logf("Roundtrip: %s", string(output))
		})
	}
}

func TestStreamChunk_ContentField(t *testing.T) {
	input := `{"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":"Hello world"}}]}`
	var chunk StreamChunk
	if err := json.Unmarshal([]byte(input), &chunk); err != nil {
		t.Fatalf("Failed to unmarshal: %v", err)
	}

	if len(chunk.Choices) != 1 {
		t.Fatalf("Expected 1 choice, got %d", len(chunk.Choices))
	}

	if chunk.Choices[0].Delta.Content != "Hello world" {
		t.Errorf("Expected content 'Hello world', got %q", chunk.Choices[0].Delta.Content)
	}
}

func TestStreamChunk_ReasoningFields(t *testing.T) {
	tests := []struct {
		name           string
		input          string
		expectedReason string
	}{
		{
			name:           "reasoning_field",
			input:          `{"id":"chatcmpl-1","choices":[{"index":0,"delta":{"reasoning":"thought"}}]}`,
			expectedReason: "thought",
		},
		{
			name:           "reasoning_content_field",
			input:          `{"id":"chatcmpl-1","choices":[{"index":0,"delta":{"reasoning_content":"thought"}}]}`,
			expectedReason: "thought",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var chunk StreamChunk
			if err := json.Unmarshal([]byte(tt.input), &chunk); err != nil {
				t.Fatalf("Failed to unmarshal: %v", err)
			}

			delta := chunk.Choices[0].Delta
			if delta.Reasoning != tt.expectedReason && delta.ReasoningContent != tt.expectedReason {
				t.Errorf("Expected reasoning %q, got reasoning=%q reasoning_content=%q",
					tt.expectedReason, delta.Reasoning, delta.ReasoningContent)
			}
		})
	}
}
