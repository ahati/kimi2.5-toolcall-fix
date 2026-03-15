package convert

import (
	"bytes"
	"encoding/json"
	"testing"

	"ai-proxy/types"

	"github.com/tmaxmax/go-sse"
)

// TestChatToAnthropicConverter_Convert tests the request conversion from OpenAI Chat to Anthropic.
func TestChatToAnthropicConverter_Convert(t *testing.T) {
	tests := []struct {
		name        string
		input       types.ChatCompletionRequest
		checkResult func(t *testing.T, result []byte)
		expectError bool
	}{
		{
			name: "simple request",
			input: types.ChatCompletionRequest{
				Model: "claude-3-opus",
				Messages: []types.Message{
					{Role: "user", Content: "Hello"},
				},
				MaxTokens: 1024,
			},
			checkResult: func(t *testing.T, result []byte) {
				var anthReq types.MessageRequest
				if err := json.Unmarshal(result, &anthReq); err != nil {
					t.Fatalf("failed to parse result: %v", err)
				}
				if anthReq.Model != "claude-3-opus" {
					t.Errorf("expected model claude-3-opus, got %s", anthReq.Model)
				}
				if anthReq.MaxTokens != 1024 {
					t.Errorf("expected max_tokens 1024, got %d", anthReq.MaxTokens)
				}
				if len(anthReq.Messages) != 1 {
					t.Errorf("expected 1 message, got %d", len(anthReq.Messages))
				}
			},
		},
		{
			name: "request with system message",
			input: types.ChatCompletionRequest{
				Model: "claude-3-opus",
				Messages: []types.Message{
					{Role: "system", Content: "You are helpful."},
					{Role: "user", Content: "Hello"},
				},
				MaxTokens: 2048,
			},
			checkResult: func(t *testing.T, result []byte) {
				var anthReq types.MessageRequest
				if err := json.Unmarshal(result, &anthReq); err != nil {
					t.Fatalf("failed to parse result: %v", err)
				}
				if anthReq.System != "You are helpful." {
					t.Errorf("expected system 'You are helpful.', got %v", anthReq.System)
				}
				if len(anthReq.Messages) != 1 {
					t.Errorf("expected 1 non-system message, got %d", len(anthReq.Messages))
				}
			},
		},
		{
			name: "request with system field",
			input: types.ChatCompletionRequest{
				Model:  "claude-3-opus",
				System: "Be concise.",
				Messages: []types.Message{
					{Role: "user", Content: "Hello"},
				},
				MaxTokens: 1024,
			},
			checkResult: func(t *testing.T, result []byte) {
				var anthReq types.MessageRequest
				if err := json.Unmarshal(result, &anthReq); err != nil {
					t.Fatalf("failed to parse result: %v", err)
				}
				if anthReq.System != "Be concise." {
					t.Errorf("expected system 'Be concise.', got %v", anthReq.System)
				}
			},
		},
		{
			name: "request with streaming",
			input: types.ChatCompletionRequest{
				Model: "claude-3-opus",
				Messages: []types.Message{
					{Role: "user", Content: "Hello"},
				},
				Stream:    true,
				MaxTokens: 1024,
			},
			checkResult: func(t *testing.T, result []byte) {
				var anthReq types.MessageRequest
				if err := json.Unmarshal(result, &anthReq); err != nil {
					t.Fatalf("failed to parse result: %v", err)
				}
				if !anthReq.Stream {
					t.Error("expected stream to be true")
				}
			},
		},
		{
			name: "request with temperature and top_p",
			input: types.ChatCompletionRequest{
				Model: "claude-3-opus",
				Messages: []types.Message{
					{Role: "user", Content: "Hello"},
				},
				Temperature: 0.7,
				TopP:        0.9,
				MaxTokens:   1024,
			},
			checkResult: func(t *testing.T, result []byte) {
				var anthReq types.MessageRequest
				if err := json.Unmarshal(result, &anthReq); err != nil {
					t.Fatalf("failed to parse result: %v", err)
				}
				if anthReq.Temperature != 0.7 {
					t.Errorf("expected temperature 0.7, got %f", anthReq.Temperature)
				}
				if anthReq.TopP != 0.9 {
					t.Errorf("expected top_p 0.9, got %f", anthReq.TopP)
				}
			},
		},
		{
			name: "request with tools",
			input: types.ChatCompletionRequest{
				Model: "claude-3-opus",
				Messages: []types.Message{
					{Role: "user", Content: "What's the weather?"},
				},
				Tools: []types.Tool{
					{
						Type: "function",
						Function: types.ToolFunction{
							Name:        "get_weather",
							Description: "Get weather info",
							Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
						},
					},
				},
				MaxTokens: 1024,
			},
			checkResult: func(t *testing.T, result []byte) {
				var anthReq types.MessageRequest
				if err := json.Unmarshal(result, &anthReq); err != nil {
					t.Fatalf("failed to parse result: %v", err)
				}
				if len(anthReq.Tools) != 1 {
					t.Errorf("expected 1 tool, got %d", len(anthReq.Tools))
				}
				if anthReq.Tools[0].Name != "get_weather" {
					t.Errorf("expected tool name 'get_weather', got %s", anthReq.Tools[0].Name)
				}
			},
		},
		{
			name: "request with tool response",
			input: types.ChatCompletionRequest{
				Model: "claude-3-opus",
				Messages: []types.Message{
					{Role: "user", Content: "What's the weather in Paris?"},
					{Role: "assistant", Content: "", ToolCalls: []types.ToolCall{
						{
							ID:   "call_123",
							Type: "function",
							Function: types.Function{
								Name:      "get_weather",
								Arguments: `{"city":"Paris"}`,
							},
						},
					}},
					{Role: "tool", ToolCallID: "call_123", Content: "Sunny, 25C"},
				},
				MaxTokens: 1024,
			},
			checkResult: func(t *testing.T, result []byte) {
				var anthReq types.MessageRequest
				if err := json.Unmarshal(result, &anthReq); err != nil {
					t.Fatalf("failed to parse result: %v", err)
				}
				// Should have 3 messages (user, assistant with tool_use, user with tool_result)
				if len(anthReq.Messages) != 3 {
					t.Errorf("expected 3 messages, got %d", len(anthReq.Messages))
				}
			},
		},
		{
			name: "default max_tokens when not set",
			input: types.ChatCompletionRequest{
				Model: "claude-3-opus",
				Messages: []types.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			checkResult: func(t *testing.T, result []byte) {
				var anthReq types.MessageRequest
				if err := json.Unmarshal(result, &anthReq); err != nil {
					t.Fatalf("failed to parse result: %v", err)
				}
				if anthReq.MaxTokens != 4096 {
					t.Errorf("expected default max_tokens 4096, got %d", anthReq.MaxTokens)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			converter := NewChatToAnthropicConverter()
			inputJSON, err := json.Marshal(tt.input)
			if err != nil {
				t.Fatalf("failed to marshal input: %v", err)
			}

			result, err := converter.Convert(inputJSON)
			if tt.expectError {
				if err == nil {
					t.Error("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			tt.checkResult(t, result)
		})
	}
}

// TestChatToAnthropicTransformer_Transform tests the SSE response transformation.
func TestChatToAnthropicTransformer_Transform(t *testing.T) {
	tests := []struct {
		name      string
		events    []types.Chunk
		checkFn   func(t *testing.T, output string)
		doneEvent bool
	}{
		{
			name: "simple text response",
			events: []types.Chunk{
				{
					ID:    "chatcmpl-123",
					Model: "claude-3-opus",
					Choices: []types.Choice{
						{
							Index: 0,
							Delta: types.Delta{
								Role:    "assistant",
								Content: "",
							},
						},
					},
				},
				{
					ID:    "chatcmpl-123",
					Model: "claude-3-opus",
					Choices: []types.Choice{
						{
							Index: 0,
							Delta: types.Delta{
								Content: "Hello",
							},
						},
					},
				},
				{
					ID:    "chatcmpl-123",
					Model: "claude-3-opus",
					Choices: []types.Choice{
						{
							Index: 0,
							Delta: types.Delta{
								Content: " world",
							},
						},
					},
				},
			},
			checkFn: func(t *testing.T, output string) {
				// Should contain message_start
				if !contains(output, `"type":"message_start"`) {
					t.Error("expected message_start event")
				}
				// Should contain text_delta events
				if !contains(output, `"type":"content_block_delta"`) {
					t.Error("expected content_block_delta event")
				}
				if !contains(output, `"text":"Hello"`) {
					t.Error("expected text 'Hello'")
				}
				if !contains(output, `"text":" world"`) {
					t.Error("expected text ' world'")
				}
			},
		},
		{
			name: "response with finish reason",
			events: []types.Chunk{
				{
					ID:    "chatcmpl-123",
					Model: "claude-3-opus",
					Choices: []types.Choice{
						{
							Index: 0,
							Delta: types.Delta{
								Role:    "assistant",
								Content: "",
							},
						},
					},
				},
				{
					ID:    "chatcmpl-123",
					Model: "claude-3-opus",
					Choices: []types.Choice{
						{
							Index: 0,
							Delta: types.Delta{
								Content: "Done",
							},
							FinishReason: ptr("stop"),
						},
					},
				},
				// Usage chunk (sent after finish_reason by upstream)
				{
					ID:      "chatcmpl-123",
					Model:   "claude-3-opus",
					Choices: []types.Choice{},
					Usage: &types.Usage{
						PromptTokens:     10,
						CompletionTokens: 5,
					},
				},
			},
			checkFn: func(t *testing.T, output string) {
				// Should contain message_delta with stop_reason
				if !contains(output, `"stop_reason":"end_turn"`) {
					t.Error("expected stop_reason end_turn")
				}
			},
		},
		{
			name: "response with tool calls",
			events: []types.Chunk{
				{
					ID:    "chatcmpl-123",
					Model: "claude-3-opus",
					Choices: []types.Choice{
						{
							Index: 0,
							Delta: types.Delta{
								Role:    "assistant",
								Content: "",
							},
						},
					},
				},
				{
					ID:    "chatcmpl-123",
					Model: "claude-3-opus",
					Choices: []types.Choice{
						{
							Index: 0,
							Delta: types.Delta{
								ToolCalls: []types.ToolCall{
									{
										Index: 0,
										ID:    "call_abc",
										Type:  "function",
										Function: types.Function{
											Name: "get_weather",
										},
									},
								},
							},
						},
					},
				},
				{
					ID:    "chatcmpl-123",
					Model: "claude-3-opus",
					Choices: []types.Choice{
						{
							Index: 0,
							Delta: types.Delta{
								ToolCalls: []types.ToolCall{
									{
										Index: 0,
										Function: types.Function{
											Arguments: `{"city":`,
										},
									},
								},
							},
						},
					},
				},
				{
					ID:    "chatcmpl-123",
					Model: "claude-3-opus",
					Choices: []types.Choice{
						{
							Index: 0,
							Delta: types.Delta{
								ToolCalls: []types.ToolCall{
									{
										Index: 0,
										Function: types.Function{
											Arguments: `"Paris"}`,
										},
									},
								},
							},
						},
					},
				},
			},
			checkFn: func(t *testing.T, output string) {
				// Should contain content_block_start for tool_use
				if !contains(output, `"type":"tool_use"`) {
					t.Error("expected tool_use content block")
				}
				if !contains(output, `"name":"get_weather"`) {
					t.Error("expected tool name get_weather")
				}
				// Should contain input_json_delta
				if !contains(output, `"type":"input_json_delta"`) {
					t.Error("expected input_json_delta")
				}
			},
		},
		{
			name: "response with finish reason tool_calls",
			events: []types.Chunk{
				{
					ID:    "chatcmpl-123",
					Model: "claude-3-opus",
					Choices: []types.Choice{
						{
							Index: 0,
							Delta: types.Delta{
								Role:    "assistant",
								Content: "",
							},
						},
					},
				},
				{
					ID:    "chatcmpl-123",
					Model: "claude-3-opus",
					Choices: []types.Choice{
						{
							Index: 0,
							Delta: types.Delta{
								ToolCalls: []types.ToolCall{
									{
										Index: 0,
										ID:    "call_xyz",
										Type:  "function",
										Function: types.Function{
											Name:      "search",
											Arguments: `{}`,
										},
									},
								},
							},
							FinishReason: ptr("tool_calls"),
						},
					},
				},
				// Usage chunk (sent after finish_reason by upstream)
				{
					ID:      "chatcmpl-123",
					Model:   "claude-3-opus",
					Choices: []types.Choice{},
					Usage: &types.Usage{
						PromptTokens:     10,
						CompletionTokens: 5,
					},
				},
			},
			checkFn: func(t *testing.T, output string) {
				// Should contain stop_reason tool_use
				if !contains(output, `"stop_reason":"tool_use"`) {
					t.Error("expected stop_reason tool_use")
				}
			},
		},
		{
			name: "done event",
			events: []types.Chunk{
				{
					ID:    "chatcmpl-123",
					Model: "claude-3-opus",
					Choices: []types.Choice{
						{
							Index: 0,
							Delta: types.Delta{
								Role:    "assistant",
								Content: "Hi",
							},
						},
					},
				},
			},
			doneEvent: true,
			checkFn: func(t *testing.T, output string) {
				// Should contain message_stop
				if !contains(output, `"type":"message_stop"`) {
					t.Error("expected message_stop event")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			transformer := NewChatToAnthropicTransformer(&buf)

			for _, chunk := range tt.events {
				data, err := json.Marshal(chunk)
				if err != nil {
					t.Fatalf("failed to marshal chunk: %v", err)
				}
				event := &sse.Event{Data: string(data)}
				if err := transformer.Transform(event); err != nil {
					t.Fatalf("transform error: %v", err)
				}
			}

			if tt.doneEvent {
				event := &sse.Event{Data: "[DONE]"}
				if err := transformer.Transform(event); err != nil {
					t.Fatalf("transform [DONE] error: %v", err)
				}
			}

			if err := transformer.Close(); err != nil {
				t.Fatalf("close error: %v", err)
			}

			tt.checkFn(t, buf.String())
		})
	}
}

// TestChatToAnthropicTransformer_ToolCallConversion tests detailed tool call conversion.
func TestChatToAnthropicTransformer_ToolCallConversion(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewChatToAnthropicTransformer(&buf)

	// First chunk with role
	chunk1 := types.Chunk{
		ID:    "chatcmpl-456",
		Model: "claude-3-opus",
		Choices: []types.Choice{
			{
				Index: 0,
				Delta: types.Delta{Role: "assistant"},
			},
		},
	}
	data1, _ := json.Marshal(chunk1)
	transformer.Transform(&sse.Event{Data: string(data1)})

	// Tool call start
	chunk2 := types.Chunk{
		ID:    "chatcmpl-456",
		Model: "claude-3-opus",
		Choices: []types.Choice{
			{
				Index: 0,
				Delta: types.Delta{
					ToolCalls: []types.ToolCall{
						{
							Index: 0,
							ID:    "tool_123",
							Type:  "function",
							Function: types.Function{
								Name: "calculate",
							},
						},
					},
				},
			},
		},
	}
	data2, _ := json.Marshal(chunk2)
	transformer.Transform(&sse.Event{Data: string(data2)})

	// Tool call arguments
	chunk3 := types.Chunk{
		ID:    "chatcmpl-456",
		Model: "claude-3-opus",
		Choices: []types.Choice{
			{
				Index: 0,
				Delta: types.Delta{
					ToolCalls: []types.ToolCall{
						{
							Index: 0,
							Function: types.Function{
								Arguments: `{"x": 1, "y": 2}`,
							},
						},
					},
				},
			},
		},
	}
	data3, _ := json.Marshal(chunk3)
	transformer.Transform(&sse.Event{Data: string(data3)})

	transformer.Close()

	output := buf.String()

	// Verify content_block_start with tool_use
	if !contains(output, `"type":"content_block_start"`) {
		t.Error("expected content_block_start event")
	}
	if !contains(output, `"type":"tool_use"`) {
		t.Error("expected tool_use type in content block")
	}
	if !contains(output, `"id":"tool_123"`) {
		t.Error("expected tool ID tool_123")
	}
	if !contains(output, `"name":"calculate"`) {
		t.Error("expected tool name calculate")
	}

	// Verify input_json_delta
	if !contains(output, `"type":"input_json_delta"`) {
		t.Error("expected input_json_delta")
	}
	if !contains(output, `"partial_json":"{\"x\": 1, \"y\": 2}"`) {
		t.Error("expected partial_json with arguments")
	}

	// Verify message_stop
	if !contains(output, `"type":"message_stop"`) {
		t.Error("expected message_stop event")
	}
}

// TestChatToAnthropicTransformer_MultipleToolCalls tests multiple tool calls in sequence.
func TestChatToAnthropicTransformer_MultipleToolCalls(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewChatToAnthropicTransformer(&buf)

	// Start message
	chunk1 := types.Chunk{
		ID:    "chatcmpl-789",
		Model: "claude-3-opus",
		Choices: []types.Choice{
			{
				Index: 0,
				Delta: types.Delta{Role: "assistant"},
			},
		},
	}
	data1, _ := json.Marshal(chunk1)
	transformer.Transform(&sse.Event{Data: string(data1)})

	// First tool call
	chunk2 := types.Chunk{
		Choices: []types.Choice{
			{
				Index: 0,
				Delta: types.Delta{
					ToolCalls: []types.ToolCall{
						{Index: 0, ID: "tool_a", Type: "function", Function: types.Function{Name: "func_a"}},
					},
				},
			},
		},
	}
	data2, _ := json.Marshal(chunk2)
	transformer.Transform(&sse.Event{Data: string(data2)})

	// Second tool call
	chunk3 := types.Chunk{
		Choices: []types.Choice{
			{
				Index: 0,
				Delta: types.Delta{
					ToolCalls: []types.ToolCall{
						{Index: 1, ID: "tool_b", Type: "function", Function: types.Function{Name: "func_b"}},
					},
				},
			},
		},
	}
	data3, _ := json.Marshal(chunk3)
	transformer.Transform(&sse.Event{Data: string(data3)})

	transformer.Close()

	output := buf.String()

	// Both tools should be present
	if !contains(output, `"name":"func_a"`) {
		t.Error("expected func_a")
	}
	if !contains(output, `"name":"func_b"`) {
		t.Error("expected func_b")
	}
}

// TestChatToAnthropicTransformer_FinishReasonMapping tests finish reason conversion.
func TestChatToAnthropicTransformer_FinishReasonMapping(t *testing.T) {
	tests := []struct {
		openaiReason    string
		anthropicReason string
	}{
		{"stop", "end_turn"},
		{"length", "max_tokens"},
		{"tool_calls", "tool_use"},
		{"content_filter", "end_turn"},
	}

	for _, tt := range tests {
		t.Run(tt.openaiReason, func(t *testing.T) {
			var buf bytes.Buffer
			transformer := NewChatToAnthropicTransformer(&buf)

			// First chunk with finish_reason
			chunk := types.Chunk{
				ID:    "chatcmpl-test",
				Model: "claude-3-opus",
				Choices: []types.Choice{
					{
						Index: 0,
						Delta: types.Delta{
							Role:    "assistant",
							Content: "test",
						},
						FinishReason: ptr(tt.openaiReason),
					},
				},
			}
			data, _ := json.Marshal(chunk)
			transformer.Transform(&sse.Event{Data: string(data)})

			// Second chunk with usage (required to trigger message_delta emission)
			usageChunk := types.Chunk{
				ID:      "chatcmpl-test",
				Model:   "claude-3-opus",
				Choices: []types.Choice{},
				Usage: &types.Usage{
					PromptTokens:     10,
					CompletionTokens: 5,
				},
			}
			usageData, _ := json.Marshal(usageChunk)
			transformer.Transform(&sse.Event{Data: string(usageData)})

			transformer.Close()

			output := buf.String()
			expected := `"stop_reason":"` + tt.anthropicReason + `"`
			if !contains(output, expected) {
				t.Errorf("expected %s, output: %s", expected, output)
			}
		})
	}
}

// TestChatToAnthropicTransformer_EmptyEvents tests handling of empty events.
func TestChatToAnthropicTransformer_EmptyEvents(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewChatToAnthropicTransformer(&buf)

	// Empty data should be ignored
	err := transformer.Transform(&sse.Event{Data: ""})
	if err != nil {
		t.Errorf("unexpected error for empty event: %v", err)
	}

	// Flush should not fail
	err = transformer.Flush()
	if err != nil {
		t.Errorf("unexpected error on flush: %v", err)
	}
}

// TestChatToAnthropicTransformer_InvalidJSON tests handling of invalid JSON.
func TestChatToAnthropicTransformer_InvalidJSON(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewChatToAnthropicTransformer(&buf)

	// Invalid JSON should be passed through
	err := transformer.Transform(&sse.Event{Data: "not valid json"})
	if err != nil {
		t.Errorf("unexpected error for invalid JSON: %v", err)
	}

	output := buf.String()
	if !contains(output, "not valid json") {
		t.Error("expected invalid JSON to be passed through")
	}
}

// TestChatToAnthropicTransformer_ThinkingToTextTransition tests that the transformer
// properly handles the transition from thinking content to text content.
// According to Anthropic SSE spec, each content block should be either thinking OR text,
// not both. When switching from thinking to text, we must close the thinking block
// and start a new text block.
func TestChatToAnthropicTransformer_ThinkingToTextTransition(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewChatToAnthropicTransformer(&buf)

	// First chunk with thinking content
	chunk1 := types.Chunk{
		ID:    "chatcmpl-123",
		Model: "claude-3-opus",
		Choices: []types.Choice{
			{
				Index: 0,
				Delta: types.Delta{
					ReasoningContent: "Let me think...",
				},
			},
		},
	}
	data1, _ := json.Marshal(chunk1)
	transformer.Transform(&sse.Event{Data: string(data1)})

	// Second chunk with text content - should close thinking and start text
	chunk2 := types.Chunk{
		ID:    "chatcmpl-123",
		Model: "claude-3-opus",
		Choices: []types.Choice{
			{
				Index: 0,
				Delta: types.Delta{
					Content: "Hello",
				},
			},
		},
	}
	data2, _ := json.Marshal(chunk2)
	transformer.Transform(&sse.Event{Data: string(data2)})

	// Close the stream
	transformer.Close()

	output := buf.String()

	// Verify we have thinking block at index 0
	if !contains(output, `"type":"thinking"`) {
		t.Error("expected thinking content block to be started")
	}

	// Verify we have text block at index 1
	if !contains(output, `"type":"text"`) {
		t.Error("expected text content block to be started")
	}

	// Verify thinking block was closed at index 0
	if !contains(output, `"index":0,"type":"content_block_stop"`) {
		t.Error("expected thinking block to be closed at index 0")
	}

	// Verify text block was closed at index 1
	if !contains(output, `"index":1,"type":"content_block_stop"`) {
		t.Error("expected text block to be closed at index 1")
	}

	// Verify thinking_delta is at index 0
	if !contains(output, `"index":0,"type":"content_block_delta"`) {
		t.Error("expected thinking delta at index 0")
	}

	// Verify text_delta is at index 1 (not 0)
	if !contains(output, `"index":1`) || !contains(output, `"type":"text_delta"`) {
		t.Error("expected text delta at index 1, not mixed with thinking at index 0")
	}
}

// TestChatToAnthropicTransformer_TextToThinkingTransition tests the reverse transition
// from text to thinking content.
func TestChatToAnthropicTransformer_TextToThinkingTransition(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewChatToAnthropicTransformer(&buf)

	// First chunk with text content
	chunk1 := types.Chunk{
		ID:    "chatcmpl-123",
		Model: "claude-3-opus",
		Choices: []types.Choice{
			{
				Index: 0,
				Delta: types.Delta{
					Content: "Hello",
				},
			},
		},
	}
	data1, _ := json.Marshal(chunk1)
	transformer.Transform(&sse.Event{Data: string(data1)})

	// Second chunk with thinking content - should close text and start thinking
	chunk2 := types.Chunk{
		ID:    "chatcmpl-123",
		Model: "claude-3-opus",
		Choices: []types.Choice{
			{
				Index: 0,
				Delta: types.Delta{
					ReasoningContent: "Let me reconsider...",
				},
			},
		},
	}
	data2, _ := json.Marshal(chunk2)
	transformer.Transform(&sse.Event{Data: string(data2)})

	// Close the stream
	transformer.Close()

	output := buf.String()

	// Verify we have text block at index 0
	if !contains(output, `"type":"text"`) {
		t.Error("expected text content block to be started")
	}

	// Verify we have thinking block at index 1
	if !contains(output, `"type":"thinking"`) {
		t.Error("expected thinking content block to be started")
	}

	// Verify text block was closed at index 0
	if !contains(output, `"index":0,"type":"content_block_stop"`) {
		t.Error("expected text block to be closed at index 0")
	}

	// Verify thinking block was closed at index 1
	if !contains(output, `"index":1,"type":"content_block_stop"`) {
		t.Error("expected thinking block to be closed at index 1")
	}
}

// Helper functions

func ptr(s string) *string {
	return &s
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsHelper(s, substr))
}

func containsHelper(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
