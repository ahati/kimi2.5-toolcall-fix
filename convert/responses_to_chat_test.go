package convert

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"ai-proxy/types"

	"github.com/tmaxmax/go-sse"
)

// TestResponsesToChatConverter_Convert tests request conversion.
func TestResponsesToChatConverter_Convert(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		wantErr  bool
		validate func(t *testing.T, output []byte)
	}{
		{
			name: "simple string input",
			input: `{
				"model": "gpt-4o",
				"input": "Hello, world!"
			}`,
			wantErr: false,
			validate: func(t *testing.T, output []byte) {
				var req types.ChatCompletionRequest
				if err := json.Unmarshal(output, &req); err != nil {
					t.Fatalf("Failed to parse output: %v", err)
				}
				if req.Model != "gpt-4o" {
					t.Errorf("Expected model gpt-4o, got %s", req.Model)
				}
				if len(req.Messages) != 1 {
					t.Errorf("Expected 1 message, got %d", len(req.Messages))
				}
				if req.Messages[0].Role != "user" {
					t.Errorf("Expected user role, got %s", req.Messages[0].Role)
				}
				if req.Messages[0].Content != "Hello, world!" {
					t.Errorf("Expected content 'Hello, world!', got %v", req.Messages[0].Content)
				}
			},
		},
		{
			name: "input with instructions",
			input: `{
				"model": "gpt-4o",
				"input": "Hello",
				"instructions": "You are a helpful assistant."
			}`,
			wantErr: false,
			validate: func(t *testing.T, output []byte) {
				var req types.ChatCompletionRequest
				if err := json.Unmarshal(output, &req); err != nil {
					t.Fatalf("Failed to parse output: %v", err)
				}
				if req.System != "You are a helpful assistant." {
					t.Errorf("Expected system message, got %s", req.System)
				}
			},
		},
		{
			name: "input with max_output_tokens",
			input: `{
				"model": "gpt-4o",
				"input": "Hello",
				"max_output_tokens": 1000
			}`,
			wantErr: false,
			validate: func(t *testing.T, output []byte) {
				var req types.ChatCompletionRequest
				if err := json.Unmarshal(output, &req); err != nil {
					t.Fatalf("Failed to parse output: %v", err)
				}
				if req.MaxTokens != 1000 {
					t.Errorf("Expected max_tokens 1000, got %d", req.MaxTokens)
				}
			},
		},
		{
			name: "input with message array",
			input: `{
				"model": "gpt-4o",
				"input": [
					{"type": "message", "role": "user", "content": "Hello"},
					{"type": "message", "role": "assistant", "content": "Hi there!"},
					{"type": "message", "role": "user", "content": "How are you?"}
				]
			}`,
			wantErr: false,
			validate: func(t *testing.T, output []byte) {
				var req types.ChatCompletionRequest
				if err := json.Unmarshal(output, &req); err != nil {
					t.Fatalf("Failed to parse output: %v", err)
				}
				if len(req.Messages) != 3 {
					t.Errorf("Expected 3 messages, got %d", len(req.Messages))
				}
				if req.Messages[0].Role != "user" {
					t.Errorf("Expected first message role user, got %s", req.Messages[0].Role)
				}
				if req.Messages[1].Role != "assistant" {
					t.Errorf("Expected second message role assistant, got %s", req.Messages[1].Role)
				}
			},
		},
		{
			name: "input with tools (flat format)",
			input: `{
				"model": "gpt-4o",
				"input": "What's the weather?",
				"tools": [
					{
						"type": "function",
						"name": "get_weather",
						"description": "Get the current weather",
						"parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
					}
				]
			}`,
			wantErr: false,
			validate: func(t *testing.T, output []byte) {
				var req types.ChatCompletionRequest
				if err := json.Unmarshal(output, &req); err != nil {
					t.Fatalf("Failed to parse output: %v", err)
				}
				if len(req.Tools) != 1 {
					t.Errorf("Expected 1 tool, got %d", len(req.Tools))
				}
				if req.Tools[0].Type != "function" {
					t.Errorf("Expected tool type function, got %s", req.Tools[0].Type)
				}
				if req.Tools[0].Function.Name != "get_weather" {
					t.Errorf("Expected function name get_weather, got %s", req.Tools[0].Function.Name)
				}
			},
		},
		{
			name: "input with tools (nested format)",
			input: `{
				"model": "gpt-4o",
				"input": "Search for something",
				"tools": [
					{
						"type": "function",
						"function": {
							"name": "search",
							"description": "Search the web",
							"parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
						}
					}
				]
			}`,
			wantErr: false,
			validate: func(t *testing.T, output []byte) {
				var req types.ChatCompletionRequest
				if err := json.Unmarshal(output, &req); err != nil {
					t.Fatalf("Failed to parse output: %v", err)
				}
				if len(req.Tools) != 1 {
					t.Errorf("Expected 1 tool, got %d", len(req.Tools))
				}
				if req.Tools[0].Function.Name != "search" {
					t.Errorf("Expected function name search, got %s", req.Tools[0].Function.Name)
				}
			},
		},
		{
			name: "input with stream flag",
			input: `{
				"model": "gpt-4o",
				"input": "Hello",
				"stream": true
			}`,
			wantErr: false,
			validate: func(t *testing.T, output []byte) {
				var req types.ChatCompletionRequest
				if err := json.Unmarshal(output, &req); err != nil {
					t.Fatalf("Failed to parse output: %v", err)
				}
				if !req.Stream {
					t.Error("Expected stream to be true")
				}
			},
		},
		{
			name: "input with temperature and top_p",
			input: `{
				"model": "gpt-4o",
				"input": "Hello",
				"temperature": 0.7,
				"top_p": 0.9
			}`,
			wantErr: false,
			validate: func(t *testing.T, output []byte) {
				var req types.ChatCompletionRequest
				if err := json.Unmarshal(output, &req); err != nil {
					t.Fatalf("Failed to parse output: %v", err)
				}
				if req.Temperature != 0.7 {
					t.Errorf("Expected temperature 0.7, got %f", req.Temperature)
				}
				if req.TopP != 0.9 {
					t.Errorf("Expected top_p 0.9, got %f", req.TopP)
				}
			},
		},
		{
			name:    "invalid JSON",
			input:   `{invalid json}`,
			wantErr: true,
		},
		{
			name: "empty input",
			input: `{
				"model": "gpt-4o",
				"input": ""
			}`,
			wantErr: false,
			validate: func(t *testing.T, output []byte) {
				var req types.ChatCompletionRequest
				if err := json.Unmarshal(output, &req); err != nil {
					t.Fatalf("Failed to parse output: %v", err)
				}
				if len(req.Messages) != 0 {
					t.Errorf("Expected 0 messages for empty input, got %d", len(req.Messages))
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			converter := NewResponsesToChatConverter()
			output, err := converter.Convert([]byte(tt.input))

			if tt.wantErr {
				if err == nil {
					t.Error("Expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			if tt.validate != nil {
				tt.validate(t, output)
			}
		})
	}
}

// TestResponsesToChatTransformer_Transform tests response streaming conversion.
func TestResponsesToChatTransformer_Transform(t *testing.T) {
	tests := []struct {
		name     string
		events   []types.ResponsesStreamEvent
		wantDone bool
		validate func(t *testing.T, output string)
	}{
		{
			name: "response.created event",
			events: []types.ResponsesStreamEvent{
				{
					Type: "response.created",
					Response: &types.ResponsesResponse{
						ID:    "resp_123",
						Model: "gpt-4o",
					},
				},
			},
			validate: func(t *testing.T, output string) {
				// response.created should not produce output
				if output != "" {
					t.Errorf("Expected no output for response.created, got: %s", output)
				}
			},
		},
		{
			name: "text delta streaming",
			events: []types.ResponsesStreamEvent{
				{
					Type: "response.created",
					Response: &types.ResponsesResponse{
						ID:    "resp_123",
						Model: "gpt-4o",
					},
				},
				{
					Type:       "response.output_text.delta",
					ItemID:     "msg_123",
					Delta:      "Hello",
				},
				{
					Type:       "response.output_text.delta",
					ItemID:     "msg_123",
					Delta:      " world",
				},
			},
			validate: func(t *testing.T, output string) {
				if !strings.Contains(output, `"content":"Hello"`) {
					t.Error("Expected content 'Hello' in delta")
				}
				if !strings.Contains(output, `"content":" world"`) {
					t.Error("Expected content ' world' in delta")
				}
				if !strings.Contains(output, `"id":"resp_123"`) {
					t.Error("Expected response ID in output")
				}
				if !strings.Contains(output, `"model":"gpt-4o"`) {
					t.Error("Expected model in output")
				}
			},
		},
		{
			name: "function call streaming",
			events: []types.ResponsesStreamEvent{
				{
					Type: "response.created",
					Response: &types.ResponsesResponse{
						ID:    "resp_123",
						Model: "gpt-4o",
					},
				},
				{
					Type: "response.output_item.added",
					OutputItem: &types.OutputItem{
						Type: "function_call",
						ID:   "call_123",
						Name: "get_weather",
					},
				},
				{
					Type:       "response.function_call_arguments.delta",
					ItemID:     "call_123",
					Delta:      `{"location"`,
				},
				{
					Type:       "response.function_call_arguments.delta",
					ItemID:     "call_123",
					Delta:      `: "Paris"}`,
				},
			},
			validate: func(t *testing.T, output string) {
				if !strings.Contains(output, `"type":"function"`) {
					t.Error("Expected function type in tool call")
				}
				if !strings.Contains(output, `"id":"call_123"`) {
					t.Error("Expected tool call ID in output")
				}
				if !strings.Contains(output, `"arguments"`) {
					t.Error("Expected arguments field in tool call")
				}
			},
		},
		{
			name: "response.completed with stop",
			events: []types.ResponsesStreamEvent{
				{
					Type: "response.created",
					Response: &types.ResponsesResponse{
						ID:    "resp_123",
						Model: "gpt-4o",
					},
				},
				{
					Type:       "response.output_text.delta",
					ItemID:     "msg_123",
					Delta:      "Hello",
				},
				{
					Type: "response.completed",
					Response: &types.ResponsesResponse{
						ID:     "resp_123",
						Model:  "gpt-4o",
						Status: "completed",
						Output: []types.OutputItem{
							{Type: "message"},
						},
						Usage: &types.ResponsesUsage{
							InputTokens:  10,
							OutputTokens: 5,
							TotalTokens:  15,
						},
					},
				},
			},
			validate: func(t *testing.T, output string) {
				if !strings.Contains(output, `"finish_reason":"stop"`) {
					t.Error("Expected finish_reason 'stop' in output")
				}
				if !strings.Contains(output, `"prompt_tokens":10`) {
					t.Error("Expected prompt_tokens in usage")
				}
				if !strings.Contains(output, `"completion_tokens":5`) {
					t.Error("Expected completion_tokens in usage")
				}
			},
		},
		{
			name: "response.completed with tool_calls",
			events: []types.ResponsesStreamEvent{
				{
					Type: "response.created",
					Response: &types.ResponsesResponse{
						ID:    "resp_123",
						Model: "gpt-4o",
					},
				},
				{
					Type: "response.output_item.added",
					OutputItem: &types.OutputItem{
						Type: "function_call",
						ID:   "call_123",
						Name: "get_weather",
					},
				},
				{
					Type:       "response.function_call_arguments.delta",
					ItemID:     "call_123",
					Delta:      `{}`,
				},
				{
					Type: "response.completed",
					Response: &types.ResponsesResponse{
						ID:     "resp_123",
						Model:  "gpt-4o",
						Status: "completed",
						Output: []types.OutputItem{
							{Type: "function_call", ID: "call_123"},
						},
					},
				},
			},
			validate: func(t *testing.T, output string) {
				if !strings.Contains(output, `"finish_reason":"tool_calls"`) {
					t.Error("Expected finish_reason 'tool_calls' in output")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			transformer := NewResponsesToChatTransformer(&buf)

			for _, event := range tt.events {
				data, err := json.Marshal(event)
				if err != nil {
					t.Fatalf("Failed to marshal event: %v", err)
				}
				err = transformer.Transform(&sse.Event{Data: string(data)})
				if err != nil {
					t.Errorf("Transform returned error: %v", err)
				}
			}

			output := buf.String()
			if tt.validate != nil {
				tt.validate(t, output)
			}
		})
	}
}

// TestResponsesToChatTransformer_DoneEvent tests [DONE] handling.
func TestResponsesToChatTransformer_DoneEvent(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesToChatTransformer(&buf)

	err := transformer.Transform(&sse.Event{Data: "[DONE]"})
	if err != nil {
		t.Errorf("Transform returned error: %v", err)
	}

	output := buf.String()
	if !strings.Contains(output, "data: [DONE]") {
		t.Errorf("Expected [DONE] marker in output, got: %s", output)
	}
}

// TestResponsesToChatTransformer_EmptyData tests empty data handling.
func TestResponsesToChatTransformer_EmptyData(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesToChatTransformer(&buf)

	err := transformer.Transform(&sse.Event{Data: ""})
	if err != nil {
		t.Errorf("Transform returned error: %v", err)
	}

	if buf.Len() != 0 {
		t.Errorf("Expected empty buffer, got: %s", buf.String())
	}
}

// TestResponsesToChatTransformer_InvalidJSON tests invalid JSON handling.
func TestResponsesToChatTransformer_InvalidJSON(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesToChatTransformer(&buf)

	err := transformer.Transform(&sse.Event{Data: "not valid json"})
	if err != nil {
		t.Errorf("Transform returned error: %v", err)
	}

	output := buf.String()
	if !strings.Contains(output, "data: not valid json") {
		t.Errorf("Expected pass-through of invalid JSON, got: %s", output)
	}
}

// TestResponsesToChatTransformer_ErrorEvent tests error event handling.
func TestResponsesToChatTransformer_ErrorEvent(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesToChatTransformer(&buf)

	event := types.ResponsesStreamEvent{
		Type: "error",
		Error: &types.ResponsesError{
			Code:    "rate_limit_exceeded",
			Message: "Rate limit exceeded",
		},
	}

	data, _ := json.Marshal(event)
	err := transformer.Transform(&sse.Event{Data: string(data)})
	if err != nil {
		t.Errorf("Transform returned error: %v", err)
	}

	output := buf.String()
	if !strings.Contains(output, `"error"`) {
		t.Errorf("Expected error in output, got: %s", output)
	}
	if !strings.Contains(output, "Rate limit exceeded") {
		t.Errorf("Expected error message in output, got: %s", output)
	}
}

// TestResponsesToChatTransformer_Flush tests flush operation.
func TestResponsesToChatTransformer_Flush(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesToChatTransformer(&buf)

	err := transformer.Flush()
	if err != nil {
		t.Errorf("Flush returned error: %v", err)
	}
}

// TestResponsesToChatTransformer_Close tests close operation.
func TestResponsesToChatTransformer_Close(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesToChatTransformer(&buf)

	err := transformer.Close()
	if err != nil {
		t.Errorf("Close returned error: %v", err)
	}
}

// TestResponsesToChatConverter_ContentParts tests content part conversion.
func TestResponsesToChatConverter_ContentParts(t *testing.T) {
	converter := NewResponsesToChatConverter()

	tests := []struct {
		name     string
		input    string
		validate func(t *testing.T, output []byte)
	}{
		{
			name: "input_text content",
			input: `{
				"model": "gpt-4o",
				"input": [
					{
						"type": "message",
						"role": "user",
						"content": [
							{"type": "input_text", "text": "Hello"}
						]
					}
				]
			}`,
			validate: func(t *testing.T, output []byte) {
				var req types.ChatCompletionRequest
				if err := json.Unmarshal(output, &req); err != nil {
					t.Fatalf("Failed to parse output: %v", err)
				}
				if len(req.Messages) != 1 {
					t.Errorf("Expected 1 message, got %d", len(req.Messages))
				}
			},
		},
		{
			name: "input_image content",
			input: `{
				"model": "gpt-4o",
				"input": [
					{
						"type": "message",
						"role": "user",
						"content": [
							{"type": "input_text", "text": "What's in this image?"},
							{"type": "input_image", "image_url": "https://example.com/image.png"}
						]
					}
				]
			}`,
			validate: func(t *testing.T, output []byte) {
				var req types.ChatCompletionRequest
				if err := json.Unmarshal(output, &req); err != nil {
					t.Fatalf("Failed to parse output: %v", err)
				}
				if len(req.Messages) != 1 {
					t.Errorf("Expected 1 message, got %d", len(req.Messages))
				}
				// Check that content was converted
				content, ok := req.Messages[0].Content.([]interface{})
				if !ok {
					t.Errorf("Expected content to be array, got %T", req.Messages[0].Content)
					return
				}
				if len(content) != 2 {
					t.Errorf("Expected 2 content parts, got %d", len(content))
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output, err := converter.Convert([]byte(tt.input))
			if err != nil {
				t.Errorf("Convert returned error: %v", err)
				return
			}
			if tt.validate != nil {
				tt.validate(t, output)
			}
		})
	}
}

// TestResponsesToChatConverter_NonFunctionTools tests that non-function tools are skipped.
func TestResponsesToChatConverter_NonFunctionTools(t *testing.T) {
	converter := NewResponsesToChatConverter()

	input := `{
		"model": "gpt-4o",
		"input": "Hello",
		"tools": [
			{"type": "file_search"},
			{"type": "function", "name": "search", "description": "Search"},
			{"type": "web_search"}
		]
	}`

	output, err := converter.Convert([]byte(input))
	if err != nil {
		t.Fatalf("Convert returned error: %v", err)
	}

	var req types.ChatCompletionRequest
	if err := json.Unmarshal(output, &req); err != nil {
		t.Fatalf("Failed to parse output: %v", err)
	}

	// Only function tools should be included
	if len(req.Tools) != 1 {
		t.Errorf("Expected 1 tool (only function type), got %d", len(req.Tools))
	}
	if len(req.Tools) > 0 && req.Tools[0].Function.Name != "search" {
		t.Errorf("Expected function name 'search', got %s", req.Tools[0].Function.Name)
	}
}

// TestResponsesToChatTransformer_FullFlow tests a complete streaming flow.
func TestResponsesToChatTransformer_FullFlow(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesToChatTransformer(&buf)

	events := []types.ResponsesStreamEvent{
		{
			Type: "response.created",
			Response: &types.ResponsesResponse{
				ID:    "resp_abc123",
				Model: "gpt-4o",
			},
		},
		{
			Type:       "response.output_text.delta",
			ItemID:     "msg_123",
			Delta:      "Hello",
		},
		{
			Type:       "response.output_text.delta",
			ItemID:     "msg_123",
			Delta:      " there",
		},
		{
			Type:       "response.output_text.delta",
			ItemID:     "msg_123",
			Delta:      "!",
		},
		{
			Type: "response.completed",
			Response: &types.ResponsesResponse{
				ID:     "resp_abc123",
				Model:  "gpt-4o",
				Status: "completed",
				Output: []types.OutputItem{
					{Type: "message"},
				},
				Usage: &types.ResponsesUsage{
					InputTokens:  5,
					OutputTokens: 3,
					TotalTokens:  8,
				},
			},
		},
	}

	for _, event := range events {
		data, _ := json.Marshal(event)
		if err := transformer.Transform(&sse.Event{Data: string(data)}); err != nil {
			t.Errorf("Transform returned error: %v", err)
		}
	}

	output := buf.String()

	// Verify all expected content
	expectedStrings := []string{
		`"id":"resp_abc123"`,
		`"model":"gpt-4o"`,
		`"content":"Hello"`,
		`"content":" there"`,
		`"content":"!"`,
		`"finish_reason":"stop"`,
		`"prompt_tokens":5`,
		`"completion_tokens":3`,
	}

	for _, expected := range expectedStrings {
		if !strings.Contains(output, expected) {
			t.Errorf("Expected %s in output", expected)
		}
	}
}

// BenchmarkResponsesToChatConverter_Convert benchmarks request conversion.
func BenchmarkResponsesToChatConverter_Convert(b *testing.B) {
	converter := NewResponsesToChatConverter()
	input := []byte(`{
		"model": "gpt-4o",
		"input": "Hello, world!",
		"instructions": "You are helpful.",
		"max_output_tokens": 1000
	}`)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		converter.Convert(input)
	}
}

// BenchmarkResponsesToChatTransformer_Transform benchmarks response transformation.
func BenchmarkResponsesToChatTransformer_Transform(b *testing.B) {
	event := types.ResponsesStreamEvent{
		Type:       "response.output_text.delta",
		ItemID:     "msg_123",
		Delta:      "Hello world",
	}
	data, _ := json.Marshal(event)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var buf bytes.Buffer
		transformer := NewResponsesToChatTransformer(&buf)
		transformer.Transform(&sse.Event{Data: string(data)})
	}
}