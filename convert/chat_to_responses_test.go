package convert

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"ai-proxy/types"

	"github.com/tmaxmax/go-sse"
)

func TestChatToResponsesTransformer_ToolCalls(t *testing.T) {
	tests := []struct {
		name           string
		chunks         []types.Chunk
		wantToolCalls  int // number of tool calls expected
		wantArgs       string // expected arguments for first tool call
		wantName       string // expected name for first tool call
		wantCallID     string // expected call_id for first tool call
	}{
		{
			name: "single tool call with streaming arguments",
			chunks: []types.Chunk{
				{
					ID:      "chatcmpl-123",
					Object:  "chat.completion.chunk",
					Model:   "test-model",
					Choices: []types.Choice{
						{
							Index: 0,
							Delta: types.Delta{
								ToolCalls: []types.ToolCall{
									{
										ID:    "call_abc123",
										Type:  "function",
										Index: 0,
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
					ID:      "chatcmpl-123",
					Object:  "chat.completion.chunk",
					Model:   "test-model",
					Choices: []types.Choice{
						{
							Index: 0,
							Delta: types.Delta{
								ToolCalls: []types.ToolCall{
									{
										Index: 0,
										Type:  "function",
										Function: types.Function{
											Arguments: "{\"loc",
										},
									},
								},
							},
						},
					},
				},
				{
					ID:      "chatcmpl-123",
					Object:  "chat.completion.chunk",
					Model:   "test-model",
					Choices: []types.Choice{
						{
							Index: 0,
							Delta: types.Delta{
								ToolCalls: []types.ToolCall{
									{
										Index: 0,
										Type:  "function",
										Function: types.Function{
											Arguments: "ation\": \"SF\"}",
										},
									},
								},
							},
						},
					},
				},
			},
			wantToolCalls: 1,
			wantArgs:      "{\"location\": \"SF\"}",
			wantName:      "get_weather",
			wantCallID:    "call_abc123",
		},
		{
			name: "multiple tool calls",
			chunks: []types.Chunk{
				{
					ID:      "chatcmpl-123",
					Object:  "chat.completion.chunk",
					Model:   "test-model",
					Choices: []types.Choice{
						{
							Index: 0,
							Delta: types.Delta{
								ToolCalls: []types.ToolCall{
									{
										ID:    "call_001",
										Type:  "function",
										Index: 0,
										Function: types.Function{
											Name: "func_a",
										},
									},
								},
							},
						},
					},
				},
				{
					ID:      "chatcmpl-123",
					Object:  "chat.completion.chunk",
					Model:   "test-model",
					Choices: []types.Choice{
						{
							Index: 0,
							Delta: types.Delta{
								ToolCalls: []types.ToolCall{
									{
										Index: 0,
										Type:  "function",
										Function: types.Function{
											Arguments: "{\"a\": 1}",
										},
									},
								},
							},
						},
					},
				},
				{
					ID:      "chatcmpl-123",
					Object:  "chat.completion.chunk",
					Model:   "test-model",
					Choices: []types.Choice{
						{
							Index: 0,
							Delta: types.Delta{
								ToolCalls: []types.ToolCall{
									{
										ID:    "call_002",
										Type:  "function",
										Index: 1,
										Function: types.Function{
											Name: "func_b",
										},
									},
								},
							},
						},
					},
				},
				{
					ID:      "chatcmpl-123",
					Object:  "chat.completion.chunk",
					Model:   "test-model",
					Choices: []types.Choice{
						{
							Index: 0,
							Delta: types.Delta{
								ToolCalls: []types.ToolCall{
									{
										Index: 1,
										Type:  "function",
										Function: types.Function{
											Arguments: "{\"b\": 2}",
										},
									},
								},
							},
						},
					},
				},
			},
			wantToolCalls: 2,
		},
		{
			name: "tool call with role assistant in first chunk",
			chunks: []types.Chunk{
				// First chunk has both role: "assistant" and the tool call
				{
					ID:      "chatcmpl-123",
					Object:  "chat.completion.chunk",
					Model:   "test-model",
					Choices: []types.Choice{
						{
							Index: 0,
							Delta: types.Delta{
								Role: "assistant",
								ToolCalls: []types.ToolCall{
									{
										ID:    "call_xyz",
										Type:  "function",
										Index: 0,
										Function: types.Function{
											Name: "read_file",
										},
									},
								},
							},
						},
					},
				},
				{
					ID:      "chatcmpl-123",
					Object:  "chat.completion.chunk",
					Model:   "test-model",
					Choices: []types.Choice{
						{
							Index: 0,
							Delta: types.Delta{
								ToolCalls: []types.ToolCall{
									{
										Index: 0,
										Type:  "function",
										Function: types.Function{
											Arguments: "{\"path\": \"test.txt\"}",
										},
									},
								},
							},
						},
					},
				},
			},
			wantToolCalls: 1,
			wantArgs:      "{\"path\": \"test.txt\"}",
			wantName:      "read_file",
			wantCallID:    "call_xyz",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			transformer := NewChatToResponsesTransformer(&buf)

			for _, chunk := range tt.chunks {
				data, err := json.Marshal(chunk)
				if err != nil {
					t.Fatalf("Failed to marshal chunk: %v", err)
				}
				event := &sse.Event{Data: string(data)}
				if err := transformer.Transform(event); err != nil {
					t.Fatalf("Transform failed: %v", err)
				}
			}

			// Close to emit final events
			if err := transformer.Close(); err != nil {
				t.Fatalf("Close failed: %v", err)
			}

			// Parse output events
			output := buf.String()
			t.Logf("Output:\n%s", output)

			// Count response.output_item.done events for function_call
			doneEvents := 0
			lines := bytes.Split(buf.Bytes(), []byte("\n\n"))
			for _, line := range lines {
				if bytes.Contains(line, []byte("\"type\":\"response.output_item.done\"")) {
					if bytes.Contains(line, []byte("\"type\":\"function_call\"")) {
						// Extract the item
						dataIdx := bytes.Index(line, []byte("data: "))
						if dataIdx >= 0 {
							data := line[dataIdx+6:]
							var event map[string]interface{}
							if err := json.Unmarshal(data, &event); err != nil {
								t.Fatalf("Failed to parse event: %v", err)
							}
							item, ok := event["item"].(map[string]interface{})
							if !ok {
								t.Fatal("No item in event")
							}

							// Check that the tool call has all fields populated
							name, _ := item["name"].(string)
							callID, _ := item["call_id"].(string)
							args, _ := item["arguments"].(string)

							t.Logf("Tool call done: name=%s, call_id=%s, args=%s", name, callID, args)

							// A valid tool call must have name and call_id
							if name == "" || callID == "" {
								t.Errorf("Tool call has empty name or call_id: name=%q, call_id=%q", name, callID)
							}

							doneEvents++
						}
					}
				}
			}

			if doneEvents != tt.wantToolCalls {
				t.Errorf("Expected %d tool calls, got %d", tt.wantToolCalls, doneEvents)
			}
		})
	}
}

func TestChatToResponsesTransformer_ContentWithRole(t *testing.T) {
	tests := []struct {
		name       string
		chunks     []types.Chunk
		wantText   string // expected text content
	}{
		{
			name: "first chunk has both role and content",
			chunks: []types.Chunk{
				// First chunk has role: "assistant" AND content - should not skip content
				{
					ID:     "chatcmpl-123",
					Object: "chat.completion.chunk",
					Model:  "test-model",
					Choices: []types.Choice{
						{
							Index: 0,
							Delta: types.Delta{
								Role:    "assistant",
								Content: "## ",
							},
						},
					},
				},
				{
					ID:     "chatcmpl-123",
					Object: "chat.completion.chunk",
					Model:  "test-model",
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
					ID:     "chatcmpl-123",
					Object: "chat.completion.chunk",
					Model:  "test-model",
					Choices: []types.Choice{
						{
							Index: 0,
							Delta: types.Delta{
								Content: " World",
							},
						},
					},
				},
			},
			wantText: "## Hello World",
		},
		{
			name: "role only in first chunk followed by content",
			chunks: []types.Chunk{
				{
					ID:     "chatcmpl-456",
					Object: "chat.completion.chunk",
					Model:  "test-model",
					Choices: []types.Choice{
						{
							Index: 0,
							Delta: types.Delta{
								Role: "assistant",
							},
						},
					},
				},
				{
					ID:     "chatcmpl-456",
					Object: "chat.completion.chunk",
					Model:  "test-model",
					Choices: []types.Choice{
						{
							Index: 0,
							Delta: types.Delta{
								Content: "Test",
							},
						},
					},
				},
			},
			wantText: "Test",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			transformer := NewChatToResponsesTransformer(&buf)

			for _, chunk := range tt.chunks {
				data, err := json.Marshal(chunk)
				if err != nil {
					t.Fatalf("Failed to marshal chunk: %v", err)
				}
				event := &sse.Event{Data: string(data)}
				if err := transformer.Transform(event); err != nil {
					t.Fatalf("Transform failed: %v", err)
				}
			}

			// Close to emit final events
			if err := transformer.Close(); err != nil {
				t.Fatalf("Close failed: %v", err)
			}

			// Extract text deltas from output
			var textBuilder strings.Builder
			lines := bytes.Split(buf.Bytes(), []byte("\n\n"))
			for _, line := range lines {
				if bytes.Contains(line, []byte("\"type\":\"response.output_text.delta\"")) {
					dataIdx := bytes.Index(line, []byte("data: "))
					if dataIdx >= 0 {
						data := line[dataIdx+6:]
						var event map[string]interface{}
						if err := json.Unmarshal(data, &event); err != nil {
							t.Fatalf("Failed to parse event: %v", err)
						}
						if delta, ok := event["delta"].(string); ok {
							textBuilder.WriteString(delta)
						}
					}
				}
			}

			gotText := textBuilder.String()
			if gotText != tt.wantText {
				t.Errorf("Expected text %q, got %q", tt.wantText, gotText)
			}
		})
	}
}