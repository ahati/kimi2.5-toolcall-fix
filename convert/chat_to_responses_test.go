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
		name          string
		chunks        []types.Chunk
		wantToolCalls int    // number of tool calls expected
		wantArgs      string // expected arguments for first tool call
		wantName      string // expected name for first tool call
		wantCallID    string // expected call_id for first tool call
	}{
		{
			name: "single tool call with streaming arguments",
			chunks: []types.Chunk{
				{
					ID:     "chatcmpl-123",
					Object: "chat.completion.chunk",
					Model:  "test-model",
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
					ID:     "chatcmpl-123",
					Object: "chat.completion.chunk",
					Model:  "test-model",
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
					ID:     "chatcmpl-123",
					Object: "chat.completion.chunk",
					Model:  "test-model",
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
					ID:     "chatcmpl-123",
					Object: "chat.completion.chunk",
					Model:  "test-model",
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
					ID:     "chatcmpl-123",
					Object: "chat.completion.chunk",
					Model:  "test-model",
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
					ID:     "chatcmpl-123",
					Object: "chat.completion.chunk",
					Model:  "test-model",
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
					ID:     "chatcmpl-123",
					Object: "chat.completion.chunk",
					Model:  "test-model",
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
					ID:     "chatcmpl-123",
					Object: "chat.completion.chunk",
					Model:  "test-model",
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
					ID:     "chatcmpl-123",
					Object: "chat.completion.chunk",
					Model:  "test-model",
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
		name     string
		chunks   []types.Chunk
		wantText string // expected text content
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

func TestChatToResponsesTransformer_UsageAfterFinishReason(t *testing.T) {
	// Test that usage is properly captured when it arrives AFTER finish_reason
	// in a separate chunk (which is the OpenAI streaming behavior with stream_options)
	finishReason := "stop"

	chunks := []types.Chunk{
		// First chunk with content
		{
			ID:     "chatcmpl-123",
			Object: "chat.completion.chunk",
			Model:  "test-model",
			Choices: []types.Choice{
				{
					Index: 0,
					Delta: types.Delta{
						Role:    "assistant",
						Content: "Hello",
					},
				},
			},
		},
		// Second chunk with finish_reason but NO usage yet
		{
			ID:     "chatcmpl-123",
			Object: "chat.completion.chunk",
			Model:  "test-model",
			Choices: []types.Choice{
				{
					Index:        0,
					Delta:        types.Delta{},
					FinishReason: &finishReason,
				},
			},
		},
		// Third chunk with ONLY usage (no choices)
		{
			ID:     "chatcmpl-123",
			Object: "chat.completion.chunk",
			Model:  "test-model",
			Usage: &types.Usage{
				PromptTokens:     100,
				CompletionTokens: 50,
				TotalTokens:      150,
			},
		},
	}

	var buf bytes.Buffer
	transformer := NewChatToResponsesTransformer(&buf)

	for _, chunk := range chunks {
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

	// Parse output and check that usage is in response.completed
	output := buf.String()
	t.Logf("Output:\n%s", output)

	// Find response.completed event and check usage
	lines := bytes.Split(buf.Bytes(), []byte("\n\n"))
	var foundCompleted bool
	var usageInput, usageOutput, usageTotal int

	for _, line := range lines {
		if bytes.Contains(line, []byte(`"type":"response.completed"`)) {
			foundCompleted = true
			dataIdx := bytes.Index(line, []byte("data: "))
			if dataIdx >= 0 {
				data := line[dataIdx+6:]
				var event map[string]interface{}
				if err := json.Unmarshal(data, &event); err != nil {
					t.Fatalf("Failed to parse event: %v", err)
				}
				resp, ok := event["response"].(map[string]interface{})
				if !ok {
					t.Fatal("No response in response.completed event")
				}
				usage, ok := resp["usage"].(map[string]interface{})
				if !ok {
					t.Fatal("No usage in response.completed event")
				}
				usageInput = int(usage["input_tokens"].(float64))
				usageOutput = int(usage["output_tokens"].(float64))
				usageTotal = int(usage["total_tokens"].(float64))
			}
		}
	}

	if !foundCompleted {
		t.Fatal("response.completed event not found")
	}

	if usageInput != 100 {
		t.Errorf("Expected input_tokens=100, got %d", usageInput)
	}
	if usageOutput != 50 {
		t.Errorf("Expected output_tokens=50, got %d", usageOutput)
	}
	if usageTotal != 150 {
		t.Errorf("Expected total_tokens=150, got %d", usageTotal)
	}
}

func TestChatToResponsesTransformer_ToolCallOnly(t *testing.T) {
	chunks := []types.Chunk{
		{
			ID:     "chatcmpl-123",
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
			ID:     "chatcmpl-123",
			Object: "chat.completion.chunk",
			Model:  "test-model",
			Choices: []types.Choice{
				{
					Index: 0,
					Delta: types.Delta{
						ToolCalls: []types.ToolCall{
							{
								ID:    "call_abc",
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
			ID:     "chatcmpl-123",
			Object: "chat.completion.chunk",
			Model:  "test-model",
			Choices: []types.Choice{
				{
					Index: 0,
					Delta: types.Delta{
						ToolCalls: []types.ToolCall{
							{
								Index: 0,
								Function: types.Function{
									Arguments: "{\"city\": \"SF\"}",
								},
							},
						},
					},
				},
			},
		},
	}
	finishReason := "tool_calls"
	chunks = append(chunks, types.Chunk{
		ID:      "chatcmpl-123",
		Choices: []types.Choice{{Index: 0, FinishReason: &finishReason}},
	})

	var buf bytes.Buffer
	transformer := NewChatToResponsesTransformer(&buf)

	for _, chunk := range chunks {
		data, err := json.Marshal(chunk)
		if err != nil {
			t.Fatalf("Failed to marshal chunk: %v", err)
		}
		event := &sse.Event{Data: string(data)}
		if err := transformer.Transform(event); err != nil {
			t.Fatalf("Transform failed: %v", err)
		}
	}

	if err := transformer.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	output := buf.String()
	t.Logf("Output:\n%s", output)

	if bytes.Contains(buf.Bytes(), []byte(`"type":"message"`)) {
		t.Error("Expected no message item in response.completed for tool-call-only response")
	}

	doneCount := bytes.Count(buf.Bytes(), []byte(`"type":"response.output_item.done"`))
	if doneCount != 1 {
		t.Errorf("Expected 1 output_item.done event for tool call, got %d", doneCount)
	}
}

func TestChatToResponsesTransformer_NoChunkID(t *testing.T) {
	chunks := []types.Chunk{
		{
			ID:     "",
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
			ID:     "",
			Object: "chat.completion.chunk",
			Model:  "test-model",
			Choices: []types.Choice{
				{
					Index: 0,
					Delta: types.Delta{
						Content: " world",
					},
				},
			},
		},
	}

	var buf bytes.Buffer
	transformer := NewChatToResponsesTransformer(&buf)

	for _, chunk := range chunks {
		data, err := json.Marshal(chunk)
		if err != nil {
			t.Fatalf("Failed to marshal chunk: %v", err)
		}
		event := &sse.Event{Data: string(data)}
		if err := transformer.Transform(event); err != nil {
			t.Fatalf("Transform failed: %v", err)
		}
	}

	if err := transformer.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	output := buf.String()
	t.Logf("Output:\n%s", output)

	if !bytes.Contains(buf.Bytes(), []byte(`"type":"response.created"`)) {
		t.Error("Expected response.created event even when chunk.ID is empty")
	}

	createdIdx := bytes.Index(buf.Bytes(), []byte(`"type":"response.created"`))
	deltaIdx := bytes.Index(buf.Bytes(), []byte(`"type":"response.output_text.delta"`))
	if createdIdx == -1 || deltaIdx == -1 {
		return
	}
	if createdIdx > deltaIdx {
		t.Error("response.created should be emitted before response.output_text.delta")
	}
}

func TestChatToResponsesTransformer_DoneEvents(t *testing.T) {
	chunks := []types.Chunk{
		{
			ID:     "chatcmpl-123",
			Object: "chat.completion.chunk",
			Model:  "test-model",
			Choices: []types.Choice{
				{
					Index: 0,
					Delta: types.Delta{
						Role:    "assistant",
						Content: "Hello",
					},
				},
			},
		},
	}
	finishReason := "stop"
	chunks = append(chunks, types.Chunk{
		ID:      "chatcmpl-123",
		Choices: []types.Choice{{Index: 0, FinishReason: &finishReason}},
	})

	var buf bytes.Buffer
	transformer := NewChatToResponsesTransformer(&buf)

	for _, chunk := range chunks {
		data, err := json.Marshal(chunk)
		if err != nil {
			t.Fatalf("Failed to marshal chunk: %v", err)
		}
		event := &sse.Event{Data: string(data)}
		if err := transformer.Transform(event); err != nil {
			t.Fatalf("Transform failed: %v", err)
		}
	}

	if err := transformer.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	output := buf.String()
	t.Logf("Output:\n%s", output)

	if !bytes.Contains(buf.Bytes(), []byte(`"type":"response.output_text.done"`)) {
		t.Error("Expected response.output_text.done event")
	}
	if !bytes.Contains(buf.Bytes(), []byte(`"type":"response.content_part.done"`)) {
		t.Error("Expected response.content_part.done event")
	}
	if !bytes.Contains(buf.Bytes(), []byte(`"type":"response.output_item.done"`)) {
		t.Error("Expected response.output_item.done event for message")
	}

	completedIdx := bytes.Index(buf.Bytes(), []byte(`"type":"response.completed"`))
	textDoneIdx := bytes.Index(buf.Bytes(), []byte(`"type":"response.output_text.done"`))
	if completedIdx == -1 || textDoneIdx == -1 {
		return
	}
	if textDoneIdx > completedIdx {
		t.Error("response.output_text.done should be emitted before response.completed")
	}
}

func TestChatToResponsesTransformer_OutputOrdering(t *testing.T) {
	chunks := []types.Chunk{
		{
			ID:     "chatcmpl-123",
			Object: "chat.completion.chunk",
			Model:  "test-model",
			Choices: []types.Choice{
				{
					Index: 0,
					Delta: types.Delta{
						ReasoningContent: "Let me think...",
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
						ToolCalls: []types.ToolCall{
							{
								ID:    "call_abc",
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
			ID:     "chatcmpl-123",
			Object: "chat.completion.chunk",
			Model:  "test-model",
			Choices: []types.Choice{
				{
					Index: 0,
					Delta: types.Delta{
						ToolCalls: []types.ToolCall{
							{
								Index: 0,
								Function: types.Function{
									Arguments: "{}",
								},
							},
						},
					},
				},
			},
		},
	}
	finishReason := "tool_calls"
	chunks = append(chunks, types.Chunk{
		ID:      "chatcmpl-123",
		Choices: []types.Choice{{Index: 0, FinishReason: &finishReason}},
	})

	var buf bytes.Buffer
	transformer := NewChatToResponsesTransformer(&buf)

	for _, chunk := range chunks {
		data, err := json.Marshal(chunk)
		if err != nil {
			t.Fatalf("Failed to marshal chunk: %v", err)
		}
		event := &sse.Event{Data: string(data)}
		if err := transformer.Transform(event); err != nil {
			t.Fatalf("Transform failed: %v", err)
		}
	}

	if err := transformer.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	output := buf.String()
	t.Logf("Output:\n%s", output)

	lines := bytes.Split(buf.Bytes(), []byte("\n\n"))
	var completedOutput []interface{}
	for _, line := range lines {
		if bytes.Contains(line, []byte(`"type":"response.completed"`)) {
			dataIdx := bytes.Index(line, []byte("data: "))
			if dataIdx >= 0 {
				data := line[dataIdx+6:]
				var event map[string]interface{}
				if err := json.Unmarshal(data, &event); err != nil {
					t.Fatalf("Failed to parse event: %v", err)
				}
				resp, ok := event["response"].(map[string]interface{})
				if !ok {
					t.Fatal("No response in response.completed event")
				}
				completedOutput = resp["output"].([]interface{})
			}
		}
	}

	if len(completedOutput) < 2 {
		t.Fatalf("Expected at least 2 output items, got %d", len(completedOutput))
	}

	reasoningIdx := -1
	messageIdx := -1
	toolCallIdx := -1
	for i, item := range completedOutput {
		itemMap, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		itemType, _ := itemMap["type"].(string)
		switch itemType {
		case "reasoning":
			reasoningIdx = i
		case "message":
			messageIdx = i
		case "function_call":
			toolCallIdx = i
		}
	}

	if reasoningIdx != -1 && reasoningIdx != 0 {
		t.Errorf("Reasoning should be at index 0, got %d", reasoningIdx)
	}
	if messageIdx != -1 && toolCallIdx != -1 && messageIdx > toolCallIdx {
		t.Errorf("Message (index %d) should come before tool calls (index %d)", messageIdx, toolCallIdx)
	}
}
