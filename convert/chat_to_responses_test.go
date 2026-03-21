package convert

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"ai-proxy/conversation"
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

// TestChatToResponsesTransformer_ToolCallsInReasoning tests tool call extraction from reasoning content.
// This verifies that when Kimi models emit tool calls inside reasoning blocks using proprietary markup,
// they are correctly extracted and converted to function_call output items.
func TestChatToResponsesTransformer_ToolCallsInReasoning(t *testing.T) {
	tests := []struct {
		name                string
		toolCallTransform   bool
		chunks              []types.Chunk
		validate            func(t *testing.T, output string)
	}{
		{
			name:              "tool call extraction enabled - single tool call",
			toolCallTransform: true,
			chunks: []types.Chunk{
				{
					ID:      "chatcmpl-123",
					Model:   "kimi-k2.5",
					Created: 1234567890,
					Choices: []types.Choice{{
						Index: 0,
						Delta: types.Delta{
							ReasoningContent: "Let me help you.<|tool_calls_section_begin|><|tool_call_begin|>bash:32<|tool_call_argument_begin|>{\"cmd\":\"ls\"}<|tool_call_end|><|tool_calls_section_end|>",
						},
					}},
				},
			},
			validate: func(t *testing.T, output string) {
				// Should contain function_call output item
				if !strings.Contains(output, `"type":"function_call"`) {
					t.Error("Expected function_call in output")
				}
				// Should contain function name
				if !strings.Contains(output, `"name":"bash"`) {
					t.Error("Expected function name 'bash' in output")
				}
				// Should contain function_call_arguments.delta for the args
				if !strings.Contains(output, `"type":"response.function_call_arguments.delta"`) {
					t.Error("Expected response.function_call_arguments.delta")
				}
				// Should contain the arguments
				if !strings.Contains(output, `"cmd\":\"ls\"`) {
					t.Error("Expected arguments in output")
				}
			},
		},
		{
			name:              "tool call extraction disabled - markup passed through",
			toolCallTransform: false,
			chunks: []types.Chunk{
				{
					ID:      "chatcmpl-456",
					Model:   "kimi-k2.5",
					Created: 1234567890,
					Choices: []types.Choice{{
						Index: 0,
						Delta: types.Delta{
							ReasoningContent: "Thinking...<|tool_calls_section_begin|><|tool_call_begin|>test<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>",
						},
					}},
				},
			},
			validate: func(t *testing.T, output string) {
				// Should NOT contain function_call - markup should be passed as reasoning
				if strings.Contains(output, `"type":"function_call"`) {
					t.Error("Should NOT contain function_call when extraction is disabled")
				}
				// Should contain reasoning type
				if !strings.Contains(output, `"type":"reasoning"`) {
					t.Errorf("Expected reasoning in output, got: %s", output)
				}
				// Should contain the raw markup in reasoning text (JSON-escaped)
				// The markup contains < and > which get JSON-escaped to \u003c and \u003e
				if !strings.Contains(output, "tool_calls_section_begin") {
					t.Errorf("Expected raw markup in output when extraction is disabled, got: %s", output)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			transformer := NewChatToResponsesTransformer(&buf)
			transformer.SetToolCallTransform(tt.toolCallTransform)

			for _, chunk := range tt.chunks {
				data, _ := json.Marshal(chunk)
				err := transformer.Transform(&sse.Event{Data: string(data)})
				if err != nil {
					t.Errorf("Transform returned error: %v", err)
				}
			}

			// Close to finalize
			transformer.Close()

			output := buf.String()
			if tt.validate != nil {
				tt.validate(t, output)
			}
		})
	}
}

func TestChatToResponses_StoreConversation_CombinesMessageAndToolCall(t *testing.T) {
	// Initialize the conversation store
	conversation.InitDefaultStore(conversation.Config{MaxSize: 1000})

	// Create a transformer
	var buf bytes.Buffer
	transformer := NewChatToResponsesTransformer(&buf)

	// Set input items
	transformer.SetInputItems([]types.InputItem{
		{Type: "message", Role: "user", Content: "Hello"},
	})

	// Simulate streaming: first text content, then tool call
	// Text content first
	transformer.Transform(&sse.Event{Data: `{"id":"test-123","model":"test-model","choices":[{"delta":{"content":"Hello there"}}]}`})

	// Then tool call
	transformer.Transform(&sse.Event{Data: `{"id":"test-123","model":"test-model","choices":[{"delta":{"tool_calls":[{"id":"call_123","type":"function","function":{"name":"test_func","arguments":"{\"arg\": \"value\"}"}}]}}]}`})

	// Finish with usage
	transformer.Transform(&sse.Event{Data: `{"id":"test-123","model":"test-model","choices":[{"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`})

	// Finish the response with [DONE]
	transformer.Transform(&sse.Event{Data: "[DONE]"})

	// Get the stored conversation
	stored := conversation.GetFromDefault("resp_test-123")
	if stored == nil {
		t.Fatal("No conversation stored")
	}

	t.Logf("Stored output items: %d", len(stored.Output))
	for i, item := range stored.Output {
		t.Logf("Item %d: type=%s, role=%s, call_id=%s, name=%s", i, item.Type, item.Role, item.CallID, item.Name)
	}

	// Now test prependHistoryToInput
	newInput := []interface{}{
		map[string]interface{}{
			"type":    "message",
			"role":    "user",
			"content": "Follow-up",
		},
	}

	result := prependHistoryToInput(stored, newInput)
	items := result.([]interface{})

	t.Logf("Result items: %d", len(items))
	for i, item := range items {
		itemJSON, _ := json.MarshalIndent(item, "", "  ")
		t.Logf("Item %d: %s", i, string(itemJSON))
	}

	// Should have: user input, combined assistant message, new user input
	if len(items) != 3 {
		t.Errorf("Expected 3 items, got %d", len(items))
	}

	// Check that the assistant message has tool_calls
	if len(items) >= 2 {
		assistantItem, ok := items[1].(map[string]interface{})
		if !ok {
			t.Fatalf("items[1] is not map[string]interface{}")
		}

		if assistantItem["type"] != "message" {
			t.Errorf("items[1] type = %v, want message", assistantItem["type"])
		}

		if assistantItem["role"] != "assistant" {
			t.Errorf("items[1] role = %v, want assistant", assistantItem["role"])
		}

		if _, hasToolCalls := assistantItem["tool_calls"]; !hasToolCalls {
			t.Errorf("Combined assistant message should have tool_calls")
		}
	}
}

func strPtr(s string) *string {
	return &s
}
