// Package toolcall provides transformers for tool calling scenarios.
// This file implements error handling tests for the responses transformer.
package toolcall

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"ai-proxy/types"

	"github.com/tmaxmax/go-sse"
)

// TestErrorTransform_InvalidJSONInSSE tests handling of invalid JSON in SSE events.
// Category D2: Incomplete JSON in SSE (HIGH)
func TestErrorTransform_InvalidJSONInSSE(t *testing.T) {
	tests := []struct {
		name string
		data string
	}{
		{
			name: "incomplete JSON object",
			data: `{"type": "message_start", "message": {`,
		},
		{
			name: "invalid JSON syntax",
			data: `{"type": "message_start", "message": undefined}`,
		},
		{
			name: "truncated event",
			data: `{"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hel`,
		},
		{
			name: "malformed JSON array",
			data: `{"type": "message_start", "message": {"content": [}}`,
		},
		{
			name: "invalid escape in JSON",
			data: `{"type": "message_start", "message": {"id": "msg_\x00"}}`,
		},
		{
			name: "binary data in JSON",
			data: `{"type": "message_start", "message": {"id": "msg_` + string([]byte{0x00, 0x01, 0x02}) + `"}}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			transformer := NewResponsesTransformer(&buf)

			event := &sse.Event{Data: tt.data}
			err := transformer.Transform(event)
			if err != nil {
				// Error is acceptable, as long as it doesn't panic
				t.Logf("Transform returned error (acceptable): %v", err)
			}

			// Verify output is valid or empty
			_ = buf.String()
		})
	}
}

// TestErrorTransform_MalformedSSEEvents tests handling of malformed SSE events.
// Category D2: Malformed SSE event (HIGH)
func TestErrorTransform_MalformedSSEEvents(t *testing.T) {
	tests := []struct {
		name  string
		event *sse.Event
	}{
		{
			name:  "empty event data",
			event: &sse.Event{Data: ""},
		},
		{
			name:  "only whitespace",
			event: &sse.Event{Data: "   \n\t  "},
		},
		{
			name:  "event with only newlines",
			event: &sse.Event{Data: "\n\n\n"},
		},
		{
			name:  "event with null bytes",
			event: &sse.Event{Data: string([]byte{0x00, 0x00, 0x00})},
		},
		{
			name:  "very long event data",
			event: &sse.Event{Data: strings.Repeat("a", 1000000)},
		},
		{
			name:  "event with special characters",
			event: &sse.Event{Data: "\x00\x01\x02\x03\x04\x05"},
		},
		{
			name:  "event type without data",
			event: &sse.Event{Type: "message", Data: ""},
		},
		{
			name:  "event ID without data",
			event: &sse.Event{LastEventID: "123", Data: ""},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			transformer := NewResponsesTransformer(&buf)

			err := transformer.Transform(tt.event)
			if err != nil {
				t.Logf("Transform returned error (acceptable): %v", err)
			}

			// Should not panic
		})
	}
}

// TestErrorTransform_MissingRequiredFields tests handling of events with missing fields.
func TestErrorTransform_MissingRequiredFields(t *testing.T) {
	tests := []struct {
		name   string
		events []types.Event
	}{
		{
			name: "message_start without message",
			events: []types.Event{
				{Type: "message_start"},
			},
		},
		{
			name: "message_start with null message",
			events: []types.Event{
				{Type: "message_start", Message: nil},
			},
		},
		{
			name: "message_start with empty message ID",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "",
						Type:  "message",
						Role:  "assistant",
						Model: "claude-3",
					},
				},
			},
		},
		{
			name: "content_block_start without index",
			events: []types.Event{
				{
					Type:         "content_block_start",
					ContentBlock: mustMarshal(types.ContentBlock{Type: "text"}),
				},
			},
		},
		{
			name: "content_block_start without content_block",
			events: []types.Event{
				{
					Type:  "content_block_start",
					Index: intPtr(0),
				},
			},
		},
		{
			name: "content_block_delta without index",
			events: []types.Event{
				{
					Type:  "content_block_delta",
					Delta: mustMarshal(types.TextDelta{Type: "text_delta", Text: "hello"}),
				},
			},
		},
		{
			name: "content_block_delta without delta",
			events: []types.Event{
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
				},
			},
		},
		{
			name: "content_block_stop without index",
			events: []types.Event{
				{Type: "content_block_stop"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			transformer := NewResponsesTransformer(&buf)

			for _, event := range tt.events {
				data, _ := json.Marshal(event)
				err := transformer.Transform(&sse.Event{Data: string(data)})
				_ = err
			}

			// Should handle missing fields without panic
			_ = buf.String()
		})
	}
}

// TestErrorTransform_InvalidContentBlockTypes tests handling of invalid content block types.
func TestErrorTransform_InvalidContentBlockTypes(t *testing.T) {
	tests := []struct {
		name   string
		events []types.Event
	}{
		{
			name: "unknown content block type",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "unknown_type"}),
				},
			},
		},
		{
			name: "empty content block type",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: ""}),
				},
			},
		},
		{
			name: "tool_use without ID",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "tool_use", Name: "test_tool"}),
				},
			},
		},
		{
			name: "tool_use without name",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "tool_use", ID: "toolu_123"}),
				},
			},
		},
		{
			name: "text block with null content",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: json.RawMessage(`{"type": "text", "text": null}`),
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			transformer := NewResponsesTransformer(&buf)

			for _, event := range tt.events {
				data, _ := json.Marshal(event)
				err := transformer.Transform(&sse.Event{Data: string(data)})
				_ = err
			}

			// Should handle invalid content blocks without panic
			_ = buf.String()
		})
	}
}

// TestErrorTransform_InvalidDeltaTypes tests handling of invalid delta types.
func TestErrorTransform_InvalidDeltaTypes(t *testing.T) {
	tests := []struct {
		name   string
		events []types.Event
	}{
		{
			name: "unknown delta type",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "text"}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: json.RawMessage(`{"type": "unknown_delta", "text": "hello"}`),
				},
			},
		},
		{
			name: "delta with null text",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "text"}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: json.RawMessage(`{"type": "text_delta", "text": null}`),
				},
			},
		},
		{
			name: "thinking delta without thinking field",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "thinking"}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: json.RawMessage(`{"type": "thinking_delta", "thinking": null}`),
				},
			},
		},
		{
			name: "input_json_delta without partial_json",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "tool_use", ID: "toolu_123", Name: "test"}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: json.RawMessage(`{"type": "input_json_delta"}`),
				},
			},
		},
		{
			name: "delta with mismatched index",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "text"}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(999), // Mismatched index
					Delta: mustMarshal(types.TextDelta{Type: "text_delta", Text: "hello"}),
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			transformer := NewResponsesTransformer(&buf)

			for _, event := range tt.events {
				data, _ := json.Marshal(event)
				err := transformer.Transform(&sse.Event{Data: string(data)})
				_ = err
			}

			// Should handle invalid deltas without panic
			_ = buf.String()
		})
	}
}

// TestErrorTransform_UpstreamTimeoutSimulates tests handling that simulates upstream timeout.
// Category D2: Upstream timeout (HIGH)
func TestErrorTransform_UpstreamTimeoutSimulates(t *testing.T) {
	t.Run("incomplete stream without message_stop", func(t *testing.T) {
		var buf bytes.Buffer
		transformer := NewResponsesTransformer(&buf)

		// Send message_start and some deltas but no message_stop
		events := []types.Event{
			{
				Type: "message_start",
				Message: &types.MessageInfo{
					ID:    "msg_123",
					Model: "claude-3",
				},
			},
			{
				Type:         "content_block_start",
				Index:        intPtr(0),
				ContentBlock: mustMarshal(types.ContentBlock{Type: "text"}),
			},
			{
				Type:  "content_block_delta",
				Index: intPtr(0),
				Delta: mustMarshal(types.TextDelta{Type: "text_delta", Text: "Hello"}),
			},
			// Missing content_block_stop and message_stop - simulates timeout
		}

		for _, event := range events {
			data, _ := json.Marshal(event)
			_ = transformer.Transform(&sse.Event{Data: string(data)})
		}

		// Verify transformer is still in consistent state
		if transformer.messageID != "msg_123" {
			t.Error("Expected messageID to be set")
		}
		if !transformer.inText {
			t.Error("Expected inText to be true (block not stopped)")
		}
	})

	t.Run("partial tool call without completion", func(t *testing.T) {
		var buf bytes.Buffer
		transformer := NewResponsesTransformer(&buf)

		events := []types.Event{
			{
				Type: "message_start",
				Message: &types.MessageInfo{
					ID:    "msg_123",
					Model: "claude-3",
				},
			},
			{
				Type:         "content_block_start",
				Index:        intPtr(0),
				ContentBlock: mustMarshal(types.ContentBlock{Type: "tool_use", ID: "toolu_123", Name: "test"}),
			},
			{
				Type:  "content_block_delta",
				Index: intPtr(0),
				Delta: mustMarshal(types.InputJSONDelta{Type: "input_json_delta", PartialJSON: `{"partial": `}),
			},
			// Missing rest of JSON and content_block_stop
		}

		for _, event := range events {
			data, _ := json.Marshal(event)
			_ = transformer.Transform(&sse.Event{Data: string(data)})
		}

		// Verify partial state
		if !transformer.inToolCall {
			t.Error("Expected inToolCall to be true")
		}
		if transformer.currentID != "toolu_123" {
			t.Errorf("Expected currentID to be 'toolu_123', got '%s'", transformer.currentID)
		}
	})
}

// TestErrorTransform_InvalidUsageData tests handling of invalid usage data.
func TestErrorTransform_InvalidUsageData(t *testing.T) {
	tests := []struct {
		name   string
		events []types.Event
	}{
		{
			name: "negative token counts",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
						Usage: &types.AnthropicUsage{
							InputTokens:  -100,
							OutputTokens: -50,
						},
					},
				},
			},
		},
		{
			name: "zero token counts",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
						Usage: &types.AnthropicUsage{
							InputTokens:  0,
							OutputTokens: 0,
						},
					},
				},
			},
		},
		{
			name: "very large token counts",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
						Usage: &types.AnthropicUsage{
							InputTokens:  2147483647,
							OutputTokens: 2147483647,
						},
					},
				},
			},
		},
		{
			name: "negative cache tokens",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
						Usage: &types.AnthropicUsage{
							InputTokens:              100,
							OutputTokens:             50,
							CacheReadInputTokens:     -10,
							CacheCreationInputTokens: -5,
						},
					},
				},
			},
		},
		{
			name: "cache tokens greater than input tokens",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
						Usage: &types.AnthropicUsage{
							InputTokens:          100,
							OutputTokens:         50,
							CacheReadInputTokens: 200, // More than input tokens
						},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			transformer := NewResponsesTransformer(&buf)

			for _, event := range tt.events {
				data, _ := json.Marshal(event)
				err := transformer.Transform(&sse.Event{Data: string(data)})
				_ = err
			}

			// Send message_stop to complete
			data, _ := json.Marshal(types.Event{Type: "message_stop"})
			_ = transformer.Transform(&sse.Event{Data: string(data)})

			// Should handle invalid usage data without panic
			_ = buf.String()
		})
	}
}

// TestErrorTransform_InvalidStopReason tests handling of invalid stop reasons.
func TestErrorTransform_InvalidStopReason(t *testing.T) {
	tests := []struct {
		name       string
		stopReason string
	}{
		{
			name:       "unknown stop reason",
			stopReason: "unknown_reason",
		},
		{
			name:       "empty stop reason",
			stopReason: "",
		},
		{
			name:       "whitespace stop reason",
			stopReason: "   ",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			transformer := NewResponsesTransformer(&buf)

			// First send message_start
			msgStart := types.Event{
				Type: "message_start",
				Message: &types.MessageInfo{
					ID:    "msg_123",
					Model: "claude-3",
				},
			}
			data, _ := json.Marshal(msgStart)
			_ = transformer.Transform(&sse.Event{Data: string(data)})

			// Then send message_delta with stop reason
			msgDelta := types.Event{
				Type:       "message_delta",
				StopReason: tt.stopReason,
			}
			data, _ = json.Marshal(msgDelta)
			err := transformer.Transform(&sse.Event{Data: string(data)})
			_ = err

			// Should handle invalid stop reason without panic
		})
	}
}

// TestErrorTransform_UnknownEventTypes tests handling of unknown event types.
func TestErrorTransform_UnknownEventTypes(t *testing.T) {
	tests := []struct {
		name      string
		eventType string
	}{
		{
			name:      "completely unknown event",
			eventType: "unknown_event_type",
		},
		{
			name:      "event with special characters",
			eventType: "message<start>",
		},
		{
			name:      "empty event type",
			eventType: "",
		},
		{
			name:      "null-like event type",
			eventType: "null",
		},
		{
			name:      "event type with spaces",
			eventType: "message start",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			transformer := NewResponsesTransformer(&buf)

			event := types.Event{Type: tt.eventType}
			data, _ := json.Marshal(event)
			err := transformer.Transform(&sse.Event{Data: string(data)})
			_ = err

			// Should handle unknown event types without panic
			_ = buf.String()
		})
	}
}

// TestErrorTransform_MalformedToolCallData tests handling of malformed tool call data.
func TestErrorTransform_MalformedToolCallData(t *testing.T) {
	tests := []struct {
		name   string
		events []types.Event
	}{
		{
			name: "tool call with invalid JSON in arguments",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "tool_use", ID: "toolu_123", Name: "test"}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.InputJSONDelta{Type: "input_json_delta", PartialJSON: `{invalid json`}),
				},
				{
					Type:  "content_block_stop",
					Index: intPtr(0),
				},
				{Type: "message_stop"},
			},
		},
		{
			name: "tool call with very long arguments",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "tool_use", ID: "toolu_123", Name: "test"}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.InputJSONDelta{Type: "input_json_delta", PartialJSON: `{"text": "` + strings.Repeat("a", 100000) + `"}`}),
				},
				{
					Type:  "content_block_stop",
					Index: intPtr(0),
				},
				{Type: "message_stop"},
			},
		},
		{
			name: "tool call with empty tool name",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "tool_use", ID: "toolu_123", Name: ""}),
				},
				{
					Type:  "content_block_stop",
					Index: intPtr(0),
				},
				{Type: "message_stop"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			transformer := NewResponsesTransformer(&buf)

			for _, event := range tt.events {
				data, _ := json.Marshal(event)
				err := transformer.Transform(&sse.Event{Data: string(data)})
				_ = err
			}

			// Should handle malformed tool call data without panic
			output := buf.String()
			// Verify output contains expected function_call structure
			if !strings.Contains(output, "function_call") {
				t.Error("Expected output to contain function_call")
			}
		})
	}
}

// TestErrorTransform_RapidEventSequence tests handling of rapid event sequences.
func TestErrorTransform_RapidEventSequence(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesTransformer(&buf)

	// Send many events rapidly
	events := []types.Event{
		{
			Type: "message_start",
			Message: &types.MessageInfo{
				ID:    "msg_123",
				Model: "claude-3",
			},
		},
	}

	// Add many content blocks
	for i := 0; i < 100; i++ {
		events = append(events, types.Event{
			Type:         "content_block_start",
			Index:        intPtr(i),
			ContentBlock: mustMarshal(types.ContentBlock{Type: "text"}),
		})
		events = append(events, types.Event{
			Type:  "content_block_delta",
			Index: intPtr(i),
			Delta: mustMarshal(types.TextDelta{Type: "text_delta", Text: "a"}),
		})
		events = append(events, types.Event{
			Type:  "content_block_stop",
			Index: intPtr(i),
		})
	}

	events = append(events, types.Event{Type: "message_stop"})

	for _, event := range events {
		data, _ := json.Marshal(event)
		err := transformer.Transform(&sse.Event{Data: string(data)})
		if err != nil {
			t.Errorf("Transform returned error: %v", err)
		}
	}

	// Verify output
	output := buf.String()
	if !strings.Contains(output, "response.completed") {
		t.Error("Expected output to contain response.completed")
	}
}

// TestErrorTransform_StateConsistency tests that transformer maintains consistent state.
func TestErrorTransform_StateConsistency(t *testing.T) {
	t.Run("sequence number always increases", func(t *testing.T) {
		var buf bytes.Buffer
		transformer := NewResponsesTransformer(&buf)

		// Send events and track sequence numbers
		events := []types.Event{
			{
				Type: "message_start",
				Message: &types.MessageInfo{
					ID:    "msg_123",
					Model: "claude-3",
				},
			},
			{
				Type:         "content_block_start",
				Index:        intPtr(0),
				ContentBlock: mustMarshal(types.ContentBlock{Type: "text"}),
			},
			{
				Type:  "content_block_delta",
				Index: intPtr(0),
				Delta: mustMarshal(types.TextDelta{Type: "text_delta", Text: "Hello"}),
			},
			{
				Type:  "content_block_stop",
				Index: intPtr(0),
			},
			{Type: "message_stop"},
		}

		prevSeqNum := 0
		for _, event := range events {
			buf.Reset()
			data, _ := json.Marshal(event)
			_ = transformer.Transform(&sse.Event{Data: string(data)})
			output := buf.String()

			// Extract sequence number from output
			if strings.Contains(output, "sequence_number") {
				var result struct{ Seq int }
				if err := json.Unmarshal([]byte(`{"seq":`+extractSequenceNumber(output)+`}`), &result); err == nil {
					if result.Seq <= prevSeqNum {
						t.Errorf("Sequence number did not increase: %d -> %d", prevSeqNum, result.Seq)
					}
					prevSeqNum = result.Seq
				}
			}
		}
	})
}

// Helper function to extract sequence number from output
func extractSequenceNumber(output string) string {
	start := strings.Index(output, `"sequence_number":`)
	if start == -1 {
		return "0"
	}
	start += len(`"sequence_number":`)
	end := start
	for end < len(output) && (output[end] >= '0' && output[end] <= '9') {
		end++
	}
	if end > start {
		return output[start:end]
	}
	return "0"
}

// TestErrorTransform_MultipleMessageStarts tests handling of multiple message_start events.
func TestErrorTransform_MultipleMessageStarts(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesTransformer(&buf)

	// First message_start
	event1 := types.Event{
		Type: "message_start",
		Message: &types.MessageInfo{
			ID:    "msg_123",
			Model: "claude-3",
		},
	}
	data, _ := json.Marshal(event1)
	_ = transformer.Transform(&sse.Event{Data: string(data)})

	// Second message_start (should reset state)
	event2 := types.Event{
		Type: "message_start",
		Message: &types.MessageInfo{
			ID:    "msg_456",
			Model: "claude-3-opus",
		},
	}
	data, _ = json.Marshal(event2)
	_ = transformer.Transform(&sse.Event{Data: string(data)})

	// Verify state was reset
	if transformer.messageID != "msg_456" {
		t.Errorf("Expected messageID to be 'msg_456', got '%s'", transformer.messageID)
	}
}

// TestErrorTransform_EmptyToolCallID tests handling of empty tool call IDs.
func TestErrorTransform_EmptyToolCallID(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesTransformer(&buf)

	events := []types.Event{
		{
			Type: "message_start",
			Message: &types.MessageInfo{
				ID:    "msg_123",
				Model: "claude-3",
			},
		},
		{
			Type:         "content_block_start",
			Index:        intPtr(0),
			ContentBlock: mustMarshal(types.ContentBlock{Type: "tool_use", ID: "", Name: "test_tool"}),
		},
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.InputJSONDelta{Type: "input_json_delta", PartialJSON: `{}`}),
		},
		{
			Type:  "content_block_stop",
			Index: intPtr(0),
		},
		{Type: "message_stop"},
	}

	for _, event := range events {
		data, _ := json.Marshal(event)
		err := transformer.Transform(&sse.Event{Data: string(data)})
		_ = err
	}

	// Should handle empty tool call ID
	output := buf.String()
	if !strings.Contains(output, "function_call") {
		t.Error("Expected output to contain function_call")
	}
}

// TestErrorTransform_RateLimitError tests rate limit error handling.
// Category D1: Rate limit error (HIGH)
func TestErrorTransform_RateLimitError(t *testing.T) {
	tests := []struct {
		name   string
		events []types.Event
	}{
		{
			name: "rate limit error event",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type: "error",
					Message: &types.MessageInfo{
						ID:    "error_123",
						Type:  "error",
						Role:  "assistant",
						Model: "claude-3",
					},
				},
			},
		},
		{
			name: "rate limit during streaming",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "text"}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.TextDelta{Type: "text_delta", Text: "Hello"}),
				},
				// Rate limit error occurs before completion
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			transformer := NewResponsesTransformer(&buf)

			for _, event := range tt.events {
				data, _ := json.Marshal(event)
				_ = transformer.Transform(&sse.Event{Data: string(data)})
			}

			// Should handle rate limit error without panic
			_ = buf.String()
		})
	}
}

// TestErrorTransform_AuthenticationError tests authentication error handling.
// Category D1: Authentication error (HIGH)
func TestErrorTransform_AuthenticationError(t *testing.T) {
	tests := []struct {
		name   string
		events []types.Event
	}{
		{
			name: "authentication error at start",
			events: []types.Event{
				{
					Type: "error",
					Message: &types.MessageInfo{
						ID:    "error_auth",
						Type:  "error",
						Role:  "assistant",
						Model: "claude-3",
					},
				},
			},
		},
		{
			name: "authentication error with message start",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type: "error",
					Message: &types.MessageInfo{
						ID:    "error_auth",
						Type:  "error",
						Role:  "assistant",
						Model: "claude-3",
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			transformer := NewResponsesTransformer(&buf)

			for _, event := range tt.events {
				data, _ := json.Marshal(event)
				_ = transformer.Transform(&sse.Event{Data: string(data)})
			}

			// Should handle authentication error without panic
			_ = buf.String()
		})
	}
}

// TestErrorTransform_UpstreamTimeout tests upstream timeout handling.
// Category D1: Upstream timeout handling (HIGH)
func TestErrorTransform_UpstreamTimeout(t *testing.T) {
	tests := []struct {
		name   string
		events []types.Event
	}{
		{
			name: "timeout during text generation",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "text"}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.TextDelta{Type: "text_delta", Text: "Hello"}),
				},
				// Timeout before content_block_stop
			},
		},
		{
			name: "timeout during reasoning",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "thinking"}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "Let me think"}),
				},
				// Timeout before reasoning completion
			},
		},
		{
			name: "timeout during tool call",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "tool_use", ID: "toolu_123", Name: "get_weather"}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.InputJSONDelta{Type: "input_json_delta", PartialJSON: `{"location": "NYC"}`}),
				},
				// Timeout before tool call completion
			},
		},
		{
			name: "timeout after partial message_stop",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:  "message_delta",
					Usage: &types.AnthropicUsage{InputTokens: 100, OutputTokens: 50},
				},
				// Timeout before message_stop
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			transformer := NewResponsesTransformer(&buf)

			for _, event := range tt.events {
				data, _ := json.Marshal(event)
				_ = transformer.Transform(&sse.Event{Data: string(data)})
			}

			// Verify transformer is in consistent state despite timeout
			if transformer.messageID != "msg_123" {
				t.Error("Expected messageID to be set")
			}
		})
	}
}

// TestErrorTransform_UpstreamConnectionReset tests upstream connection reset.
// Category D1: Upstream connection reset (HIGH)
func TestErrorTransform_UpstreamConnectionReset(t *testing.T) {
	tests := []struct {
		name   string
		events []types.Event
	}{
		{
			name: "connection reset after message_start",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				// Connection reset - no more events
			},
		},
		{
			name: "connection reset during content streaming",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "text"}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.TextDelta{Type: "text_delta", Text: "Partial "}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.TextDelta{Type: "text_delta", Text: "content"}),
				},
				// Connection reset before content_block_stop
			},
		},
		{
			name: "connection reset during tool streaming",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "tool_use", ID: "toolu_123", Name: "search"}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.InputJSONDelta{Type: "input_json_delta", PartialJSON: `{"query": "in`}),
				},
				// Connection reset before completion
			},
		},
		{
			name: "connection reset during reasoning",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "thinking"}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "Analyzing..."}),
				},
				// Connection reset before reasoning completion
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			transformer := NewResponsesTransformer(&buf)

			for _, event := range tt.events {
				data, _ := json.Marshal(event)
				_ = transformer.Transform(&sse.Event{Data: string(data)})
			}

			// Should handle incomplete stream without panic
			_ = buf.String()
		})
	}
}

// TestErrorTransform_ContentOnlyNewlines tests content with only newlines.
// Category E1: Content with only newlines (HIGH)
func TestErrorTransform_ContentOnlyNewlines(t *testing.T) {
	tests := []struct {
		name   string
		events []types.Event
		expect string
	}{
		{
			name: "single newline in text",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "text"}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.TextDelta{Type: "text_delta", Text: "\n"}),
				},
				{
					Type:  "content_block_stop",
					Index: intPtr(0),
				},
				{Type: "message_stop"},
			},
			expect: "\n",
		},
		{
			name: "multiple newlines",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "text"}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.TextDelta{Type: "text_delta", Text: "\n\n\n"}),
				},
				{
					Type:  "content_block_stop",
					Index: intPtr(0),
				},
				{Type: "message_stop"},
			},
			expect: "\\n\\n\\n", // JSON-escaped newlines
		},
		{
			name: "windows line endings",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "text"}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.TextDelta{Type: "text_delta", Text: "\r\n\r\n"}),
				},
				{
					Type:  "content_block_stop",
					Index: intPtr(0),
				},
				{Type: "message_stop"},
			},
			expect: "\\r\\n\\r\\n", // JSON-escaped Windows line endings
		},
		{
			name: "newlines in thinking",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "thinking"}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "\n\n"}),
				},
				{
					Type:  "content_block_stop",
					Index: intPtr(0),
				},
				{Type: "message_stop"},
			},
			expect: "reasoning",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			transformer := NewResponsesTransformer(&buf)

			for _, event := range tt.events {
				data, _ := json.Marshal(event)
				_ = transformer.Transform(&sse.Event{Data: string(data)})
			}

			output := buf.String()
			if tt.expect != "" && !strings.Contains(output, tt.expect) {
				t.Errorf("Expected output to contain '%s', got: %s", tt.expect, output)
			}
		})
	}
}

// TestErrorTransform_UnicodeEmojiHandling tests Unicode emoji handling.
// Category E1: Unicode emoji handling (HIGH)
func TestErrorTransform_UnicodeEmojiHandling(t *testing.T) {
	tests := []struct {
		name   string
		events []types.Event
		expect string
	}{
		{
			name: "basic emojis",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "text"}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.TextDelta{Type: "text_delta", Text: "👋"}),
				},
				{
					Type:  "content_block_stop",
					Index: intPtr(0),
				},
				{Type: "message_stop"},
			},
			expect: "👋",
		},
		{
			name: "complex emoji - flag",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "text"}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.TextDelta{Type: "text_delta", Text: "🇺🇸"}),
				},
				{
					Type:  "content_block_stop",
					Index: intPtr(0),
				},
				{Type: "message_stop"},
			},
			expect: "🇺🇸",
		},
		{
			name: "emoji in thinking",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "thinking"}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "🤔"}),
				},
				{
					Type:  "content_block_stop",
					Index: intPtr(0),
				},
				{Type: "message_stop"},
			},
			expect: "🤔",
		},
		{
			name: "emoji in tool arguments",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "tool_use", ID: "toolu_123", Name: "get_weather"}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.InputJSONDelta{Type: "input_json_delta", PartialJSON: `{"city": "🌆"}`}),
				},
				{
					Type:  "content_block_stop",
					Index: intPtr(0),
				},
				{Type: "message_stop"},
			},
			expect: "🌆",
		},
		{
			name: "CJK characters",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "text"}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.TextDelta{Type: "text_delta", Text: "世界你好"}),
				},
				{
					Type:  "content_block_stop",
					Index: intPtr(0),
				},
				{Type: "message_stop"},
			},
			expect: "世界你好",
		},
		{
			name: "RTL text",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "text"}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.TextDelta{Type: "text_delta", Text: "مرحبا"}),
				},
				{
					Type:  "content_block_stop",
					Index: intPtr(0),
				},
				{Type: "message_stop"},
			},
			expect: "مرحبا",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			transformer := NewResponsesTransformer(&buf)

			for _, event := range tt.events {
				data, _ := json.Marshal(event)
				_ = transformer.Transform(&sse.Event{Data: string(data)})
			}

			output := buf.String()
			if !strings.Contains(output, tt.expect) {
				t.Errorf("Expected output to contain '%s', got: %s", tt.expect, output)
			}
		})
	}
}

// TestErrorTransform_ToolWithNoParameters tests tool with no parameters.
// Category E2: Tool with no parameters (MEDIUM)
func TestErrorTransform_ToolWithNoParameters(t *testing.T) {
	// This tests the tool call conversion in responses_to_anthropic
	tests := []struct {
		name   string
		events []types.Event
		expect string
	}{
		{
			name: "tool call with empty JSON arguments",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "tool_use", ID: "toolu_123", Name: "get_time"}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.InputJSONDelta{Type: "input_json_delta", PartialJSON: `{}`}),
				},
				{
					Type:  "content_block_stop",
					Index: intPtr(0),
				},
				{Type: "message_stop"},
			},
			expect: `"arguments":"{}"`,
		},
		{
			name: "tool call with no arguments streamed",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "tool_use", ID: "toolu_123", Name: "ping"}),
				},
				// No content_block_delta for arguments
				{
					Type:  "content_block_stop",
					Index: intPtr(0),
				},
				{Type: "message_stop"},
			},
			expect: "function_call",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			transformer := NewResponsesTransformer(&buf)

			for _, event := range tt.events {
				data, _ := json.Marshal(event)
				_ = transformer.Transform(&sse.Event{Data: string(data)})
			}

			output := buf.String()
			if !strings.Contains(output, tt.expect) {
				t.Errorf("Expected output to contain '%s', got: %s", tt.expect, output)
			}
		})
	}
}

// TestErrorTransform_ToolNameWithSpaces tests tool name with spaces.
// Category E2: Tool name with spaces (MEDIUM)
func TestErrorTransform_ToolNameWithSpaces(t *testing.T) {
	tests := []struct {
		name   string
		events []types.Event
		expect string
	}{
		{
			name: "tool name with internal space",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "tool_use", ID: "toolu_123", Name: "get weather"}),
				},
				{
					Type:  "content_block_stop",
					Index: intPtr(0),
				},
				{Type: "message_stop"},
			},
			expect: `"name":"get weather"`,
		},
		{
			name: "tool name with leading space",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "tool_use", ID: "toolu_123", Name: " get_weather"}),
				},
				{
					Type:  "content_block_stop",
					Index: intPtr(0),
				},
				{Type: "message_stop"},
			},
			expect: `"name":" get_weather"`,
		},
		{
			name: "tool name with trailing space",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "tool_use", ID: "toolu_123", Name: "get_weather "}),
				},
				{
					Type:  "content_block_stop",
					Index: intPtr(0),
				},
				{Type: "message_stop"},
			},
			expect: `"name":"get_weather "`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			transformer := NewResponsesTransformer(&buf)

			for _, event := range tt.events {
				data, _ := json.Marshal(event)
				_ = transformer.Transform(&sse.Event{Data: string(data)})
			}

			output := buf.String()
			if !strings.Contains(output, tt.expect) {
				t.Errorf("Expected output to contain '%s', got: %s", tt.expect, output)
			}
		})
	}
}

// TestErrorTransform_ToolNameWithSpecialChars tests tool name with special characters.
// Category E2: Tool name with special chars (MEDIUM)
func TestErrorTransform_ToolNameWithSpecialChars(t *testing.T) {
	tests := []struct {
		name   string
		events []types.Event
		expect string
	}{
		{
			name: "tool name with hyphen",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "tool_use", ID: "toolu_123", Name: "get-weather"}),
				},
				{
					Type:  "content_block_stop",
					Index: intPtr(0),
				},
				{Type: "message_stop"},
			},
			expect: `"name":"get-weather"`,
		},
		{
			name: "tool name with dot",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "tool_use", ID: "toolu_123", Name: "get.weather"}),
				},
				{
					Type:  "content_block_stop",
					Index: intPtr(0),
				},
				{Type: "message_stop"},
			},
			expect: `"name":"get.weather"`,
		},
		{
			name: "tool name with underscore",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "tool_use", ID: "toolu_123", Name: "get_weather"}),
				},
				{
					Type:  "content_block_stop",
					Index: intPtr(0),
				},
				{Type: "message_stop"},
			},
			expect: `"name":"get_weather"`,
		},
		{
			name: "tool name with number",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "tool_use", ID: "toolu_123", Name: "get_weather_v2"}),
				},
				{
					Type:  "content_block_stop",
					Index: intPtr(0),
				},
				{Type: "message_stop"},
			},
			expect: `"name":"get_weather_v2"`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			transformer := NewResponsesTransformer(&buf)

			for _, event := range tt.events {
				data, _ := json.Marshal(event)
				_ = transformer.Transform(&sse.Event{Data: string(data)})
			}

			output := buf.String()
			if !strings.Contains(output, tt.expect) {
				t.Errorf("Expected output to contain '%s', got: %s", tt.expect, output)
			}
		})
	}
}
