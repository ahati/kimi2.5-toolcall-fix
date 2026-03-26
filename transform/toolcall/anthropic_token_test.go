package toolcall

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/tmaxmax/go-sse"
)

// TestAnthropicTransformer_TokenAccounting_MessageStart tests token normalization in message_start events
func TestAnthropicTransformer_TokenAccounting_MessageStart(t *testing.T) {
	tests := []struct {
		name            string
		inputJSON       string
		wantContains    []string
		wantNotContains []string
	}{
		{
			name:      "Anthropic format - already correct field names",
			inputJSON: `{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3","usage":{"input_tokens":25,"output_tokens":1}}}`,
			wantContains: []string{
				`"input_tokens":25`,
				`"output_tokens":1`,
			},
			wantNotContains: []string{
				`"prompt_tokens"`,
				`"completion_tokens"`,
				`"total_tokens"`,
			},
		},
		{
			name:      "OpenAI format - needs normalization",
			inputJSON: `{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3","usage":{"prompt_tokens":25,"completion_tokens":1,"total_tokens":26}}}`,
			wantContains: []string{
				`"input_tokens":25`,
				`"output_tokens":1`,
			},
			wantNotContains: []string{
				`"prompt_tokens"`,
				`"completion_tokens"`,
				`"total_tokens"`,
			},
		},
		{
			name:      "Mixed format - partial normalization",
			inputJSON: `{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3","usage":{"input_tokens":25,"completion_tokens":1}}}`,
			wantContains: []string{
				`"input_tokens":25`,
				`"output_tokens":1`,
			},
			wantNotContains: []string{
				`"completion_tokens"`,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			tr := NewAnthropicTransformer(&buf)
			tr.SetKimiToolCallTransform(true) // Enable Kimi tool call transformation

			event := &sse.Event{
				Data: tt.inputJSON,
			}

			if err := tr.Transform(event); err != nil {
				t.Fatalf("Transform failed: %v", err)
			}

			output := buf.String()

			for _, want := range tt.wantContains {
				if !strings.Contains(output, want) {
					t.Errorf("Expected output to contain %q, got: %s", want, output)
				}
			}

			for _, wantNot := range tt.wantNotContains {
				if strings.Contains(output, wantNot) {
					t.Errorf("Expected output NOT to contain %q, got: %s", wantNot, output)
				}
			}
		})
	}
}

// TestAnthropicTransformer_TokenAccounting_MessageDelta tests token normalization in message_delta events
func TestAnthropicTransformer_TokenAccounting_MessageDelta(t *testing.T) {
	tests := []struct {
		name            string
		inputJSON       string
		toolsEmitted    bool
		wantContains    []string
		wantNotContains []string
	}{
		{
			name:         "Anthropic format - output_tokens only",
			inputJSON:    `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":15}}`,
			toolsEmitted: false,
			wantContains: []string{
				`"output_tokens":15`,
			},
			wantNotContains: []string{
				`"completion_tokens"`,
				`"total_tokens"`,
			},
		},
		{
			name:         "OpenAI format - needs completion_tokens normalization",
			inputJSON:    `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"completion_tokens":15}}`,
			toolsEmitted: false,
			wantContains: []string{
				`"output_tokens":15`,
			},
			wantNotContains: []string{
				`"completion_tokens"`,
				`"total_tokens"`,
			},
		},
		{
			name:         "OpenAI format - with input and completion tokens",
			inputJSON:    `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"prompt_tokens":100,"completion_tokens":50,"total_tokens":150}}`,
			toolsEmitted: false,
			wantContains: []string{
				`"input_tokens":100`,
				`"output_tokens":50`,
			},
			wantNotContains: []string{
				`"prompt_tokens"`,
				`"completion_tokens"`,
				`"total_tokens"`,
			},
		},
		{
			name:         "Tool use - stop_reason conversion with token normalization",
			inputJSON:    `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"completion_tokens":89}}`,
			toolsEmitted: true,
			wantContains: []string{
				`"output_tokens":89`,
				`"stop_reason":"tool_use"`,
			},
			wantNotContains: []string{
				`"completion_tokens"`,
				`"end_turn"`,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			tr := NewAnthropicTransformer(&buf)
			tr.SetKimiToolCallTransform(true) // Enable Kimi tool call transformation
			tr.toolsEmitted = tt.toolsEmitted

			event := &sse.Event{
				Data: tt.inputJSON,
			}

			if err := tr.Transform(event); err != nil {
				t.Fatalf("Transform failed: %v", err)
			}

			output := buf.String()

			for _, want := range tt.wantContains {
				if !strings.Contains(output, want) {
					t.Errorf("Expected output to contain %q, got: %s", want, output)
				}
			}

			for _, wantNot := range tt.wantNotContains {
				if strings.Contains(output, wantNot) {
					t.Errorf("Expected output NOT to contain %q, got: %s", wantNot, output)
				}
			}
		})
	}
}

// TestAnthropicTransformer_MultiTurn_TokenAccounting tests token accounting across multiple turns
func TestAnthropicTransformer_MultiTurn_TokenAccounting(t *testing.T) {
	tests := []struct {
		name        string
		events      []*sse.Event
		eventChecks []map[string]string // Expected content for each event (empty map = no check)
	}{
		{
			name: "Multi-turn conversation with cumulative tokens",
			events: []*sse.Event{
				// Turn 1 - message_start
				{
					Data: `{"type":"message_start","message":{"id":"msg_turn1","type":"message","role":"assistant","content":[],"model":"claude-3","usage":{"input_tokens":25,"output_tokens":1}}}`,
				},
				// Turn 1 - content blocks
				{
					Data: `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
				},
				{
					Data: `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`,
				},
				{
					Data: `{"type":"content_block_stop","index":0}`,
				},
				// Turn 1 - message_delta with cumulative output tokens
				{
					Data: `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":15}}`,
				},
				{
					Data: `{"type":"message_stop"}`,
				},
				// Turn 2 - message_start with new message ID
				{
					Data: `{"type":"message_start","message":{"id":"msg_turn2","type":"message","role":"assistant","content":[],"model":"claude-3","usage":{"input_tokens":50,"output_tokens":1}}}`,
				},
				// Turn 2 - content blocks
				{
					Data: `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
				},
				{
					Data: `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi there"}}`,
				},
				{
					Data: `{"type":"content_block_stop","index":0}`,
				},
				// Turn 2 - message_delta with cumulative output tokens
				{
					Data: `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":25}}`,
				},
				{
					Data: `{"type":"message_stop"}`,
				},
			},
			eventChecks: []map[string]string{
				// Turn 1 - message_start
				{"input_tokens": "25", "output_tokens": "1"},
				// content_block_start - no check
				{},
				// content_block_delta - no check
				{},
				// content_block_stop - no check
				{},
				// message_delta
				{"output_tokens": "15"},
				// message_stop - no check
				{},
				// Turn 2 - message_start
				{"input_tokens": "50", "output_tokens": "1"},
				// content_block_start - no check
				{},
				// content_block_delta - no check
				{},
				// content_block_stop - no check
				{},
				// message_delta
				{"output_tokens": "25"},
				// message_stop - no check
				{},
			},
		},
		{
			name: "Multi-turn with OpenAI format tokens",
			events: []*sse.Event{
				// Turn 1
				{
					Data: `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-3","usage":{"prompt_tokens":100,"completion_tokens":2}}}`,
				},
				{
					Data: `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
				},
				{
					Data: `{"type":"content_block_stop","index":0}`,
				},
				{
					Data: `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"completion_tokens":20}}`,
				},
				{
					Data: `{"type":"message_stop"}`,
				},
				// Turn 2
				{
					Data: `{"type":"message_start","message":{"id":"msg_2","type":"message","role":"assistant","content":[],"model":"claude-3","usage":{"prompt_tokens":150,"completion_tokens":2}}}`,
				},
				{
					Data: `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
				},
				{
					Data: `{"type":"content_block_stop","index":0}`,
				},
				{
					Data: `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"completion_tokens":35}}`,
				},
				{
					Data: `{"type":"message_stop"}`,
				},
			},
			eventChecks: []map[string]string{
				// Turn 1 - message_start (should be normalized)
				{"input_tokens": "100", "output_tokens": "2"},
				// content_block_start - no check
				{},
				// content_block_stop - no check
				{},
				// message_delta (should be normalized)
				{"output_tokens": "20"},
				// message_stop - no check
				{},
				// Turn 2 - message_start (should be normalized)
				{"input_tokens": "150", "output_tokens": "2"},
				// content_block_start - no check
				{},
				// content_block_stop - no check
				{},
				// message_delta (should be normalized)
				{"output_tokens": "35"},
				// message_stop - no check
				{},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			tr := NewAnthropicTransformer(&buf)
			tr.SetKimiToolCallTransform(true) // Enable Kimi tool call transformation

			if len(tt.events) != len(tt.eventChecks) {
				t.Fatalf("Number of events (%d) must match number of eventChecks (%d)", len(tt.events), len(tt.eventChecks))
			}

			for i, event := range tt.events {
				buf.Reset()
				if err := tr.Transform(event); err != nil {
					t.Fatalf("Transform failed at event %d: %v", i, err)
				}

				output := buf.String()
				checks := tt.eventChecks[i]

				// Only check events that have expectations
				if len(checks) > 0 {
					for key, value := range checks {
						expected := fmt.Sprintf(`"%s":%s`, key, value)
						if !strings.Contains(output, expected) {
							t.Errorf("Event %d (%s): Expected to contain %q, got: %s", i, event.Data[:50], expected, output)
						}
					}
				}
			}
		})
	}
}

// TestAnthropicTransformer_TokenAccounting_EdgeCases tests edge cases in token accounting
func TestAnthropicTransformer_TokenAccounting_EdgeCases(t *testing.T) {
	tests := []struct {
		name      string
		inputJSON string
		wantOK    bool
	}{
		{
			name:      "Empty usage object",
			inputJSON: `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{}}`,
			wantOK:    true,
		},
		{
			name:      "Missing usage field",
			inputJSON: `{"type":"message_delta","delta":{"stop_reason":"end_turn"}}`,
			wantOK:    true,
		},
		{
			name:      "Zero token counts",
			inputJSON: `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":0}}`,
			wantOK:    true,
		},
		{
			name:      "Large token counts",
			inputJSON: `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":100000}}`,
			wantOK:    true,
		},
		{
			name:      "Only input_tokens in message_delta",
			inputJSON: `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":100}}`,
			wantOK:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			tr := NewAnthropicTransformer(&buf)
			tr.SetKimiToolCallTransform(true) // Enable Kimi tool call transformation

			event := &sse.Event{
				Data: tt.inputJSON,
			}

			err := tr.Transform(event)

			if tt.wantOK && err != nil {
				t.Errorf("Expected success, got error: %v", err)
			}
			if !tt.wantOK && err == nil {
				t.Error("Expected error, got success")
			}
		})
	}
}

// TestAnthropicTransformer_TokenAccounting_ToolCalls tests token accounting with tool calls
func TestAnthropicTransformer_TokenAccounting_ToolCalls(t *testing.T) {
	tests := []struct {
		name            string
		events          []*sse.Event
		wantContains    []string
		wantNotContains []string
	}{
		{
			name: "Tool call with token normalization - Kimi markup in thinking block",
			events: []*sse.Event{
				// message_start with OpenAI-style token fields
				{
					Data: `{"type":"message_start","message":{"id":"msg_tool","type":"message","role":"assistant","content":[],"model":"claude-3","usage":{"prompt_tokens":50,"completion_tokens":2,"total_tokens":52}}}`,
				},
				// Thinking block start
				{
					Data: `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
				},
				// Thinking delta with Kimi tool call markup
				{
					Data: `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me think about this<|tool_calls_section_begin|><|tool_call_begin|>bash<|tool_call_argument_begin|>{\"command\":\"ls\"}<|tool_call_end|><|tool_calls_section_end|>"}}`,
				},
				// Thinking block stop
				{
					Data: `{"type":"content_block_stop","index":0}`,
				},
				// message_delta with OpenAI-style token fields
				{
					Data: `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"completion_tokens":50}}`,
				},
				{
					Data: `{"type":"message_stop"}`,
				},
			},
			wantContains: []string{
				`"input_tokens":50`,
				`"output_tokens":2`,
				`"output_tokens":50`,
				`"stop_reason":"tool_use"`,
				`"type":"tool_use"`,
			},
			wantNotContains: []string{
				`"prompt_tokens"`,
				`"completion_tokens"`,
				`"total_tokens"`,
				`"stop_reason":"end_turn"`,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			tr := NewAnthropicTransformer(&buf)
			tr.SetKimiToolCallTransform(true) // Enable Kimi tool call transformation

			var allOutput strings.Builder

			for i, event := range tt.events {
				buf.Reset()
				if err := tr.Transform(event); err != nil {
					t.Fatalf("Transform failed at event %d: %v", i, err)
				}
				allOutput.WriteString(buf.String())
			}

			output := allOutput.String()

			for _, want := range tt.wantContains {
				if !strings.Contains(output, want) {
					t.Errorf("Expected output to contain %q, got: %s", want, output)
				}
			}

			for _, wantNot := range tt.wantNotContains {
				if strings.Contains(output, wantNot) {
					t.Errorf("Expected output NOT to contain %q, got: %s", wantNot, output)
				}
			}
		})
	}
}

// TestAnthropicTransformer_TokenAccounting_ConsistencyAcrossTurns verifies token counts don't reset incorrectly
func TestAnthropicTransformer_TokenAccounting_ConsistencyAcrossTurns(t *testing.T) {
	// This test simulates a multi-turn conversation where token counts should only increase
	// or stay the same within a single message stream (not decrease)

	var buf bytes.Buffer
	tr := NewAnthropicTransformer(&buf)

	// Simulate a single message stream with multiple message_delta events
	events := []*sse.Event{
		{Data: `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-3","usage":{"input_tokens":100,"output_tokens":1}}}`},
		{Data: `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
		{Data: `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Part 1"}}`},
		{Data: `{"type":"content_block_stop","index":0}`},
		// First delta with cumulative count
		{Data: `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":10}}`},
		{Data: `{"type":"message_stop"}`},
	}

	var outputTokens []int

	for _, event := range events {
		buf.Reset()
		if err := tr.Transform(event); err != nil {
			t.Fatalf("Transform failed: %v", err)
		}

		output := buf.String()

		// Extract output_tokens from the output
		if strings.Contains(output, "output_tokens") {
			var eventData map[string]interface{}
			// Extract JSON from SSE format
			lines := strings.Split(output, "\n")
			for _, line := range lines {
				if strings.HasPrefix(line, "data: ") {
					jsonStr := strings.TrimPrefix(line, "data: ")
					if err := json.Unmarshal([]byte(jsonStr), &eventData); err == nil {
						// Check for usage in message_start
						if msg, ok := eventData["message"].(map[string]interface{}); ok {
							if usage, ok := msg["usage"].(map[string]interface{}); ok {
								if tokens, ok := usage["output_tokens"].(float64); ok {
									outputTokens = append(outputTokens, int(tokens))
								}
							}
						}
						// Check for usage in message_delta
						if usage, ok := eventData["usage"].(map[string]interface{}); ok {
							if tokens, ok := usage["output_tokens"].(float64); ok {
								outputTokens = append(outputTokens, int(tokens))
							}
						}
					}
				}
			}
		}
	}

	// Verify token counts are monotonically increasing (cumulative)
	for i := 1; i < len(outputTokens); i++ {
		if outputTokens[i] < outputTokens[i-1] {
			t.Errorf("Token count decreased from %d to %d - tokens should be cumulative",
				outputTokens[i-1], outputTokens[i])
		}
	}

	// Verify we captured token counts
	if len(outputTokens) == 0 {
		t.Error("No token counts were captured from the stream")
	}
}
