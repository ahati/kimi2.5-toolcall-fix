package toolcall

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"ai-proxy/types"

	"github.com/tmaxmax/go-sse"
)

// TestToolUseBlockWithExtractedArgs reproduces a scenario where the model
// sends a tool_use block (generating output_item.added with correct ID)
// but then the arguments come embedded in thinking content (which would
// need to extract them properly).
//
// This tests the case where Kimi outputs:
// 1. An explicit tool_use block start (with correct ID)
// 2. Arguments inside thinking_delta instead of input_json_delta
func TestToolUseBlockWithExtractedArgs(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesTransformer(&buf)
	// Enable tool call extraction for this test
	transformer.SetToolCallTransform(true)

	events := []types.Event{
		{
			Type: "message_start",
			Message: &types.MessageInfo{
				ID:    "msg_test123",
				Model: "kimi-k2.5",
			},
		},
		{
			Type:         "content_block_start",
			Index:        intPtr(0),
			ContentBlock: mustMarshal(types.ContentBlock{Type: "thinking", Thinking: ""}),
		},
		// Thinking with tool call markup - tool call gets extracted
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "<|tool_calls_section_begin|><|tool_call_begin|>exec_command:0<|tool_call_argument_begin|>"}),
		},
		// First args chunk
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: `{"cmd":"ls"}`}),
		},
		// End tool call
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "<|tool_call_end|><|tool_calls_section_end|>"}),
		},
		{
			Type:  "content_block_stop",
			Index: intPtr(0),
		},
		// Now an explicit tool_use block (this is another tool call)
		{
			Type:         "content_block_start",
			Index:        intPtr(1),
			ContentBlock: json.RawMessage(`{"type":"tool_use","id":"toolu_abc123","name":"read_file"}`),
		},
		{
			Type:  "content_block_delta",
			Index: intPtr(1),
			Delta: mustMarshal(types.InputJSONDelta{Type: "input_json_delta", PartialJSON: `{"path":"/tmp/test.txt"}`}),
		},
		{
			Type:  "content_block_stop",
			Index: intPtr(1),
		},
		{
			Type:       "message_delta",
			StopReason: "end_turn",
			Usage:      &types.AnthropicUsage{OutputTokens: 10},
		},
		{
			Type: "message_stop",
		},
	}

	for _, event := range events {
		data, _ := json.Marshal(event)
		err := transformer.Transform(&sse.Event{Data: string(data)})
		if err != nil {
			t.Fatalf("Transform error: %v", err)
		}
	}

	output := buf.String()
	t.Logf("Output:\n%s", output)

	// Check that all function_call_arguments.delta events have proper IDs
	lines := strings.Split(output, "\n")
	for i := 0; i < len(lines); i++ {
		line := lines[i]
		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			var event map[string]interface{}
			if err := json.Unmarshal([]byte(data), &event); err != nil {
				continue
			}

			eventType, _ := event["type"].(string)
			if eventType == "response.function_call_arguments.delta" {
				itemID, _ := event["item_id"].(string)
				callID, _ := event["call_id"].(string)
				delta, _ := event["delta"].(string)

				t.Logf("delta: item_id=%s, call_id=%s, delta=%s", itemID, callID, delta)

				if itemID == "" {
					t.Errorf("BUG: function_call_arguments.delta has empty 'item_id' field")
				}
				if callID == "" {
					t.Errorf("BUG: function_call_arguments.delta has empty 'call_id' field")
				}
			}
		}
	}
}

// TestToolCallIDConsistency verifies that the ID in output_item.added
// matches the ID in all subsequent delta events for the same tool call.
func TestToolCallIDConsistency(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesTransformer(&buf)
	// Enable tool call extraction for this test
	transformer.SetToolCallTransform(true)

	events := []types.Event{
		{
			Type: "message_start",
			Message: &types.MessageInfo{
				ID:    "msg_consistency",
				Model: "kimi-k2.5",
			},
		},
		{
			Type:         "content_block_start",
			Index:        intPtr(0),
			ContentBlock: mustMarshal(types.ContentBlock{Type: "thinking", Thinking: ""}),
		},
		// Single tool call split across multiple chunks
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "<|tool_calls_section_begin|>"}),
		},
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "<|tool_call_begin|>bash"}),
		},
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "<|tool_call_argument_begin|>"}),
		},
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: `{"com`}),
		},
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: `mand":"echo hello"}`}),
		},
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "<|tool_call_end|>"}),
		},
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "<|tool_calls_section_end|>"}),
		},
		{
			Type:  "content_block_stop",
			Index: intPtr(0),
		},
		{
			Type:       "message_delta",
			StopReason: "end_turn",
			Usage:      &types.AnthropicUsage{OutputTokens: 10},
		},
		{
			Type: "message_stop",
		},
	}

	for _, event := range events {
		data, _ := json.Marshal(event)
		err := transformer.Transform(&sse.Event{Data: string(data)})
		if err != nil {
			t.Fatalf("Transform error: %v", err)
		}
	}

	output := buf.String()
	t.Logf("Output:\n%s", output)

	// Extract the tool call ID from output_item.added
	var expectedID string
	lines := strings.Split(output, "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			var event map[string]interface{}
			if err := json.Unmarshal([]byte(data), &event); err != nil {
				continue
			}

			eventType, _ := event["type"].(string)
			if eventType == "response.output_item.added" {
				item, _ := event["item"].(map[string]interface{})
				if item != nil {
					itemType, _ := item["type"].(string)
					if itemType == "function_call" {
						expectedID, _ = item["id"].(string)
						t.Logf("Found tool call ID: %s", expectedID)
						break
					}
				}
			}
		}
	}

	if expectedID == "" {
		t.Fatal("Could not find tool call ID in output_item.added")
	}

	// Verify all delta events use the same ID
	lines = strings.Split(output, "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			var event map[string]interface{}
			if err := json.Unmarshal([]byte(data), &event); err != nil {
				continue
			}

			eventType, _ := event["type"].(string)
			if eventType == "response.function_call_arguments.delta" {
				itemID, _ := event["item_id"].(string)
				callID, _ := event["call_id"].(string)

				if itemID != expectedID {
					t.Errorf("BUG: function_call_arguments.delta has item_id=%s, expected %s", itemID, expectedID)
				}
				if callID != expectedID {
					t.Errorf("BUG: function_call_arguments.delta has call_id=%s, expected %s", callID, expectedID)
				}
			}
		}
	}
}
