package toolcall

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"ai-proxy/types"
)

// TestExtractedToolCallsPreserveIDAndName reproduces the bug where tool calls
// extracted from thinking content lose their ID and Name in the output_item.done event.
func TestExtractedToolCallsPreserveIDAndName(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesTransformer(&buf)
	// Enable tool call extraction for this test
	transformer.SetKimiToolCallTransform(true)

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
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "<|tool_calls_section_begin|>"}),
		},
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "<|tool_call_begin|>"}),
		},
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "exec_command:4"}),
		},
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "<|tool_call_argument_begin|>"}),
		},
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: `{"cmd":"ls -la"}`}),
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
		if err := transformer.handleEvent(event); err != nil {
			t.Fatalf("handleEvent error: %v", err)
		}
	}

	output := buf.String()
	t.Logf("Output:\n%s", output)

	// Parse each SSE event and check for the bug
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
			if eventType == "response.output_item.done" {
				item, _ := event["item"].(map[string]interface{})
				if item == nil {
					continue
				}
				itemType, _ := item["type"].(string)
				if itemType == "function_call" {
					// Check that name, id, call_id, and item_id are not empty
					name, _ := item["name"].(string)
					id, _ := item["id"].(string)
					callID, _ := item["call_id"].(string)
					itemID, _ := event["item_id"].(string)

					if name == "" {
						t.Errorf("BUG: function_call output_item.done has empty 'name' field")
					}
					if id == "" {
						t.Errorf("BUG: function_call output_item.done has empty 'id' field")
					}
					if callID == "" {
						t.Errorf("BUG: function_call output_item.done has empty 'call_id' field")
					}
					if itemID == "" {
						t.Errorf("BUG: function_call output_item.done has empty 'item_id' field")
					}

					// Verify the function name is correct
					if name != "exec_command" {
						t.Errorf("Expected function name 'exec_command', got '%s'", name)
					}

					t.Logf("SUCCESS: function_call output_item.done has correct values: id=%s, name=%s, call_id=%s, item_id=%s", id, name, callID, itemID)
				}
			}
		}
	}
}

// TestExtractedToolCallsPreserveIDAndName_MultipleToolCalls tests multiple tool calls
// extracted from thinking content.
func TestExtractedToolCallsPreserveIDAndName_MultipleToolCalls(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesTransformer(&buf)
	// Enable tool call extraction for this test
	transformer.SetKimiToolCallTransform(true)

	events := []types.Event{
		{
			Type: "message_start",
			Message: &types.MessageInfo{
				ID:    "msg_test456",
				Model: "kimi-k2.5",
			},
		},
		{
			Type:         "content_block_start",
			Index:        intPtr(0),
			ContentBlock: mustMarshal(types.ContentBlock{Type: "thinking", Thinking: ""}),
		},
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "<|tool_calls_section_begin|>"}),
		},
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "<|tool_call_begin|>"}),
		},
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "exec_command:0"}),
		},
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "<|tool_call_argument_begin|>"}),
		},
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: `{"cmd":"ls"}`}),
		},
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "<|tool_call_end|>"}),
		},
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "<|tool_call_begin|>"}),
		},
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "read_file:1"}),
		},
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "<|tool_call_argument_begin|>"}),
		},
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: `{"path":"/tmp/test.txt"}`}),
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
		if err := transformer.handleEvent(event); err != nil {
			t.Fatalf("handleEvent error: %v", err)
		}
	}

	output := buf.String()
	t.Logf("Output:\n%s", output)

	// Count tool calls and verify they all have correct names
	toolCallCount := 0
	toolNames := make(map[string]bool)

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
			if eventType == "response.output_item.done" {
				item, _ := event["item"].(map[string]interface{})
				if item == nil {
					continue
				}
				itemType, _ := item["type"].(string)
				if itemType == "function_call" {
					toolCallCount++
					name, _ := item["name"].(string)
					id, _ := item["id"].(string)

					if name == "" {
						t.Errorf("BUG: function_call %d has empty 'name' field", toolCallCount)
					} else {
						toolNames[name] = true
						t.Logf("Tool call %d: name=%s, id=%s", toolCallCount, name, id)
					}
				}
			}
		}
	}

	if toolCallCount != 2 {
		t.Errorf("Expected 2 tool calls, got %d", toolCallCount)
	}

	// Verify we have both exec_command and read_file
	if !toolNames["exec_command"] {
		t.Errorf("Missing tool call 'exec_command'")
	}
	if !toolNames["read_file"] {
		t.Errorf("Missing tool call 'read_file'")
	}
}
