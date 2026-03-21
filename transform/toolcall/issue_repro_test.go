package toolcall

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"ai-proxy/types"
)

// TestExtractedToolCalls_SplitAcrossChunks reproduces the bug where tool calls
// extracted from thinking content have empty item_id/call_id in delta events
// when the markup is split across multiple streaming chunks.
func TestExtractedToolCalls_SplitAcrossChunks(t *testing.T) {
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
		// Chunk 1: Section begin
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "<|tool_calls_section_begin|>"}),
		},
		// Chunk 2: Tool call begin
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "<|tool_call_begin|>"}),
		},
		// Chunk 3: ID and name (split)
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "exec_"}),
		},
		// Chunk 4: Rest of name
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "command:0"}),
		},
		// Chunk 5: Arg begin
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "<|tool_call_argument_begin|>"}),
		},
		// Chunk 6: First part of args
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: `{"`}),
		},
		// Chunk 7: More args
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: `cmd":"ls -la"}`}),
		},
		// Chunk 8: Tool call end
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "<|tool_call_end|>"}),
		},
		// Chunk 9: Section end
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
			Type:      "message_delta",
			StopReason: "end_turn",
			Usage:     &types.AnthropicUsage{OutputTokens: 10},
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

			// Check output_item.added event
			if eventType == "response.output_item.added" {
				item, _ := event["item"].(map[string]interface{})
				if item != nil {
					itemType, _ := item["type"].(string)
					if itemType == "function_call" {
						id, _ := item["id"].(string)
						name, _ := item["name"].(string)
						callID, _ := item["call_id"].(string)
						itemID, _ := event["item_id"].(string)

						t.Logf("output_item.added: id=%s, name=%s, call_id=%s, item_id=%s", id, name, callID, itemID)

						if id == "" {
							t.Errorf("BUG: output_item.added has empty 'id' field")
						}
						if name == "" {
							t.Errorf("BUG: output_item.added has empty 'name' field")
						}
						if callID == "" {
							t.Errorf("BUG: output_item.added has empty 'call_id' field")
						}
						if itemID == "" {
							t.Errorf("BUG: output_item.added has empty 'item_id' field")
						}
					}
				}
			}

			// Check function_call_arguments.delta event
			if eventType == "response.function_call_arguments.delta" {
				itemID, _ := event["item_id"].(string)
				callID, _ := event["call_id"].(string)
				delta, _ := event["delta"].(string)

				t.Logf("function_call_arguments.delta: item_id=%s, call_id=%s, delta=%s", itemID, callID, delta)

				if itemID == "" {
					t.Errorf("BUG: function_call_arguments.delta has empty 'item_id' field")
				}
				if callID == "" {
					t.Errorf("BUG: function_call_arguments.delta has empty 'call_id' field")
				}
			}

			// Check output_item.done event
			if eventType == "response.output_item.done" {
				item, _ := event["item"].(map[string]interface{})
				if item != nil {
					itemType, _ := item["type"].(string)
					if itemType == "function_call" {
						id, _ := item["id"].(string)
						name, _ := item["name"].(string)
						callID, _ := item["call_id"].(string)
						arguments, _ := item["arguments"].(string)
						itemID, _ := event["item_id"].(string)

						t.Logf("output_item.done: id=%s, name=%s, call_id=%s, arguments=%s, item_id=%s", id, name, callID, arguments, itemID)

						if id == "" {
							t.Errorf("BUG: output_item.done has empty 'id' field")
						}
						if name == "" {
							t.Errorf("BUG: output_item.done has empty 'name' field")
						}
						if callID == "" {
							t.Errorf("BUG: output_item.done has empty 'call_id' field")
						}
						if arguments == "" {
							t.Errorf("BUG: output_item.done has empty 'arguments' field")
						}
						if itemID == "" {
							t.Errorf("BUG: output_item.done has empty 'item_id' field")
						}
					}
				}
			}
		}
	}
}

