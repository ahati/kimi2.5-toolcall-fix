package toolcall

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"ai-proxy/types"
)

// TestReproduceIssueFromCapturedLogs reproduces the bug from the captured logs
// where tool calls extracted from thinking content have empty id/name in output_item.done
func TestReproduceIssueFromCapturedLogs(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesTransformer(&buf)

	// Exact sequence from captured logs: function.exec_command:2
	// The upstream sends: "function" ".exec" "_command" ":" "2"
	// Which gets concatenated to "function.exec_command:2" before ArgBegin
	upstreamEvents := []string{
		`event: message_start
data: {"message":{"model":"kimi-k2.5","id":"msg_13c8256b-60de-4eee-a2a2-e0e17d16e1e2","role":"assistant","type":"message","content":[],"usage":{"input_tokens":5942,"output_tokens":0}},"type":"message_start"}`,
		`event: ping
data: {"type":"ping"}`,
		`event: content_block_start
data: {"type":"content_block_start","content_block":{"type":"thinking","signature":"","thinking":""},"index":0}`,
		`event: content_block_delta
data: {"delta":{"type":"thinking_delta","thinking":"<|tool_calls_section_begin|>"},"type":"content_block_delta","index":0}`,
		`event: content_block_delta
data: {"delta":{"type":"thinking_delta","thinking":"<|tool_call_begin|>"},"type":"content_block_delta","index":0}`,
		`event: content_block_delta
data: {"delta":{"type":"thinking_delta","thinking":"function"},"type":"content_block_delta","index":0}`,
		`event: content_block_delta
data: {"delta":{"type":"thinking_delta","thinking":".exec"},"type":"content_block_delta","index":0}`,
		`event: content_block_delta
data: {"delta":{"type":"thinking_delta","thinking":"_command"},"type":"content_block_delta","index":0}`,
		`event: content_block_delta
data: {"delta":{"type":"thinking_delta","thinking":":"},"type":"content_block_delta","index":0}`,
		`event: content_block_delta
data: {"delta":{"type":"thinking_delta","thinking":"2"},"type":"content_block_delta","index":0}`,
		`event: content_block_delta
data: {"delta":{"type":"thinking_delta","thinking":"<|tool_call_argument_begin|>"},"type":"content_block_delta","index":0}`,
		`event: content_block_delta
data: {"delta":{"type":"thinking_delta","thinking":"{\"cmd\": \"ls -d /usr/include/*/ 2>/dev/null | head -30\", \"max_output_tokens\": 1000}"},"type":"content_block_delta","index":0}`,
		`event: content_block_delta
data: {"delta":{"type":"thinking_delta","thinking":"<|tool_call_end|>"},"type":"content_block_delta","index":0}`,
		`event: content_block_delta
data: {"delta":{"type":"thinking_delta","thinking":"<|tool_calls_section_end|>"},"type":"content_block_delta","index":0}`,
		`event: content_block_stop
data: {"type":"content_block_stop","index":0}`,
		`event: message_delta
data: {"delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":100}}`,
		`event: message_stop
data: {"type":"message_stop"}`,
	}

	// Process events using handleEvent (internal method)
	for _, eventStr := range upstreamEvents {
		// Parse the SSE event
		lines := strings.Split(eventStr, "\n")
		var eventType, dataStr string
		for _, line := range lines {
			if strings.HasPrefix(line, "event: ") {
				eventType = strings.TrimPrefix(line, "event: ")
			} else if strings.HasPrefix(line, "data: ") {
				dataStr = strings.TrimPrefix(line, "data: ")
			}
		}

		var event types.Event
		if dataStr != "" {
			if err := json.Unmarshal([]byte(dataStr), &event); err != nil {
				t.Fatalf("Failed to parse event: %v", err)
			}
		}
		event.Type = eventType

		if err := transformer.handleEvent(event); err != nil {
			t.Fatalf("handleEvent error: %v", err)
		}
	}

	output := buf.String()
	t.Logf("Output:\n%s", output)

	// Parse output and check for the bug
	lines := strings.Split(output, "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			var event map[string]interface{}
			if err := json.Unmarshal([]byte(data), &event); err != nil {
				continue
			}

			eventType, _ := event["type"].(string)
			t.Logf("Event: %s", eventType)

			if eventType == "response.output_item.done" {
				item, _ := event["item"].(map[string]interface{})
				if item == nil {
					continue
				}
				itemType, _ := item["type"].(string)
				t.Logf("  item type: %s", itemType)
				if itemType == "function_call" {
					name, _ := item["name"].(string)
					id, _ := item["id"].(string)
					callID, _ := item["call_id"].(string)
					itemID, _ := event["item_id"].(string)

					t.Logf("  function_call: name=%q, id=%q, call_id=%q, item_id=%q", name, id, callID, itemID)

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

					// The function name should be exec_command (stripped of module prefix)
					if name != "" && name != "exec_command" {
						t.Errorf("Expected function name 'exec_command', got '%s'", name)
					}
				}
			}
		}
	}
}
