package downstream

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/tmaxmax/go-sse"
)

func TestAnthropicToolCallTransformer_MessageStart(t *testing.T) {
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	event := &sse.Event{
		Data: `{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet-20241022","usage":{"input_tokens":25,"output_tokens":1}}}`,
	}

	transformer.Transform(event)

	result := output.String()
	if !strings.Contains(result, "event: message_start") {
		t.Errorf("Expected message_start event, got: %s", result)
	}
	if !strings.Contains(result, "msg_123") {
		t.Errorf("Expected message ID in output, got: %s", result)
	}
}

func TestAnthropicToolCallTransformer_TextContent(t *testing.T) {
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	events := []string{
		`{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet-20241022"}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"}}`,
		`{"type":"message_stop"}`,
	}

	for _, data := range events {
		transformer.Transform(&sse.Event{Data: data})
	}

	result := output.String()
	if !strings.Contains(result, "Hello") || !strings.Contains(result, " world") {
		t.Errorf("Expected text content in output, got: %s", result)
	}
	// Text content should pass through (may have 1 or 2 delta events depending on buffering)
	deltaCount := strings.Count(result, "event: content_block_delta")
	if deltaCount < 1 {
		t.Errorf("Expected at least 1 content_block_delta event, got: %d", deltaCount)
	}
}

func TestAnthropicToolCallTransformer_ThinkingContent(t *testing.T) {
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	events := []string{
		`{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet-20241022"}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me think about this"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":" step by step"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"The answer is 42"}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"message_stop"}`,
	}

	for _, data := range events {
		transformer.Transform(&sse.Event{Data: data})
	}

	result := output.String()
	if !strings.Contains(result, "Let me think about this step by step") {
		t.Errorf("Expected thinking content in output, got: %s", result)
	}
	if !strings.Contains(result, "The answer is 42") {
		t.Errorf("Expected text content in output, got: %s", result)
	}
}

func TestAnthropicToolCallTransformer_ToolCall_Basic(t *testing.T) {
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	events := []string{
		`{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet-20241022"}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"I need to call a function<|tool_calls_section_begin|><|tool_call_begin|>get_weather:1<|tool_call_argument_begin|>{\"city\": \"San Francisco\"}<|tool_call_end|><|tool_calls_section_end|>"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_stop"}`,
	}

	for _, data := range events {
		transformer.Transform(&sse.Event{Data: data})
	}

	result := output.String()
	if !strings.Contains(result, "event: content_block_start") {
		t.Errorf("Expected content_block_start event for tool_use")
	}
	if !strings.Contains(result, "tool_use") {
		t.Errorf("Expected tool_use in output, got: %s", result)
	}
	if !strings.Contains(result, "get_weather") {
		t.Errorf("Expected function name in output, got: %s", result)
	}
	if !strings.Contains(result, "San Francisco") {
		t.Errorf("Expected function arguments in output, got: %s", result)
	}
}

func TestAnthropicToolCallTransformer_ToolCall_Multiple(t *testing.T) {
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	events := []string{
		`{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet-20241022"}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"<|tool_calls_section_begin|><|tool_call_begin|>bash:1<|tool_call_argument_begin|>{\"command\": \"ls\"}<|tool_call_end|><|tool_call_begin|>read:2<|tool_call_argument_begin|>{\"path\": \"/test.txt\"}<|tool_call_end|><|tool_calls_section_end|>"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_stop"}`,
	}

	for _, data := range events {
		transformer.Transform(&sse.Event{Data: data})
	}

	result := output.String()
	if strings.Count(result, "tool_use") < 2 {
		t.Errorf("Expected at least 2 tool_use blocks, got: %s", result)
	}
	if !strings.Contains(result, "bash") {
		t.Errorf("Expected bash function name")
	}
	if !strings.Contains(result, "read") {
		t.Errorf("Expected read function name")
	}
}

func TestAnthropicToolCallTransformer_ToolCall_WithTextBefore(t *testing.T) {
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	events := []string{
		`{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet-20241022"}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me check<|tool_calls_section_begin|><|tool_call_begin|>get_weather:1<|tool_call_argument_begin|>{\"city\": \"NYC\"}<|tool_call_end|><|tool_calls_section_end|> Done."}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_stop"}`,
	}

	for _, data := range events {
		transformer.Transform(&sse.Event{Data: data})
	}

	result := output.String()
	if !strings.Contains(result, "Let me check") {
		t.Errorf("Expected thinking text before tool call")
	}
	if !strings.Contains(result, "Done.") {
		t.Errorf("Expected thinking text after tool call section")
	}
}

func TestAnthropicToolCallTransformer_ToolCall_ArgumentsSplitAcrossChunks(t *testing.T) {
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	events := []string{
		`{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet-20241022"}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"<|tool_calls_section_begin|><|tool_call_begin|>search:1<|tool_call_argument_begin|>{\"query\": \""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"weather in Boston"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"\"}<|tool_call_end|><|tool_calls_section_end|>"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_stop"}`,
	}

	for _, data := range events {
		transformer.Transform(&sse.Event{Data: data})
	}

	result := output.String()
	if !strings.Contains(result, "weather in Boston") {
		t.Errorf("Expected split arguments to be reassembled, got: %s", result)
	}
}

func TestAnthropicToolCallTransformer_ToolCall_NoThinking(t *testing.T) {
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	events := []string{
		`{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet-20241022"}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_stop"}`,
	}

	for _, data := range events {
		transformer.Transform(&sse.Event{Data: data})
	}

	result := output.String()
	if !strings.Contains(result, "Hello") {
		t.Errorf("Expected text content to pass through unchanged")
	}
	if strings.Contains(result, "tool_use") {
		t.Errorf("Unexpected tool_use in non-tool response")
	}
}

func TestAnthropicToolCallTransformer_PingEvent(t *testing.T) {
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	event := &sse.Event{
		Data: `{"type":"ping"}`,
	}

	transformer.Transform(event)

	result := output.String()
	if !strings.Contains(result, "event: ping") {
		t.Errorf("Expected ping event to pass through, got: %s", result)
	}
}

func TestAnthropicToolCallTransformer_MessageDelta(t *testing.T) {
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	event := &sse.Event{
		Data: `{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":89}}`,
	}

	transformer.Transform(event)

	result := output.String()
	if !strings.Contains(result, "event: message_delta") {
		t.Errorf("Expected message_delta event, got: %s", result)
	}
	if !strings.Contains(result, "tool_use") {
		t.Errorf("Expected stop_reason in output")
	}
}

func TestAnthropicToolCallTransformer_InvalidJSON(t *testing.T) {
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	event := &sse.Event{
		Data: `{"type": "invalid json`,
	}

	transformer.Transform(event)

	result := output.String()
	if !strings.Contains(result, "event: error") {
		t.Errorf("Expected error event for invalid JSON, got: %s", result)
	}
}

func TestAnthropicToolCallTransformer_EmptyData(t *testing.T) {
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	event := &sse.Event{
		Data: "",
	}

	transformer.Transform(event)

	result := output.String()
	if result != "" {
		t.Errorf("Expected no output for empty data, got: %s", result)
	}
}

func TestAnthropicToolCallTransformer_FunctionNameParsing(t *testing.T) {
	testCases := []struct {
		name     string
		input    string
		expected string
	}{
		{"simple_name", "bash:1", "bash"},
		{"with_namespace", "functions.bash:1", "bash"},
		{"with_colon_only", "bash:", "bash"},
		{"no_suffix", "bash", "bash"},
		{"complex_namespace", "tools.utils.bash:1", "bash"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var output bytes.Buffer
			transformer := NewAnthropicToolCallTransformer(&output)

			events := []string{
				`{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet-20241022"}}`,
				`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
				`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"<|tool_calls_section_begin|><|tool_call_begin|>` + tc.input + `<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>"}}`,
				`{"type":"content_block_stop","index":0}`,
				`{"type":"message_stop"}`,
			}

			for _, data := range events {
				transformer.Transform(&sse.Event{Data: data})
			}

			result := output.String()
			if !strings.Contains(result, tc.expected) {
				t.Errorf("Expected function name %q in output, got: %s", tc.expected, result)
			}
		})
	}
}

func TestAnthropicToolCallTransformer_ToolIDGeneration(t *testing.T) {
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	events := []string{
		`{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet-20241022"}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"<|tool_calls_section_begin|><|tool_call_begin|>bash:1<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_stop"}`,
	}

	for _, data := range events {
		transformer.Transform(&sse.Event{Data: data})
	}

	result := output.String()
	if !strings.Contains(result, "toolu_") {
		t.Errorf("Expected toolu_ prefix in tool ID, got: %s", result)
	}
}

func TestAnthropicToolCallTransformer_ToolIDPreservation(t *testing.T) {
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	events := []string{
		`{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet-20241022"}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"<|tool_calls_section_begin|><|tool_call_begin|>toolu_abc123<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_stop"}`,
	}

	for _, data := range events {
		transformer.Transform(&sse.Event{Data: data})
	}

	result := output.String()
	if !strings.Contains(result, "toolu_abc123") {
		t.Errorf("Expected toolu_abc123 to be preserved, got: %s", result)
	}
}

func TestAnthropicToolCallTransformer_BlockIndexTracking(t *testing.T) {
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	events := []string{
		`{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet-20241022"}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"<|tool_calls_section_begin|><|tool_call_begin|>tool1:1<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_call_begin|>tool2:2<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_stop"}`,
	}

	for _, data := range events {
		transformer.Transform(&sse.Event{Data: data})
	}

	result := output.String()
	lines := strings.Split(result, "\n")
	toolUseIndices := make(map[int]bool)

	for _, line := range lines {
		if strings.Contains(line, "tool_use") && strings.Contains(line, "content_block_start") {
			var event AnthropicEvent
			if strings.HasPrefix(line, "data: ") {
				jsonStr := strings.TrimPrefix(line, "data: ")
				if err := json.Unmarshal([]byte(jsonStr), &event); err == nil {
					if event.Index != nil {
						toolUseIndices[*event.Index] = true
					}
				}
			}
		}
	}

	if len(toolUseIndices) != 2 {
		t.Errorf("Expected 2 unique tool_use block indices, got %d", len(toolUseIndices))
	}
}

func TestAnthropicToolCallTransformer_ThinkingReopenAfterToolSection(t *testing.T) {
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	events := []string{
		`{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet-20241022"}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Before<|tool_calls_section_begin|><|tool_call_begin|>tool:1<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>After"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_stop"}`,
	}

	for _, data := range events {
		transformer.Transform(&sse.Event{Data: data})
	}

	result := output.String()
	if !strings.Contains(result, "Before") {
		t.Errorf("Expected thinking content before tool section")
	}
	if !strings.Contains(result, "After") {
		t.Errorf("Expected thinking content after tool section to reopen thinking block")
	}
}

func TestAnthropicToolCallTransformer_ArgumentsWithSpecialCharacters(t *testing.T) {
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	events := []string{
		`{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet-20241022"}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"<|tool_calls_section_begin|><|tool_call_begin|>bash:1<|tool_call_argument_begin|>{\"cmd\": \"echo \\\"hello world\\\" && ls -la\"}<|tool_call_end|><|tool_calls_section_end|>"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_stop"}`,
	}

	for _, data := range events {
		transformer.Transform(&sse.Event{Data: data})
	}

	result := output.String()
	if !strings.Contains(result, "echo") {
		t.Errorf("Expected special characters in arguments")
	}
}

func TestAnthropicToolCallTransformer_StateMachine_Reset(t *testing.T) {
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	events1 := []string{
		`{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet-20241022"}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"<|tool_calls_section_begin|><|tool_call_begin|>tool:1<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>"}}`,
		`{"type":"content_block_stop","index":0}`,
	}

	for _, data := range events1 {
		transformer.Transform(&sse.Event{Data: data})
	}

	transformer2 := NewAnthropicToolCallTransformer(&output)
	events2 := []string{
		`{"type":"message_start","message":{"id":"msg_456","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet-20241022"}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"<|tool_calls_section_begin|><|tool_call_begin|>tool2:1<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_stop"}`,
	}

	for _, data := range events2 {
		transformer2.Transform(&sse.Event{Data: data})
	}

	result := output.String()
	if strings.Count(result, "tool_use") != 2 {
		t.Errorf("Expected 2 tool_use blocks total, got different count")
	}
}

func TestAnthropicToolCallTransformer_UnknownEventType(t *testing.T) {
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	event := &sse.Event{
		Data: `{"type":"unknown_event_type","data":{}}`,
	}

	transformer.Transform(event)

	result := output.String()
	if !strings.Contains(result, "event: unknown_event_type") {
		t.Errorf("Expected unknown event type to pass through, got: %s", result)
	}
}

func TestAnthropicToolCallTransformer_Flush(t *testing.T) {
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	transformer.Transform(&sse.Event{
		Data: `{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet-20241022"}}`,
	})

	transformer.Flush()

	result := output.String()
	if !strings.Contains(result, "message_start") {
		t.Errorf("Expected message_start event after flush")
	}
}

func TestAnthropicToolCallTransformer_Close(t *testing.T) {
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	transformer.Transform(&sse.Event{
		Data: `{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet-20241022"}}`,
	})

	transformer.Close()

	result := output.String()
	if !strings.Contains(result, "message_start") {
		t.Errorf("Expected message_start event after close")
	}
}

func TestAnthropicToolCallTransformer_ContentBlockStart_NonThinking(t *testing.T) {
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	events := []string{
		`{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet-20241022"}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_stop"}`,
	}

	for _, data := range events {
		transformer.Transform(&sse.Event{Data: data})
	}

	result := output.String()
	if !strings.Contains(result, "event: content_block_start") {
		t.Errorf("Expected content_block_start event for text block")
	}
	if !strings.Contains(result, "Hello") {
		t.Errorf("Expected text content")
	}
}

func TestAnthropicToolCallTransformer_SSEEventFormat(t *testing.T) {
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	event := &sse.Event{
		Data: `{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet-20241022"}}`,
	}

	transformer.Transform(event)

	result := output.String()
	lines := strings.Split(strings.TrimSpace(result), "\n")

	if len(lines) < 2 {
		t.Fatalf("Expected at least 2 lines in SSE format, got: %d", len(lines))
	}

	if !strings.HasPrefix(lines[0], "event: ") {
		t.Errorf("Expected first line to start with 'event: ', got: %s", lines[0])
	}

	if !strings.HasPrefix(lines[1], "data: ") {
		t.Errorf("Expected second line to start with 'data: ', got: %s", lines[1])
	}
}

func TestAnthropicToolCallTransformer_ThinkingDeltaWithSignature(t *testing.T) {
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	events := []string{
		`{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet-20241022"}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me think"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"abc123"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_stop"}`,
	}

	for _, data := range events {
		transformer.Transform(&sse.Event{Data: data})
	}

	result := output.String()
	if !strings.Contains(result, "Let me think") {
		t.Errorf("Expected thinking content")
	}
	if strings.Contains(result, "signature") {
		t.Logf("Signature delta passed through (may be expected)")
	}
}

func TestAnthropicToolCallTransformer_MultipleThinkingBlocks(t *testing.T) {
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	events := []string{
		`{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet-20241022"}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"First thought"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Response"}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"content_block_start","index":2,"content_block":{"type":"thinking","thinking":""}}`,
		`{"type":"content_block_delta","index":2,"delta":{"type":"thinking_delta","thinking":"Second thought"}}`,
		`{"type":"content_block_stop","index":2}`,
		`{"type":"message_stop"}`,
	}

	for _, data := range events {
		transformer.Transform(&sse.Event{Data: data})
	}

	result := output.String()
	if !strings.Contains(result, "First thought") {
		t.Errorf("Expected first thinking block")
	}
	if !strings.Contains(result, "Second thought") {
		t.Errorf("Expected second thinking block")
	}
	if !strings.Contains(result, "Response") {
		t.Errorf("Expected text response")
	}
}
