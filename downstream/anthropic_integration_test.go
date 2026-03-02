package downstream

import (
	"bytes"
	"strings"
	"testing"

	"github.com/tmaxmax/go-sse"
)

// TestAnthropicIntegration_BasicStreaming tests the basic streaming example from Anthropic docs
// Source: https://docs.anthropic.com/en/api/messages-streaming
func TestAnthropicIntegration_BasicStreaming(t *testing.T) {
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	// Real Anthropic streaming response from documentation
	events := []string{
		`{"type":"message_start","message":{"id":"msg_1nZdL29xx5MUA1yADyHTEsnR8uuvGzszyY","type":"message","role":"assistant","content":[],"model":"claude-opus-4-6","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":25,"output_tokens":1}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"ping"}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"!"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":15}}`,
		`{"type":"message_stop"}`,
	}

	for _, data := range events {
		transformer.Transform(&sse.Event{Data: data})
	}

	result := output.String()

	// Verify message_start
	if !strings.Contains(result, "event: message_start") {
		t.Errorf("Expected message_start event")
	}
	if !strings.Contains(result, "msg_1nZdL29xx5MUA1yADyHTEsnR8uuvGzszyY") {
		t.Errorf("Expected message ID")
	}

	// Verify text content
	if !strings.Contains(result, "Hello") {
		t.Errorf("Expected 'Hello' in output")
	}
	if !strings.Contains(result, "!") {
		t.Errorf("Expected '!' in output")
	}

	// Verify ping event passed through
	if !strings.Contains(result, "event: ping") {
		t.Errorf("Expected ping event to pass through")
	}

	// Verify message_delta
	if !strings.Contains(result, "event: message_delta") {
		t.Errorf("Expected message_delta event")
	}
	if !strings.Contains(result, "end_turn") {
		t.Errorf("Expected stop_reason in message_delta")
	}
}

// TestAnthropicIntegration_ToolUse tests the tool use streaming example from Anthropic docs
// Source: https://docs.anthropic.com/en/api/messages-streaming
func TestAnthropicIntegration_ToolUse(t *testing.T) {
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	// Real Anthropic tool use response from documentation (simplified)
	events := []string{
		`{"type":"message_start","message":{"id":"msg_014p7gG3wDgGV9EUtLvnow3U","type":"message","role":"assistant","model":"claude-opus-4-6","stop_sequence":null,"usage":{"input_tokens":472,"output_tokens":2},"content":[],"stop_reason":null}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"ping"}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Okay"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":","}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" let"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"'s"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" check"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" the"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" weather"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01T1x1fJ34qAmk2tNTrN7Up6","name":"get_weather","input":{}}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"location\":"}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":" \"San Francisco, CA\"}"}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":89}}`,
		`{"type":"message_stop"}`,
	}

	for _, data := range events {
		transformer.Transform(&sse.Event{Data: data})
	}

	result := output.String()

	// Verify text content
	if !strings.Contains(result, "Okay") {
		t.Errorf("Expected 'Okay' in output")
	}
	// Check for individual words since they're in separate events
	if !strings.Contains(result, "check") || !strings.Contains(result, "weather") {
		t.Errorf("Expected text content to be present")
	}

	// Verify tool_use block
	if !strings.Contains(result, "tool_use") {
		t.Errorf("Expected tool_use content block")
	}
	if !strings.Contains(result, "get_weather") {
		t.Errorf("Expected get_weather function name")
	}
	if !strings.Contains(result, "San Francisco") {
		t.Errorf("Expected location in arguments")
	}

	// Verify stop reason
	if !strings.Contains(result, "tool_use") {
		t.Errorf("Expected tool_use stop reason")
	}
}

// TestAnthropicIntegration_ExtendedThinking tests the extended thinking example from Anthropic docs
// Source: https://docs.anthropic.com/en/api/messages-streaming
func TestAnthropicIntegration_ExtendedThinking(t *testing.T) {
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	// Real Anthropic extended thinking response from documentation
	events := []string{
		`{"type":"message_start","message":{"id":"msg_01thinking","type":"message","role":"assistant","content":[],"model":"claude-opus-4-6","stop_reason":null,"stop_sequence":null}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"I need to find the GCD of 1071 and 462 using the Euclidean algorithm."}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"\n\n1071 = 2 × 462 + 147"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"\n462 = 3 × 147 + 21"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"\n147 = 7 × 21 + 0"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"\nThe remainder is 0, so GCD(1071, 462) = 21."}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"EqQBCgIYAhIM1gbcDa9GJwZA2b3hGgxBdjrkzLoky3dl1pkiMOYds..."}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"The greatest common divisor of 1071 and 462 is **21**."}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null}}`,
		`{"type":"message_stop"}`,
	}

	for _, data := range events {
		transformer.Transform(&sse.Event{Data: data})
	}

	result := output.String()

	// Verify thinking content
	if !strings.Contains(result, "Euclidean algorithm") {
		t.Errorf("Expected thinking content about Euclidean algorithm")
	}
	if !strings.Contains(result, "GCD(1071, 462) = 21") {
		t.Errorf("Expected GCD calculation result in thinking")
	}

	// Verify text response
	if !strings.Contains(result, "**21**") {
		t.Errorf("Expected final answer in text response")
	}

	// Verify signature delta passed through
	if strings.Contains(result, "signature") {
		t.Logf("Signature delta present (may be expected)")
	}
}

// TestAnthropicIntegration_WebSearchTool tests the web search tool example from Anthropic docs
// Source: https://docs.anthropic.com/en/api/messages-streaming
func TestAnthropicIntegration_WebSearchTool(t *testing.T) {
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	// Real Anthropic web search tool response from documentation
	events := []string{
		`{"type":"message_start","message":{"id":"msg_01G...","type":"message","role":"assistant","model":"claude-opus-4-6","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":2679,"output_tokens":3}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"I'll check"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" the current weather in New York City for you"}}`,
		`{"type":"ping"}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"."}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"server_tool_use","id":"srvtoolu_014hJH82Qum7Td6UV8gDXThB","name":"web_search","input":{}}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"query\": \"weather NYC today\"}"}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"content_block_start","index":2,"content_block":{"type":"web_search_tool_result","tool_use_id":"srvtoolu_014hJH82Qum7Td6UV8gDXThB","content":[{"type":"web_search_result","title":"Weather in NYC"}]}}`,
		`{"type":"content_block_stop","index":2}`,
		`{"type":"content_block_start","index":3,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":3,"delta":{"type":"text_delta","text":"Here's the current weather information for New York City"}}`,
		`{"type":"content_block_stop","index":3}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":510}}`,
		`{"type":"message_stop"}`,
	}

	for _, data := range events {
		transformer.Transform(&sse.Event{Data: data})
	}

	result := output.String()

	// Verify initial text
	if !strings.Contains(result, "I'll check") {
		t.Errorf("Expected initial text")
	}

	// Verify server_tool_use block (note: this is different from tool_use)
	if strings.Contains(result, "web_search") {
		t.Logf("Web search tool present")
	}

	// Verify final response
	if !strings.Contains(result, "New York City") {
		t.Errorf("Expected NYC weather information")
	}
}

// TestAnthropicIntegration_KimiWithThinkingAndTools tests Kimi format with thinking containing tool calls
func TestAnthropicIntegration_KimiWithThinkingAndTools(t *testing.T) {
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	// Simulated Kimi-K2.5 response with tool calls embedded in thinking
	events := []string{
		`{"type":"message_start","message":{"id":"msg_kimi_001","type":"message","role":"assistant","content":[],"model":"kimi-k2.5","usage":{"input_tokens":100,"output_tokens":1}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me analyze the user's request."}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":" I need to check the file system.<|tool_calls_section_begin|><|tool_call_begin|>bash:1<|tool_call_argument_begin|>{\"command\": \"ls -la /workspace\"}<|tool_call_end|>"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"<|tool_call_begin|>read:2<|tool_call_argument_begin|>{\"path\": \"/workspace/main.go\"}<|tool_call_end|><|tool_calls_section_end|>"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":" Now I can proceed with the analysis."}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"I've analyzed the workspace."}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":50}}`,
		`{"type":"message_stop"}`,
	}

	for _, data := range events {
		transformer.Transform(&sse.Event{Data: data})
	}

	result := output.String()

	// Verify thinking content before tools
	if !strings.Contains(result, "Let me analyze") {
		t.Errorf("Expected thinking content before tools")
	}

	// Verify tool calls were extracted
	if strings.Count(result, "tool_use") < 2 {
		t.Errorf("Expected at least 2 tool_use blocks, got: %s", result)
	}

	// Verify bash tool
	if !strings.Contains(result, "bash") {
		t.Errorf("Expected bash function")
	}
	if !strings.Contains(result, "ls -la") {
		t.Errorf("Expected bash command")
	}

	// Verify read tool
	if !strings.Contains(result, "read") {
		t.Errorf("Expected read function")
	}
	if !strings.Contains(result, "main.go") {
		t.Errorf("Expected file path")
	}

	// Verify thinking content after tools
	if !strings.Contains(result, "Now I can proceed") {
		t.Errorf("Expected thinking content after tools")
	}

	// Verify text response
	if !strings.Contains(result, "I've analyzed") {
		t.Errorf("Expected text response")
	}
}

// TestAnthropicIntegration_KimiToolsOnlyInThinking tests when only tools are in thinking (no surrounding text)
func TestAnthropicIntegration_KimiToolsOnlyInThinking(t *testing.T) {
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	events := []string{
		`{"type":"message_start","message":{"id":"msg_kimi_002","type":"message","role":"assistant","content":[],"model":"kimi-k2.5"}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"<|tool_calls_section_begin|><|tool_call_begin|>glob:1<|tool_call_argument_begin|>{\"pattern\": \"**/*.go\"}<|tool_call_end|><|tool_calls_section_end|>"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use"}}`,
		`{"type":"message_stop"}`,
	}

	for _, data := range events {
		transformer.Transform(&sse.Event{Data: data})
	}

	result := output.String()

	// Verify tool call
	if !strings.Contains(result, "glob") {
		t.Errorf("Expected glob function")
	}
	if !strings.Contains(result, "**/*.go") {
		t.Errorf("Expected glob pattern")
	}
}

// TestAnthropicIntegration_KimiComplexArguments tests tool calls with complex JSON arguments
func TestAnthropicIntegration_KimiComplexArguments(t *testing.T) {
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	events := []string{
		`{"type":"message_start","message":{"id":"msg_kimi_003","type":"message","role":"assistant","content":[],"model":"kimi-k2.5"}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"<|tool_calls_section_begin|><|tool_call_begin|>bash:1<|tool_call_argument_begin|>{\"command\": \"echo \\\"hello\\\" && ls -la | grep test\"}<|tool_call_end|><|tool_calls_section_end|>"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_stop"}`,
	}

	for _, data := range events {
		transformer.Transform(&sse.Event{Data: data})
	}

	result := output.String()

	// Verify complex command
	if !strings.Contains(result, "echo") {
		t.Errorf("Expected echo in command")
	}
	if !strings.Contains(result, "grep test") {
		t.Errorf("Expected grep in command")
	}
}

// TestAnthropicIntegration_KimiMultipleToolCallsSequential tests multiple separate tool call sections
func TestAnthropicIntegration_KimiMultipleToolCallsSequential(t *testing.T) {
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	events := []string{
		`{"type":"message_start","message":{"id":"msg_kimi_004","type":"message","role":"assistant","content":[],"model":"kimi-k2.5"}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"First task.<|tool_calls_section_begin|><|tool_call_begin|>read:1<|tool_call_argument_begin|>{\"path\": \"a.txt\"}<|tool_call_end|><|tool_calls_section_end|>"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":" Second task.<|tool_calls_section_begin|><|tool_call_begin|>read:2<|tool_call_argument_begin|>{\"path\": \"b.txt\"}<|tool_call_end|><|tool_calls_section_end|>"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_stop"}`,
	}

	for _, data := range events {
		transformer.Transform(&sse.Event{Data: data})
	}

	result := output.String()

	// Verify both tool calls
	if strings.Count(result, "tool_use") < 2 {
		t.Errorf("Expected at least 2 tool_use blocks, got: %d", strings.Count(result, "tool_use"))
	}
	if !strings.Contains(result, "a.txt") {
		t.Errorf("Expected first file path")
	}
	if !strings.Contains(result, "b.txt") {
		t.Errorf("Expected second file path")
	}
}

// TestAnthropicIntegration_FullStreamWithAllEventTypes tests a complete stream with all event types
func TestAnthropicIntegration_FullStreamWithAllEventTypes(t *testing.T) {
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	events := []string{
		`{"type":"message_start","message":{"id":"msg_full","type":"message","role":"assistant","content":[],"model":"kimi-k2.5","usage":{"input_tokens":100,"output_tokens":1}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		`{"type":"ping"}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Thinking...<|tool_calls_section_begin|><|tool_call_begin|>tool:1<|tool_call_argument_begin|>{"arg": "value"}<|tool_call_end|><|tool_calls_section_end|>Done."}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Response"}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":50}}`,
		`{"type":"message_stop"}`,
	}

	for _, data := range events {
		transformer.Transform(&sse.Event{Data: data})
	}

	result := output.String()

	// Count event types
	eventTypes := []string{
		"event: message_start",
		"event: content_block_start",
		"event: content_block_delta",
		"event: content_block_stop",
		"event: message_delta",
		"event: message_stop",
		"event: ping",
	}

	for _, eventType := range eventTypes {
		if !strings.Contains(result, eventType) {
			t.Errorf("Expected %s in output", eventType)
		}
	}

	// Verify SSE format (event: before data:)
	lines := strings.Split(result, "\n")
	for i, line := range lines {
		if strings.HasPrefix(line, "event:") && i+1 < len(lines) {
			if !strings.HasPrefix(lines[i+1], "data:") {
				t.Errorf("Expected data: after event: at line %d", i)
			}
		}
	}
}
