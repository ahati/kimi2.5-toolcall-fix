package toolcall

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"ai-proxy/types"
)

func TestAnthropicOutput_ToolCall_Basic(t *testing.T) {
	var buf bytes.Buffer
	output := NewAnthropicOutput(&buf, ContextThinking, 0)
	output.SetBlockOpen(true)
	output.currentIndex = 0

	parser := NewParser(DefaultTokenSet(), output)
	parser.Feed("I need to call a function<|tool_calls_section_begin|><|tool_call_begin|>get_weather:1<|tool_call_argument_begin|>{\"city\": \"San Francisco\"}<|tool_call_end|><|tool_calls_section_end|>")
	parser.Flush()

	result := buf.String()
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

func TestAnthropicOutput_ToolCall_Multiple(t *testing.T) {
	var buf bytes.Buffer
	output := NewAnthropicOutput(&buf, ContextThinking, 0)
	output.SetBlockOpen(true)
	output.currentIndex = 0

	parser := NewParser(DefaultTokenSet(), output)
	parser.Feed("<|tool_calls_section_begin|><|tool_call_begin|>bash:1<|tool_call_argument_begin|>{\"command\": \"ls\"}<|tool_call_end|><|tool_call_begin|>read:2<|tool_call_argument_begin|>{\"path\": \"/test.txt\"}<|tool_call_end|><|tool_calls_section_end|>")
	parser.Flush()

	result := buf.String()
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

func TestAnthropicOutput_ToolCall_ArgumentsSplitAcrossChunks(t *testing.T) {
	var buf bytes.Buffer
	output := NewAnthropicOutput(&buf, ContextThinking, 0)
	output.SetBlockOpen(true)
	output.currentIndex = 0

	parser := NewParser(DefaultTokenSet(), output)
	parser.Feed("<|tool_calls_section_begin|><|tool_call_begin|>search:1<|tool_call_argument_begin|>{\"query\": \"")
	parser.Feed("weather in Boston")
	parser.Feed("\"}<|tool_call_end|><|tool_calls_section_end|>")
	parser.Flush()

	result := buf.String()
	if !strings.Contains(result, "weather in Boston") {
		t.Errorf("Expected split arguments to be reassembled, got: %s", result)
	}
}

func TestAnthropicOutput_BlockIndexTracking(t *testing.T) {
	var buf bytes.Buffer
	output := NewAnthropicOutput(&buf, ContextThinking, 0)
	output.SetBlockOpen(true)
	output.currentIndex = 0

	parser := NewParser(DefaultTokenSet(), output)
	parser.Feed("<|tool_calls_section_begin|><|tool_call_begin|>tool1:1<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_call_begin|>tool2:2<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>")
	parser.Flush()

	result := buf.String()
	lines := strings.Split(result, "\n")
	toolUseIndices := make(map[int]bool)

	for _, line := range lines {
		if strings.Contains(line, "tool_use") && strings.Contains(line, "content_block_start") {
			var event types.Event
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

	seen := make(map[int]bool)
	for idx := range toolUseIndices {
		if seen[idx] {
			t.Errorf("Duplicate block index detected: %d", idx)
		}
		seen[idx] = true
	}
}

func TestAnthropicOutput_StopReasonTransformation(t *testing.T) {
	var buf bytes.Buffer
	output := NewAnthropicOutput(&buf, ContextThinking, 0)
	output.SetBlockOpen(true)
	output.currentIndex = 0

	parser := NewParser(DefaultTokenSet(), output)
	parser.Feed("<|tool_calls_section_begin|><|tool_call_begin|>bash:1<|tool_call_argument_begin|>{\"cmd\": \"ls\"}<|tool_call_end|><|tool_calls_section_end|>")
	parser.Flush()

	if !output.ToolsEmitted() {
		t.Errorf("Expected ToolsEmitted to be true after tool calls")
	}
}

func TestAnthropicOutput_TextContext(t *testing.T) {
	var buf bytes.Buffer
	output := NewAnthropicOutput(&buf, ContextText, 0)
	output.SetBlockOpen(true)
	output.currentIndex = 0

	output.OnText("Hello world")

	result := buf.String()
	if !strings.Contains(result, "text_delta") {
		t.Errorf("Expected text_delta in output, got: %s", result)
	}
	if !strings.Contains(result, "Hello world") {
		t.Errorf("Expected text content in output, got: %s", result)
	}
}

func TestAnthropicOutput_ThinkingContext(t *testing.T) {
	var buf bytes.Buffer
	output := NewAnthropicOutput(&buf, ContextThinking, 0)
	output.SetBlockOpen(true)
	output.currentIndex = 0

	output.OnText("Let me think")

	result := buf.String()
	if !strings.Contains(result, "thinking_delta") {
		t.Errorf("Expected thinking_delta in output, got: %s", result)
	}
	if !strings.Contains(result, "Let me think") {
		t.Errorf("Expected thinking content in output, got: %s", result)
	}
}

func TestAnthropicOutput_BlockIndexIncrements(t *testing.T) {
	var buf bytes.Buffer
	output := NewAnthropicOutput(&buf, ContextThinking, 5)
	output.SetBlockOpen(false)

	parser := NewParser(DefaultTokenSet(), output)
	parser.Feed("<|tool_calls_section_begin|><|tool_call_begin|>tool:1<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>")
	parser.Flush()

	result := buf.String()

	var toolIndex int
	lines := strings.Split(result, "\n")
	for _, line := range lines {
		if strings.Contains(line, "tool_use") && strings.Contains(line, "content_block_start") {
			if strings.HasPrefix(line, "data: ") {
				jsonStr := strings.TrimPrefix(line, "data: ")
				var event types.Event
				if err := json.Unmarshal([]byte(jsonStr), &event); err == nil {
					if event.Index != nil {
						toolIndex = *event.Index
					}
				}
			}
		}
	}

	if toolIndex != 5 {
		t.Errorf("Expected tool index 5 (from initial block index), got %d", toolIndex)
	}
}

func TestAnthropicOutput_CloseOpenBlockBeforeToolCall(t *testing.T) {
	var buf bytes.Buffer
	output := NewAnthropicOutput(&buf, ContextThinking, 0)
	output.SetBlockOpen(true)
	output.currentIndex = 0

	output.OnToolCallStart("toolu_123", "bash", 0)

	result := buf.String()
	stopCount := strings.Count(result, "event: content_block_stop")
	startCount := strings.Count(result, "event: content_block_start")

	if stopCount != 1 {
		t.Errorf("Expected 1 content_block_stop event (for closing open block), got %d", stopCount)
	}
	if startCount != 1 {
		t.Errorf("Expected 1 content_block_start event (for tool_use), got %d", startCount)
	}
}

func TestAnthropicOutput_NoCloseWhenBlockNotOpen(t *testing.T) {
	var buf bytes.Buffer
	output := NewAnthropicOutput(&buf, ContextThinking, 0)
	output.SetBlockOpen(false)

	output.OnToolCallStart("toolu_123", "bash", 0)

	result := buf.String()
	stopCount := strings.Count(result, "content_block_stop")

	if stopCount != 0 {
		t.Errorf("Expected 0 content_block_stop when no block was open, got %d", stopCount)
	}
}

func TestAnthropicOutput_ThreeToolCallsSequential(t *testing.T) {
	var buf bytes.Buffer
	output := NewAnthropicOutput(&buf, ContextThinking, 0)
	output.SetBlockOpen(true)
	output.currentIndex = 0

	parser := NewParser(DefaultTokenSet(), output)
	parser.Feed("<|tool_calls_section_begin|><|tool_call_begin|>bash:1<|tool_call_argument_begin|>{\"cmd\": \"cmd1\"}<|tool_call_end|><|tool_call_begin|>bash:2<|tool_call_argument_begin|>{\"cmd\": \"cmd2\"}<|tool_call_end|><|tool_call_begin|>bash:3<|tool_call_argument_begin|>{\"cmd\": \"cmd3\"}<|tool_call_end|><|tool_calls_section_end|>")
	parser.Flush()

	result := buf.String()

	toolUseCount := strings.Count(result, `"type":"tool_use"`)
	if toolUseCount != 3 {
		t.Errorf("Expected 3 tool_use blocks, got %d. Output:\n%s", toolUseCount, result)
	}

	var indices []int
	lines := strings.Split(result, "\n")
	for _, line := range lines {
		if strings.Contains(line, `"type":"content_block_start"`) && strings.Contains(line, `"type":"tool_use"`) {
			jsonStr := strings.TrimPrefix(line, "data: ")
			var event struct {
				Index int `json:"index"`
			}
			if err := json.Unmarshal([]byte(jsonStr), &event); err == nil {
				indices = append(indices, event.Index)
			}
		}
	}

	if len(indices) != 3 {
		t.Fatalf("Expected 3 tool_use block indices, got %d", len(indices))
	}

	for i := 0; i < len(indices)-1; i++ {
		if indices[i] >= indices[i+1] {
			t.Errorf("Tool indices not strictly increasing: %v", indices)
		}
	}

	seen := make(map[int]bool)
	for _, idx := range indices {
		if seen[idx] {
			t.Errorf("Duplicate tool index: %d", idx)
		}
		seen[idx] = true
	}
}

func TestAnthropicOutput_SSEEventFormat(t *testing.T) {
	var buf bytes.Buffer
	output := NewAnthropicOutput(&buf, ContextThinking, 0)

	output.OnText("test")

	result := buf.String()
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

func TestAnthropicOutput_EmptyTextIgnored(t *testing.T) {
	var buf bytes.Buffer
	output := NewAnthropicOutput(&buf, ContextThinking, 0)

	output.OnText("")

	if buf.Len() > 0 {
		t.Errorf("Expected no output for empty text, got: %s", buf.String())
	}
}
