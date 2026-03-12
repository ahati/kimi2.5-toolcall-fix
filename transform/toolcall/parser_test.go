package toolcall

import (
	"strings"
	"testing"
)

type captureHandler struct {
	events []string
}

func (h *captureHandler) OnText(text string) {
	h.events = append(h.events, "text:"+text)
}

func (h *captureHandler) OnToolCallStart(id, name string, index int) {
	h.events = append(h.events, "start:"+id+":"+name)
}

func (h *captureHandler) OnToolCallArgs(args string, index int) {
	h.events = append(h.events, "args:"+args)
}

func (h *captureHandler) OnToolCallEnd(index int) {
	h.events = append(h.events, "end")
}

func TestParser_ContentBeforeAndAfterToolCall(t *testing.T) {
	handler := &captureHandler{}
	parser := NewParser(DefaultTokenSet(), handler)

	parser.Feed("Hello ")
	parser.Feed("<|tool_calls_section_begin|>")
	parser.Feed("<|tool_call_begin|>bash:1<|tool_call_argument_begin|>{\"cmd\":\"ls\"}<|tool_call_end|> ")
	parser.Feed("<|tool_calls_section_end|> end")
	parser.Flush()

	hasHello := false
	hasEnd := false
	for _, e := range handler.events {
		if strings.HasPrefix(e, "text:Hello") {
			hasHello = true
		}
		if strings.Contains(e, " end") {
			hasEnd = true
		}
	}

	if !hasHello {
		t.Error("Expected 'Hello ' text before tool calls")
	}
	if !hasEnd {
		t.Error("Expected ' end' text after tool section")
	}

	startCount := 0
	for _, e := range handler.events {
		if strings.HasPrefix(e, "start:") {
			startCount++
		}
	}
	if startCount != 1 {
		t.Errorf("Expected 1 tool call start, got %d", startCount)
	}
}

func TestParser_ArgumentsSplitAcrossChunks(t *testing.T) {
	handler := &captureHandler{}
	parser := NewParser(DefaultTokenSet(), handler)

	parser.Feed("<|tool_calls_section_begin|><|tool_call_begin|>search:1<|tool_call_argument_begin|>{\"query\": \"")
	parser.Feed("weather in Boston")
	parser.Feed("\"}<|tool_call_end|><|tool_calls_section_end|>")
	parser.Flush()

	found := false
	for _, e := range handler.events {
		if strings.Contains(e, "weather in Boston") {
			found = true
		}
	}
	if !found {
		t.Error("Expected split arguments to be accumulated")
	}
}

func TestParser_MultipleToolCalls(t *testing.T) {
	handler := &captureHandler{}
	parser := NewParser(DefaultTokenSet(), handler)

	parser.Feed("<|tool_calls_section_begin|>")
	parser.Feed("<|tool_call_begin|>bash:1<|tool_call_argument_begin|>{\"command\": \"ls\"}<|tool_call_end|>")
	parser.Feed("<|tool_call_begin|>read:2<|tool_call_argument_begin|>{\"path\": \"/test.txt\"}<|tool_call_end|>")
	parser.Feed("<|tool_calls_section_end|>")
	parser.Flush()

	startCount := 0
	endCount := 0
	for _, e := range handler.events {
		if strings.HasPrefix(e, "start:") {
			startCount++
		}
		if e == "end" {
			endCount++
		}
	}

	if startCount != 2 {
		t.Errorf("Expected 2 tool call starts, got %d", startCount)
	}
	if endCount != 2 {
		t.Errorf("Expected 2 tool call ends, got %d", endCount)
	}
}

func TestParser_FunctionNameParsing(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"bash:1", "bash"},
		{"functions.bash:1", "bash"},
		{"tools.utils.bash:1", "bash"},
		{"bash:", "bash"},
		{"bash", "bash"},
	}

	for _, tc := range tests {
		t.Run(tc.input, func(t *testing.T) {
			handler := &captureHandler{}
			parser := NewParser(DefaultTokenSet(), handler)

			parser.Feed("<|tool_calls_section_begin|><|tool_call_begin|>" + tc.input + "<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>")
			parser.Flush()

			found := false
			for _, e := range handler.events {
				if strings.Contains(e, ":"+tc.expected) {
					found = true
				}
			}
			if !found {
				t.Errorf("Expected function name %q in events %v", tc.expected, handler.events)
			}
		})
	}
}

func TestParser_IncompleteSection(t *testing.T) {
	handler := &captureHandler{}
	parser := NewParser(DefaultTokenSet(), handler)

	parser.Feed("<|tool_calls_section_begin|>")
	parser.Feed("<|tool_call_begin|>bash")
	parser.Flush()

	startCount := 0
	for _, e := range handler.events {
		if strings.HasPrefix(e, "start:") {
			startCount++
		}
	}

	if startCount != 0 {
		t.Errorf("Expected 0 tool call starts for incomplete section, got %d", startCount)
	}
}

func TestParser_MissingEndTokens(t *testing.T) {
	handler := &captureHandler{}
	parser := NewParser(DefaultTokenSet(), handler)

	parser.Feed("<|tool_calls_section_begin|><|tool_call_begin|>bash<|tool_call_argument_begin|>{\"x\":1}")
	parser.Flush()

	for _, e := range handler.events {
		if strings.Contains(e, "{\"x\":1}") {
			return
		}
	}
	t.Error("Expected partial args to be flushed")
}

func TestParser_Reset(t *testing.T) {
	handler := &captureHandler{}
	parser := NewParser(DefaultTokenSet(), handler)

	parser.Feed("<|tool_calls_section_begin|><|tool_call_begin|>tool:1<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>")
	parser.Flush()

	parser.Reset()

	handler.events = nil
	parser.Feed("<|tool_calls_section_begin|><|tool_call_begin|>tool2:1<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>")
	parser.Flush()

	startCount := 0
	for _, e := range handler.events {
		if strings.HasPrefix(e, "start:") {
			startCount++
		}
	}
	if startCount != 1 {
		t.Errorf("Expected 1 tool call after reset, got %d", startCount)
	}
}

func TestParser_NoToolCalls(t *testing.T) {
	handler := &captureHandler{}
	parser := NewParser(DefaultTokenSet(), handler)

	parser.Feed("Just some regular text")
	parser.Flush()

	if len(handler.events) != 1 {
		t.Errorf("Expected 1 text event, got %d", len(handler.events))
	}
	if handler.events[0] != "text:Just some regular text" {
		t.Errorf("Expected text event, got %v", handler.events)
	}
}

func TestParser_EmptyInput(t *testing.T) {
	handler := &captureHandler{}
	parser := NewParser(DefaultTokenSet(), handler)

	parser.Feed("")
	parser.Flush()

	if len(handler.events) != 0 {
		t.Errorf("Expected 0 events for empty input, got %d", len(handler.events))
	}
}

func TestParser_ComplexArguments(t *testing.T) {
	handler := &captureHandler{}
	parser := NewParser(DefaultTokenSet(), handler)

	parser.Feed(`<|tool_calls_section_begin|><|tool_call_begin|>bash:1<|tool_call_argument_begin|>{"command": "echo \"hello\" && ls -la"}<|tool_call_end|><|tool_calls_section_end|>`)
	parser.Flush()

	found := false
	for _, e := range handler.events {
		if strings.Contains(e, "echo") && strings.Contains(e, "ls -la") {
			found = true
		}
	}
	if !found {
		t.Error("Expected complex arguments to be preserved")
	}
}
