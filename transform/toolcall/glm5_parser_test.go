package toolcall

import (
	"strings"
	"testing"
)

func TestGLM5Parser_SimpleToolCall(t *testing.T) {
	p := NewGLM5Parser()
	events := p.Parse(`<tool_call>exec_command<arg_key>cmd</arg_key><arg_value>echo hello</arg_value></tool_call>`)

	if len(events) != 3 {
		t.Fatalf("expected 3 events, got %d", len(events))
	}

	if events[0].Type != EventToolStart {
		t.Errorf("expected EventToolStart, got %v", events[0].Type)
	}
	if events[0].Name != "exec_command" {
		t.Errorf("expected tool name 'exec_command', got %q", events[0].Name)
	}

	if events[1].Type != EventToolArgs {
		t.Errorf("expected EventToolArgs, got %v", events[1].Type)
	}
	if events[1].Args != `{"cmd":"echo hello"}` {
		t.Errorf("expected args '{\"cmd\":\"echo hello\"}', got %q", events[1].Args)
	}

	if events[2].Type != EventToolEnd {
		t.Errorf("expected EventToolEnd, got %v", events[2].Type)
	}
}

func TestGLM5Parser_MultipleArgs(t *testing.T) {
	p := NewGLM5Parser()
	events := p.Parse(`<tool_call>search<arg_key>query</arg_key><arg_value>python docs</arg_value><arg_key>limit</arg_key><arg_value>10</arg_value></tool_call>`)

	if len(events) != 3 {
		t.Fatalf("expected 3 events, got %d", len(events))
	}

	if events[0].Name != "search" {
		t.Errorf("expected tool name 'search', got %q", events[0].Name)
	}

	// Args might be in any order due to map iteration
	args := events[1].Args
	if !(strings.Contains(args, `"query":"python docs"`) && strings.Contains(args, `"limit":"10"`)) {
		t.Errorf("expected args to contain query and limit, got %q", args)
	}
}

func TestGLM5Parser_NoArgs(t *testing.T) {
	p := NewGLM5Parser()
	events := p.Parse(`<tool_call>get_status</tool_call>`)

	if len(events) != 3 {
		t.Fatalf("expected 3 events, got %d", len(events))
	}

	if events[0].Name != "get_status" {
		t.Errorf("expected tool name 'get_status', got %q", events[0].Name)
	}

	if events[1].Args != "{}" {
		t.Errorf("expected empty args '{}', got %q", events[1].Args)
	}
}

func TestGLM5Parser_WithPrefix(t *testing.T) {
	p := NewGLM5Parser()
	events := p.Parse(`I'll run a command now. <tool_call>exec_command<arg_key>cmd</arg_key><arg_value>ls -la</arg_value></tool_call>`)

	if len(events) != 4 { // content + 3 tool events
		t.Fatalf("expected 4 events, got %d", len(events))
	}

	if events[0].Type != EventContent {
		t.Errorf("expected EventContent for prefix, got %v", events[0].Type)
	}
	if events[0].Text != "I'll run a command now. " {
		t.Errorf("expected prefix text, got %q", events[0].Text)
	}

	if events[1].Type != EventToolStart {
		t.Errorf("expected EventToolStart, got %v", events[1].Type)
	}
	if events[1].Name != "exec_command" {
		t.Errorf("expected tool name 'exec_command', got %q", events[1].Name)
	}

	if events[2].Type != EventToolArgs {
		t.Errorf("expected EventToolArgs, got %v", events[2].Type)
	}

	if events[3].Type != EventToolEnd {
		t.Errorf("expected EventToolEnd, got %v", events[3].Type)
	}
}

func TestGLM5Parser_ChunkedInput(t *testing.T) {
	p := NewGLM5Parser()

	// Simulate streaming chunks
	events1 := p.Parse(`<tool_call>exec`)
	if len(events1) != 0 {
		t.Errorf("expected no events from partial input, got %d", len(events1))
	}

	events2 := p.Parse(`_command<arg_key>c`)
	if len(events2) != 0 {
		t.Errorf("expected no events from partial input, got %d", len(events2))
	}

	events3 := p.Parse(`md</arg_key><arg_value>echo hello</arg_value></tool_call>`)
	if len(events3) != 3 {
		t.Fatalf("expected 3 events, got %d", len(events3))
	}

	if events3[0].Name != "exec_command" {
		t.Errorf("expected tool name 'exec_command', got %q", events3[0].Name)
	}
}

func TestGLM5Parser_MultipleToolCalls(t *testing.T) {
	p := NewGLM5Parser()
	events := p.Parse(`<tool_call>search<arg_key>q</arg_key><arg_value>test</arg_value></tool_call> and <tool_call>exec<arg_key>c</arg_key><arg_value>run</arg_value></tool_call>`)

	if len(events) != 7 { // content + 3 events + content + 3 events
		t.Fatalf("expected 7 events, got %d", len(events))
	}

	// First tool call
	if events[0].Type != EventToolStart || events[0].Name != "search" {
		t.Errorf("expected first tool 'search', got %v / %q", events[0].Type, events[0].Name)
	}

	// Middle content
	if events[3].Type != EventContent {
		t.Errorf("expected content between tools, got %v", events[3].Type)
	}

	// Second tool call
	if events[4].Type != EventToolStart || events[4].Name != "exec" {
		t.Errorf("expected second tool 'exec', got %v / %q", events[4].Type, events[4].Name)
	}
}

func TestGLM5Parser_EmptyInput(t *testing.T) {
	p := NewGLM5Parser()
	events := p.Parse("")

	if len(events) != 0 {
		t.Errorf("expected no events for empty input, got %d", len(events))
	}
}

func TestGLM5Parser_NoToolCall(t *testing.T) {
	p := NewGLM5Parser()
	events := p.Parse("This is just regular text without any tool calls.")

	if len(events) != 1 {
		t.Fatalf("expected 1 event, got %d", len(events))
	}

	if events[0].Type != EventContent {
		t.Errorf("expected EventContent, got %v", events[0].Type)
	}
	if events[0].Text != "This is just regular text without any tool calls." {
		t.Errorf("expected full text, got %q", events[0].Text)
	}
}

func TestGLM5Parser_StateTransitions(t *testing.T) {
	p := NewGLM5Parser()

	if p.State() != glm5StateIdle {
		t.Errorf("expected initial state idle, got %v", p.State())
	}

	p.Parse(`<tool_call>`)
	if p.State() != glm5StateInToolCall {
		t.Errorf("expected state inToolCall after <tool_call>, got %v", p.State())
	}

	p.Parse(`func<arg_key>`)
	if p.State() != glm5StateReadingArgKey {
		t.Errorf("expected state readingArgKey after <arg_key>, got %v", p.State())
	}

	p.Parse(`key</arg_key><arg_value>`)
	if p.State() != glm5StateReadingArgValue {
		t.Errorf("expected state readingArgValue after <arg_value>, got %v", p.State())
	}

	p.Parse(`value</arg_value></tool_call>`)
	if p.State() != glm5StateIdle {
		t.Errorf("expected state idle after </tool_call>, got %v", p.State())
	}
}

func TestGLM5Parser_Reset(t *testing.T) {
	p := NewGLM5Parser()
	p.Parse(`<tool_call>func<arg_key>k</arg_key><arg_value>v`)

	if p.IsIdle() {
		t.Error("expected parser to not be idle mid-parse")
	}

	p.Reset()

	if !p.IsIdle() {
		t.Error("expected parser to be idle after reset")
	}

	// Should be able to parse new input
	events := p.Parse(`<tool_call>new<arg_key>x</arg_key><arg_value>y</arg_value></tool_call>`)
	if len(events) != 3 {
		t.Fatalf("expected 3 events after reset, got %d", len(events))
	}
}
