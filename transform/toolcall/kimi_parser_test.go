package toolcall

import (
	"fmt"
	"strings"
	"testing"
)

func TestParser_SimpleContent(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected []Event
	}{
		{
			name:     "empty input",
			input:    "",
			expected: nil,
		},
		{
			name:  "simple text",
			input: "Hello, world!",
			expected: []Event{
				{Type: EventContent, Text: "Hello, world!"},
			},
		},
		{
			name:  "text with newlines",
			input: "Line 1\nLine 2\nLine 3",
			expected: []Event{
				{Type: EventContent, Text: "Line 1\nLine 2\nLine 3"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := NewParser(DefaultTokens)
			events := p.Parse(tt.input)
			if len(events) != len(tt.expected) {
				t.Errorf("expected %d events, got %d", len(tt.expected), len(events))
				return
			}
			for i, e := range events {
				if e.Type != tt.expected[i].Type {
					t.Errorf("event %d: expected type %v, got %v", i, tt.expected[i].Type, e.Type)
				}
				if e.Text != tt.expected[i].Text {
					t.Errorf("event %d: expected text %q, got %q", i, tt.expected[i].Text, e.Text)
				}
			}
		})
	}
}

func TestParser_SingleToolCall(t *testing.T) {
	p := NewParser(DefaultTokens)

	events := p.Parse("Before<|tool_calls_section_begin|>")
	events = append(events, p.Parse("<|tool_call_begin|>bash<|tool_call_argument_begin|>")...)
	events = append(events, p.Parse(`{"command":"ls"}<|tool_call_end|>`)...)
	events = append(events, p.Parse("<|tool_calls_section_end|>")...)
	events = append(events, p.Parse("After")...)

	expected := []struct {
		Type EventType
		Text string
		Name string
		Args string
	}{
		{Type: EventContent, Text: "Before"},
		{Type: EventToolStart, Name: "bash"},
		{Type: EventToolArgs, Args: `{"command":"ls"}`},
		{Type: EventToolEnd},
		{Type: EventSectionEnd},
		{Type: EventContent, Text: "After"},
	}

	if len(events) != len(expected) {
		t.Fatalf("expected %d events, got %d\n%+v", len(expected), len(events), events)
	}

	for i, e := range events {
		if e.Type != expected[i].Type {
			t.Errorf("event %d: expected type %v, got %v", i, expected[i].Type, e.Type)
		}
		if expected[i].Text != "" && e.Text != expected[i].Text {
			t.Errorf("event %d: expected text %q, got %q", i, expected[i].Text, e.Text)
		}
		if expected[i].Name != "" && e.Name != expected[i].Name {
			t.Errorf("event %d: expected name %q, got %q", i, expected[i].Name, e.Name)
		}
		if expected[i].Args != "" && e.Args != expected[i].Args {
			t.Errorf("event %d: expected args %q, got %q", i, expected[i].Args, e.Args)
		}
	}
}

func TestParser_MultipleToolCalls(t *testing.T) {
	p := NewParser(DefaultTokens)

	events := p.Parse("<|tool_calls_section_begin|>")
	events = append(events, p.Parse("<|tool_call_begin|>bash<|tool_call_argument_begin|>")...)
	events = append(events, p.Parse(`{"command":"ls"}<|tool_call_end|>`)...)
	events = append(events, p.Parse("<|tool_call_begin|>read<|tool_call_argument_begin|>")...)
	events = append(events, p.Parse(`{"file":"test.txt"}<|tool_call_end|>`)...)
	events = append(events, p.Parse("<|tool_calls_section_end|>")...)

	expected := []struct {
		Type  EventType
		Name  string
		Args  string
		Index int
	}{
		{Type: EventToolStart, Name: "bash", Index: 0},
		{Type: EventToolArgs, Args: `{"command":"ls"}`, Index: 0},
		{Type: EventToolEnd, Index: 0},
		{Type: EventToolStart, Name: "read", Index: 1},
		{Type: EventToolArgs, Args: `{"file":"test.txt"}`, Index: 1},
		{Type: EventToolEnd, Index: 1},
		{Type: EventSectionEnd},
	}

	if len(events) != len(expected) {
		t.Fatalf("expected %d events, got %d\n%+v", len(expected), len(events), events)
	}

	for i, e := range events {
		if e.Type != expected[i].Type {
			t.Errorf("event %d: expected type %v, got %v", i, expected[i].Type, e.Type)
		}
		if expected[i].Name != "" && e.Name != expected[i].Name {
			t.Errorf("event %d: expected name %q, got %q", i, expected[i].Name, e.Name)
		}
		if expected[i].Args != "" && e.Args != expected[i].Args {
			t.Errorf("event %d: expected args %q, got %q", i, expected[i].Args, e.Args)
		}
	}
}

func TestParser_ChunkedInput(t *testing.T) {
	p := NewParser(DefaultTokens)

	events := p.Parse("Start<|tool_calls_section_begin|>")
	events = append(events, p.Parse("<|tool_call_begin|>bash<|tool_call_argument_begin|>")...)
	events = append(events, p.Parse(`{"command":`)...)
	events = append(events, p.Parse(`"ls"}<|tool_call_end|>`)...)
	events = append(events, p.Parse("<|tool_calls_section_end|>")...)
	events = append(events, p.Parse("End")...)

	expected := []EventType{
		EventContent,
		EventToolStart,
		EventToolArgs,
		EventToolArgs,
		EventToolEnd,
		EventSectionEnd,
		EventContent,
	}

	if len(events) != len(expected) {
		t.Fatalf("expected %d events, got %d\n%+v", len(expected), len(events), events)
	}

	for i, e := range events {
		if e.Type != expected[i] {
			t.Errorf("event %d: expected type %v, got %v", i, expected[i], e.Type)
		}
	}
}

func TestParser_IncompleteToolCall(t *testing.T) {
	tests := []struct {
		name      string
		inputs    []string
		wantState state
	}{
		{
			name:      "only section begin",
			inputs:    []string{"<|tool_calls_section_begin|>"},
			wantState: stateInSection,
		},
		{
			name:      "section and call begin",
			inputs:    []string{"<|tool_calls_section_begin|>", "<|tool_call_begin|>"},
			wantState: stateReadingID,
		},
		{
			name:      "missing arg begin",
			inputs:    []string{"<|tool_calls_section_begin|>", "<|tool_call_begin|>", "bash"},
			wantState: stateReadingID,
		},
		{
			name:      "missing call end",
			inputs:    []string{"<|tool_calls_section_begin|>", "<|tool_call_begin|>", "bash<|tool_call_argument_begin|>", "{\"cmd\":\"ls\"}"},
			wantState: stateReadingArgs,
		},
		{
			name:      "missing section end",
			inputs:    []string{"<|tool_calls_section_begin|>", "<|tool_call_begin|>", "bash<|tool_call_argument_begin|>", "{\"cmd\":\"ls\"}<|tool_call_end|>"},
			wantState: stateInSection,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := NewParser(DefaultTokens)
			for _, input := range tt.inputs {
				p.Parse(input)
			}
			if p.State() != tt.wantState {
				t.Errorf("expected state %v, got %v", tt.wantState, p.State())
			}
		})
	}
}

func TestParser_StateTransitions(t *testing.T) {
	t.Run("idle to in_section", func(t *testing.T) {
		p := NewParser(DefaultTokens)
		if p.State() != stateIdle {
			t.Errorf("expected initial state to be idle, got %v", p.State())
		}
		p.Parse("<|tool_calls_section_begin|>")
		if p.State() != stateInSection {
			t.Errorf("expected state to be in_section, got %v", p.State())
		}
	})

	t.Run("in_section to reading_id", func(t *testing.T) {
		p := NewParser(DefaultTokens)
		p.Parse("<|tool_calls_section_begin|>")
		p.Parse("<|tool_call_begin|>")
		if p.State() != stateReadingID {
			t.Errorf("expected state to be reading_id, got %v", p.State())
		}
	})

	t.Run("reading_id to reading_args", func(t *testing.T) {
		p := NewParser(DefaultTokens)
		p.Parse("<|tool_calls_section_begin|>")
		p.Parse("<|tool_call_begin|>")
		p.Parse("bash<|tool_call_argument_begin|>")
		if p.State() != stateReadingArgs {
			t.Errorf("expected state to be reading_args, got %v", p.State())
		}
	})

	t.Run("reading_args back to in_section", func(t *testing.T) {
		p := NewParser(DefaultTokens)
		p.Parse("<|tool_calls_section_begin|>")
		p.Parse("<|tool_call_begin|>")
		p.Parse("bash<|tool_call_argument_begin|>")
		p.Parse("{}<|tool_call_end|>")
		if p.State() != stateInSection {
			t.Errorf("expected state to be in_section, got %v", p.State())
		}
	})

	t.Run("in_section to trailing", func(t *testing.T) {
		p := NewParser(DefaultTokens)
		p.Parse("<|tool_calls_section_begin|>")
		p.Parse("<|tool_calls_section_end|>")
		if p.State() != stateTrailing {
			t.Errorf("expected state to be trailing, got %v", p.State())
		}
	})

	t.Run("trailing to in_section", func(t *testing.T) {
		p := NewParser(DefaultTokens)
		p.Parse("<|tool_calls_section_begin|>")
		p.Parse("<|tool_calls_section_end|>")
		p.Parse("text<|tool_calls_section_begin|>")
		if p.State() != stateInSection {
			t.Errorf("expected state to be in_section, got %v", p.State())
		}
	})
}

func TestParser_ToolCallIDParsing(t *testing.T) {
	tests := []struct {
		name         string
		toolID       string
		wantIDPrefix string
		wantName     string
	}{
		{
			name:         "simple function name",
			toolID:       "bash",
			wantIDPrefix: "call_",
			wantName:     "bash",
		},
		{
			name:         "prefixed call_ ID",
			toolID:       "call_abc123",
			wantIDPrefix: "call_abc123",
			wantName:     "call_abc123",
		},
		{
			name:         "prefixed toolu_ ID",
			toolID:       "toolu_xyz789",
			wantIDPrefix: "toolu_xyz789",
			wantName:     "toolu_xyz789",
		},
		{
			name:         "function with dot notation",
			toolID:       "tools.bash",
			wantIDPrefix: "call_",
			wantName:     "bash",
		},
		{
			name:         "function with colon",
			toolID:       "bash:123",
			wantIDPrefix: "call_",
			wantName:     "bash",
		},
		{
			name:         "function with dot and colon",
			toolID:       "tools.bash:123",
			wantIDPrefix: "call_",
			wantName:     "bash",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := NewParser(DefaultTokens)
			id, name := p.parseToolCallID(tt.toolID)

			if !strings.HasPrefix(id, tt.wantIDPrefix) && id != tt.wantIDPrefix {
				t.Errorf("expected ID to have prefix %q, got %q", tt.wantIDPrefix, id)
			}
			if name != tt.wantName {
				t.Errorf("expected name %q, got %q", tt.wantName, name)
			}
		})
	}
}

func TestParser_ExtractFunctionName(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"bash", "bash"},
		{"tools.bash", "bash"},
		{"tools.nested.bash", "nested.bash"},
		{"bash:123", "bash"},
		{"tools.bash:123", "bash"},
		{"  bash  ", "bash"},
		{"  tools.bash:123  ", "bash"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			p := NewParser(DefaultTokens)
			result := p.extractFunctionName(tt.input)
			if result != tt.expected {
				t.Errorf("expected %q, got %q", tt.expected, result)
			}
		})
	}
}

func TestParser_Reset(t *testing.T) {
	p := NewParser(DefaultTokens)
	p.Parse("<|tool_calls_section_begin|><|tool_call_begin|>bash<|tool_call_argument_begin|>{}")

	if p.State() == stateIdle {
		t.Error("expected parser to be in non-idle state before reset")
	}

	p.Reset()

	if p.State() != stateIdle {
		t.Errorf("expected state to be idle after reset, got %v", p.State())
	}
	if p.Buffer() != "" {
		t.Errorf("expected buffer to be empty after reset, got %q", p.Buffer())
	}
	if p.toolIndex != 0 {
		t.Errorf("expected toolIndex to be 0 after reset, got %d", p.toolIndex)
	}
}

func TestParser_Buffer(t *testing.T) {
	p := NewParser(DefaultTokens)
	p.Parse("<|tool_calls_section_begin|>")

	if p.Buffer() != "" {
		t.Errorf("expected buffer to be empty after section begin, got %q", p.Buffer())
	}

	p.Parse("partial")
	if p.Buffer() != "partial" {
		t.Errorf("expected buffer to be %q, got %q", "partial", p.Buffer())
	}

	p.Parse("_data")
	if p.Buffer() != "partial_data" {
		t.Errorf("expected buffer to be %q, got %q", "partial_data", p.Buffer())
	}
}

func TestParser_EmptySection(t *testing.T) {
	p := NewParser(DefaultTokens)
	p.Parse("<|tool_calls_section_begin|>")
	events := p.Parse("<|tool_calls_section_end|>")

	expected := []EventType{EventSectionEnd}
	if len(events) != len(expected) {
		t.Fatalf("expected %d events, got %d", len(expected), len(events))
	}

	for i, e := range events {
		if e.Type != expected[i] {
			t.Errorf("event %d: expected type %v, got %v", i, expected[i], e.Type)
		}
	}
}

func TestParser_TrailingContent(t *testing.T) {
	p := NewParser(DefaultTokens)
	p.Parse("<|tool_calls_section_begin|>")
	events := p.Parse("<|tool_calls_section_end|>")
	events = append(events, p.Parse("trailing text")...)

	expected := []struct {
		Type EventType
		Text string
	}{
		{Type: EventSectionEnd},
		{Type: EventContent, Text: "trailing text"},
	}

	if len(events) != len(expected) {
		t.Fatalf("expected %d events, got %d", len(expected), len(events))
	}

	for i, e := range events {
		if e.Type != expected[i].Type {
			t.Errorf("event %d: expected type %v, got %v", i, expected[i].Type, e.Type)
		}
		if expected[i].Text != "" && e.Text != expected[i].Text {
			t.Errorf("event %d: expected text %q, got %q", i, expected[i].Text, e.Text)
		}
	}
}

func TestParser_MultipleSections(t *testing.T) {
	p := NewParser(DefaultTokens)

	events := p.Parse("text1<|tool_calls_section_begin|>")
	events = append(events, p.Parse("<|tool_call_begin|>bash<|tool_call_argument_begin|>{}<|tool_call_end|>")...)
	events = append(events, p.Parse("<|tool_calls_section_end|>")...)
	events = append(events, p.Parse("text2<|tool_calls_section_begin|>")...)
	events = append(events, p.Parse("<|tool_call_begin|>read<|tool_call_argument_begin|>{}<|tool_call_end|>")...)
	events = append(events, p.Parse("<|tool_calls_section_end|>")...)
	events = append(events, p.Parse("text3")...)

	expected := []struct {
		Type EventType
		Text string
		Name string
	}{
		{Type: EventContent, Text: "text1"},
		{Type: EventToolStart, Name: "bash"},
		{Type: EventToolArgs},
		{Type: EventToolEnd},
		{Type: EventSectionEnd},
		{Type: EventContent, Text: "text2"},
		{Type: EventToolStart, Name: "read"},
		{Type: EventToolArgs},
		{Type: EventToolEnd},
		{Type: EventSectionEnd},
		{Type: EventContent, Text: "text3"},
	}

	if len(events) != len(expected) {
		t.Fatalf("expected %d events, got %d\n%+v", len(expected), len(events), events)
	}

	for i, e := range events {
		if e.Type != expected[i].Type {
			t.Errorf("event %d: expected type %v, got %v", i, expected[i].Type, e.Type)
		}
		if expected[i].Text != "" && e.Text != expected[i].Text {
			t.Errorf("event %d: expected text %q, got %q", i, expected[i].Text, e.Text)
		}
		if expected[i].Name != "" && e.Name != expected[i].Name {
			t.Errorf("event %d: expected name %q, got %q", i, expected[i].Name, e.Name)
		}
	}
}

func TestParser_EmptyToolArgs(t *testing.T) {
	p := NewParser(DefaultTokens)
	events := p.Parse("<|tool_calls_section_begin|>")
	events = append(events, p.Parse("<|tool_call_begin|>bash<|tool_call_argument_begin|>")...)
	events = append(events, p.Parse("<|tool_call_end|>")...)
	events = append(events, p.Parse("<|tool_calls_section_end|>")...)

	expected := []EventType{
		EventToolStart,
		EventToolEnd,
		EventSectionEnd,
	}

	if len(events) != len(expected) {
		t.Fatalf("expected %d events, got %d\n%+v", len(expected), len(events), events)
	}

	for i, e := range events {
		if e.Type != expected[i] {
			t.Errorf("event %d: expected type %v, got %v", i, expected[i], e.Type)
		}
	}
}

func TestParser_WhitespaceInToolID(t *testing.T) {
	p := NewParser(DefaultTokens)
	events := p.Parse("<|tool_calls_section_begin|>")
	events = append(events, p.Parse("<|tool_call_begin|>  bash  <|tool_call_argument_begin|>{}<|tool_call_end|>")...)
	events = append(events, p.Parse("<|tool_calls_section_end|>")...)

	if len(events) < 1 {
		t.Fatal("expected at least one event")
	}

	if events[0].Type != EventToolStart {
		t.Errorf("expected first event to be ToolStart, got %v", events[0].Type)
	}

	if events[0].Name != "bash" {
		t.Errorf("expected name to be %q, got %q", "bash", events[0].Name)
	}
}

func TestParser_CustomTokens(t *testing.T) {
	customTokens := Tokens{
		SectionBegin: "[SECTION]",
		CallBegin:    "[CALL]",
		ArgBegin:     "[ARGS]",
		CallEnd:      "[/CALL]",
		SectionEnd:   "[/SECTION]",
	}

	p := NewParser(customTokens)
	events := p.Parse("text[SECTION]")
	events = append(events, p.Parse("[CALL]bash[ARGS]{}[/CALL]")...)
	events = append(events, p.Parse("[/SECTION]")...)
	events = append(events, p.Parse("trailing")...)

	expected := []struct {
		Type EventType
		Text string
		Name string
	}{
		{Type: EventContent, Text: "text"},
		{Type: EventToolStart, Name: "bash"},
		{Type: EventToolArgs},
		{Type: EventToolEnd},
		{Type: EventSectionEnd},
		{Type: EventContent, Text: "trailing"},
	}

	if len(events) != len(expected) {
		t.Fatalf("expected %d events, got %d\n%+v", len(expected), len(events), events)
	}

	for i, e := range events {
		if e.Type != expected[i].Type {
			t.Errorf("event %d: expected type %v, got %v", i, expected[i].Type, e.Type)
		}
	}
}

func TestParser_ToolIndexIncrement(t *testing.T) {
	p := NewParser(DefaultTokens)
	events := p.Parse("<|tool_calls_section_begin|>")
	events = append(events, p.Parse("<|tool_call_begin|>a<|tool_call_argument_begin|>{}<|tool_call_end|>")...)
	events = append(events, p.Parse("<|tool_call_begin|>b<|tool_call_argument_begin|>{}<|tool_call_end|>")...)
	events = append(events, p.Parse("<|tool_call_begin|>c<|tool_call_argument_begin|>{}<|tool_call_end|>")...)
	events = append(events, p.Parse("<|tool_calls_section_end|>")...)

	toolStarts := 0
	for _, e := range events {
		if e.Type == EventToolStart {
			if e.Index != toolStarts {
				t.Errorf("expected tool index %d, got %d", toolStarts, e.Index)
			}
			toolStarts++
		}
	}

	if toolStarts != 3 {
		t.Errorf("expected 3 tool starts, got %d", toolStarts)
	}
}

// TestParser_InvalidToolCallID tests that invalid tool call IDs are rejected
// and emitted as regular content instead.
func TestParser_InvalidToolCallID(t *testing.T) {
	p := NewParser(DefaultTokens)

	// Test with a very long ID (should be rejected)
	longID := strings.Repeat("a", 300)
	input := fmt.Sprintf("<|tool_calls_section_begin|><|tool_call_begin|>%s<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>", longID)
	events := p.Parse(input)

	// Should emit content, not tool call
	if len(events) != 2 {
		t.Fatalf("expected 2 events (content + section end), got %d: %+v", len(events), events)
	}
	if events[0].Type != EventContent {
		t.Errorf("expected EventContent for invalid ID, got %v", events[0].Type)
	}
	if events[1].Type != EventSectionEnd {
		t.Errorf("expected EventSectionEnd, got %v", events[1].Type)
	}

	// Test with ID containing newlines (should be rejected)
	p.Reset()
	input = "<|tool_calls_section_begin|><|tool_call_begin|>func\nwith\nnewlines<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>"
	events = p.Parse(input)

	if len(events) != 2 {
		t.Fatalf("expected 2 events for newline ID, got %d: %+v", len(events), events)
	}
	if events[0].Type != EventContent {
		t.Errorf("expected EventContent for newline ID, got %v", events[0].Type)
	}

	// Test with ID containing markdown (should be rejected)
	p.Reset()
	input = "<|tool_calls_section_begin|><|tool_call_begin|>**bold**<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>"
	events = p.Parse(input)

	if len(events) != 2 {
		t.Fatalf("expected 2 events for markdown ID, got %d: %+v", len(events), events)
	}
	if events[0].Type != EventContent {
		t.Errorf("expected EventContent for markdown ID, got %v", events[0].Type)
	}

	// Test that valid IDs still work
	p.Reset()
	input = "<|tool_calls_section_begin|><|tool_call_begin|>bash<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>"
	events = p.Parse(input)

	if len(events) != 4 {
		t.Fatalf("expected 4 events for valid ID, got %d: %+v", len(events), events)
	}
	if events[0].Type != EventToolStart {
		t.Errorf("expected EventToolStart for valid ID, got %v", events[0].Type)
	}
	if events[0].Name != "bash" {
		t.Errorf("expected name 'bash', got %s", events[0].Name)
	}
}
