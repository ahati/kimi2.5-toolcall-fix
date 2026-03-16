package toolcall

import (
	"encoding/json"
	"strings"
	"testing"

	"ai-proxy/types"
)

func TestAnthropicFormatter_FormatContent(t *testing.T) {
	f := NewAnthropicFormatter("msg-123", "claude-3")
	f.IncrementBlockIndex()

	output := f.FormatContent("Hello, world!")

	if !strings.HasPrefix(string(output), "event: content_block_delta\n") {
		t.Error("expected output to start with 'event: content_block_delta'")
	}
	if !strings.HasSuffix(string(output), "\n\n") {
		t.Error("expected output to end with '\\n\\n'")
	}

	dataLine := ""
	lines := strings.Split(string(output), "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "data: ") {
			dataLine = strings.TrimPrefix(line, "data: ")
			break
		}
	}

	var event types.Event
	if err := json.Unmarshal([]byte(dataLine), &event); err != nil {
		t.Fatalf("failed to parse JSON: %v", err)
	}

	if event.Type != "content_block_delta" {
		t.Errorf("expected type 'content_block_delta', got %q", event.Type)
	}
	if event.Index == nil || *event.Index != 0 {
		t.Errorf("expected index 0, got %v", event.Index)
	}

	var delta types.TextDelta
	if err := json.Unmarshal(event.Delta, &delta); err != nil {
		t.Fatalf("failed to parse delta: %v", err)
	}

	if delta.Type != "text_delta" {
		t.Errorf("expected delta type 'text_delta', got %q", delta.Type)
	}
	if delta.Text != "Hello, world!" {
		t.Errorf("expected text 'Hello, world!', got %q", delta.Text)
	}
}

func TestAnthropicFormatter_FormatToolStart(t *testing.T) {
	f := NewAnthropicFormatter("msg-456", "claude-3")
	output := f.FormatToolStart("toolu_abc", "bash", 0)

	if !strings.HasPrefix(string(output), "event: content_block_start\n") {
		t.Error("expected output to start with 'event: content_block_start'")
	}

	dataLine := ""
	lines := strings.Split(string(output), "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "data: ") {
			dataLine = strings.TrimPrefix(line, "data: ")
			break
		}
	}

	var event types.Event
	if err := json.Unmarshal([]byte(dataLine), &event); err != nil {
		t.Fatalf("failed to parse JSON: %v", err)
	}

	if event.Type != "content_block_start" {
		t.Errorf("expected type 'content_block_start', got %q", event.Type)
	}

	var block types.ContentBlock
	if err := json.Unmarshal(event.Delta, &block); err != nil {
		t.Fatalf("failed to parse block: %v", err)
	}

	if block.Type != "tool_use" {
		t.Errorf("expected block type 'tool_use', got %q", block.Type)
	}
	if block.ID != "toolu_abc" {
		t.Errorf("expected ID 'toolu_abc', got %q", block.ID)
	}
	if block.Name != "bash" {
		t.Errorf("expected Name 'bash', got %q", block.Name)
	}

	if !f.ToolsEmitted() {
		t.Error("expected toolsEmitted to be true after FormatToolStart")
	}
}

func TestAnthropicFormatter_FormatToolArgs(t *testing.T) {
	f := NewAnthropicFormatter("msg-789", "claude-3")
	f.FormatToolStart("toolu_xyz", "bash", 0)

	output := f.FormatToolArgs(`{"command":"ls"}`, 0)

	if !strings.HasPrefix(string(output), "event: content_block_delta\n") {
		t.Error("expected output to start with 'event: content_block_delta'")
	}

	dataLine := ""
	lines := strings.Split(string(output), "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "data: ") {
			dataLine = strings.TrimPrefix(line, "data: ")
			break
		}
	}

	var event types.Event
	if err := json.Unmarshal([]byte(dataLine), &event); err != nil {
		t.Fatalf("failed to parse JSON: %v", err)
	}

	var delta types.InputJSONDelta
	if err := json.Unmarshal(event.Delta, &delta); err != nil {
		t.Fatalf("failed to parse delta: %v", err)
	}

	if delta.Type != "input_json_delta" {
		t.Errorf("expected delta type 'input_json_delta', got %q", delta.Type)
	}
	if delta.PartialJSON != `{"command":"ls"}` {
		t.Errorf("expected PartialJSON %q, got %q", `{"command":"ls"}`, delta.PartialJSON)
	}
}

func TestAnthropicFormatter_FormatToolEnd(t *testing.T) {
	f := NewAnthropicFormatter("msg-123", "claude-3")
	f.FormatToolStart("toolu_abc", "bash", 0)

	output := f.FormatToolEnd(0)

	if !strings.HasPrefix(string(output), "event: content_block_stop\n") {
		t.Error("expected output to start with 'event: content_block_stop'")
	}

	dataLine := ""
	lines := strings.Split(string(output), "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "data: ") {
			dataLine = strings.TrimPrefix(line, "data: ")
			break
		}
	}

	var event types.Event
	if err := json.Unmarshal([]byte(dataLine), &event); err != nil {
		t.Fatalf("failed to parse JSON: %v", err)
	}

	if event.Type != "content_block_stop" {
		t.Errorf("expected type 'content_block_stop', got %q", event.Type)
	}
}

func TestAnthropicFormatter_FormatSectionEnd(t *testing.T) {
	f := NewAnthropicFormatter("msg-123", "claude-3")
	output := f.FormatSectionEnd()

	if output != nil {
		t.Errorf("expected nil output for FormatSectionEnd, got %v", output)
	}
}

func TestAnthropicFormatter_FormatDone(t *testing.T) {
	f := NewAnthropicFormatter("msg-123", "claude-3")
	output := f.FormatDone()

	expected := "event: message_stop\ndata: {}\n\n"
	if string(output) != expected {
		t.Errorf("expected %q, got %q", expected, string(output))
	}
}

func TestAnthropicFormatter_BlockIndexIncrement(t *testing.T) {
	f := NewAnthropicFormatter("msg-123", "claude-3")

	if f.BlockIndex() != -1 {
		t.Errorf("expected initial block index -1, got %d", f.BlockIndex())
	}

	f.FormatToolStart("call_0", "bash", 0)
	if f.BlockIndex() != 0 {
		t.Errorf("expected block index 0 after first tool, got %d", f.BlockIndex())
	}

	f.FormatToolStart("call_1", "read", 1)
	if f.BlockIndex() != 1 {
		t.Errorf("expected block index 1 after second tool, got %d", f.BlockIndex())
	}
}

func TestAnthropicFormatter_MultipleToolCalls(t *testing.T) {
	f := NewAnthropicFormatter("msg-multi", "claude-3")

	output0 := f.FormatToolStart("call_0", "bash", 0)
	output1 := f.FormatToolStart("call_1", "read", 1)

	var event0, event1 types.Event
	dataLine0 := extractDataLine(string(output0))
	dataLine1 := extractDataLine(string(output1))

	json.Unmarshal([]byte(dataLine0), &event0)
	json.Unmarshal([]byte(dataLine1), &event1)

	if event0.Index == nil || *event0.Index != 0 {
		t.Errorf("expected index 0 for first tool, got %v", event0.Index)
	}
	if event1.Index == nil || *event1.Index != 1 {
		t.Errorf("expected index 1 for second tool, got %v", event1.Index)
	}
}

func TestAnthropicFormatter_SetMessageID(t *testing.T) {
	f := NewAnthropicFormatter("old-id", "claude-3")
	f.SetMessageID("new-id")

	if f.messageID != "new-id" {
		t.Errorf("expected messageID 'new-id', got %q", f.messageID)
	}
}

func TestAnthropicFormatter_SetModel(t *testing.T) {
	f := NewAnthropicFormatter("msg-id", "old-model")
	f.SetModel("new-model")

	if f.model != "new-model" {
		t.Errorf("expected model 'new-model', got %q", f.model)
	}
}

func TestAnthropicFormatter_SetBlockIndex(t *testing.T) {
	f := NewAnthropicFormatter("msg-123", "claude-3")

	f.SetBlockIndex(5)
	if f.BlockIndex() != 5 {
		t.Errorf("expected block index 5, got %d", f.BlockIndex())
	}
}

func TestAnthropicFormatter_IncrementBlockIndex(t *testing.T) {
	f := NewAnthropicFormatter("msg-123", "claude-3")

	f.IncrementBlockIndex()
	if f.BlockIndex() != 0 {
		t.Errorf("expected block index 0, got %d", f.BlockIndex())
	}

	f.IncrementBlockIndex()
	if f.BlockIndex() != 1 {
		t.Errorf("expected block index 1, got %d", f.BlockIndex())
	}
}

func TestAnthropicFormatter_ToolsEmitted(t *testing.T) {
	f := NewAnthropicFormatter("msg-123", "claude-3")

	if f.ToolsEmitted() {
		t.Error("expected toolsEmitted to be false initially")
	}

	f.FormatToolStart("call_0", "bash", 0)

	if !f.ToolsEmitted() {
		t.Error("expected toolsEmitted to be true after tool start")
	}
}

func TestAnthropicFormatter_EmptyContent(t *testing.T) {
	f := NewAnthropicFormatter("msg-123", "claude-3")
	f.IncrementBlockIndex()

	output := f.FormatContent("")

	dataLine := extractDataLine(string(output))
	var event types.Event
	json.Unmarshal([]byte(dataLine), &event)

	var delta types.TextDelta
	json.Unmarshal(event.Delta, &delta)

	if delta.Text != "" {
		t.Errorf("expected empty text, got %q", delta.Text)
	}
}

func TestAnthropicFormatter_SpecialCharacters(t *testing.T) {
	f := NewAnthropicFormatter("msg-123", "claude-3")
	f.IncrementBlockIndex()

	specialContent := "Hello\nWorld\t\"Quotes\"\\Backslash"
	output := f.FormatContent(specialContent)

	dataLine := extractDataLine(string(output))
	var event types.Event
	json.Unmarshal([]byte(dataLine), &event)

	var delta types.TextDelta
	json.Unmarshal(event.Delta, &delta)

	if delta.Text != specialContent {
		t.Errorf("expected text %q, got %q", specialContent, delta.Text)
	}
}

func TestAnthropicFormatter_UnicodeContent(t *testing.T) {
	f := NewAnthropicFormatter("msg-123", "claude-3")
	f.IncrementBlockIndex()

	unicodeContent := "Hello 世界 🌍"
	output := f.FormatContent(unicodeContent)

	dataLine := extractDataLine(string(output))
	var event types.Event
	json.Unmarshal([]byte(dataLine), &event)

	var delta types.TextDelta
	json.Unmarshal(event.Delta, &delta)

	if delta.Text != unicodeContent {
		t.Errorf("expected text %q, got %q", unicodeContent, delta.Text)
	}
}

func TestAnthropicFormatter_EmptyArgs(t *testing.T) {
	f := NewAnthropicFormatter("msg-123", "claude-3")
	f.FormatToolStart("call_0", "bash", 0)

	output := f.FormatToolArgs("", 0)

	dataLine := extractDataLine(string(output))
	var event types.Event
	json.Unmarshal([]byte(dataLine), &event)

	var delta types.InputJSONDelta
	json.Unmarshal(event.Delta, &delta)

	if delta.PartialJSON != "" {
		t.Errorf("expected empty PartialJSON, got %q", delta.PartialJSON)
	}
}

func TestAnthropicFormatter_PartialJSON(t *testing.T) {
	f := NewAnthropicFormatter("msg-123", "claude-3")
	f.FormatToolStart("call_0", "bash", 0)

	partial := `{"command": "ls`
	output := f.FormatToolArgs(partial, 0)

	dataLine := extractDataLine(string(output))
	var event types.Event
	json.Unmarshal([]byte(dataLine), &event)

	var delta types.InputJSONDelta
	json.Unmarshal(event.Delta, &delta)

	if delta.PartialJSON != partial {
		t.Errorf("expected PartialJSON %q, got %q", partial, delta.PartialJSON)
	}
}

func extractDataLine(output string) string {
	lines := strings.Split(output, "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "data: ") {
			return strings.TrimPrefix(line, "data: ")
		}
	}
	return ""
}

// TestAnthropicFormatter_UnknownContentBlockType tests handling of unknown content block types.
// Category D3: Unknown content block type (MEDIUM)
func TestAnthropicFormatter_UnknownContentBlockType(t *testing.T) {
	// The AnthropicFormatter doesn't directly handle content block types from input,
	// but it formats output events. This test verifies it handles edge cases.
	f := NewAnthropicFormatter("msg-123", "claude-3")
	f.IncrementBlockIndex()

	// Format normal content - should work fine
	output := f.FormatContent("Hello")
	if !strings.Contains(string(output), "content_block_delta") {
		t.Error("expected content_block_delta event")
	}

	// Format empty content - should handle gracefully
	emptyOutput := f.FormatContent("")
	if emptyOutput == nil {
		t.Error("expected non-nil output for empty content")
	}
}

// TestAnthropicFormatter_InvalidToolCallID tests handling of invalid tool call IDs.
// Category D3: Invalid tool call ID (MEDIUM)
func TestAnthropicFormatter_InvalidToolCallID(t *testing.T) {
	tests := []struct {
		name string
		id   string
	}{
		{
			name: "empty ID",
			id:   "",
		},
		{
			name: "ID with special characters",
			id:   "call_!@#$%^&*()",
		},
		{
			name: "ID with spaces",
			id:   "call 123",
		},
		{
			name: "ID with unicode",
			id:   "call_你好",
		},
		{
			name: "ID with emoji",
			id:   "call_🚀",
		},
		{
			name: "ID with newlines",
			id:   "call\n123",
		},
		{
			name: "ID with tab",
			id:   "call\t123",
		},
		{
			name: "very long ID",
			id:   strings.Repeat("a", 1000),
		},
		{
			name: "ID starting with number",
			id:   "123call",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := NewAnthropicFormatter("msg-123", "claude-3")
			output := f.FormatToolStart(tt.id, "test_function", 0)

			// Should handle all IDs without error
			if output == nil {
				t.Error("expected non-nil output for tool start")
				return
			}

			// Verify the output is valid SSE
			if !strings.HasPrefix(string(output), "event: content_block_start") {
				t.Error("expected output to start with 'event: content_block_start'")
			}

			// For non-empty IDs, verify the ID is present (may be JSON escaped)
			if tt.id != "" {
				// Check that the output contains an ID field with some content
				if !strings.Contains(string(output), `"id":`) {
					t.Error("expected output to contain 'id' field")
				}
			}
		})
	}
}

// TestAnthropicFormatter_MismatchedToolCallIDs tests handling of mismatched tool call IDs.
// Category D3: Mismatched tool call IDs (MEDIUM)
func TestAnthropicFormatter_MismatchedToolCallIDs(t *testing.T) {
	// The formatter doesn't track tool call IDs internally, it just formats events.
	// But we should test that it handles different tools at different indices.
	f := NewAnthropicFormatter("msg-123", "claude-3")

	// Start first tool
	output1 := f.FormatToolStart("call_abc", "func_a", 0)
	if !strings.Contains(string(output1), `"id":"call_abc"`) {
		t.Error("expected first tool to have ID call_abc")
	}

	// Arguments for first tool
	argsOutput1 := f.FormatToolArgs(`{"x":1}`, 0)
	if !strings.Contains(string(argsOutput1), `"partial_json":"{\"x\":1}"`) {
		t.Error("expected arguments for first tool")
	}

	// End first tool
	endOutput1 := f.FormatToolEnd(0)
	if !strings.Contains(string(endOutput1), `"type":"content_block_stop"`) {
		t.Error("expected content_block_stop for first tool")
	}

	// Start second tool with different ID
	output2 := f.FormatToolStart("call_xyz", "func_b", 1)
	if !strings.Contains(string(output2), `"id":"call_xyz"`) {
		t.Error("expected second tool to have ID call_xyz")
	}

	// Arguments for second tool
	argsOutput2 := f.FormatToolArgs(`{"y":2}`, 1)
	if !strings.Contains(string(argsOutput2), `"partial_json":"{\"y\":2}"`) {
		t.Error("expected arguments for second tool")
	}

	// Verify both tools are in output with correct indices
	fullOutput := string(output1) + string(argsOutput1) + string(endOutput1) + string(output2) + string(argsOutput2)

	// Count occurrences
	if strings.Count(fullOutput, "call_abc") != 1 {
		t.Error("expected exactly one occurrence of call_abc")
	}
	if strings.Count(fullOutput, "call_xyz") != 1 {
		t.Error("expected exactly one occurrence of call_xyz")
	}
}
