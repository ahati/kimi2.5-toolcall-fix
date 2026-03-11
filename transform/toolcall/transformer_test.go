package toolcall

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"ai-proxy/types"

	"github.com/tmaxmax/go-sse"
)

func TestNewOpenAITransformer(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewOpenAITransformer(buf, "msg-123", "gpt-4")

	if tr == nil {
		t.Fatal("expected transformer to be non-nil")
	}
	if tr.messageID != "msg-123" {
		t.Errorf("expected messageID 'msg-123', got %q", tr.messageID)
	}
	if tr.model != "gpt-4" {
		t.Errorf("expected model 'gpt-4', got %q", tr.model)
	}
	if tr.parser == nil {
		t.Error("expected parser to be non-nil")
	}
	if tr.formatter == nil {
		t.Error("expected formatter to be non-nil")
	}
}

func TestNewAnthropicTransformer(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewAnthropicTransformer(buf, "msg-456", "claude-3")

	if tr == nil {
		t.Fatal("expected transformer to be non-nil")
	}
	if tr.messageID != "msg-456" {
		t.Errorf("expected messageID 'msg-456', got %q", tr.messageID)
	}
	if tr.model != "claude-3" {
		t.Errorf("expected model 'claude-3', got %q", tr.model)
	}
}

func TestTransformer_Transform_SimpleContent(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewOpenAITransformer(buf, "msg-123", "gpt-4")

	chunk := types.Chunk{
		ID:      "msg-123",
		Object:  "chat.completion.chunk",
		Model:   "gpt-4",
		Choices: []types.Choice{{Delta: types.Delta{Content: "Hello"}}},
	}
	data, _ := json.Marshal(chunk)

	event := &sse.Event{Data: string(data)}
	if err := tr.Transform(event); err != nil {
		t.Fatalf("Transform failed: %v", err)
	}

	output := buf.String()
	if !strings.Contains(output, `"content":"Hello"`) {
		t.Errorf("expected output to contain content, got %q", output)
	}
}

func TestTransformer_Transform_EmptyData(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewOpenAITransformer(buf, "msg-123", "gpt-4")

	event := &sse.Event{Data: ""}
	if err := tr.Transform(event); err != nil {
		t.Fatalf("Transform failed: %v", err)
	}

	if buf.Len() != 0 {
		t.Errorf("expected empty output, got %q", buf.String())
	}
}

func TestTransformer_Transform_DoneMarker(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewOpenAITransformer(buf, "msg-123", "gpt-4")

	event := &sse.Event{Data: "[DONE]"}
	if err := tr.Transform(event); err != nil {
		t.Fatalf("Transform failed: %v", err)
	}

	if buf.Len() != 0 {
		t.Errorf("expected empty output for [DONE], got %q", buf.String())
	}
}

func TestTransformer_Transform_InvalidJSON(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewOpenAITransformer(buf, "msg-123", "gpt-4")

	event := &sse.Event{Data: "not valid json"}
	if err := tr.Transform(event); err != nil {
		t.Fatalf("Transform failed: %v", err)
	}

	output := buf.String()
	if !strings.Contains(output, "not valid json") {
		t.Errorf("expected output to contain raw data, got %q", output)
	}
}

func TestTransformer_Transform_ExtractsMessageID(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewOpenAITransformer(buf, "", "")

	chunk := types.Chunk{
		ID:      "extracted-id",
		Object:  "chat.completion.chunk",
		Model:   "extracted-model",
		Choices: []types.Choice{{Delta: types.Delta{Content: "test"}}},
	}
	data, _ := json.Marshal(chunk)

	event := &sse.Event{Data: string(data)}
	if err := tr.Transform(event); err != nil {
		t.Fatalf("Transform failed: %v", err)
	}

	if tr.messageID != "extracted-id" {
		t.Errorf("expected messageID 'extracted-id', got %q", tr.messageID)
	}
	if tr.model != "extracted-model" {
		t.Errorf("expected model 'extracted-model', got %q", tr.model)
	}
}

func TestTransformer_Transform_NoChoices(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewOpenAITransformer(buf, "msg-123", "gpt-4")

	chunk := types.Chunk{
		ID:      "msg-123",
		Object:  "chat.completion.chunk",
		Choices: []types.Choice{},
	}
	data, _ := json.Marshal(chunk)

	event := &sse.Event{Data: string(data)}
	if err := tr.Transform(event); err != nil {
		t.Fatalf("Transform failed: %v", err)
	}

	if buf.Len() != 0 {
		t.Errorf("expected empty output for no choices, got %q", buf.String())
	}
}

func TestTransformer_Transform_WithToolCalls(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewOpenAITransformer(buf, "msg-123", "gpt-4")

	input := "Before" +
		"<|tool_calls_section_begin|>" +
		"<|tool_call_begin|>" +
		"bash" +
		"<|tool_call_argument_begin|>" +
		`{"command":"ls"}` +
		"<|tool_call_end|>" +
		"<|tool_calls_section_end|>" +
		"After"

	chunk := types.Chunk{
		ID:      "msg-123",
		Object:  "chat.completion.chunk",
		Model:   "gpt-4",
		Choices: []types.Choice{{Delta: types.Delta{Reasoning: input}}},
	}
	data, _ := json.Marshal(chunk)

	event := &sse.Event{Data: string(data)}
	if err := tr.Transform(event); err != nil {
		t.Fatalf("Transform failed: %v", err)
	}
	if err := tr.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	output := buf.String()
	if !strings.Contains(output, `"name":"bash"`) {
		t.Errorf("expected output to contain tool name, got %q", output)
	}
	if !strings.Contains(output, `"arguments":"{\"command\":\"ls\"}"`) {
		t.Errorf("expected output to contain tool arguments, got %q", output)
	}
}

func TestTransformer_Transform_ReasoningContent(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewOpenAITransformer(buf, "msg-123", "gpt-4")

	chunk := types.Chunk{
		ID:      "msg-123",
		Object:  "chat.completion.chunk",
		Model:   "gpt-4",
		Choices: []types.Choice{{Delta: types.Delta{ReasoningContent: "reasoning text"}}},
	}
	data, _ := json.Marshal(chunk)

	event := &sse.Event{Data: string(data)}
	if err := tr.Transform(event); err != nil {
		t.Fatalf("Transform failed: %v", err)
	}

	output := buf.String()
	if !strings.Contains(output, "reasoning text") {
		t.Errorf("expected output to contain reasoning content, got %q", output)
	}
}

func TestTransformer_Flush(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewOpenAITransformer(buf, "msg-123", "gpt-4")

	chunk := types.Chunk{
		ID:      "msg-123",
		Object:  "chat.completion.chunk",
		Model:   "gpt-4",
		Choices: []types.Choice{{Delta: types.Delta{Content: "unflushed content"}}},
	}
	data, _ := json.Marshal(chunk)

	event := &sse.Event{Data: string(data)}
	if err := tr.Transform(event); err != nil {
		t.Fatalf("Transform failed: %v", err)
	}

	if err := tr.Flush(); err != nil {
		t.Fatalf("Flush failed: %v", err)
	}

	output := buf.String()
	if !strings.Contains(output, "unflushed content") {
		t.Errorf("expected output to contain flushed content, got %q", output)
	}
}

func TestTransformer_Flush_EmptyBuffer(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewOpenAITransformer(buf, "msg-123", "gpt-4")

	if err := tr.Flush(); err != nil {
		t.Fatalf("Flush failed: %v", err)
	}

	if buf.Len() != 0 {
		t.Errorf("expected empty output, got %q", buf.String())
	}
}

func TestTransformer_Close(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewOpenAITransformer(buf, "msg-123", "gpt-4")

	chunk := types.Chunk{
		ID:      "msg-123",
		Object:  "chat.completion.chunk",
		Model:   "gpt-4",
		Choices: []types.Choice{{Delta: types.Delta{Content: "content to flush"}}},
	}
	data, _ := json.Marshal(chunk)

	event := &sse.Event{Data: string(data)}
	if err := tr.Transform(event); err != nil {
		t.Fatalf("Transform failed: %v", err)
	}

	if err := tr.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	output := buf.String()
	if !strings.Contains(output, "content to flush") {
		t.Errorf("expected output to contain flushed content, got %q", output)
	}
}

func TestTransformer_Parser(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewOpenAITransformer(buf, "msg-123", "gpt-4")

	if tr.Parser() == nil {
		t.Error("expected Parser to return non-nil")
	}
}

func TestTransformer_Formatter(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewOpenAITransformer(buf, "msg-123", "gpt-4")

	if tr.Formatter() == nil {
		t.Error("expected Formatter to return non-nil")
	}
}

func TestTransformer_OpenAI_EndToEnd(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewOpenAITransformer(buf, "", "")

	chunks := []string{
		"Start",
		"<|tool_calls_section_begin|>",
		"<|tool_call_begin|>",
		"bash",
		"<|tool_call_argument_begin|>",
		`{"command":"ls -la"}`,
		"<|tool_call_end|>",
		"<|tool_calls_section_end|>",
		"End",
	}

	for i, text := range chunks {
		chunk := types.Chunk{
			ID:      "e2e-msg",
			Object:  "chat.completion.chunk",
			Model:   "gpt-4",
			Choices: []types.Choice{{Delta: types.Delta{Reasoning: text}}},
		}
		data, _ := json.Marshal(chunk)

		event := &sse.Event{Data: string(data)}
		if err := tr.Transform(event); err != nil {
			t.Fatalf("Transform failed at chunk %d: %v", i, err)
		}
	}

	if err := tr.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	output := buf.String()

	if !strings.Contains(output, `"content":"Start"`) {
		t.Error("expected output to contain 'Start' content")
	}
	if !strings.Contains(output, `"name":"bash"`) {
		t.Error("expected output to contain bash tool name")
	}
	if !strings.Contains(output, `"arguments":"{\"command\":\"ls -la\"}"`) {
		t.Error("expected output to contain tool arguments")
	}
	if !strings.Contains(output, `"content":"End"`) {
		t.Error("expected output to contain 'End' content")
	}
}

func TestTransformer_Anthropic_EndToEnd(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewAnthropicTransformer(buf, "", "")

	chunks := []string{
		"Start",
		"<|tool_calls_section_begin|>",
		"<|tool_call_begin|>",
		"read",
		"<|tool_call_argument_begin|>",
		`{"file":"test.txt"}`,
		"<|tool_call_end|>",
		"<|tool_calls_section_end|>",
		"End",
	}

	for i, text := range chunks {
		chunk := types.Chunk{
			ID:      "anthropic-msg",
			Object:  "chat.completion.chunk",
			Model:   "claude-3",
			Choices: []types.Choice{{Delta: types.Delta{Reasoning: text}}},
		}
		data, _ := json.Marshal(chunk)

		event := &sse.Event{Data: string(data)}
		if err := tr.Transform(event); err != nil {
			t.Fatalf("Transform failed at chunk %d: %v", i, err)
		}
	}

	if err := tr.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	output := buf.String()

	if !strings.Contains(output, "content_block_delta") {
		t.Error("expected output to contain content_block_delta event")
	}
	if !strings.Contains(output, "content_block_start") {
		t.Error("expected output to contain content_block_start event")
	}
	if !strings.Contains(output, `"name":"read"`) {
		t.Error("expected output to contain read tool name")
	}
	if !strings.Contains(output, "input_json_delta") {
		t.Error("expected output to contain input_json_delta")
	}
	if !strings.Contains(output, "content_block_stop") {
		t.Error("expected output to contain content_block_stop event")
	}
}

func TestTransformer_WriteEvent_AllTypes(t *testing.T) {
	tests := []struct {
		name      string
		event     Event
		wantEmpty bool
	}{
		{
			name:      "EventContent",
			event:     Event{Type: EventContent, Text: "hello"},
			wantEmpty: false,
		},
		{
			name:      "EventToolStart",
			event:     Event{Type: EventToolStart, ID: "call_1", Name: "bash", Index: 0},
			wantEmpty: false,
		},
		{
			name:      "EventToolArgs",
			event:     Event{Type: EventToolArgs, Args: "{}", Index: 0},
			wantEmpty: false,
		},
		{
			name:      "EventToolEnd",
			event:     Event{Type: EventToolEnd, Index: 0},
			wantEmpty: true,
		},
		{
			name:      "EventSectionEnd",
			event:     Event{Type: EventSectionEnd},
			wantEmpty: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			buf := &bytes.Buffer{}
			tr := NewOpenAITransformer(buf, "msg-123", "gpt-4")

			if err := tr.writeEvent(tt.event); err != nil {
				t.Fatalf("writeEvent failed: %v", err)
			}

			if tt.wantEmpty && buf.Len() != 0 {
				t.Errorf("expected empty output, got %q", buf.String())
			}
			if !tt.wantEmpty && buf.Len() == 0 {
				t.Error("expected non-empty output")
			}
		})
	}
}

func TestTransformer_MultipleToolCalls(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewOpenAITransformer(buf, "msg-123", "gpt-4")

	input := "<|tool_calls_section_begin|>" +
		"<|tool_call_begin|>bash<|tool_call_argument_begin|>{}<|tool_call_end|>" +
		"<|tool_call_begin|>read<|tool_call_argument_begin|>{}<|tool_call_end|>" +
		"<|tool_call_begin|>write<|tool_call_argument_begin|>{}<|tool_call_end|>" +
		"<|tool_calls_section_end|>"

	chunk := types.Chunk{
		ID:      "msg-123",
		Object:  "chat.completion.chunk",
		Model:   "gpt-4",
		Choices: []types.Choice{{Delta: types.Delta{Reasoning: input}}},
	}
	data, _ := json.Marshal(chunk)

	event := &sse.Event{Data: string(data)}
	if err := tr.Transform(event); err != nil {
		t.Fatalf("Transform failed: %v", err)
	}
	if err := tr.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	output := buf.String()

	if strings.Count(output, `"name":"bash"`) != 1 {
		t.Error("expected exactly one bash tool call")
	}
	if strings.Count(output, `"name":"read"`) != 1 {
		t.Error("expected exactly one read tool call")
	}
	if strings.Count(output, `"name":"write"`) != 1 {
		t.Error("expected exactly one write tool call")
	}
}

func TestTransformer_ContentPreference(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewOpenAITransformer(buf, "msg-123", "gpt-4")

	chunk := types.Chunk{
		ID:     "msg-123",
		Object: "chat.completion.chunk",
		Model:  "gpt-4",
		Choices: []types.Choice{{Delta: types.Delta{
			Content:   "content field",
			Reasoning: "should be ignored when content present",
		}}},
	}
	data, _ := json.Marshal(chunk)

	event := &sse.Event{Data: string(data)}
	if err := tr.Transform(event); err != nil {
		t.Fatalf("Transform failed: %v", err)
	}

	output := buf.String()
	if !strings.Contains(output, "content field") {
		t.Error("expected output to contain content field")
	}
	if strings.Contains(output, "should be ignored") {
		t.Error("expected reasoning to be ignored when content present")
	}
}

func TestTransformer_Write(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewOpenAITransformer(buf, "msg-123", "gpt-4")

	if err := tr.write([]byte("test data")); err != nil {
		t.Fatalf("write failed: %v", err)
	}

	if buf.String() != "test data" {
		t.Errorf("expected 'test data', got %q", buf.String())
	}
}

func TestTransformer_Write_EmptyData(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewOpenAITransformer(buf, "msg-123", "gpt-4")

	if err := tr.write([]byte{}); err != nil {
		t.Fatalf("write failed: %v", err)
	}

	if buf.Len() != 0 {
		t.Errorf("expected empty buffer, got %q", buf.String())
	}
}

func TestTransformer_Write_NilData(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewOpenAITransformer(buf, "msg-123", "gpt-4")

	if err := tr.write(nil); err != nil {
		t.Fatalf("write failed: %v", err)
	}

	if buf.Len() != 0 {
		t.Errorf("expected empty buffer, got %q", buf.String())
	}
}

func TestTransformer_AnthropicFormatter_Methods(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewAnthropicTransformer(buf, "msg-123", "claude-3")

	af, ok := tr.Formatter().(*AnthropicFormatter)
	if !ok {
		t.Fatal("expected AnthropicFormatter")
	}

	af.SetMessageID("new-id")
	af.SetModel("new-model")
	af.SetBlockIndex(5)
	af.IncrementBlockIndex()

	if af.messageID != "new-id" {
		t.Errorf("expected messageID 'new-id', got %q", af.messageID)
	}
	if af.model != "new-model" {
		t.Errorf("expected model 'new-model', got %q", af.model)
	}
	if af.BlockIndex() != 6 {
		t.Errorf("expected block index 6, got %d", af.BlockIndex())
	}
}

func TestTransformer_OpenAIFormatter_Methods(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewOpenAITransformer(buf, "msg-123", "gpt-4")

	of, ok := tr.Formatter().(*OpenAIFormatter)
	if !ok {
		t.Fatal("expected OpenAIFormatter")
	}

	of.SetMessageID("new-id")
	of.SetModel("new-model")

	if of.messageID != "new-id" {
		t.Errorf("expected messageID 'new-id', got %q", of.messageID)
	}
	if of.model != "new-model" {
		t.Errorf("expected model 'new-model', got %q", of.model)
	}
}
