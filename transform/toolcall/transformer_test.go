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
	tr := NewOpenAITransformer(buf)

	if tr == nil {
		t.Fatal("expected transformer to be non-nil")
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
	tr := NewAnthropicTransformer(buf)

	if tr == nil {
		t.Fatal("expected transformer to be non-nil")
	}
	if tr.formatter == nil {
		t.Error("expected formatter to be non-nil")
	}
}

func TestTransformer_Transform_SimpleContent(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewOpenAITransformer(buf)

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
	if !strings.Contains(output, "Hello") {
		t.Errorf("expected output to contain Hello, got %q", output)
	}
}

func TestTransformer_Transform_DoneMarker(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewOpenAITransformer(buf)

	event := &sse.Event{Data: "[DONE]"}
	if err := tr.Transform(event); err != nil {
		t.Fatalf("Transform failed: %v", err)
	}

	output := buf.String()
	expected := "data: [DONE]\n\n"
	if output != expected {
		t.Errorf("expected %q, got %q", expected, output)
	}
}

func TestTransformer_Transform_InvalidJSON(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewOpenAITransformer(buf)

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
	tr := NewOpenAITransformer(buf)

	chunk := types.Chunk{
		ID:      "extracted-id",
		Model:   "gpt-4",
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
}

func TestTransformer_Transform_NoChoices(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewOpenAITransformer(buf)

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

	output := buf.String()
	if !strings.Contains(output, "data:") {
		t.Error("expected output to contain data: prefix")
	}
	if !strings.Contains(output, "msg-123") {
		t.Error("expected output to contain msg-123")
	}
}

func TestTransformer_Transform_WithToolCalls(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewOpenAITransformer(buf)

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
		ID:      "test-id",
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
	t.Logf("Output: %s", output)

	if !strings.Contains(output, "tool_calls") {
		t.Error("expected output to contain tool_calls")
	}
	if !strings.Contains(output, `"name":"bash"`) {
		t.Error("expected output to contain bash tool name")
	}
	if !strings.Contains(output, `"arguments":"{\"command\":\"ls\"}"`) {
		t.Error("expected output to contain arguments")
	}
}

func TestTransformer_Transform_MultipleToolCalls(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewOpenAITransformer(buf)

	input := "<|tool_calls_section_begin|>" +
		"<|tool_call_begin|>read<|tool_call_argument_begin|>{\"file\":\"test.txt\"}<|tool_call_end|>" +
		"<|tool_call_begin|>write<|tool_call_argument_begin|>{\"file\":\"out.txt\"}<|tool_call_end|>" +
		"<|tool_calls_section_end|>"

	chunk := types.Chunk{
		ID:      "test-id",
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
	t.Logf("Output: %s", output)

	if !strings.Contains(output, `"name":"read"`) {
		t.Error("expected output to contain read tool name")
	}
	if !strings.Contains(output, `"name":"write"`) {
		t.Error("expected output to contain write tool name")
	}
}

func TestTransformer_Transform_StreamingToolCalls(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewOpenAITransformer(buf)

	chunks := []string{
		"Start",
		"<|tool_calls_section_begin|>",
		"<|tool_call_begin|>",
		"bash",
		"<|tool_call_argument_begin|>",
		`{"command":"ls"}`,
		"<|tool_call_end|>",
		"<|tool_calls_section_end|>",
		"End",
	}

	for i, text := range chunks {
		chunk := types.Chunk{
			ID:      "test-id",
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
	t.Logf("Output: %s", output)

	if !strings.Contains(output, `"name":"bash"`) {
		t.Error("expected output to contain bash tool name")
	}
}

func TestTransformer_OpenAI_EndToEnd(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewOpenAITransformer(buf)

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
			ID:      "test-id",
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
	t.Logf("Output: %s", output)

	if !strings.Contains(output, `"name":"read"`) {
		t.Error("expected output to contain read tool name")
	}
}

func TestTransformer_Anthropic_EndToEnd(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewAnthropicTransformer(buf)

	events := []types.Event{
		{Type: "message_start", Message: &types.MessageInfo{ID: "msg-123", Model: "kimi-k2.5"}},
		{Type: "content_block_start", Index: intPtr(0), ContentBlock: json.RawMessage(`{"type":"thinking","thinking":""}`)},
		{Type: "content_block_delta", Index: intPtr(0), Delta: json.RawMessage(`{"type":"thinking_delta","thinking":"Some reasoning<|tool_calls_section_begin|><|tool_call_begin|>bash<|tool_call_argument_begin|>{\"cmd\":\"ls\"}<|tool_call_end|><|tool_calls_section_end|>"}`)},
		{Type: "content_block_stop", Index: intPtr(0)},
		{Type: "message_delta", Delta: json.RawMessage(`{"stop_reason":"end_turn"}`)},
		{Type: "message_stop"},
	}

	for i, event := range events {
		data, _ := json.Marshal(event)
		sseEvent := &sse.Event{Data: string(data)}
		if err := tr.Transform(sseEvent); err != nil {
			t.Fatalf("Transform failed at event %d: %v", i, err)
		}
	}

	if err := tr.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	output := buf.String()
	t.Logf("Output: %s", output)

	if !strings.Contains(output, `"type":"tool_use"`) {
		t.Error("expected output to contain tool_use block")
	}
	if !strings.Contains(output, `"name":"bash"`) {
		t.Error("expected output to contain bash tool name")
	}
	if !strings.Contains(output, "input_json_delta") {
		t.Error("expected output to contain input_json_delta")
	}
	if !strings.Contains(output, `"stop_reason":"tool_use"`) {
		t.Error("expected stop_reason to be changed to tool_use")
	}
}

func TestTransformer_AnthropicPassthrough(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewAnthropicTransformer(buf)

	event := types.Event{
		Type:  "content_block_delta",
		Index: intPtr(0),
		Delta: json.RawMessage(`{"type":"text_delta","text":"Hello"}`),
	}
	data, _ := json.Marshal(event)

	sseEvent := &sse.Event{Data: string(data)}
	if err := tr.Transform(sseEvent); err != nil {
		t.Fatalf("Transform failed: %v", err)
	}

	output := buf.String()
	if !strings.Contains(output, "data:") {
		t.Error("expected output to contain SSE data: prefix")
	}
	if !strings.Contains(output, "text_delta") {
		t.Error("expected output to contain text_delta")
	}
	if !strings.Contains(output, "Hello") {
		t.Error("expected output to contain Hello")
	}
}

func TestTransformer_AnthropicToolCallMarkup(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewAnthropicTransformer(buf)

	delta := map[string]interface{}{
		"type": "text_delta",
		"text": "<|tool_calls_section_begin|><|tool_call_begin|>bash<|tool_call_argument_begin|>{\"cmd\":\"ls\"}<|tool_call_end|><|tool_calls_section_end|>",
	}
	deltaJSON, _ := json.Marshal(delta)

	event := types.Event{
		Type:  "content_block_delta",
		Index: intPtr(0),
		Delta: deltaJSON,
	}
	data, _ := json.Marshal(event)

	sseEvent := &sse.Event{Data: string(data)}
	if err := tr.Transform(sseEvent); err != nil {
		t.Fatalf("Transform failed: %v", err)
	}
	if err := tr.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	output := buf.String()
	t.Logf("Output: %s", output)
	if !strings.Contains(output, "bash") {
		t.Error("expected output to contain bash tool name")
	}
}

func TestTransformer_OpenAIPassthrough(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewOpenAITransformer(buf)

	chunk := types.Chunk{
		ID:    "test-id",
		Model: "gpt-4",
		Choices: []types.Choice{{
			Delta: types.Delta{Content: "Hello"},
		}},
	}
	data, _ := json.Marshal(chunk)

	event := &sse.Event{Data: string(data)}
	if err := tr.Transform(event); err != nil {
		t.Fatalf("Transform failed: %v", err)
	}

	output := buf.String()
	if !strings.Contains(output, "Hello") {
		t.Errorf("expected output to contain Hello, got %q", output)
	}
}

func TestTransformer_OpenAIToolCallFromReasoning(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewOpenAITransformer(buf)

	chunk := types.Chunk{
		ID:    "test-id",
		Model: "gpt-4",
		Choices: []types.Choice{{
			Delta: types.Delta{Reasoning: "<|tool_calls_section_begin|><|tool_call_begin|>bash<|tool_call_argument_begin|>{\"command\":\"ls -la\"}<|tool_call_end|><|tool_calls_section_end|>"},
		}},
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
	t.Logf("Output: %s", output)

	if !strings.Contains(output, `"name":"bash"`) {
		t.Error("expected output to contain bash tool name")
	}
	if !strings.Contains(output, `"arguments":"{\"command\":\"ls -la\"}"`) {
		t.Error("expected output to contain arguments")
	}
}

func TestTransformer_Anthropic_ToolCallsInThinking(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewAnthropicTransformer(buf)

	events := []types.Event{
		{Type: "message_start", Message: &types.MessageInfo{ID: "msg-123", Model: "kimi-k2.5"}},
		{Type: "content_block_start", Index: intPtr(0), ContentBlock: json.RawMessage(`{"type":"thinking","thinking":""}`)},
		{Type: "content_block_delta", Index: intPtr(0), Delta: json.RawMessage(`{"type":"thinking_delta","thinking":"Some reasoning<|tool_calls_section_begin|><|tool_call_begin|>bash<|tool_call_argument_begin|>{\"cmd\":\"ls\"}<|tool_call_end|><|tool_calls_section_end|>"}`)},
		{Type: "content_block_stop", Index: intPtr(0)},
		{Type: "message_delta", Delta: json.RawMessage(`{"stop_reason":"end_turn"}`)},
		{Type: "message_stop"},
	}

	for i, event := range events {
		data, _ := json.Marshal(event)
		sseEvent := &sse.Event{Data: string(data)}
		if err := tr.Transform(sseEvent); err != nil {
			t.Fatalf("Transform failed at event %d: %v", i, err)
		}
	}

	if err := tr.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	output := buf.String()
	t.Logf("Output: %s", output)

	if !strings.Contains(output, `"type":"tool_use"`) {
		t.Error("expected output to contain tool_use block")
	}
	if !strings.Contains(output, `"name":"bash"`) {
		t.Error("expected output to contain bash tool name")
	}
	if !strings.Contains(output, "input_json_delta") {
		t.Error("expected output to contain input_json_delta")
	}
	if !strings.Contains(output, `"stop_reason":"tool_use"`) {
		t.Error("expected stop_reason to be changed to tool_use")
	}
}

func TestTransformer_Anthropic_ToolCallsInText(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewAnthropicTransformer(buf)

	events := []types.Event{
		{Type: "message_start", Message: &types.MessageInfo{ID: "msg-456", Model: "kimi-k2.5"}},
		{Type: "content_block_start", Index: intPtr(0), ContentBlock: json.RawMessage(`{"type":"text","text":""}`)},
		{Type: "content_block_delta", Index: intPtr(0), Delta: json.RawMessage(`{"type":"text_delta","text":"Hello<|tool_calls_section_begin|><|tool_call_begin|>read<|tool_call_argument_begin|>{\"file\":\"test.txt\"}<|tool_call_end|><|tool_calls_section_end|>World"}`)},
		{Type: "content_block_stop", Index: intPtr(0)},
		{Type: "message_delta", Delta: json.RawMessage(`{"stop_reason":"end_turn"}`)},
		{Type: "message_stop"},
	}

	for i, event := range events {
		data, _ := json.Marshal(event)
		sseEvent := &sse.Event{Data: string(data)}
		if err := tr.Transform(sseEvent); err != nil {
			t.Fatalf("Transform failed at event %d: %v", i, err)
		}
	}

	if err := tr.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	output := buf.String()
	t.Logf("Output: %s", output)

	if !strings.Contains(output, `"type":"tool_use"`) {
		t.Error("expected output to contain tool_use block")
	}
	if !strings.Contains(output, `"name":"read"`) {
		t.Error("expected output to contain read tool name")
	}
	if !strings.Contains(output, `"stop_reason":"tool_use"`) {
		t.Error("expected stop_reason to be changed to tool_use")
	}
}

func TestTransformer_Anthropic_TextPassthrough(t *testing.T) {
	buf := &bytes.Buffer{}
	tr := NewAnthropicTransformer(buf)

	events := []types.Event{
		{Type: "message_start", Message: &types.MessageInfo{ID: "msg-789", Model: "kimi-k2.5"}},
		{Type: "content_block_start", Index: intPtr(0), ContentBlock: json.RawMessage(`{"type":"text","text":""}`)},
		{Type: "content_block_delta", Index: intPtr(0), Delta: json.RawMessage(`{"type":"text_delta","text":"Hello World"}`)},
		{Type: "content_block_stop", Index: intPtr(0)},
		{Type: "message_delta", Delta: json.RawMessage(`{"stop_reason":"end_turn"}`)},
		{Type: "message_stop"},
	}

	for i, event := range events {
		data, _ := json.Marshal(event)
		sseEvent := &sse.Event{Data: string(data)}
		if err := tr.Transform(sseEvent); err != nil {
			t.Fatalf("Transform failed at event %d: %v", i, err)
		}
	}

	if err := tr.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	output := buf.String()
	t.Logf("Output: %s", output)

	if !strings.Contains(output, "Hello World") {
		t.Error("expected output to contain Hello World")
	}
	if !strings.Contains(output, `"stop_reason":"end_turn"`) {
		t.Error("expected stop_reason to remain end_turn")
	}
}

func TestParser_Parse(t *testing.T) {
	p := NewParser(DefaultTokens)

	events := p.Parse("<|tool_calls_section_begin|><|tool_call_begin|>bash<|tool_call_argument_begin|>{\"cmd\":\"ls\"}<|tool_call_end|><|tool_calls_section_end|>")
	if len(events) == 0 {
		t.Fatal("expected events to be parsed")
	}

	foundToolStart := false
	for _, e := range events {
		if e.Type == EventToolStart && e.Name == "bash" {
			foundToolStart = true
		}
	}

	if !foundToolStart {
		t.Error("expected to find tool start event for bash")
	}
}

func TestParser_IsIdle(t *testing.T) {
	p := NewParser(DefaultTokens)

	if !p.IsIdle() {
		t.Error("expected parser to be idle initially")
	}

	p.Parse("<|tool_calls_section_begin|>")
	if p.IsIdle() {
		t.Error("expected parser to not be idle after section begin")
	}
}
