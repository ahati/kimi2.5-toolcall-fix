package toolcall

import (
	"encoding/json"
	"strings"
	"testing"

	"ai-proxy/types"
)

func TestOpenAIFormatter_FormatContent(t *testing.T) {
	f := NewOpenAIFormatter("msg-123", "gpt-4")
	output := f.FormatContent("Hello, world!")

	if !strings.HasPrefix(string(output), "data: ") {
		t.Error("expected output to start with 'data: '")
	}
	if !strings.HasSuffix(string(output), "\n\n") {
		t.Error("expected output to end with '\\n\\n'")
	}

	jsonStr := strings.TrimPrefix(string(output), "data: ")
	jsonStr = strings.TrimSuffix(jsonStr, "\n\n")

	var chunk types.Chunk
	if err := json.Unmarshal([]byte(jsonStr), &chunk); err != nil {
		t.Fatalf("failed to parse JSON: %v", err)
	}

	if chunk.ID != "msg-123" {
		t.Errorf("expected ID 'msg-123', got %q", chunk.ID)
	}
	if chunk.Object != "chat.completion.chunk" {
		t.Errorf("expected Object 'chat.completion.chunk', got %q", chunk.Object)
	}
	if chunk.Model != "gpt-4" {
		t.Errorf("expected Model 'gpt-4', got %q", chunk.Model)
	}
	if len(chunk.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(chunk.Choices))
	}
	if chunk.Choices[0].Delta.Content != "Hello, world!" {
		t.Errorf("expected content 'Hello, world!', got %q", chunk.Choices[0].Delta.Content)
	}
}

func TestOpenAIFormatter_FormatToolStart(t *testing.T) {
	f := NewOpenAIFormatter("msg-456", "gpt-4")
	output := f.FormatToolStart("call_abc", "bash", 0)

	jsonStr := strings.TrimPrefix(string(output), "data: ")
	jsonStr = strings.TrimSuffix(jsonStr, "\n\n")

	var chunk types.Chunk
	if err := json.Unmarshal([]byte(jsonStr), &chunk); err != nil {
		t.Fatalf("failed to parse JSON: %v", err)
	}

	if len(chunk.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(chunk.Choices))
	}

	toolCalls := chunk.Choices[0].Delta.ToolCalls
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(toolCalls))
	}

	tc := toolCalls[0]
	if tc.ID != "call_abc" {
		t.Errorf("expected ID 'call_abc', got %q", tc.ID)
	}
	if tc.Type != "function" {
		t.Errorf("expected Type 'function', got %q", tc.Type)
	}
	if tc.Index != 0 {
		t.Errorf("expected Index 0, got %d", tc.Index)
	}
	if tc.Function.Name != "bash" {
		t.Errorf("expected Function.Name 'bash', got %q", tc.Function.Name)
	}
	if tc.Function.Arguments != "" {
		t.Errorf("expected empty Arguments, got %q", tc.Function.Arguments)
	}
}

func TestOpenAIFormatter_FormatToolArgs(t *testing.T) {
	f := NewOpenAIFormatter("msg-789", "gpt-4")
	output := f.FormatToolArgs(`{"command":"ls"}`, 0)

	jsonStr := strings.TrimPrefix(string(output), "data: ")
	jsonStr = strings.TrimSuffix(jsonStr, "\n\n")

	var chunk types.Chunk
	if err := json.Unmarshal([]byte(jsonStr), &chunk); err != nil {
		t.Fatalf("failed to parse JSON: %v", err)
	}

	toolCalls := chunk.Choices[0].Delta.ToolCalls
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(toolCalls))
	}

	tc := toolCalls[0]
	if tc.Index != 0 {
		t.Errorf("expected Index 0, got %d", tc.Index)
	}
	if tc.Function.Arguments != `{"command":"ls"}` {
		t.Errorf("expected Arguments %q, got %q", `{"command":"ls"}`, tc.Function.Arguments)
	}
	if tc.ID != "" {
		t.Errorf("expected empty ID in args chunk, got %q", tc.ID)
	}
	if tc.Function.Name != "" {
		t.Errorf("expected empty Name in args chunk, got %q", tc.Function.Name)
	}
}

func TestOpenAIFormatter_FormatToolEnd(t *testing.T) {
	f := NewOpenAIFormatter("msg-123", "gpt-4")
	output := f.FormatToolEnd(0)

	if output != nil {
		t.Errorf("expected nil output for FormatToolEnd, got %v", output)
	}
}

func TestOpenAIFormatter_FormatSectionEnd(t *testing.T) {
	f := NewOpenAIFormatter("msg-123", "gpt-4")
	output := f.FormatSectionEnd()

	if output != nil {
		t.Errorf("expected nil output for FormatSectionEnd, got %v", output)
	}
}

func TestOpenAIFormatter_FormatDone(t *testing.T) {
	f := NewOpenAIFormatter("msg-123", "gpt-4")
	output := f.FormatDone()

	expected := "data: [DONE]\n\n"
	if string(output) != expected {
		t.Errorf("expected %q, got %q", expected, string(output))
	}
}

func TestOpenAIFormatter_MultipleToolCalls(t *testing.T) {
	f := NewOpenAIFormatter("msg-multi", "gpt-4")

	start0 := f.FormatToolStart("call_0", "bash", 0)
	start1 := f.FormatToolStart("call_1", "read", 1)
	args0 := f.FormatToolArgs("{}", 0)
	args1 := f.FormatToolArgs("{}", 1)

	var chunk0, chunk1 types.Chunk
	json.Unmarshal([]byte(strings.TrimPrefix(strings.TrimSuffix(string(start0), "\n\n"), "data: ")), &chunk0)
	json.Unmarshal([]byte(strings.TrimPrefix(strings.TrimSuffix(string(start1), "\n\n"), "data: ")), &chunk1)

	if chunk0.Choices[0].Delta.ToolCalls[0].Index != 0 {
		t.Error("expected index 0 for first tool")
	}
	if chunk1.Choices[0].Delta.ToolCalls[0].Index != 1 {
		t.Error("expected index 1 for second tool")
	}

	var argsChunk0, argsChunk1 types.Chunk
	json.Unmarshal([]byte(strings.TrimPrefix(strings.TrimSuffix(string(args0), "\n\n"), "data: ")), &argsChunk0)
	json.Unmarshal([]byte(strings.TrimPrefix(strings.TrimSuffix(string(args1), "\n\n"), "data: ")), &argsChunk1)

	if argsChunk0.Choices[0].Delta.ToolCalls[0].Index != 0 {
		t.Error("expected index 0 for first tool args")
	}
	if argsChunk1.Choices[0].Delta.ToolCalls[0].Index != 1 {
		t.Error("expected index 1 for second tool args")
	}
}

func TestOpenAIFormatter_SetMessageID(t *testing.T) {
	f := NewOpenAIFormatter("old-id", "gpt-4")
	f.SetMessageID("new-id")

	output := f.FormatContent("test")
	jsonStr := strings.TrimPrefix(string(output), "data: ")
	jsonStr = strings.TrimSuffix(jsonStr, "\n\n")

	var chunk types.Chunk
	json.Unmarshal([]byte(jsonStr), &chunk)

	if chunk.ID != "new-id" {
		t.Errorf("expected ID 'new-id', got %q", chunk.ID)
	}
}

func TestOpenAIFormatter_SetModel(t *testing.T) {
	f := NewOpenAIFormatter("msg-id", "old-model")
	f.SetModel("new-model")

	output := f.FormatContent("test")
	jsonStr := strings.TrimPrefix(string(output), "data: ")
	jsonStr = strings.TrimSuffix(jsonStr, "\n\n")

	var chunk types.Chunk
	json.Unmarshal([]byte(jsonStr), &chunk)

	if chunk.Model != "new-model" {
		t.Errorf("expected Model 'new-model', got %q", chunk.Model)
	}
}

func TestOpenAIFormatter_EmptyContent(t *testing.T) {
	f := NewOpenAIFormatter("msg-123", "gpt-4")
	output := f.FormatContent("")

	jsonStr := strings.TrimPrefix(string(output), "data: ")
	jsonStr = strings.TrimSuffix(jsonStr, "\n\n")

	var chunk types.Chunk
	if err := json.Unmarshal([]byte(jsonStr), &chunk); err != nil {
		t.Fatalf("failed to parse JSON: %v", err)
	}

	if chunk.Choices[0].Delta.Content != "" {
		t.Errorf("expected empty content, got %q", chunk.Choices[0].Delta.Content)
	}
}

func TestOpenAIFormatter_SpecialCharacters(t *testing.T) {
	f := NewOpenAIFormatter("msg-123", "gpt-4")
	specialContent := "Hello\nWorld\t\"Quotes\"\\Backslash"
	output := f.FormatContent(specialContent)

	jsonStr := strings.TrimPrefix(string(output), "data: ")
	jsonStr = strings.TrimSuffix(jsonStr, "\n\n")

	var chunk types.Chunk
	if err := json.Unmarshal([]byte(jsonStr), &chunk); err != nil {
		t.Fatalf("failed to parse JSON: %v", err)
	}

	if chunk.Choices[0].Delta.Content != specialContent {
		t.Errorf("expected content %q, got %q", specialContent, chunk.Choices[0].Delta.Content)
	}
}

func TestOpenAIFormatter_UnicodeContent(t *testing.T) {
	f := NewOpenAIFormatter("msg-123", "gpt-4")
	unicodeContent := "Hello 世界 🌍"
	output := f.FormatContent(unicodeContent)

	jsonStr := strings.TrimPrefix(string(output), "data: ")
	jsonStr = strings.TrimSuffix(jsonStr, "\n\n")

	var chunk types.Chunk
	if err := json.Unmarshal([]byte(jsonStr), &chunk); err != nil {
		t.Fatalf("failed to parse JSON: %v", err)
	}

	if chunk.Choices[0].Delta.Content != unicodeContent {
		t.Errorf("expected content %q, got %q", unicodeContent, chunk.Choices[0].Delta.Content)
	}
}

func TestOpenAIFormatter_ChunkHasCreatedTimestamp(t *testing.T) {
	f := NewOpenAIFormatter("msg-123", "gpt-4")
	output := f.FormatContent("test")

	jsonStr := strings.TrimPrefix(string(output), "data: ")
	jsonStr = strings.TrimSuffix(jsonStr, "\n\n")

	var chunk types.Chunk
	json.Unmarshal([]byte(jsonStr), &chunk)

	if chunk.Created == 0 {
		t.Error("expected Created timestamp to be non-zero")
	}
}

func TestOpenAIFormatter_LargeArgs(t *testing.T) {
	f := NewOpenAIFormatter("msg-123", "gpt-4")

	largeArgs := make(map[string]string)
	for i := 0; i < 100; i++ {
		largeArgs[string(rune('a'+i%26))] = strings.Repeat("x", 100)
	}
	argsJSON, _ := json.Marshal(largeArgs)

	output := f.FormatToolStart("call-large", "test", 0)
	if len(output) == 0 {
		t.Error("expected non-empty output")
	}

	argsOutput := f.FormatToolArgs(string(argsJSON), 0)
	if len(argsOutput) == 0 {
		t.Error("expected non-empty args output")
	}
}
