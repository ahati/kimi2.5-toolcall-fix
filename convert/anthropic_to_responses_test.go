package convert

import (
	"bytes"
	"strings"
	"testing"

	"ai-proxy/types"

	"github.com/tmaxmax/go-sse"
)

func TestAnthropicToResponsesRequest_Simple(t *testing.T) {
	req := &types.MessageRequest{
		Model:     "claude-3-opus",
		MaxTokens: 1024,
		Messages: []types.MessageInput{
			{Role: "user", Content: "Hello"},
		},
	}

	out, err := AnthropicToResponsesRequest(req)
	if err != nil {
		t.Fatalf("AnthropicToResponsesRequest failed: %v", err)
	}

	if out.Model != "claude-3-opus" {
		t.Errorf("expected model claude-3-opus, got %s", out.Model)
	}
	if out.MaxOutputTokens != 1024 {
		t.Errorf("expected max_output_tokens 1024, got %d", out.MaxOutputTokens)
	}

	// Input should be a simple string
	if str, ok := out.Input.(string); !ok || str != "Hello" {
		t.Errorf("expected input 'Hello', got %v", out.Input)
	}
}

func TestAnthropicToResponsesRequest_WithThinking(t *testing.T) {
	req := &types.MessageRequest{
		Model:     "claude-3-opus",
		MaxTokens: 1024,
		Messages: []types.MessageInput{
			{Role: "user", Content: "Hello"},
		},
		Thinking: &types.ThinkingConfig{
			Type:         "enabled",
			BudgetTokens: 8000,
		},
	}

	out, err := AnthropicToResponsesRequest(req)
	if err != nil {
		t.Fatalf("AnthropicToResponsesRequest failed: %v", err)
	}

	if out.Reasoning == nil {
		t.Fatal("expected reasoning to be set")
	}
	// BudgetToReasoningEffort maps 8000 to "medium"
	if out.Reasoning.Effort != "medium" {
		t.Errorf("expected reasoning effort medium, got %s", out.Reasoning.Effort)
	}
}

func TestAnthropicToResponsesRequest_WithTools(t *testing.T) {
	req := &types.MessageRequest{
		Model:     "claude-3-opus",
		MaxTokens: 1024,
		Messages: []types.MessageInput{
			{Role: "user", Content: "What's the weather?"},
		},
		Tools: []types.ToolDef{
			{
				Name:        "get_weather",
				Description: "Get the current weather",
			},
		},
		ToolChoice: &types.ToolChoice{Type: "auto"},
	}

	out, err := AnthropicToResponsesRequest(req)
	if err != nil {
		t.Fatalf("AnthropicToResponsesRequest failed: %v", err)
	}

	if len(out.Tools) != 1 {
		t.Errorf("expected 1 tool, got %d", len(out.Tools))
	}
	if out.Tools[0].Name != "get_weather" {
		t.Errorf("expected tool name get_weather, got %s", out.Tools[0].Name)
	}
}

func TestAnthropicToResponsesRequest_ToolResult(t *testing.T) {
	req := &types.MessageRequest{
		Model:     "claude-3-opus",
		MaxTokens: 1024,
		Messages: []types.MessageInput{
			{Role: "user", Content: []interface{}{
				map[string]interface{}{
					"type":        "tool_result",
					"tool_use_id": "tool_123",
					"content":     "The weather is sunny",
				},
			}},
		},
	}

	out, err := AnthropicToResponsesRequest(req)
	if err != nil {
		t.Fatalf("AnthropicToResponsesRequest failed: %v", err)
	}

	items, ok := out.Input.([]types.InputItem)
	if !ok {
		t.Fatalf("expected input to be []InputItem, got %T", out.Input)
	}

	if len(items) != 1 {
		t.Errorf("expected 1 item, got %d", len(items))
	}
	if items[0].Type != "function_call_output" {
		t.Errorf("expected type function_call_output, got %s", items[0].Type)
	}
	if items[0].CallID != "tool_123" {
		t.Errorf("expected call_id tool_123, got %s", items[0].CallID)
	}
}

func TestAnthropicToResponsesRequest_AssistantWithToolUse(t *testing.T) {
	req := &types.MessageRequest{
		Model:     "claude-3-opus",
		MaxTokens: 1024,
		Messages: []types.MessageInput{
			{Role: "assistant", Content: []interface{}{
				map[string]interface{}{
					"type": "text",
					"text": "Let me check the weather.",
				},
				map[string]interface{}{
					"type":  "tool_use",
					"id":    "tool_123",
					"name":  "get_weather",
					"input": map[string]interface{}{"location": "SF"},
				},
			}},
		},
	}

	out, err := AnthropicToResponsesRequest(req)
	if err != nil {
		t.Fatalf("AnthropicToResponsesRequest failed: %v", err)
	}

	items, ok := out.Input.([]types.InputItem)
	if !ok {
		t.Fatalf("expected input to be []InputItem, got %T", out.Input)
	}

	// Should have message item and function_call item
	if len(items) != 2 {
		t.Errorf("expected 2 items, got %d", len(items))
	}

	// First item should be message
	if items[0].Type != "message" {
		t.Errorf("expected first item type message, got %s", items[0].Type)
	}

	// Second item should be function_call
	if items[1].Type != "function_call" {
		t.Errorf("expected second item type function_call, got %s", items[1].Type)
	}
	if items[1].Name != "get_weather" {
		t.Errorf("expected function name get_weather, got %s", items[1].Name)
	}
}

func TestReasoningEffortToBudget(t *testing.T) {
	tests := []struct {
		effort   string
		expected int
	}{
		{"low", 4000},
		{"medium", 8000},
		{"high", 16000},
		{"unknown", 8000}, // default
	}

	for _, tt := range tests {
		t.Run(tt.effort, func(t *testing.T) {
			result := ReasoningEffortToBudget(tt.effort)
			if result != tt.expected {
				t.Errorf("ReasoningEffortToBudget(%s) = %d, want %d", tt.effort, result, tt.expected)
			}
		})
	}
}

func TestBudgetToReasoningEffort(t *testing.T) {
	tests := []struct {
		budget   int
		expected string
	}{
		{2000, "low"},
		{4000, "low"},
		{5000, "medium"},
		{8000, "medium"},
		{10000, "medium"},
		{15000, "high"},
		{20000, "high"},
	}

	for _, tt := range tests {
		t.Run(string(rune(tt.budget)), func(t *testing.T) {
			result := BudgetToReasoningEffort(tt.budget)
			if result != tt.expected {
				t.Errorf("BudgetToReasoningEffort(%d) = %s, want %s", tt.budget, result, tt.expected)
			}
		})
	}
}

// TestAnthropicToResponsesTransformer_TokenCounts verifies that token counts
// from message_start and message_delta are properly propagated to the final response.
func TestAnthropicToResponsesTransformer_TokenCounts(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewAnthropicToResponsesTransformer(&buf)

	// Simulate message_start with usage data
	messageStart := `{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","model":"claude-3-opus","usage":{"input_tokens":3639,"output_tokens":0,"cache_read_input_tokens":1408,"cache_creation_input_tokens":0}}}`
	event1 := &sse.Event{Data: messageStart}
	if err := transformer.Transform(event1); err != nil {
		t.Fatalf("Transform message_start failed: %v", err)
	}

	// Simulate content block for text output
	contentBlockStart := `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`
	event2 := &sse.Event{Data: contentBlockStart}
	if err := transformer.Transform(event2); err != nil {
		t.Fatalf("Transform content_block_start failed: %v", err)
	}

	// Simulate content block delta
	contentBlockDelta := `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`
	event3 := &sse.Event{Data: contentBlockDelta}
	if err := transformer.Transform(event3); err != nil {
		t.Fatalf("Transform content_block_delta failed: %v", err)
	}

	// Simulate content block stop
	contentBlockStop := `{"type":"content_block_stop","index":0}`
	event4 := &sse.Event{Data: contentBlockStop}
	if err := transformer.Transform(event4); err != nil {
		t.Fatalf("Transform content_block_stop failed: %v", err)
	}

	// Simulate message_delta with output token count
	messageDelta := `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":3639,"output_tokens":188}}`
	event5 := &sse.Event{Data: messageDelta}
	if err := transformer.Transform(event5); err != nil {
		t.Fatalf("Transform message_delta failed: %v", err)
	}

	output := buf.String()

	// Verify the final response.completed event contains all token counts
	// Expected: input_tokens=5047 (3639+1408 cache), output_tokens=188, total_tokens=5235, cached_tokens=1408
	if !strings.Contains(output, `"input_tokens":5047`) {
		t.Errorf("Expected input_tokens=5047 (3639+1408 cache) in output, got:\n%s", output)
	}
	if !strings.Contains(output, `"output_tokens":188`) {
		t.Errorf("Expected output_tokens=188 in output, got:\n%s", output)
	}
	if !strings.Contains(output, `"total_tokens":5235`) {
		t.Errorf("Expected total_tokens=5235 (5047+188) in output, got:\n%s", output)
	}
	if !strings.Contains(output, `"cached_tokens":1408`) {
		t.Errorf("Expected cached_tokens=1408 in output, got:\n%s", output)
	}
}
