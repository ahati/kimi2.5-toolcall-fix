package convert

import (
	"testing"

	"ai-proxy/types"
)

func TestAnthropicToResponsesRequest_Simple(t *testing.T) {
	req := &types.MessageRequest{
		Model:    "claude-3-opus",
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
		Model:    "claude-3-opus",
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
		Model:    "claude-3-opus",
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
		Model:    "claude-3-opus",
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
		Model:    "claude-3-opus",
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