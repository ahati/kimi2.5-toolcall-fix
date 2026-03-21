package convert

import (
	"encoding/json"
	"testing"

	"ai-proxy/types"
)

func TestAnthropicToChatRequest_Simple(t *testing.T) {
	req := &types.MessageRequest{
		Model:    "claude-3-opus",
		MaxTokens: 1024,
		Messages: []types.MessageInput{
			{Role: "user", Content: "Hello"},
		},
	}

	out, err := AnthropicToChatRequest(req)
	if err != nil {
		t.Fatalf("AnthropicToChatRequest failed: %v", err)
	}

	if out.Model != "claude-3-opus" {
		t.Errorf("expected model claude-3-opus, got %s", out.Model)
	}
	if out.MaxTokens != 1024 {
		t.Errorf("expected max_tokens 1024, got %d", out.MaxTokens)
	}
	if len(out.Messages) != 1 {
		t.Errorf("expected 1 message, got %d", len(out.Messages))
	}
	if out.Messages[0].Role != "user" {
		t.Errorf("expected user role, got %s", out.Messages[0].Role)
	}
}

func TestAnthropicToChatRequest_WithTools(t *testing.T) {
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
				InputSchema: json.RawMessage(`{"type": "object", "properties": {"location": {"type": "string"}}}`),
			},
		},
		ToolChoice: &types.ToolChoice{Type: "auto"},
	}

	out, err := AnthropicToChatRequest(req)
	if err != nil {
		t.Fatalf("AnthropicToChatRequest failed: %v", err)
	}

	if len(out.Tools) != 1 {
		t.Errorf("expected 1 tool, got %d", len(out.Tools))
	}
	if out.Tools[0].Function.Name != "get_weather" {
		t.Errorf("expected tool name get_weather, got %s", out.Tools[0].Function.Name)
	}
}

func TestAnthropicToChatRequest_WithSystem(t *testing.T) {
	req := &types.MessageRequest{
		Model:    "claude-3-opus",
		MaxTokens: 1024,
		System:   "You are a helpful assistant.",
		Messages: []types.MessageInput{
			{Role: "user", Content: "Hello"},
		},
	}

	out, err := AnthropicToChatRequest(req)
	if err != nil {
		t.Fatalf("AnthropicToChatRequest failed: %v", err)
	}

	if len(out.Messages) != 2 {
		t.Errorf("expected 2 messages (system + user), got %d", len(out.Messages))
	}
	if out.Messages[0].Role != "system" {
		t.Errorf("expected first message to be system, got %s", out.Messages[0].Role)
	}
}

func TestAnthropicToChatRequest_ToolResult(t *testing.T) {
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

	out, err := AnthropicToChatRequest(req)
	if err != nil {
		t.Fatalf("AnthropicToChatRequest failed: %v", err)
	}

	// Should have one tool message
	if len(out.Messages) != 1 {
		t.Errorf("expected 1 message, got %d", len(out.Messages))
	}
	if out.Messages[0].Role != "tool" {
		t.Errorf("expected tool role, got %s", out.Messages[0].Role)
	}
	if out.Messages[0].ToolCallID != "tool_123" {
		t.Errorf("expected tool_call_id tool_123, got %s", out.Messages[0].ToolCallID)
	}
}

func TestAnthropicToChatRequest_AssistantWithToolUse(t *testing.T) {
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

	out, err := AnthropicToChatRequest(req)
	if err != nil {
		t.Fatalf("AnthropicToChatRequest failed: %v", err)
	}

	if len(out.Messages) != 1 {
		t.Errorf("expected 1 message, got %d", len(out.Messages))
	}
	if out.Messages[0].Role != "assistant" {
		t.Errorf("expected assistant role, got %s", out.Messages[0].Role)
	}
	if len(out.Messages[0].ToolCalls) != 1 {
		t.Errorf("expected 1 tool call, got %d", len(out.Messages[0].ToolCalls))
	}
	if out.Messages[0].ToolCalls[0].Function.Name != "get_weather" {
		t.Errorf("expected function name get_weather, got %s", out.Messages[0].ToolCalls[0].Function.Name)
	}
}