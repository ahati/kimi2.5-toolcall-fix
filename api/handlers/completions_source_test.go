package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"ai-proxy/config"
	"ai-proxy/router"

	"github.com/gin-gonic/gin"
)

// =============================================================================
// Row 2: Chat Completions Source Tests
// =============================================================================
//
// This test file covers protocol conversions where OpenAI Chat Completions API is the
// source protocol. It tests the CompletionsHandler's TransformRequest method.
//
// Matrix:
//   Row 2, Col 1: Chat Completions → Anthropic
//   Row 2, Col 2: Chat Completions → Chat Completions (Passthrough)
//   Row 2, Col 3: Chat Completions → Responses
//
// Reference: docs/protocol_conversion_guide.md Sections 4 and 7

// -----------------------------------------------------------------------------
// Column 1: Chat Completions → Anthropic
// -----------------------------------------------------------------------------

// TestOpenAIToAnthropic_BasicRequest tests basic conversion from Chat Completions
// to Anthropic Messages format. System messages extracted to system field,
// max_tokens preserved.
func TestOpenAIToAnthropic_BasicRequest(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"anthropic": "https://api.test.com/anthropic/v1/messages",
	}, "anthropic")

	handler := &CompletionsHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "anthropic",
		},
	}

	request := `{
		"model": "test-model",
		"messages": [
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": "Hello, world!"}
		],
		"max_tokens": 1024,
		"stream": true
	}`

	transformed, err := handler.TransformRequest([]byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse transformed request: %v", err)
	}

	// Verify system field extracted from system message
	if req["system"] != "You are a helpful assistant." {
		t.Errorf("Expected system field from system message, got %v", req["system"])
	}

	// Verify messages without system message
	messages, ok := req["messages"].([]interface{})
	if !ok {
		t.Fatal("Expected messages array")
	}
	if len(messages) != 1 {
		t.Errorf("Expected 1 message (user only), got %d", len(messages))
	}

	// Verify max_tokens is preserved
	if req["max_tokens"].(float64) != 1024 {
		t.Errorf("Expected max_tokens 1024, got %v", req["max_tokens"])
	}
}

// TestOpenAIToAnthropic_TemperatureClamping tests that OpenAI temperature (0-2)
// is clamped to Anthropic range (0-1).
func TestOpenAIToAnthropic_TemperatureClamping(t *testing.T) {
	tests := []struct {
		name     string
		input    float64
		expected float64
	}{
		{"Within range", 0.5, 0.5},
		{"At Anthropic max", 1.0, 1.0},
		{"Over Anthropic max", 1.8, 1.0},
		{"At OpenAI max", 2.0, 1.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := mockMultiProtocolProvider("test", map[string]string{
				"anthropic": "https://api.test.com/anthropic/v1/messages",
			}, "anthropic")

			handler := &CompletionsHandler{
				cfg: &config.Config{},
				route: &router.ResolvedRoute{
					Provider:       provider,
					Model:          "test-model",
					OutputProtocol: "anthropic",
				},
			}

			request := `{
				"model": "test-model",
				"messages": [{"role": "user", "content": "hi"}],
				"temperature": ` + jsonMarshal(tt.input) + `,
				"stream": true
			}`

			transformed, err := handler.TransformRequest([]byte(request))
			if err != nil {
				t.Fatalf("TransformRequest failed: %v", err)
			}

			var req map[string]interface{}
			if err := json.Unmarshal(transformed, &req); err != nil {
				t.Fatalf("Failed to parse: %v", err)
			}

			got, ok := req["temperature"].(float64)
			if !ok {
				t.Fatalf("Expected temperature to be float64, got %T", req["temperature"])
			}
			if got != tt.expected {
				t.Errorf("Temperature: got %v, want %v", got, tt.expected)
			}
		})
	}
}

// TestOpenAIToAnthropic_ToolChoiceNone tests that tool_choice:"none" causes
// tools to be dropped from the request.
func TestOpenAIToAnthropic_ToolChoiceNone(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"anthropic": "https://api.test.com/anthropic/v1/messages",
	}, "anthropic")

	handler := &CompletionsHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "anthropic",
		},
	}

	request := `{
		"model": "test-model",
		"messages": [{"role": "user", "content": "hi"}],
		"tools": [{
			"type": "function",
			"function": {
				"name": "test",
				"description": "A test tool",
				"parameters": {"type": "object"}
			}
		}],
		"tool_choice": "none",
		"stream": true
	}`

	transformed, err := handler.TransformRequest([]byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse: %v", err)
	}

	// Tools should be dropped when tool_choice is "none"
	if _, hasTools := req["tools"]; hasTools {
		t.Error("Expected tools to be dropped when tool_choice is 'none'")
	}

	// tool_choice should also be nil/absent
	if _, hasToolChoice := req["tool_choice"]; hasToolChoice {
		t.Error("Expected tool_choice to be absent when 'none'")
	}
}

// TestOpenAIToAnthropic_DefaultMaxTokens tests that default max_tokens is injected
// if not provided in the request.
func TestOpenAIToAnthropic_DefaultMaxTokens(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"anthropic": "https://api.test.com/anthropic/v1/messages",
	}, "anthropic")

	handler := &CompletionsHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "anthropic",
		},
	}

	request := `{
		"model": "test-model",
		"messages": [{"role": "user", "content": "hi"}],
		"stream": true
	}`

	transformed, err := handler.TransformRequest([]byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse: %v", err)
	}

	// max_tokens should be injected with default value
	if _, hasMaxTokens := req["max_tokens"]; !hasMaxTokens {
		t.Error("Expected max_tokens to be injected (Anthropic requires it)")
	}
}

// TestOpenAIToAnthropic_MultipleSystemMessages tests that multiple system messages
// are joined with double newlines.
func TestOpenAIToAnthropic_MultipleSystemMessages(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"anthropic": "https://api.test.com/anthropic/v1/messages",
	}, "anthropic")

	handler := &CompletionsHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "anthropic",
		},
	}

	request := `{
		"model": "test-model",
		"messages": [
			{"role": "system", "content": "Be helpful"},
			{"role": "system", "content": "Be concise"},
			{"role": "user", "content": "hi"}
		],
		"stream": true
	}`

	transformed, err := handler.TransformRequest([]byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse: %v", err)
	}

	// System messages should be joined
	expected := "Be helpful\n\nBe concise"
	if req["system"] != expected {
		t.Errorf("Expected system '%s', got %v", expected, req["system"])
	}

	// Messages should not contain system role
	messages := req["messages"].([]interface{})
	for _, m := range messages {
		msg := m.(map[string]interface{})
		if msg["role"] == "system" {
			t.Error("System messages should be extracted, not in messages array")
		}
	}
}

// TestOpenAIToAnthropic_ImageDataURI tests that image_url with data URI is converted
// to Anthropic image block format.
func TestOpenAIToAnthropic_ImageDataURI(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"anthropic": "https://api.test.com/anthropic/v1/messages",
	}, "anthropic")

	handler := &CompletionsHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "anthropic",
		},
	}

	request := `{
		"model": "test-model",
		"messages": [{
			"role": "user",
			"content": [
				{"type": "text", "text": "What is this?"},
				{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}}
			]
		}],
		"stream": true
	}`

	transformed, err := handler.TransformRequest([]byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse: %v", err)
	}

	messages := req["messages"].([]interface{})
	if len(messages) != 1 {
		t.Fatalf("Expected 1 message, got %d", len(messages))
	}

	msg := messages[0].(map[string]interface{})
	content, ok := msg["content"].([]interface{})
	if !ok {
		t.Fatal("Expected content array")
	}

	// Find image block
	var foundImage bool
	for _, c := range content {
		block := c.(map[string]interface{})
		if block["type"] == "image" {
			foundImage = true
			source := block["source"].(map[string]interface{})
			if source["type"] != "base64" {
				t.Errorf("Expected source type 'base64', got %v", source["type"])
			}
			if source["media_type"] != "image/png" {
				t.Errorf("Expected media_type 'image/png', got %v", source["media_type"])
			}
			if source["data"] != "abc123" {
				t.Errorf("Expected data 'abc123', got %v", source["data"])
			}
		}
	}

	if !foundImage {
		t.Error("Expected to find image block from image_url")
	}
}

// TestOpenAIToAnthropic_ConsecutiveToolMessages tests that consecutive tool messages
// are batched into a single user message with multiple tool_result blocks.
func TestOpenAIToAnthropic_ConsecutiveToolMessages(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"anthropic": "https://api.test.com/anthropic/v1/messages",
	}, "anthropic")

	handler := &CompletionsHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "anthropic",
		},
	}

	request := `{
		"model": "test-model",
		"messages": [
			{"role": "user", "content": "Check both"},
			{
				"role": "assistant",
				"content": null,
				"tool_calls": [
					{"id": "call_1", "type": "function", "function": {"name": "fn1", "arguments": "{}"}},
					{"id": "call_2", "type": "function", "function": {"name": "fn2", "arguments": "{}"}}
				]
			},
			{"role": "tool", "tool_call_id": "call_1", "content": "result1"},
			{"role": "tool", "tool_call_id": "call_2", "content": "result2"}
		],
		"stream": true
	}`

	transformed, err := handler.TransformRequest([]byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse: %v", err)
	}

	messages := req["messages"].([]interface{})

	// Should have: user, assistant with tool_use, user with tool_results
	if len(messages) < 3 {
		t.Errorf("Expected at least 3 messages, got %d", len(messages))
	}

	// Last message should be user with tool_result blocks
	lastMsg := messages[len(messages)-1].(map[string]interface{})
	if lastMsg["role"] != "user" {
		t.Errorf("Expected last message role 'user', got %v", lastMsg["role"])
	}

	content, ok := lastMsg["content"].([]interface{})
	if !ok {
		t.Fatal("Expected content array in last message")
	}

	// Count tool_result blocks
	var toolResultCount int
	for _, c := range content {
		block := c.(map[string]interface{})
		if block["type"] == "tool_result" {
			toolResultCount++
		}
	}

	if toolResultCount != 2 {
		t.Errorf("Expected 2 tool_result blocks (batched), got %d", toolResultCount)
	}
}

// TestOpenAIToAnthropic_ToolCallsToToolUse tests that tool_calls in assistant messages
// are converted to tool_use blocks.
func TestOpenAIToAnthropic_ToolCallsToToolUse(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"anthropic": "https://api.test.com/anthropic/v1/messages",
	}, "anthropic")

	handler := &CompletionsHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "anthropic",
		},
	}

	request := `{
		"model": "test-model",
		"messages": [
			{"role": "user", "content": "Calculate 2+2"},
			{
				"role": "assistant",
				"content": null,
				"tool_calls": [{
					"id": "call_123",
					"type": "function",
					"function": {
						"name": "calculator",
						"arguments": "{\"expr\": \"2+2\"}"
					}
				}]
			}
		],
		"stream": true
	}`

	transformed, err := handler.TransformRequest([]byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse: %v", err)
	}

	messages := req["messages"].([]interface{})
	if len(messages) < 2 {
		t.Fatalf("Expected at least 2 messages, got %d", len(messages))
	}

	// Assistant message should have tool_use blocks
	assistantMsg := messages[1].(map[string]interface{})
	if assistantMsg["role"] != "assistant" {
		t.Errorf("Expected assistant role, got %v", assistantMsg["role"])
	}

	content, ok := assistantMsg["content"].([]interface{})
	if !ok {
		t.Fatalf("Expected content array, got %v", assistantMsg["content"])
	}

	var foundToolUse bool
	for _, c := range content {
		block := c.(map[string]interface{})
		if block["type"] == "tool_use" {
			foundToolUse = true
			if block["id"] != "call_123" {
				t.Errorf("Expected tool_use id 'call_123', got %v", block["id"])
			}
			if block["name"] != "calculator" {
				t.Errorf("Expected tool_use name 'calculator', got %v", block["name"])
			}
		}
	}

	if !foundToolUse {
		t.Error("Expected to find tool_use block from tool_calls")
	}
}

// TestOpenAIToAnthropic_ToolsConversion tests that OpenAI tools format is converted
// to Anthropic format (parameters → input_schema, drop type field).
func TestOpenAIToAnthropic_ToolsConversion(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"anthropic": "https://api.test.com/anthropic/v1/messages",
	}, "anthropic")

	handler := &CompletionsHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "anthropic",
		},
	}

	request := `{
		"model": "test-model",
		"messages": [{"role": "user", "content": "Search for cats"}],
		"tools": [{
			"type": "function",
			"function": {
				"name": "search",
				"description": "Search the web",
				"parameters": {
					"type": "object",
					"properties": {
						"query": {"type": "string"}
					},
					"required": ["query"]
				}
			}
		}],
		"stream": true
	}`

	transformed, err := handler.TransformRequest([]byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse: %v", err)
	}

	tools := req["tools"].([]interface{})
	if len(tools) != 1 {
		t.Fatalf("Expected 1 tool, got %d", len(tools))
	}

	tool := tools[0].(map[string]interface{})

	// Should have name at top level
	if tool["name"] != "search" {
		t.Errorf("Expected tool name 'search', got %v", tool["name"])
	}

	// Should have input_schema (not parameters)
	if _, hasInputSchema := tool["input_schema"]; !hasInputSchema {
		t.Error("Expected input_schema field (Anthropic format)")
	}

	// Should NOT have type="function" at top level (Anthropic doesn't use this)
	if tool["type"] == "function" {
		t.Error("Anthropic tools should not have type='function' at top level")
	}
}

// TestOpenAIToAnthropic_ToolChoiceRequired tests that tool_choice:"required" maps to "any".
func TestOpenAIToAnthropic_ToolChoiceRequired(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"anthropic": "https://api.test.com/anthropic/v1/messages",
	}, "anthropic")

	handler := &CompletionsHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "anthropic",
		},
	}

	request := `{
		"model": "test-model",
		"messages": [{"role": "user", "content": "hi"}],
		"tools": [{"type": "function", "function": {"name": "test", "parameters": {"type": "object"}}}],
		"tool_choice": "required",
		"stream": true
	}`

	transformed, err := handler.TransformRequest([]byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse: %v", err)
	}

	// tool_choice should be "any" (Anthropic equivalent of "required")
	tc := req["tool_choice"]
	switch v := tc.(type) {
	case string:
		if v != "any" {
			t.Errorf("Expected tool_choice 'any', got %v", v)
		}
	case map[string]interface{}:
		if v["type"] != "any" {
			t.Errorf("Expected tool_choice type 'any', got %v", v["type"])
		}
	default:
		t.Errorf("Unexpected tool_choice type: %T", tc)
	}
}

// TestOpenAIToAnthropic_UserToMetadata tests that user field maps to metadata.user_id.
func TestOpenAIToAnthropic_UserToMetadata(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"anthropic": "https://api.test.com/anthropic/v1/messages",
	}, "anthropic")

	handler := &CompletionsHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "anthropic",
		},
	}

	request := `{
		"model": "test-model",
		"messages": [{"role": "user", "content": "hi"}],
		"user": "user_123",
		"stream": true
	}`

	transformed, err := handler.TransformRequest([]byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse: %v", err)
	}

	metadata, ok := req["metadata"].(map[string]interface{})
	if !ok {
		t.Fatal("Expected metadata object")
	}

	if metadata["user_id"] != "user_123" {
		t.Errorf("Expected metadata.user_id 'user_123', got %v", metadata["user_id"])
	}
}

// TestOpenAIToAnthropic_DroppedFields tests that unsupported fields are dropped.
func TestOpenAIToAnthropic_DroppedFields(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"anthropic": "https://api.test.com/anthropic/v1/messages",
	}, "anthropic")

	handler := &CompletionsHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "anthropic",
		},
	}

	request := `{
		"model": "test-model",
		"messages": [{"role": "user", "content": "hi"}],
		"frequency_penalty": 0.5,
		"presence_penalty": 0.3,
		"n": 2,
		"response_format": {"type": "json_object"},
		"stream": true
	}`

	transformed, err := handler.TransformRequest([]byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse: %v", err)
	}

	// These fields should be dropped
	droppedFields := []string{"frequency_penalty", "presence_penalty", "n", "response_format"}
	for _, field := range droppedFields {
		if _, exists := req[field]; exists {
			t.Errorf("Field '%s' should be dropped for Anthropic target", field)
		}
	}
}

// -----------------------------------------------------------------------------
// Column 2: Chat Completions → Chat Completions (Passthrough)
// -----------------------------------------------------------------------------

// TestOpenAIToOpenAI_Passthrough tests that Chat Completions → Chat Completions
// is passthrough with model replacement.
func TestOpenAIToOpenAI_Passthrough(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"openai": "https://api.test.com/v1/chat/completions",
	}, "openai")

	handler := &CompletionsHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "upstream-model",
			OutputProtocol: "openai",
			IsPassthrough:  true,
		},
	}

	request := `{
		"model": "downstream-model",
		"messages": [
			{"role": "system", "content": "Be helpful"},
			{"role": "user", "content": "hi"}
		],
		"tools": [{
			"type": "function",
			"function": {
				"name": "test",
				"parameters": {"type": "object"}
			}
		}],
		"stream": true
	}`

	transformed, err := handler.TransformRequest([]byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse: %v", err)
	}

	// Model should be updated to upstream model
	if req["model"] != "upstream-model" {
		t.Errorf("Expected model 'upstream-model', got %v", req["model"])
	}

	// Messages should remain unchanged
	messages := req["messages"].([]interface{})
	if len(messages) != 2 {
		t.Errorf("Expected 2 messages, got %d", len(messages))
	}

	// Tools should remain in OpenAI format
	tools := req["tools"].([]interface{})
	if len(tools) != 1 {
		t.Errorf("Expected 1 tool, got %d", len(tools))
	}

	tool := tools[0].(map[string]interface{})
	if tool["type"] != "function" {
		t.Errorf("Expected tool type 'function', got %v", tool["type"])
	}
}

// -----------------------------------------------------------------------------
// Column 3: Chat Completions → Responses
// -----------------------------------------------------------------------------

// TestOpenAIToResponses_BasicRequest tests basic conversion from Chat Completions
// to Responses API format. System messages → instructions, messages → input.
func TestOpenAIToResponses_BasicRequest(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"responses": "https://api.test.com/v1/responses",
	}, "responses")

	handler := &CompletionsHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "responses",
		},
	}

	request := `{
		"model": "test-model",
		"messages": [
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": "Hello, world!"}
		],
		"max_tokens": 1024,
		"stream": true
	}`

	transformed, err := handler.TransformRequest([]byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse transformed request: %v", err)
	}

	// Verify instructions field from system message
	if req["instructions"] != "You are a helpful assistant." {
		t.Errorf("Expected instructions from system message, got %v", req["instructions"])
	}

	// Verify input field (can be string or array)
	input := req["input"]
	switch v := input.(type) {
	case string:
		if v != "Hello, world!" {
			t.Errorf("Expected input 'Hello, world!', got %v", v)
		}
	case []interface{}:
		if len(v) != 1 {
			t.Errorf("Expected 1 input item, got %d", len(v))
		}
	}

	// Verify max_output_tokens from max_tokens
	if req["max_output_tokens"].(float64) != 1024 {
		t.Errorf("Expected max_output_tokens 1024, got %v", req["max_output_tokens"])
	}
}

// TestOpenAIToResponses_SystemToInstructions tests that system messages become instructions.
func TestOpenAIToResponses_SystemToInstructions(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"responses": "https://api.test.com/v1/responses",
	}, "responses")

	handler := &CompletionsHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "responses",
		},
	}

	request := `{
		"model": "test-model",
		"messages": [
			{"role": "system", "content": "Be helpful"},
			{"role": "system", "content": "Be concise"},
			{"role": "user", "content": "hi"}
		],
		"stream": true
	}`

	transformed, err := handler.TransformRequest([]byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse: %v", err)
	}

	// Multiple system messages should be joined
	expected := "Be helpful\n\nBe concise"
	if req["instructions"] != expected {
		t.Errorf("Expected instructions '%s', got %v", expected, req["instructions"])
	}

	// Input should not contain system messages
	input, ok := req["input"].([]interface{})
	if ok {
		for _, item := range input {
			i := item.(map[string]interface{})
			if i["role"] == "system" {
				t.Error("System messages should be extracted to instructions, not in input")
			}
		}
	}
}

// TestOpenAIToResponses_ToolCalls tests that tool_calls become function_call items.
func TestOpenAIToResponses_ToolCalls(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"responses": "https://api.test.com/v1/responses",
	}, "responses")

	handler := &CompletionsHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "responses",
		},
	}

	request := `{
		"model": "test-model",
		"messages": [
			{"role": "user", "content": "Calculate 2+2"},
			{
				"role": "assistant",
				"content": null,
				"tool_calls": [{
					"id": "call_123",
					"type": "function",
					"function": {
						"name": "calculator",
						"arguments": "{\"expr\": \"2+2\"}"
					}
				}]
			}
		],
		"stream": true
	}`

	transformed, err := handler.TransformRequest([]byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse: %v", err)
	}

	input := req["input"].([]interface{})

	// Find function_call item
	var foundFunctionCall bool
	for _, item := range input {
		i := item.(map[string]interface{})
		if i["type"] == "function_call" {
			foundFunctionCall = true
			if i["call_id"] != "call_123" {
				t.Errorf("Expected call_id 'call_123', got %v", i["call_id"])
			}
			if i["name"] != "calculator" {
				t.Errorf("Expected name 'calculator', got %v", i["name"])
			}
			if i["arguments"] != "{\"expr\": \"2+2\"}" {
				t.Errorf("Expected arguments, got %v", i["arguments"])
			}
		}
	}

	if !foundFunctionCall {
		t.Error("Expected to find function_call item from tool_calls")
	}
}

// TestOpenAIToResponses_ToolMessage tests that tool role messages become function_call_output items.
func TestOpenAIToResponses_ToolMessage(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"responses": "https://api.test.com/v1/responses",
	}, "responses")

	handler := &CompletionsHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "responses",
		},
	}

	request := `{
		"model": "test-model",
		"messages": [
			{"role": "user", "content": "Calculate"},
			{
				"role": "assistant",
				"content": null,
				"tool_calls": [{"id": "call_456", "type": "function", "function": {"name": "calc", "arguments": "{}"}}]
			},
			{"role": "tool", "tool_call_id": "call_456", "content": "4"}
		],
		"stream": true
	}`

	transformed, err := handler.TransformRequest([]byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse: %v", err)
	}

	input := req["input"].([]interface{})

	// Find function_call_output item
	var foundFunctionCallOutput bool
	for _, item := range input {
		i := item.(map[string]interface{})
		if i["type"] == "function_call_output" {
			foundFunctionCallOutput = true
			if i["call_id"] != "call_456" {
				t.Errorf("Expected call_id 'call_456', got %v", i["call_id"])
			}
			if i["output"] != "4" {
				t.Errorf("Expected output '4', got %v", i["output"])
			}
		}
	}

	if !foundFunctionCallOutput {
		t.Error("Expected to find function_call_output item from tool message")
	}
}

// TestOpenAIToResponses_Tools tests that tools are converted to Responses format.
func TestOpenAIToResponses_Tools(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"responses": "https://api.test.com/v1/responses",
	}, "responses")

	handler := &CompletionsHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "responses",
		},
	}

	request := `{
		"model": "test-model",
		"messages": [{"role": "user", "content": "Search"}],
		"tools": [{
			"type": "function",
			"function": {
				"name": "search",
				"description": "Search the web",
				"parameters": {
					"type": "object",
					"properties": {"query": {"type": "string"}}
				}
			}
		}],
		"stream": true
	}`

	transformed, err := handler.TransformRequest([]byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse: %v", err)
	}

	tools := req["tools"].([]interface{})
	if len(tools) != 1 {
		t.Fatalf("Expected 1 tool, got %d", len(tools))
	}

	tool := tools[0].(map[string]interface{})

	// Should have type, name, description, parameters at top level (flattened)
	if tool["type"] != "function" {
		t.Errorf("Expected type 'function', got %v", tool["type"])
	}
	if tool["name"] != "search" {
		t.Errorf("Expected name 'search', got %v", tool["name"])
	}
	if tool["description"] != "Search the web" {
		t.Errorf("Expected description, got %v", tool["description"])
	}

	// Parameters should be at top level (not nested in function)
	if _, hasParameters := tool["parameters"]; !hasParameters {
		t.Error("Expected parameters at top level")
	}

	// Should NOT have nested function object
	if _, hasFunction := tool["function"]; hasFunction {
		t.Error("Should not have nested 'function' object in Responses format")
	}
}

// TestOpenAIToResponses_ToolChoice tests tool_choice conversion.
func TestOpenAIToResponses_ToolChoice(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected interface{}
	}{
		{
			name:     "auto",
			input:    `"tool_choice": "auto"`,
			expected: "auto",
		},
		{
			name:     "required",
			input:    `"tool_choice": "required"`,
			expected: "required",
		},
		{
			name:     "none",
			input:    `"tool_choice": "none"`,
			expected: "none",
		},
		{
			name:  "specific function",
			input: `"tool_choice": {"type": "function", "function": {"name": "search"}}`,
			expected: map[string]interface{}{
				"type": "function",
				"name": "search",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := mockMultiProtocolProvider("test", map[string]string{
				"responses": "https://api.test.com/v1/responses",
			}, "responses")

			handler := &CompletionsHandler{
				cfg: &config.Config{},
				route: &router.ResolvedRoute{
					Provider:       provider,
					Model:          "test-model",
					OutputProtocol: "responses",
				},
			}

			request := `{
				"model": "test-model",
				"messages": [{"role": "user", "content": "hi"}],
				"tools": [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}],
				` + tt.input + `,
				"stream": true
			}`

			transformed, err := handler.TransformRequest([]byte(request))
			if err != nil {
				t.Fatalf("TransformRequest failed: %v", err)
			}

			var req map[string]interface{}
			if err := json.Unmarshal(transformed, &req); err != nil {
				t.Fatalf("Failed to parse: %v", err)
			}

			tc := req["tool_choice"]
			switch expected := tt.expected.(type) {
			case string:
				if tc != expected {
					t.Errorf("Expected tool_choice '%s', got %v", expected, tc)
				}
			case map[string]interface{}:
				tcMap, ok := tc.(map[string]interface{})
				if !ok {
					t.Fatalf("Expected tool_choice object, got %T", tc)
				}
				if tcMap["type"] != expected["type"] {
					t.Errorf("Expected type '%v', got %v", expected["type"], tcMap["type"])
				}
				if tcMap["name"] != expected["name"] {
					t.Errorf("Expected name '%v', got %v", expected["name"], tcMap["name"])
				}
			}
		})
	}
}

// TestOpenAIToResponses_UserToMetadata tests that user field maps to metadata.user_id.
func TestOpenAIToResponses_UserToMetadata(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"responses": "https://api.test.com/v1/responses",
	}, "responses")

	handler := &CompletionsHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "responses",
		},
	}

	request := `{
		"model": "test-model",
		"messages": [{"role": "user", "content": "hi"}],
		"user": "user_456",
		"stream": true
	}`

	transformed, err := handler.TransformRequest([]byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse: %v", err)
	}

	metadata, ok := req["metadata"].(map[string]interface{})
	if !ok {
		t.Fatal("Expected metadata object")
	}

	if metadata["user_id"] != "user_456" {
		t.Errorf("Expected metadata.user_id 'user_456', got %v", metadata["user_id"])
	}
}

// TestOpenAIToResponses_DroppedFields tests that unsupported fields are dropped.
func TestOpenAIToResponses_DroppedFields(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"responses": "https://api.test.com/v1/responses",
	}, "responses")

	handler := &CompletionsHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "responses",
		},
	}

	request := `{
		"model": "test-model",
		"messages": [{"role": "user", "content": "hi"}],
		"stop": ["END"],
		"n": 2,
		"response_format": {"type": "json_object"},
		"stream": true
	}`

	transformed, err := handler.TransformRequest([]byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse: %v", err)
	}

	// These fields should be dropped
	droppedFields := []string{"stop", "n", "response_format"}
	for _, field := range droppedFields {
		if _, exists := req[field]; exists {
			t.Errorf("Field '%s' should be dropped for Responses target", field)
		}
	}
}

// -----------------------------------------------------------------------------
// Cross-Protocol Tests
// -----------------------------------------------------------------------------

// TestCompletionsSource_UpstreamURL tests correct endpoint selection.
func TestCompletionsSource_UpstreamURL(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"openai":    "https://api.test.com/v1/chat/completions",
		"anthropic": "https://api.test.com/anthropic/v1/messages",
		"responses": "https://api.test.com/v1/responses",
	}, "openai")

	tests := []struct {
		name           string
		outputProtocol string
		wantURL        string
	}{
		{
			name:           "OpenAI protocol",
			outputProtocol: "openai",
			wantURL:        "https://api.test.com/v1/chat/completions",
		},
		{
			name:           "Anthropic protocol",
			outputProtocol: "anthropic",
			wantURL:        "https://api.test.com/anthropic/v1/messages",
		},
		{
			name:           "Responses protocol",
			outputProtocol: "responses",
			wantURL:        "https://api.test.com/v1/responses",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			handler := &CompletionsHandler{
				cfg: &config.Config{},
				route: &router.ResolvedRoute{
					Provider:       provider,
					Model:          "test-model",
					OutputProtocol: tt.outputProtocol,
				},
			}

			got := handler.UpstreamURL()
			if got != tt.wantURL {
				t.Errorf("UpstreamURL() = %s, want %s", got, tt.wantURL)
			}
		})
	}
}

// TestCompletionsSource_ForwardHeaders tests header forwarding based on OutputProtocol.
func TestCompletionsSource_ForwardHeaders(t *testing.T) {
	tests := []struct {
		name                   string
		outputProtocol         string
		expectAnthropicHeaders bool
	}{
		{
			name:                   "OpenAI target - no Anthropic headers",
			outputProtocol:         "openai",
			expectAnthropicHeaders: false,
		},
		{
			name:                   "Anthropic target - forward Anthropic headers",
			outputProtocol:         "anthropic",
			expectAnthropicHeaders: true,
		},
		{
			name:                   "Responses target - no Anthropic headers",
			outputProtocol:         "responses",
			expectAnthropicHeaders: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := mockMultiProtocolProvider("test", map[string]string{
				"openai":    "https://api.test.com/v1/chat/completions",
				"anthropic": "https://api.test.com/anthropic/v1/messages",
				"responses": "https://api.test.com/v1/responses",
			}, "openai")

			handler := &CompletionsHandler{
				cfg: &config.Config{},
				route: &router.ResolvedRoute{
					Provider:       provider,
					Model:          "test-model",
					OutputProtocol: tt.outputProtocol,
				},
			}

			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			c.Request = httptest.NewRequest(http.MethodPost, "/", nil)
			c.Request.Header.Set("X-Custom", "value")
			c.Request.Header.Set("Anthropic-Version", "2023-06-01")
			c.Request.Header.Set("Anthropic-Beta", "messages-2024-01-01")

			upstreamReq := httptest.NewRequest(http.MethodPost, "https://upstream.example.com", nil)
			handler.ForwardHeaders(c, upstreamReq)

			if upstreamReq.Header.Get("X-Custom") != "value" {
				t.Error("X-Custom header should be forwarded")
			}

			gotAnthropicVersion := upstreamReq.Header.Get("Anthropic-Version")
			if tt.expectAnthropicHeaders {
				if gotAnthropicVersion != "2023-06-01" {
					t.Errorf("Anthropic-Version should be forwarded for Anthropic target, got '%s'", gotAnthropicVersion)
				}
			} else {
				if gotAnthropicVersion != "" {
					t.Errorf("Anthropic-Version should NOT be forwarded for non-Anthropic target, got '%s'", gotAnthropicVersion)
				}
			}
		})
	}
}

// TestCompletionsSource_NilRoutePassthrough tests nil route behavior.
func TestCompletionsSource_NilRoutePassthrough(t *testing.T) {
	handler := &CompletionsHandler{
		cfg:   &config.Config{},
		route: nil,
	}

	request := `{"model": "test", "messages": [{"role": "user", "content": "hi"}], "stream": true}`

	transformed, err := handler.TransformRequest([]byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	// Nil route should pass through unchanged
	if string(transformed) != request {
		t.Errorf("Nil route should pass through unchanged")
	}
}

// TestCompletionsSource_MultipleToolCallsInAssistant tests multiple tool_calls handling.
func TestCompletionsSource_MultipleToolCallsInAssistant(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"anthropic": "https://api.test.com/anthropic/v1/messages",
	}, "anthropic")

	handler := &CompletionsHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "anthropic",
		},
	}

	request := `{
		"model": "test-model",
		"messages": [
			{"role": "user", "content": "Do both"},
			{
				"role": "assistant",
				"content": null,
				"tool_calls": [
					{"id": "call_1", "type": "function", "function": {"name": "fn1", "arguments": "{\"a\": 1}"}},
					{"id": "call_2", "type": "function", "function": {"name": "fn2", "arguments": "{\"b\": 2}"}}
				]
			}
		],
		"stream": true
	}`

	transformed, err := handler.TransformRequest([]byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse: %v", err)
	}

	messages := req["messages"].([]interface{})
	assistantMsg := messages[1].(map[string]interface{})
	content := assistantMsg["content"].([]interface{})

	// Count tool_use blocks
	var toolUseCount int
	for _, c := range content {
		block := c.(map[string]interface{})
		if block["type"] == "tool_use" {
			toolUseCount++
		}
	}

	if toolUseCount != 2 {
		t.Errorf("Expected 2 tool_use blocks, got %d", toolUseCount)
	}
}

// TestCompletionsSource_MaxTokensFallback tests max_tokens handling.
func TestCompletionsSource_MaxTokensFallback(t *testing.T) {
	tests := []struct {
		name          string
		request       string
		expectedValue float64
	}{
		{
			name:          "max_tokens provided",
			request:       `{"model": "test", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 256}`,
			expectedValue: 256,
		},
		{
			name:          "no max_tokens uses default",
			request:       `{"model": "test", "messages": [{"role": "user", "content": "hi"}]}`,
			expectedValue: 32768, // DefaultAnthropicMaxTokens
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := mockMultiProtocolProvider("test", map[string]string{
				"anthropic": "https://api.test.com/anthropic/v1/messages",
			}, "anthropic")

			handler := &CompletionsHandler{
				cfg: &config.Config{},
				route: &router.ResolvedRoute{
					Provider:       provider,
					Model:          "test-model",
					OutputProtocol: "anthropic",
				},
			}

			transformed, err := handler.TransformRequest([]byte(tt.request))
			if err != nil {
				t.Fatalf("TransformRequest failed: %v", err)
			}

			var req map[string]interface{}
			if err := json.Unmarshal(transformed, &req); err != nil {
				t.Fatalf("Failed to parse: %v", err)
			}

			if req["max_tokens"].(float64) != tt.expectedValue {
				t.Errorf("Expected max_tokens %v, got %v", tt.expectedValue, req["max_tokens"])
			}
		})
	}
}

// Helper function to safely marshal values
func jsonMarshal(v interface{}) string {
	b, _ := json.Marshal(v)
	return string(b)
}
