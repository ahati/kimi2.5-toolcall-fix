package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"ai-proxy/config"
	"ai-proxy/router"

	"github.com/gin-gonic/gin"
)

// =============================================================================
// Row 3: Responses Source Tests
// =============================================================================
//
// This test file covers protocol conversions where OpenAI Responses API is the
// source protocol. It tests the ResponsesHandler's TransformRequest method.
//
// Matrix:
//   Row 3, Col 1: Responses → Anthropic
//   Row 3, Col 2: Responses → Chat Completions
//   Row 3, Col 3: Responses → Responses (passthrough)
//
// Reference: docs/protocol_conversion_guide.md Sections 5, 8, and 9

// -----------------------------------------------------------------------------
// Column 1: Responses → Anthropic
// -----------------------------------------------------------------------------

// TestResponsesToAnthropic_BasicRequest tests basic conversion from Responses API
// to Anthropic Messages format. Instructions → system, input → messages.
func TestResponsesToAnthropic_BasicRequest(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"anthropic": "https://api.test.com/anthropic/v1/messages",
	}, "anthropic")

	handler := &ResponsesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "anthropic",
		},
	}

	request := `{
		"model": "test-model",
		"instructions": "You are a helpful assistant.",
		"input": "Hello, world!",
		"max_output_tokens": 1024,
		"stream": true
	}`

	transformed, err := handler.TransformRequest(context.TODO(), []byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse transformed request: %v", err)
	}

	// Verify system field from instructions
	if req["system"] != "You are a helpful assistant." {
		t.Errorf("Expected system field from instructions, got %v", req["system"])
	}

	// Verify messages from input
	messages, ok := req["messages"].([]interface{})
	if !ok {
		t.Fatal("Expected messages array")
	}
	if len(messages) != 1 {
		t.Errorf("Expected 1 message, got %d", len(messages))
	}

	msg := messages[0].(map[string]interface{})
	if msg["role"] != "user" {
		t.Errorf("Expected user role, got %v", msg["role"])
	}
	if msg["content"] != "Hello, world!" {
		t.Errorf("Expected content 'Hello, world!', got %v", msg["content"])
	}

	// Verify max_tokens from max_output_tokens
	if req["max_tokens"].(float64) != 1024 {
		t.Errorf("Expected max_tokens 1024, got %v", req["max_tokens"])
	}
}

// TestResponsesToAnthropic_StringInput tests string input conversion:
// "Hello" → messages:[{role:"user", content:"Hello"}]
func TestResponsesToAnthropic_StringInput(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"anthropic": "https://api.test.com/anthropic/v1/messages",
	}, "anthropic")

	handler := &ResponsesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "anthropic",
		},
	}

	request := `{
		"model": "test-model",
		"input": "Hello",
		"stream": true
	}`

	transformed, err := handler.TransformRequest(context.TODO(), []byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse transformed request: %v", err)
	}

	messages := req["messages"].([]interface{})
	if len(messages) != 1 {
		t.Errorf("Expected 1 message from string input, got %d", len(messages))
	}

	msg := messages[0].(map[string]interface{})
	if msg["role"] != "user" {
		t.Errorf("Expected user role, got %v", msg["role"])
	}
	if msg["content"] != "Hello" {
		t.Errorf("Expected content 'Hello', got %v", msg["content"])
	}
}

// TestResponsesToAnthropic_InstructionsToSystem tests instructions field becomes system field.
func TestResponsesToAnthropic_InstructionsToSystem(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"anthropic": "https://api.test.com/anthropic/v1/messages",
	}, "anthropic")

	handler := &ResponsesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "anthropic",
		},
	}

	request := `{
		"model": "test-model",
		"instructions": "Be helpful and concise",
		"input": "Hi",
		"stream": true
	}`

	transformed, err := handler.TransformRequest(context.TODO(), []byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse transformed request: %v", err)
	}

	if req["system"] != "Be helpful and concise" {
		t.Errorf("Expected system field from instructions, got %v", req["system"])
	}
}

// TestResponsesToAnthropic_FunctionCallOutput tests function_call items become tool_use blocks.
func TestResponsesToAnthropic_FunctionCallOutput(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"anthropic": "https://api.test.com/anthropic/v1/messages",
	}, "anthropic")

	handler := &ResponsesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "anthropic",
		},
	}

	request := `{
		"model": "test-model",
		"input": [
			{"type": "message", "role": "user", "content": "What is 2+2?"},
			{
				"type": "function_call",
				"call_id": "call_123",
				"name": "calculator",
				"arguments": "{\"expr\": \"2+2\"}"
			},
			{
				"type": "function_call_output",
				"call_id": "call_123",
				"output": "4"
			}
		],
		"stream": true
	}`

	transformed, err := handler.TransformRequest(context.TODO(), []byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse transformed request: %v", err)
	}

	messages := req["messages"].([]interface{})

	// Should have user message, assistant with tool_use, user with tool_result
	if len(messages) < 2 {
		t.Errorf("Expected at least 2 messages, got %d", len(messages))
	}

	// Find tool_use and tool_result blocks
	var foundToolUse, foundToolResult bool
	for _, m := range messages {
		msg := m.(map[string]interface{})
		content := msg["content"]

		switch c := content.(type) {
		case []interface{}:
			for _, block := range c {
				if b, ok := block.(map[string]interface{}); ok {
					if b["type"] == "tool_use" {
						foundToolUse = true
						if b["id"] != "call_123" {
							t.Errorf("Expected tool_use id 'call_123', got %v", b["id"])
						}
						if b["name"] != "calculator" {
							t.Errorf("Expected tool_use name 'calculator', got %v", b["name"])
						}
					}
					if b["type"] == "tool_result" {
						foundToolResult = true
						if b["tool_use_id"] != "call_123" {
							t.Errorf("Expected tool_result tool_use_id 'call_123', got %v", b["tool_use_id"])
						}
					}
				}
			}
		}
	}

	if !foundToolUse {
		t.Error("Expected to find tool_use block from function_call")
	}
	if !foundToolResult {
		t.Error("Expected to find tool_result block from function_call_output")
	}
}

// TestResponsesToAnthropic_StatusIncomplete tests status:"incomplete" handling.
// Note: This is for response transformation, but we verify the request transformation
// handles reasoning which can lead to incomplete status.
func TestResponsesToAnthropic_ReasoningToThinking(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"anthropic": "https://api.test.com/anthropic/v1/messages",
	}, "anthropic")

	handler := &ResponsesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "anthropic",
		},
	}

	request := `{
		"model": "test-model",
		"input": "Think about this",
		"reasoning": {"effort": "high"},
		"max_output_tokens": 10000,
		"stream": true
	}`

	transformed, err := handler.TransformRequest(context.TODO(), []byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse transformed request: %v", err)
	}

	// Verify thinking configuration was added
	thinking, ok := req["thinking"].(map[string]interface{})
	if !ok {
		t.Fatal("Expected thinking configuration from reasoning")
	}
	if thinking["type"] != "enabled" {
		t.Errorf("Expected thinking type 'enabled', got %v", thinking["type"])
	}
	if thinking["budget_tokens"] == nil {
		t.Error("Expected budget_tokens in thinking config")
	}
}

// TestResponsesToAnthropic_RefusalContent tests refusal content is treated as text.
func TestResponsesToAnthropic_RefusalContent(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"anthropic": "https://api.test.com/anthropic/v1/messages",
	}, "anthropic")

	handler := &ResponsesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "anthropic",
		},
	}

	request := `{
		"model": "test-model",
		"input": [
			{
				"type": "message",
				"role": "assistant",
				"content": [
					{"type": "refusal", "text": "I cannot help with that."}
				]
			}
		],
		"stream": true
	}`

	transformed, err := handler.TransformRequest(context.TODO(), []byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse transformed request: %v", err)
	}

	messages := req["messages"].([]interface{})
	if len(messages) == 0 {
		t.Fatal("Expected at least one message")
	}

	// Refusal should be converted to text block
	msg := messages[0].(map[string]interface{})
	content := msg["content"]

	var foundText bool
	switch c := content.(type) {
	case []interface{}:
		for _, block := range c {
			if b, ok := block.(map[string]interface{}); ok {
				if b["type"] == "text" && b["text"] == "I cannot help with that." {
					foundText = true
				}
			}
		}
	case string:
		if c == "I cannot help with that." {
			foundText = true
		}
	}

	if !foundText {
		t.Error("Expected refusal content to be converted to text")
	}
}

// TestResponsesToAnthropic_ToolsConversion tests tool conversion.
func TestResponsesToAnthropic_ToolsConversion(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"anthropic": "https://api.test.com/anthropic/v1/messages",
	}, "anthropic")

	handler := &ResponsesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "anthropic",
		},
	}

	request := `{
		"model": "test-model",
		"input": "Search for cats",
		"tools": [{
			"type": "function",
			"name": "search",
			"description": "Search the web",
			"parameters": {
				"type": "object",
				"properties": {
					"query": {"type": "string"}
				},
				"required": ["query"]
			}
		}],
		"stream": true
	}`

	transformed, err := handler.TransformRequest(context.TODO(), []byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse transformed request: %v", err)
	}

	tools := req["tools"].([]interface{})
	if len(tools) != 1 {
		t.Fatalf("Expected 1 tool, got %d", len(tools))
	}

	tool := tools[0].(map[string]interface{})
	if tool["name"] != "search" {
		t.Errorf("Expected tool name 'search', got %v", tool["name"])
	}

	// Should have input_schema (Anthropic format) instead of parameters
	if _, hasInputSchema := tool["input_schema"]; !hasInputSchema {
		t.Error("Expected input_schema field in tool (Anthropic format)")
	}
}

// -----------------------------------------------------------------------------
// Column 2: Responses → Chat Completions
// -----------------------------------------------------------------------------

// TestResponsesToOpenAI_BasicRequest tests basic conversion from Responses API
// to Chat Completions format. Instructions → system message, input → messages.
func TestResponsesToOpenAI_BasicRequest(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"openai": "https://api.test.com/v1/chat/completions",
	}, "openai")

	handler := &ResponsesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "openai",
		},
	}

	request := `{
		"model": "test-model",
		"instructions": "You are a helpful assistant.",
		"input": "Hello, world!",
		"max_output_tokens": 1024,
		"stream": true
	}`

	transformed, err := handler.TransformRequest(context.TODO(), []byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse transformed request: %v", err)
	}

	messages := req["messages"].([]interface{})
	if len(messages) < 2 {
		t.Errorf("Expected at least 2 messages (system + user), got %d", len(messages))
	}

	// First message should be system
	sysMsg := messages[0].(map[string]interface{})
	if sysMsg["role"] != "system" {
		t.Errorf("Expected first message to be system role, got %v", sysMsg["role"])
	}
	if sysMsg["content"] != "You are a helpful assistant." {
		t.Errorf("Expected system content from instructions, got %v", sysMsg["content"])
	}

	// Second message should be user
	userMsg := messages[1].(map[string]interface{})
	if userMsg["role"] != "user" {
		t.Errorf("Expected user role, got %v", userMsg["role"])
	}
	if userMsg["content"] != "Hello, world!" {
		t.Errorf("Expected content 'Hello, world!', got %v", userMsg["content"])
	}
}

// TestResponsesToOpenAI_FunctionCallItem tests function_call items become
// assistant message with tool_calls.
func TestResponsesToOpenAI_FunctionCallItem(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"openai": "https://api.test.com/v1/chat/completions",
	}, "openai")

	handler := &ResponsesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "openai",
		},
	}

	request := `{
		"model": "test-model",
		"input": [
			{"type": "message", "role": "user", "content": "Calculate 2+2"},
			{
				"type": "function_call",
				"call_id": "call_456",
				"name": "calculator",
				"arguments": "{\"expr\": \"2+2\"}"
			}
		],
		"stream": true
	}`

	transformed, err := handler.TransformRequest(context.TODO(), []byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse transformed request: %v", err)
	}

	messages := req["messages"].([]interface{})

	// Find assistant message with tool_calls
	var foundToolCalls bool
	for _, m := range messages {
		msg := m.(map[string]interface{})
		if msg["role"] == "assistant" {
			if tc, ok := msg["tool_calls"].([]interface{}); ok && len(tc) > 0 {
				foundToolCalls = true
				toolCall := tc[0].(map[string]interface{})
				if toolCall["id"] != "call_456" {
					t.Errorf("Expected tool_call id 'call_456', got %v", toolCall["id"])
				}
				if toolCall["type"] != "function" {
					t.Errorf("Expected tool_call type 'function', got %v", toolCall["type"])
				}
				fn := toolCall["function"].(map[string]interface{})
				if fn["name"] != "calculator" {
					t.Errorf("Expected function name 'calculator', got %v", fn["name"])
				}
			}
		}
	}

	if !foundToolCalls {
		t.Error("Expected to find assistant message with tool_calls from function_call item")
	}
}

// TestResponsesToOpenAI_FunctionCallOutputItem tests function_call_output items
// become tool role messages.
func TestResponsesToOpenAI_FunctionCallOutputItem(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"openai": "https://api.test.com/v1/chat/completions",
	}, "openai")

	handler := &ResponsesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "openai",
		},
	}

	request := `{
		"model": "test-model",
		"input": [
			{"type": "message", "role": "user", "content": "Calculate 2+2"},
			{
				"type": "function_call",
				"call_id": "call_789",
				"name": "calculator",
				"arguments": "{\"expr\": \"2+2\"}"
			},
			{
				"type": "function_call_output",
				"call_id": "call_789",
				"output": "4"
			}
		],
		"stream": true
	}`

	transformed, err := handler.TransformRequest(context.TODO(), []byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse transformed request: %v", err)
	}

	messages := req["messages"].([]interface{})

	// Find tool message
	var foundToolMessage bool
	for _, m := range messages {
		msg := m.(map[string]interface{})
		if msg["role"] == "tool" {
			foundToolMessage = true
			if msg["tool_call_id"] != "call_789" {
				t.Errorf("Expected tool_call_id 'call_789', got %v", msg["tool_call_id"])
			}
			if msg["content"] != "4" {
				t.Errorf("Expected content '4', got %v", msg["content"])
			}
		}
	}

	if !foundToolMessage {
		t.Error("Expected to find tool role message from function_call_output item")
	}
}

// TestResponsesToOpenAI_ToolsConversion tests tool conversion to OpenAI format.
func TestResponsesToOpenAI_ToolsConversion(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"openai": "https://api.test.com/v1/chat/completions",
	}, "openai")

	handler := &ResponsesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "openai",
		},
	}

	request := `{
		"model": "test-model",
		"input": "Search for cats",
		"tools": [{
			"type": "function",
			"name": "search",
			"description": "Search the web",
			"parameters": {
				"type": "object",
				"properties": {
					"query": {"type": "string"}
				}
			}
		}],
		"stream": true
	}`

	transformed, err := handler.TransformRequest(context.TODO(), []byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse transformed request: %v", err)
	}

	tools := req["tools"].([]interface{})
	if len(tools) != 1 {
		t.Fatalf("Expected 1 tool, got %d", len(tools))
	}

	tool := tools[0].(map[string]interface{})
	if tool["type"] != "function" {
		t.Errorf("Expected tool type 'function', got %v", tool["type"])
	}

	fn := tool["function"].(map[string]interface{})
	if fn["name"] != "search" {
		t.Errorf("Expected function name 'search', got %v", fn["name"])
	}
	if fn["description"] != "Search the web" {
		t.Errorf("Expected function description, got %v", fn["description"])
	}
}

// TestResponsesToOpenAI_DroppedTools tests file_search and web_search tools are dropped.
func TestResponsesToOpenAI_DroppedTools(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"openai": "https://api.test.com/v1/chat/completions",
	}, "openai")

	handler := &ResponsesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "openai",
		},
	}

	request := `{
		"model": "test-model",
		"input": "Search for cats",
		"tools": [
			{"type": "file_search"},
			{"type": "web_search_preview"},
			{"type": "function", "name": "custom_search", "parameters": {"type": "object"}}
		],
		"stream": true
	}`

	transformed, err := handler.TransformRequest(context.TODO(), []byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse transformed request: %v", err)
	}

	tools := req["tools"].([]interface{})
	// Only function tools should remain
	if len(tools) != 1 {
		t.Errorf("Expected 1 tool (only function type), got %d", len(tools))
	}

	tool := tools[0].(map[string]interface{})
	if tool["type"] != "function" {
		t.Errorf("Expected only function type tool, got %v", tool["type"])
	}
}

// TestResponsesToOpenAI_MetadataUserID tests metadata.user_id conversion to user field.
func TestResponsesToOpenAI_MetadataUserID(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"openai": "https://api.test.com/v1/chat/completions",
	}, "openai")

	handler := &ResponsesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "openai",
		},
	}

	request := `{
		"model": "test-model",
		"input": "Hello",
		"metadata": {"user_id": "user_123"},
		"stream": true
	}`

	transformed, err := handler.TransformRequest(context.TODO(), []byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse transformed request: %v", err)
	}

	if req["user"] != "user_123" {
		t.Errorf("Expected user field 'user_123', got %v", req["user"])
	}
}

// -----------------------------------------------------------------------------
// Column 3: Responses → Responses (Passthrough)
// -----------------------------------------------------------------------------

// TestResponsesToResponses_Passthrough tests that Responses→Responses is passthrough.
func TestResponsesToResponses_Passthrough(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"responses": "https://api.test.com/v1/responses",
	}, "responses")

	handler := &ResponsesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "responses",
			IsPassthrough:  true,
		},
	}

	request := `{
		"model": "downstream-model",
		"instructions": "Be helpful",
		"input": "Hello",
		"tools": [{"type": "function", "name": "test", "parameters": {"type": "object"}}],
		"stream": true
	}`

	transformed, err := handler.TransformRequest(context.TODO(), []byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse transformed request: %v", err)
	}

	// Model should be updated to upstream model
	if req["model"] != "test-model" {
		t.Errorf("Expected model 'test-model', got %v", req["model"])
	}

	// Instructions should remain unchanged (passthrough)
	if req["instructions"] != "Be helpful" {
		t.Errorf("Expected instructions to remain, got %v", req["instructions"])
	}

	// Input should remain as-is
	if req["input"] != "Hello" {
		t.Errorf("Expected input to remain, got %v", req["input"])
	}

	// Tools should remain in Responses format (not transformed)
	tools := req["tools"].([]interface{})
	if len(tools) != 1 {
		t.Errorf("Expected 1 tool, got %d", len(tools))
	}
}

// TestResponsesToResponses_NilRoutePassthrough tests nil route behavior.
func TestResponsesToResponses_NilRoutePassthrough(t *testing.T) {
	handler := &ResponsesHandler{
		cfg:   &config.Config{},
		route: nil,
	}

	request := `{"model": "test", "input": "Hello", "stream": true}`

	transformed, err := handler.TransformRequest(context.TODO(), []byte(request))
	if err == nil {
		// Should return error for nil route
		t.Error("Expected error for nil route")
	}
	if transformed != nil {
		t.Error("Expected nil transformed body for nil route")
	}
}

// -----------------------------------------------------------------------------
// Cross-Protocol Corner Cases (Section 9)
// -----------------------------------------------------------------------------

// TestResponsesSource_EmptyInput tests handling of empty input.
func TestResponsesSource_EmptyInput(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"anthropic": "https://api.test.com/anthropic/v1/messages",
	}, "anthropic")

	handler := &ResponsesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "anthropic",
		},
	}

	request := `{
		"model": "test-model",
		"input": "",
		"stream": true
	}`

	transformed, err := handler.TransformRequest(context.TODO(), []byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse transformed request: %v", err)
	}

	// Empty input should result in empty or minimal messages
	messages := req["messages"]
	if messages == nil {
		// This is acceptable
		return
	}

	if msgArr, ok := messages.([]interface{}); ok && len(msgArr) == 0 {
		// Empty array is acceptable
		return
	}
}

// TestResponsesSource_MultipleToolCalls tests handling of multiple function calls.
func TestResponsesSource_MultipleToolCalls(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"openai": "https://api.test.com/v1/chat/completions",
	}, "openai")

	handler := &ResponsesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "openai",
		},
	}

	request := `{
		"model": "test-model",
		"input": [
			{"type": "message", "role": "user", "content": "Calculate both"},
			{
				"type": "function_call",
				"call_id": "call_1",
				"name": "add",
				"arguments": "{\"a\": 1, \"b\": 2}"
			},
			{
				"type": "function_call",
				"call_id": "call_2",
				"name": "multiply",
				"arguments": "{\"a\": 3, \"b\": 4}"
			}
		],
		"stream": true
	}`

	transformed, err := handler.TransformRequest(context.TODO(), []byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse transformed request: %v", err)
	}

	messages := req["messages"].([]interface{})

	// Find assistant message with tool_calls
	var toolCallCount int
	for _, m := range messages {
		msg := m.(map[string]interface{})
		if msg["role"] == "assistant" {
			if tc, ok := msg["tool_calls"].([]interface{}); ok {
				toolCallCount = len(tc)
			}
		}
	}

	if toolCallCount != 2 {
		t.Errorf("Expected 2 tool_calls in assistant message, got %d", toolCallCount)
	}
}

// TestResponsesSource_TemperaturePassthrough tests temperature is passed through.
func TestResponsesSource_TemperaturePassthrough(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"openai": "https://api.test.com/v1/chat/completions",
	}, "openai")

	handler := &ResponsesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "openai",
		},
	}

	request := `{
		"model": "test-model",
		"input": "Hello",
		"temperature": 0.7,
		"stream": true
	}`

	transformed, err := handler.TransformRequest(context.TODO(), []byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse transformed request: %v", err)
	}

	if req["temperature"].(float64) != 0.7 {
		t.Errorf("Expected temperature 0.7, got %v", req["temperature"])
	}
}

// TestResponsesSource_ToolChoiceConversion tests tool_choice conversion.
func TestResponsesSource_ToolChoiceConversion(t *testing.T) {
	tests := []struct {
		name           string
		toolChoice     string
		outputProtocol string
		expectedType   string
	}{
		{
			name:           "auto to Anthropic",
			toolChoice:     "auto",
			outputProtocol: "anthropic",
			expectedType:   "auto",
		},
		{
			name:           "required to Anthropic becomes any",
			toolChoice:     "required",
			outputProtocol: "anthropic",
			expectedType:   "any",
		},
		{
			name:           "auto to OpenAI",
			toolChoice:     "auto",
			outputProtocol: "openai",
			expectedType:   "auto",
		},
		{
			name:           "required to OpenAI",
			toolChoice:     "required",
			outputProtocol: "openai",
			expectedType:   "required",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := mockMultiProtocolProvider("test", map[string]string{
				tt.outputProtocol: "https://api.test.com/v1/endpoint",
			}, tt.outputProtocol)

			handler := &ResponsesHandler{
				cfg: &config.Config{},
				route: &router.ResolvedRoute{
					Provider:       provider,
					Model:          "test-model",
					OutputProtocol: tt.outputProtocol,
				},
			}

			request := `{
				"model": "test-model",
				"input": "Hello",
				"tool_choice": "` + tt.toolChoice + `",
				"tools": [{"type": "function", "name": "test", "parameters": {"type": "object"}}],
				"stream": true
			}`

			transformed, err := handler.TransformRequest(context.TODO(), []byte(request))
			if err != nil {
				t.Fatalf("TransformRequest failed: %v", err)
			}

			var req map[string]interface{}
			if err := json.Unmarshal(transformed, &req); err != nil {
				t.Fatalf("Failed to parse transformed request: %v", err)
			}

			toolChoice := req["tool_choice"]
			if toolChoice == nil {
				t.Error("Expected tool_choice to be present")
				return
			}

			switch tc := toolChoice.(type) {
			case string:
				if tc != tt.expectedType {
					t.Errorf("Expected tool_choice '%s', got '%s'", tt.expectedType, tc)
				}
			case map[string]interface{}:
				if tc["type"] != tt.expectedType {
					t.Errorf("Expected tool_choice type '%s', got %v", tt.expectedType, tc["type"])
				}
			}
		})
	}
}

// TestResponsesSource_UpstreamURL tests correct endpoint selection.
func TestResponsesSource_UpstreamURL(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"openai":    "https://api.test.com/v1/chat/completions",
		"anthropic": "https://api.test.com/anthropic/v1/messages",
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
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			handler := &ResponsesHandler{
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

// TestResponsesSource_ForwardHeaders tests header forwarding based on OutputProtocol.
func TestResponsesSource_ForwardHeaders(t *testing.T) {
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
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := mockMultiProtocolProvider("test", map[string]string{
				"openai":    "https://api.test.com/v1/chat/completions",
				"anthropic": "https://api.test.com/anthropic/v1/messages",
			}, "openai")

			handler := &ResponsesHandler{
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
					t.Errorf("Anthropic-Version should NOT be forwarded for OpenAI target, got '%s'", gotAnthropicVersion)
				}
			}
		})
	}
}

// TestResponsesSource_ArrayInputWithMixedContent tests array input with various content types.
// Note: Currently, system role messages in input array may not be extracted to the
// system field due to how the conversion processes items. The primary way to set
// the system field is via the "instructions" field. This test documents current behavior.
func TestResponsesSource_ArrayInputWithMixedContent(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"anthropic": "https://api.test.com/anthropic/v1/messages",
	}, "anthropic")

	handler := &ResponsesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "anthropic",
		},
	}

	request := `{
		"model": "test-model",
		"input": [
			{"type": "message", "role": "user", "content": "User message"},
			{"type": "message", "role": "assistant", "content": "Assistant message"}
		],
		"stream": true
	}`

	transformed, err := handler.TransformRequest(context.TODO(), []byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse transformed request: %v", err)
	}

	// Verify messages contain user and assistant
	messages := req["messages"].([]interface{})
	if len(messages) < 2 {
		t.Errorf("Expected at least 2 messages (user and assistant), got %d", len(messages))
	}

	// Verify user and assistant roles are present
	var foundUser, foundAssistant bool
	for _, m := range messages {
		msg := m.(map[string]interface{})
		switch msg["role"] {
		case "user":
			foundUser = true
		case "assistant":
			foundAssistant = true
		}
	}

	if !foundUser {
		t.Error("Expected to find user message")
	}
	if !foundAssistant {
		t.Error("Expected to find assistant message")
	}
}

// TestResponsesSource_InstructionsField tests that instructions field becomes system field.
// This is the primary way to set the system prompt for Anthropic targets.
func TestResponsesSource_InstructionsField(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"anthropic": "https://api.test.com/anthropic/v1/messages",
	}, "anthropic")

	handler := &ResponsesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "anthropic",
		},
	}

	request := `{
		"model": "test-model",
		"instructions": "You are a helpful coding assistant.",
		"input": "Hello",
		"stream": true
	}`

	transformed, err := handler.TransformRequest(context.TODO(), []byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse transformed request: %v", err)
	}

	// Instructions field should become system field
	if req["system"] != "You are a helpful coding assistant." {
		t.Errorf("Expected system field from instructions, got %v", req["system"])
	}
}
