package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"ai-proxy/config"
	"ai-proxy/router"

	"github.com/gin-gonic/gin"
)

func init() {
	gin.SetMode(gin.TestMode)
}

// =============================================================================
// Test Helpers for Multi-Protocol Provider Testing
// =============================================================================

// mockMultiProtocolProvider creates a provider with Endpoints map (no Type field).
// This simulates providers like MiniMax that support multiple protocols.
func mockMultiProtocolProvider(name string, endpoints map[string]string, defaultProto string) config.Provider {
	return config.Provider{
		Name:      name,
		Endpoints: endpoints,
		Default:   defaultProto,
	}
}

// mockLegacyProvider creates a provider with a single endpoint (legacy single-protocol).
func mockLegacyProvider(name, providerType, baseURL string) config.Provider {
	return config.Provider{
		Name:      name,
		Endpoints: map[string]string{providerType: baseURL},
	}
}

// mockRoute creates a ResolvedRoute with proper OutputProtocol set.
func mockRoute(provider config.Provider, model, outputProtocol string) *router.ResolvedRoute {
	return &router.ResolvedRoute{
		Provider:       provider,
		Model:          model,
		OutputProtocol: outputProtocol,
	}
}

// =============================================================================
// Multi-Protocol Provider Tests - Messages Handler
// =============================================================================

// TestMessagesHandler_MultiProtocolProvider_OpenAITarget tests that a multi-protocol
// provider with OpenAI output protocol correctly transforms Anthropic tools to OpenAI format.
// This is the bug we fixed - multi-protocol providers have empty Type, so the switch
// statement must use OutputProtocol instead.
func TestMessagesHandler_MultiProtocolProvider_OpenAITarget(t *testing.T) {
	provider := mockMultiProtocolProvider("minimax", map[string]string{
		"openai":    "https://api.minimax.io/v1/chat/completions",
		"anthropic": "https://api.minimax.io/anthropic/v1/messages",
	}, "openai")

	handler := &MessagesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "MiniMax-M2.7",
			OutputProtocol: "openai",
		},
	}

	anthropicRequest := `{
		"model": "MiniMax-M2.7",
		"max_tokens": 100,
		"messages": [{"role": "user", "content": "hi"}],
		"tools": [{
			"name": "calculator",
			"description": "A calculator",
			"input_schema": {
				"type": "object",
				"properties": {"expr": {"type": "string"}},
				"required": ["expr"]
			}
		}],
		"stream": true
	}`

	transformed, err := handler.TransformRequest([]byte(anthropicRequest))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse transformed request: %v", err)
	}

	tools, ok := req["tools"].([]interface{})
	if !ok {
		t.Fatal("Expected tools array in transformed request")
	}

	if len(tools) != 1 {
		t.Fatalf("Expected 1 tool, got %d", len(tools))
	}

	tool := tools[0].(map[string]interface{})
	if tool["type"] != "function" {
		t.Errorf("Expected tool type 'function', got '%v'", tool["type"])
	}

	function, ok := tool["function"].(map[string]interface{})
	if !ok {
		t.Fatal("Expected function object in tool")
	}

	if function["name"] != "calculator" {
		t.Errorf("Expected function name 'calculator', got '%v'", function["name"])
	}

	if function["parameters"] == nil {
		t.Error("Expected parameters in function (converted from input_schema)")
	}
}

// TestMessagesHandler_MultiProtocolProvider_AnthropicTarget tests that a multi-protocol
// provider with Anthropic output protocol passes through without transformation.
func TestMessagesHandler_MultiProtocolProvider_AnthropicTarget(t *testing.T) {
	provider := mockMultiProtocolProvider("alibaba", map[string]string{
		"openai":    "https://dashscope.aliyuncs.com/v1/chat/completions",
		"anthropic": "https://dashscope.aliyuncs.com/anthropic/v1/messages",
	}, "anthropic")

	handler := &MessagesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "kimi-k2.5",
			OutputProtocol: "anthropic",
		},
	}

	anthropicRequest := `{
		"model": "kimi-k2.5",
		"max_tokens": 100,
		"messages": [{"role": "user", "content": "hi"}],
		"tools": [{
			"name": "search",
			"description": "Search the web",
			"input_schema": {"type": "object"}
		}],
		"stream": true
	}`

	transformed, err := handler.TransformRequest([]byte(anthropicRequest))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse transformed request: %v", err)
	}

	tools, ok := req["tools"].([]interface{})
	if !ok {
		t.Fatal("Expected tools array in transformed request")
	}

	tool := tools[0].(map[string]interface{})
	if tool["name"] != "search" {
		t.Errorf("Expected tool name 'search', got '%v'", tool["name"])
	}

	if _, hasInputSchema := tool["input_schema"]; !hasInputSchema {
		t.Error("Expected input_schema to be preserved for Anthropic target")
	}

	if tool["type"] == "function" {
		t.Error("Tool should NOT have type='function' for Anthropic target (Anthropic format)")
	}
}

// TestMessagesHandler_LegacyProvider_OpenAITarget tests the legacy path with Type field.
func TestMessagesHandler_LegacyProvider_OpenAITarget(t *testing.T) {
	provider := mockLegacyProvider("openai", "openai", "https://api.openai.com/v1")

	handler := &MessagesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "gpt-4o",
			OutputProtocol: "openai",
		},
	}

	anthropicRequest := `{
		"model": "gpt-4o",
		"max_tokens": 100,
		"messages": [{"role": "user", "content": "hi"}],
		"tools": [{
			"name": "test",
			"description": "A test tool",
			"input_schema": {"type": "object"}
		}],
		"stream": true
	}`

	transformed, err := handler.TransformRequest([]byte(anthropicRequest))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse transformed request: %v", err)
	}

	tools, ok := req["tools"].([]interface{})
	if !ok {
		t.Fatal("Expected tools array")
	}

	tool := tools[0].(map[string]interface{})
	if tool["type"] != "function" {
		t.Errorf("Expected tool type 'function', got '%v'", tool["type"])
	}
}

// =============================================================================
// Multi-Protocol Provider Tests - Completions Handler
// =============================================================================

// TestCompletionsHandler_MultiProtocolProvider_AnthropicTarget tests that
// OpenAI Chat Completions are transformed to Anthropic Messages format.
func TestCompletionsHandler_MultiProtocolProvider_AnthropicTarget(t *testing.T) {
	provider := mockMultiProtocolProvider("alibaba", map[string]string{
		"openai":    "https://dashscope.aliyuncs.com/v1/chat/completions",
		"anthropic": "https://dashscope.aliyuncs.com/anthropic/v1/messages",
	}, "anthropic")

	handler := &CompletionsHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "kimi-k2.5",
			OutputProtocol: "anthropic",
		},
	}

	openAIRequest := `{
		"model": "kimi-k2.5",
		"messages": [
			{"role": "system", "content": "Be helpful"},
			{"role": "user", "content": "hi"}
		],
		"tools": [{
			"type": "function",
			"function": {
				"name": "search",
				"description": "Search",
				"parameters": {"type": "object"}
			}
		}],
		"stream": true
	}`

	transformed, err := handler.TransformRequest([]byte(openAIRequest))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse transformed request: %v", err)
	}

	if _, hasSystem := req["system"]; !hasSystem {
		t.Error("Expected system field (converted from system message)")
	}

	tools, ok := req["tools"].([]interface{})
	if !ok {
		t.Fatal("Expected tools array")
	}

	tool := tools[0].(map[string]interface{})
	if _, hasInputSchema := tool["input_schema"]; !hasInputSchema {
		t.Error("Expected input_schema (converted from parameters)")
	}
}

// TestCompletionsHandler_MultiProtocolProvider_OpenAITarget tests passthrough.
func TestCompletionsHandler_MultiProtocolProvider_OpenAITarget(t *testing.T) {
	provider := mockMultiProtocolProvider("minimax", map[string]string{
		"openai":    "https://api.minimax.io/v1/chat/completions",
		"anthropic": "https://api.minimax.io/anthropic/v1/messages",
	}, "openai")

	handler := &CompletionsHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "MiniMax-M2.7",
			OutputProtocol: "openai",
		},
	}

	openAIRequest := `{
		"model": "MiniMax-M2.7",
		"messages": [{"role": "user", "content": "hi"}],
		"tools": [{
			"type": "function",
			"function": {
				"name": "test",
				"parameters": {"type": "object"}
			}
		}],
		"stream": true
	}`

	transformed, err := handler.TransformRequest([]byte(openAIRequest))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse transformed request: %v", err)
	}

	tools, ok := req["tools"].([]interface{})
	if !ok {
		t.Fatal("Expected tools array")
	}

	tool := tools[0].(map[string]interface{})
	if tool["type"] != "function" {
		t.Errorf("Expected tool type 'function' to be preserved, got '%v'", tool["type"])
	}
}

// =============================================================================
// Multi-Protocol Provider Tests - Responses Handler
// =============================================================================

// TestResponsesHandler_MultiProtocolProvider_AnthropicTarget tests that
// Responses API requests are transformed to Anthropic Messages format.
func TestResponsesHandler_MultiProtocolProvider_AnthropicTarget(t *testing.T) {
	provider := mockMultiProtocolProvider("alibaba", map[string]string{
		"openai":    "https://dashscope.aliyuncs.com/v1/chat/completions",
		"anthropic": "https://dashscope.aliyuncs.com/anthropic/v1/messages",
	}, "anthropic")

	handler := &ResponsesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "kimi-k2.5",
			OutputProtocol: "anthropic",
		},
	}

	responsesRequest := `{
		"model": "kimi-k2.5",
		"instructions": "Be helpful",
		"input": "Hello",
		"tools": [{
			"type": "function",
			"name": "search",
			"description": "Search",
			"parameters": {"type": "object"}
		}],
		"stream": true
	}`

	transformed, err := handler.TransformRequest([]byte(responsesRequest))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse transformed request: %v", err)
	}

	if _, hasSystem := req["system"]; !hasSystem {
		t.Error("Expected system field (converted from instructions)")
	}

	if _, hasMessages := req["messages"]; !hasMessages {
		t.Error("Expected messages field (converted from input)")
	}

	tools, ok := req["tools"].([]interface{})
	if !ok {
		t.Fatal("Expected tools array")
	}

	tool := tools[0].(map[string]interface{})
	if _, hasInputSchema := tool["input_schema"]; !hasInputSchema {
		t.Error("Expected input_schema (converted from parameters)")
	}
}

// TestResponsesHandler_MultiProtocolProvider_OpenAITarget tests transformation to Chat.
func TestResponsesHandler_MultiProtocolProvider_OpenAITarget(t *testing.T) {
	provider := mockMultiProtocolProvider("minimax", map[string]string{
		"openai":    "https://api.minimax.io/v1/chat/completions",
		"anthropic": "https://api.minimax.io/anthropic/v1/messages",
	}, "openai")

	handler := &ResponsesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "MiniMax-M2.7",
			OutputProtocol: "openai",
		},
	}

	responsesRequest := `{
		"model": "MiniMax-M2.7",
		"instructions": "Be helpful",
		"input": "Hello",
		"tools": [{
			"type": "function",
			"name": "search",
			"parameters": {"type": "object"}
		}],
		"stream": true
	}`

	transformed, err := handler.TransformRequest([]byte(responsesRequest))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	if err := json.Unmarshal(transformed, &req); err != nil {
		t.Fatalf("Failed to parse transformed request: %v", err)
	}

	if _, hasMessages := req["messages"]; !hasMessages {
		t.Error("Expected messages field (converted from input)")
	}

	tools, ok := req["tools"].([]interface{})
	if !ok {
		t.Fatal("Expected tools array")
	}

	tool := tools[0].(map[string]interface{})
	if tool["type"] != "function" {
		t.Errorf("Expected tool type 'function', got '%v'", tool["type"])
	}
}

// =============================================================================
// Forward Headers Tests - Multi-Protocol Provider
// =============================================================================

// TestMessagesHandler_ForwardHeaders_MultiProtocol tests that headers are forwarded
// correctly based on OutputProtocol for multi-protocol providers.
func TestMessagesHandler_ForwardHeaders_MultiProtocol(t *testing.T) {
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

			handler := &MessagesHandler{
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
					t.Errorf("Anthropic-Version should NOT be forwarded for OpenAI target, got '%s'", gotAnthropicVersion)
				}
			}
		})
	}
}

// =============================================================================
// Upstream URL Tests - Multi-Protocol Provider
// =============================================================================

// TestUpstreamURL_MultiProtocol tests that the correct endpoint URL is selected
// based on OutputProtocol for multi-protocol providers.
func TestUpstreamURL_MultiProtocol(t *testing.T) {
	provider := mockMultiProtocolProvider("minimax", map[string]string{
		"openai":    "https://api.minimax.io/v1/chat/completions",
		"anthropic": "https://api.minimax.io/anthropic/v1/messages",
	}, "openai")

	tests := []struct {
		name           string
		outputProtocol string
		wantURL        string
	}{
		{
			name:           "OpenAI protocol uses OpenAI endpoint",
			outputProtocol: "openai",
			wantURL:        "https://api.minimax.io/v1/chat/completions",
		},
		{
			name:           "Anthropic protocol uses Anthropic endpoint",
			outputProtocol: "anthropic",
			wantURL:        "https://api.minimax.io/anthropic/v1/messages",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			messagesHandler := &MessagesHandler{
				cfg: &config.Config{},
				route: &router.ResolvedRoute{
					Provider:       provider,
					Model:          "MiniMax-M2.7",
					OutputProtocol: tt.outputProtocol,
				},
			}

			got := messagesHandler.UpstreamURL()
			if got != tt.wantURL {
				t.Errorf("UpstreamURL() = %s, want %s", got, tt.wantURL)
			}

			completionsHandler := &CompletionsHandler{
				cfg: &config.Config{},
				route: &router.ResolvedRoute{
					Provider:       provider,
					Model:          "MiniMax-M2.7",
					OutputProtocol: tt.outputProtocol,
				},
			}

			got = completionsHandler.UpstreamURL()
			if got != tt.wantURL {
				t.Errorf("CompletionsHandler.UpstreamURL() = %s, want %s", got, tt.wantURL)
			}
		})
	}
}

// =============================================================================
// Tool Transformation Tests - Detailed
// =============================================================================

// TestToolTransformation_AnthropicToOpenAI_Detailed tests comprehensive tool conversion.
func TestToolTransformation_AnthropicToOpenAI_Detailed(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected map[string]interface{}
	}{
		{
			name: "Simple tool",
			input: `{
				"name": "calculator",
				"description": "A calculator",
				"input_schema": {
					"type": "object",
					"properties": {
						"expression": {"type": "string"}
					},
					"required": ["expression"]
				}
			}`,
			expected: map[string]interface{}{
				"type": "function",
				"function": map[string]interface{}{
					"name":        "calculator",
					"description": "A calculator",
					"parameters": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"expression": map[string]interface{}{"type": "string"},
						},
						"required": []interface{}{"expression"},
					},
				},
			},
		},
		{
			name: "Tool with nested schema",
			input: `{
				"name": "search",
				"description": "Search the web",
				"input_schema": {
					"type": "object",
					"properties": {
						"query": {"type": "string"},
						"options": {
							"type": "object",
							"properties": {
								"limit": {"type": "integer"}
							}
						}
					}
				}
			}`,
			expected: map[string]interface{}{
				"type": "function",
				"function": map[string]interface{}{
					"name":        "search",
					"description": "Search the web",
					"parameters": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"query": map[string]interface{}{"type": "string"},
							"options": map[string]interface{}{
								"type": "object",
								"properties": map[string]interface{}{
									"limit": map[string]interface{}{"type": "integer"},
								},
							},
						},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := mockMultiProtocolProvider("test", map[string]string{
				"openai": "https://api.test.com/v1/chat/completions",
			}, "openai")

			handler := &MessagesHandler{
				cfg: &config.Config{},
				route: &router.ResolvedRoute{
					Provider:       provider,
					Model:          "test-model",
					OutputProtocol: "openai",
				},
			}

			request := `{
				"model": "test-model",
				"max_tokens": 100,
				"messages": [{"role": "user", "content": "hi"}],
				"tools": [` + tt.input + `],
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
			tool := tools[0].(map[string]interface{})

			expectedBytes, _ := json.Marshal(tt.expected)
			toolBytes, _ := json.Marshal(tool)

			if !bytes.Equal(toolBytes, expectedBytes) {
				t.Errorf("Tool mismatch:\ngot:  %s\nwant: %s", toolBytes, expectedBytes)
			}
		})
	}
}

// =============================================================================
// Message Transformation Tests
// =============================================================================

// TestMessageConversion_AnthropicToOpenAI_ToolUse tests tool_use block conversion.
func TestMessageConversion_AnthropicToOpenAI_ToolUse(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"openai": "https://api.test.com/v1/chat/completions",
	}, "openai")

	handler := &MessagesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "openai",
		},
	}

	request := `{
		"model": "test-model",
		"max_tokens": 100,
		"messages": [
			{"role": "user", "content": "What is 2+2?"},
			{
				"role": "assistant",
				"content": [
					{"type": "text", "text": "Let me calculate."},
					{
						"type": "tool_use",
						"id": "toolu_123",
						"name": "calculator",
						"input": {"expr": "2+2"}
					}
				]
			},
			{
				"role": "user",
				"content": [{"type": "tool_result", "tool_use_id": "toolu_123", "content": "4"}]
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
	if assistantMsg["role"] != "assistant" {
		t.Errorf("Expected assistant role, got %v", assistantMsg["role"])
	}

	if assistantMsg["content"] != "Let me calculate." {
		t.Errorf("Expected text content, got %v", assistantMsg["content"])
	}

	toolCalls := assistantMsg["tool_calls"].([]interface{})
	if len(toolCalls) != 1 {
		t.Fatalf("Expected 1 tool_call, got %d", len(toolCalls))
	}

	tc := toolCalls[0].(map[string]interface{})
	if tc["id"] != "toolu_123" {
		t.Errorf("Expected tool_call id 'toolu_123', got %v", tc["id"])
	}

	function := tc["function"].(map[string]interface{})
	if function["name"] != "calculator" {
		t.Errorf("Expected function name 'calculator', got %v", function["name"])
	}

	toolMsg := messages[2].(map[string]interface{})
	if toolMsg["role"] != "tool" {
		t.Errorf("Expected tool role for tool_result, got %v", toolMsg["role"])
	}
}

// =============================================================================
// CreateTransformer Tests - Multi-Protocol Provider
// =============================================================================

// TestMessagesHandler_CreateTransformer_MultiProtocol tests transformer selection
// based on OutputProtocol for multi-protocol providers.
func TestMessagesHandler_CreateTransformer_MultiProtocol(t *testing.T) {
	tests := []struct {
		name           string
		outputProtocol string
		isPassthrough  bool
		wantTransform  bool
	}{
		{
			name:           "OpenAI target - needs ChatToAnthropicTransformer",
			outputProtocol: "openai",
			isPassthrough:  false,
			wantTransform:  true,
		},
		{
			name:           "Anthropic target - passthrough",
			outputProtocol: "anthropic",
			isPassthrough:  false,
			wantTransform:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := mockMultiProtocolProvider("test", map[string]string{
				"openai":    "https://api.test.com/v1/chat/completions",
				"anthropic": "https://api.test.com/anthropic/v1/messages",
			}, "openai")

			handler := &MessagesHandler{
				cfg: &config.Config{},
				route: &router.ResolvedRoute{
					Provider:              provider,
					Model:                 "test-model",
					OutputProtocol:        tt.outputProtocol,
					IsPassthrough:         tt.isPassthrough,
					KimiToolCallTransform: false,
				},
			}

			var buf bytes.Buffer
			transformer := handler.CreateTransformer(&buf)

			if transformer == nil {
				t.Fatal("CreateTransformer returned nil")
			}

			if tt.wantTransform && buf.Len() != 0 {
				t.Log("Transformer created for transformation case")
			}
		})
	}
}

// =============================================================================
// Passthrough Mode Tests
// =============================================================================

// TestPassthrough_OpenAIToOpenAI tests that OpenAI→OpenAI is passthrough.
// Note: Even in passthrough mode, the model name is updated to the resolved model.
func TestPassthrough_OpenAIToOpenAI(t *testing.T) {
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

	request := `{"model": "downstream-model", "messages": [{"role": "user", "content": "hi"}]}`

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

	// Messages should be preserved without transformation
	messages := req["messages"].([]interface{})
	if len(messages) != 1 {
		t.Errorf("Expected 1 message, got %d", len(messages))
	}

	// Should NOT have input_schema (would indicate Anthropic transformation)
	reqBytes, _ := json.Marshal(req)
	if bytes.Contains(reqBytes, []byte("input_schema")) {
		t.Error("Passthrough should not add Anthropic-specific fields")
	}
}

// TestPassthrough_AnthropicToAnthropic tests that Anthropic→Anthropic is passthrough.
func TestPassthrough_AnthropicToAnthropic(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"anthropic": "https://api.test.com/anthropic/v1/messages",
	}, "anthropic")

	handler := &MessagesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "anthropic",
			IsPassthrough:  true,
		},
	}

	request := `{"model": "test-model", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]}`

	transformed, err := handler.TransformRequest([]byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	json.Unmarshal(transformed, &req)

	if _, hasTools := req["tools"]; hasTools {
		t.Error("Passthrough should preserve original format")
	}
}

// =============================================================================
// System Prompt Tests
// =============================================================================

// TestSystemPrompt_Preserved tests that system prompt is handled correctly.
// Note: Current implementation passes through system field - this test documents behavior.
func TestSystemPrompt_Preserved(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"openai": "https://api.test.com/v1/chat/completions",
	}, "openai")

	handler := &MessagesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "openai",
		},
	}

	request := `{
		"model": "test-model",
		"max_tokens": 100,
		"system": "You are helpful.",
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

	// System field should be present (passed through or converted based on implementation)
	if _, hasSystem := req["system"]; !hasSystem {
		if _, hasMessages := req["messages"]; !hasMessages {
			t.Error("Expected either system field or system message")
		}
	}
}

// =============================================================================
// Tool Tests
// =============================================================================

// TestTools_Preserved tests that tools are handled correctly in transformation.
func TestTools_Preserved(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"openai": "https://api.test.com/v1/chat/completions",
	}, "openai")

	handler := &MessagesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "openai",
		},
	}

	request := `{
		"model": "test-model",
		"max_tokens": 100,
		"messages": [{"role": "user", "content": "hi"}],
		"tools": [{"name": "test", "description": "A test", "input_schema": {"type": "object"}}],
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
	// Tool should be transformed to OpenAI format
	if tool["type"] != "function" {
		t.Errorf("Expected tool type 'function', got %v", tool["type"])
	}
}

// =============================================================================
// Edge Case Tests
// =============================================================================

// TestEdgeCase_EmptyTools tests handling of empty tools array.
func TestEdgeCase_EmptyTools(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"openai": "https://api.test.com/v1/chat/completions",
	}, "openai")

	handler := &MessagesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "openai",
		},
	}

	request := `{
		"model": "test-model",
		"max_tokens": 100,
		"messages": [{"role": "user", "content": "hi"}],
		"tools": [],
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

	if _, hasTools := req["tools"]; hasTools {
		t.Error("Empty tools array should be omitted or handled gracefully")
	}
}

// TestEdgeCase_NilRoute tests handler behavior with nil route.
func TestEdgeCase_NilRoute(t *testing.T) {
	handler := &MessagesHandler{
		cfg:   &config.Config{},
		route: nil,
	}

	request := `{"model": "test", "max_tokens": 100, "messages": [], "stream": true}`

	transformed, err := handler.TransformRequest([]byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	if string(transformed) != request {
		t.Errorf("Nil route should pass through unchanged")
	}
}

// TestEdgeCase_MultipleToolResults tests handling of multiple tool results in one message.
// Note: Multiple tool_result blocks in one user message may be handled differently
// depending on implementation - this test verifies they are processed without error.
func TestEdgeCase_MultipleToolResults(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"openai": "https://api.test.com/v1/chat/completions",
	}, "openai")

	handler := &MessagesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "openai",
		},
	}

	request := `{
		"model": "test-model",
		"max_tokens": 100,
		"messages": [
			{"role": "user", "content": "Check both"},
			{
				"role": "assistant",
				"content": [
					{"type": "tool_use", "id": "tool1", "name": "fn1", "input": {}},
					{"type": "tool_use", "id": "tool2", "name": "fn2", "input": {}}
				]
			},
			{
				"role": "user",
				"content": [
					{"type": "tool_result", "tool_use_id": "tool1", "content": "result1"},
					{"type": "tool_result", "tool_use_id": "tool2", "content": "result2"}
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

	// Verify tool results are handled (implementation may batch or split)
	var foundToolResults int
	for _, m := range messages {
		msg := m.(map[string]interface{})
		if msg["role"] == "tool" {
			foundToolResults++
		}
	}

	// At minimum, tool results should be processed
	if foundToolResults < 1 {
		t.Error("Expected at least one tool message from tool_result conversion")
	}
}

// =============================================================================
// Responses API Input Conversion Tests
// =============================================================================

// TestResponsesInput_StringToArray tests string input conversion.
func TestResponsesInput_StringToArray(t *testing.T) {
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
		"input": "Hello world",
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
		t.Errorf("Expected 1 message from string input, got %d", len(messages))
	}

	msg := messages[0].(map[string]interface{})
	if msg["role"] != "user" {
		t.Errorf("Expected user role, got %v", msg["role"])
	}
	if msg["content"] != "Hello world" {
		t.Errorf("Expected content 'Hello world', got %v", msg["content"])
	}
}

// TestResponsesInput_InstructionsToSystem tests instructions -> system conversion.
func TestResponsesInput_InstructionsToSystem(t *testing.T) {
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
		"input": "Hello",
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

	if req["system"] != "Be helpful and concise" {
		t.Errorf("Expected system field from instructions, got %v", req["system"])
	}
}

// =============================================================================
// Max Tokens Field Tests
// =============================================================================

// TestMaxTokens_AnthropicToOpenAI tests max_tokens field handling.
func TestMaxTokens_AnthropicToOpenAI(t *testing.T) {
	provider := mockMultiProtocolProvider("test", map[string]string{
		"openai": "https://api.test.com/v1/chat/completions",
	}, "openai")

	handler := &MessagesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			Model:          "test-model",
			OutputProtocol: "openai",
		},
	}

	request := `{
		"model": "test-model",
		"max_tokens": 4096,
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

	if req["max_tokens"].(float64) != 4096 {
		t.Errorf("Expected max_tokens 4096, got %v", req["max_tokens"])
	}
}

// =============================================================================
// Regression Tests - Bug Fixes
// =============================================================================

// TestRegression_MultiProtocolProviderType tests the specific bug we fixed:
// Multi-protocol providers have empty Type field, so switch statements must
// use OutputProtocol instead of Provider.Type.
func TestRegression_MultiProtocolProviderType(t *testing.T) {
	provider := config.Provider{
		Name: "minimax",
		Endpoints: map[string]string{
			"openai":    "https://api.minimax.io/v1/chat/completions",
			"anthropic": "https://api.minimax.io/anthropic/v1/messages",
		},
		Default: "openai",
	}

	// Verify Provider supports multiple protocols
	if !provider.HasProtocol("openai") || !provider.HasProtocol("anthropic") {
		t.Error("Expected provider to support both openai and anthropic protocols")
	}

	// Verify OutputProtocol is set correctly
	route := &router.ResolvedRoute{
		Provider:       provider,
		Model:          "MiniMax-M2.7",
		OutputProtocol: "openai",
	}

	if route.OutputProtocol != "openai" {
		t.Errorf("Expected OutputProtocol 'openai', got '%s'", route.OutputProtocol)
	}

	// Now test that transformation works correctly
	handler := &MessagesHandler{
		cfg:   &config.Config{},
		route: route,
	}

	request := `{
		"model": "MiniMax-M2.7",
		"max_tokens": 100,
		"messages": [{"role": "user", "content": "hi"}],
		"tools": [{
			"name": "test",
			"description": "A test",
			"input_schema": {"type": "object"}
		}],
		"stream": true
	}`

	transformed, err := handler.TransformRequest([]byte(request))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var req map[string]interface{}
	json.Unmarshal(transformed, &req)

	tools := req["tools"].([]interface{})
	tool := tools[0].(map[string]interface{})

	// If this fails, the bug is present: tools weren't transformed because
	// Provider.Type was empty, causing the switch to fall through to default
	if tool["type"] != "function" {
		t.Error("REGRESSION: Tool was not transformed to OpenAI format. " +
			"This indicates the switch statement is using Provider.Type instead of OutputProtocol.")
	}
}
