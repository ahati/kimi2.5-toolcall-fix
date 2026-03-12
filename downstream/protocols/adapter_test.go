package protocols

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"ai-proxy/config"
	"ai-proxy/transform/toolcall"
	"ai-proxy/types"

	"github.com/gin-gonic/gin"
	"github.com/tmaxmax/go-sse"
)

func TestProtocolAdapter_Interface(t *testing.T) {
	var _ ProtocolAdapter = (*OpenAIAdapter)(nil)
	var _ ProtocolAdapter = (*AnthropicAdapter)(nil)
	var _ ProtocolAdapter = (*BridgeAdapter)(nil)
}

func TestNewAnthropicAdapter(t *testing.T) {
	adapter := NewAnthropicAdapter()
	if adapter == nil {
		t.Error("Expected non-nil adapter")
	}
}

func TestNewOpenAIAdapter(t *testing.T) {
	adapter := NewOpenAIAdapter()
	if adapter == nil {
		t.Error("Expected non-nil adapter")
	}
}

func TestNewBridgeAdapter(t *testing.T) {
	adapter := NewBridgeAdapter()
	if adapter == nil {
		t.Error("Expected non-nil adapter")
	}
}

func TestAnthropicAdapter_TransformRequest(t *testing.T) {
	adapter := &AnthropicAdapter{}
	body := []byte(`{"model":"test","stream":true}`)
	result, err := adapter.TransformRequest(body)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if string(result) != string(body) {
		t.Error("Expected body to be passed through unchanged")
	}
}

func TestAnthropicAdapter_ValidateRequest(t *testing.T) {
	adapter := &AnthropicAdapter{}
	streamingReq := []byte(`{"model":"test","stream":true}`)
	if err := adapter.ValidateRequest(streamingReq); err != nil {
		t.Errorf("Expected no error for streaming request, got: %v", err)
	}
	nonStreamingReq := []byte(`{"model":"test","stream":false}`)
	if err := adapter.ValidateRequest(nonStreamingReq); err != ErrNonStreamingNotSupported {
		t.Errorf("Expected ErrNonStreamingNotSupported, got: %v", err)
	}
}

func TestAnthropicAdapter_UpstreamURL(t *testing.T) {
	adapter := &AnthropicAdapter{}
	cfg := &config.Config{AnthropicUpstreamURL: "https://api.anthropic.com/v1/messages"}
	if url := adapter.UpstreamURL(cfg); url != "https://api.anthropic.com/v1/messages" {
		t.Errorf("Expected correct upstream URL, got: %s", url)
	}
}

func TestAnthropicAdapter_UpstreamAPIKey(t *testing.T) {
	adapter := &AnthropicAdapter{}
	cfg := &config.Config{AnthropicAPIKey: "test-key-123"}
	if key := adapter.UpstreamAPIKey(cfg); key != "test-key-123" {
		t.Errorf("Expected correct API key, got: %s", key)
	}
}

func TestAnthropicAdapter_ForwardHeaders(t *testing.T) {
	adapter := &AnthropicAdapter{}
	src := http.Header{}
	dst := http.Header{}
	src.Set("X-Custom-Header", "value1")
	src.Set("Anthropic-Version", "2023-06-01")
	src.Set("Anthropic-Beta", "prompt-caching")
	src.Set("Authorization", "secret")
	src.Set("Connection", "keep-alive")
	src.Set("Keep-Alive", "timeout=5")
	adapter.ForwardHeaders(src, dst)
	if dst.Get("X-Custom-Header") != "value1" {
		t.Error("Expected X-Custom-Header to be forwarded")
	}
	if dst.Get("Anthropic-Version") != "2023-06-01" {
		t.Error("Expected Anthropic-Version to be forwarded")
	}
	if dst.Get("Anthropic-Beta") != "prompt-caching" {
		t.Error("Expected Anthropic-Beta to be forwarded")
	}
	if dst.Get("Authorization") != "" {
		t.Error("Expected Authorization not to be forwarded")
	}
	if dst.Get("Connection") != "keep-alive" {
		t.Error("Expected Connection to be forwarded")
	}
	if dst.Get("Keep-Alive") != "timeout=5" {
		t.Error("Expected Keep-Alive to be forwarded")
	}
}

func TestAnthropicAdapter_SendError(t *testing.T) {
	gin.SetMode(gin.TestMode)
	adapter := &AnthropicAdapter{}
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	adapter.SendError(c, http.StatusBadRequest, "test error message")
	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected status %d, got %d", http.StatusBadRequest, w.Code)
	}
	var resp map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &resp)
	if resp["type"] != "error" {
		t.Error("Expected error type")
	}
	errorDetail, ok := resp["error"].(map[string]interface{})
	if !ok {
		t.Fatal("Expected error detail object")
	}
	if errorDetail["message"] != "test error message" {
		t.Errorf("Expected correct error message, got: %v", errorDetail["message"])
	}
}

func TestAnthropicAdapter_CreateTransformer(t *testing.T) {
	adapter := &AnthropicAdapter{}
	var buf bytes.Buffer
	base := types.StreamChunk{ID: "msg-123"}

	transformer := adapter.CreateTransformer(&buf, base)
	if transformer == nil {
		t.Fatal("Expected transformer to be created")
	}

	transformer.Transform(&sse.Event{
		Data: `{"type":"message_start","message":{"id":"msg-123","type":"message","role":"assistant"}}`,
	})
	transformer.Transform(&sse.Event{
		Data: `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
	})
	transformer.Transform(&sse.Event{
		Data: `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`,
	})
	transformer.Transform(&sse.Event{
		Data: `{"type":"content_block_stop","index":0}`,
	})
	transformer.Flush()

	output := buf.String()
	if !bytes.Contains([]byte(output), []byte("Hello")) {
		t.Errorf("Expected output to contain 'Hello', got: %s", output)
	}
}

func TestAnthropicAdapter_IsStreamingRequest(t *testing.T) {
	adapter := &AnthropicAdapter{}

	streamingReq := `{"model":"test","stream":true}`
	if !adapter.IsStreamingRequest([]byte(streamingReq)) {
		t.Error("Expected streaming request to be detected")
	}

	nonStreamingReq := `{"model":"test","stream":false}`
	if adapter.IsStreamingRequest([]byte(nonStreamingReq)) {
		t.Error("Expected non-streaming request to return false")
	}

	invalidReq := `{invalid json}`
	if adapter.IsStreamingRequest([]byte(invalidReq)) {
		t.Error("Expected invalid JSON to return false (default)")
	}
}

func TestOpenAIAdapter_TransformRequest(t *testing.T) {
	adapter := &OpenAIAdapter{}
	body := []byte(`{"model":"test","stream":true}`)
	result, err := adapter.TransformRequest(body)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if string(result) != string(body) {
		t.Error("Expected body to be passed through unchanged")
	}
}

func TestOpenAIAdapter_ValidateRequest(t *testing.T) {
	adapter := &OpenAIAdapter{}
	streamingReq := []byte(`{"model":"test","stream":true}`)
	if err := adapter.ValidateRequest(streamingReq); err != nil {
		t.Errorf("Expected no error for streaming request, got: %v", err)
	}
	nonStreamingReq := []byte(`{"model":"test","stream":false}`)
	if err := adapter.ValidateRequest(nonStreamingReq); err != ErrNonStreamingNotSupported {
		t.Errorf("Expected ErrNonStreamingNotSupported, got: %v", err)
	}
}

func TestOpenAIAdapter_UpstreamURL(t *testing.T) {
	adapter := &OpenAIAdapter{}
	cfg := &config.Config{OpenAIUpstreamURL: "https://api.openai.com/v1/chat/completions"}
	if url := adapter.UpstreamURL(cfg); url != "https://api.openai.com/v1/chat/completions" {
		t.Errorf("Expected correct upstream URL, got: %s", url)
	}
}

func TestOpenAIAdapter_UpstreamAPIKey(t *testing.T) {
	adapter := &OpenAIAdapter{}
	cfg := &config.Config{OpenAIUpstreamAPIKey: "sk-test-key"}
	if key := adapter.UpstreamAPIKey(cfg); key != "sk-test-key" {
		t.Errorf("Expected correct API key, got: %s", key)
	}
}

func TestOpenAIAdapter_ForwardHeaders(t *testing.T) {
	adapter := &OpenAIAdapter{}
	src := http.Header{}
	dst := http.Header{}
	src.Set("X-Custom-Header", "value1")
	src.Set("Extra", "extra-value")
	src.Set("Authorization", "secret")
	src.Set("Connection", "keep-alive")
	src.Set("Upgrade", "websocket")
	src.Set("TE", "trailers")
	adapter.ForwardHeaders(src, dst)
	if dst.Get("X-Custom-Header") != "value1" {
		t.Error("Expected X-Custom-Header to be forwarded")
	}
	if dst.Get("Extra") != "extra-value" {
		t.Error("Expected Extra to be forwarded")
	}
	if dst.Get("Authorization") != "" {
		t.Error("Expected Authorization not to be forwarded")
	}
	if dst.Get("Connection") != "keep-alive" {
		t.Error("Expected Connection to be forwarded")
	}
	if dst.Get("Upgrade") != "websocket" {
		t.Error("Expected Upgrade to be forwarded")
	}
	if dst.Get("TE") != "trailers" {
		t.Error("Expected TE to be forwarded")
	}
}

func TestOpenAIAdapter_SendError(t *testing.T) {
	gin.SetMode(gin.TestMode)
	adapter := &OpenAIAdapter{}
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	adapter.SendError(c, http.StatusBadRequest, "test error message")
	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected status %d, got %d", http.StatusBadRequest, w.Code)
	}
	var resp map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &resp)
	errorObj, ok := resp["error"].(map[string]interface{})
	if !ok {
		t.Fatal("Expected error object")
	}
	if errorObj["message"] != "test error message" {
		t.Errorf("Expected correct error message, got: %v", errorObj["message"])
	}
	if errorObj["type"] != "invalid_request_error" {
		t.Errorf("Expected correct error type, got: %v", errorObj["type"])
	}
}

func TestOpenAIAdapter_CreateTransformer(t *testing.T) {
	adapter := &OpenAIAdapter{}
	var buf bytes.Buffer
	base := types.StreamChunk{ID: "test-123", Model: "test-model"}

	transformer := adapter.CreateTransformer(&buf, base)
	if transformer == nil {
		t.Fatal("Expected transformer to be created")
	}

	transformer.Transform(&sse.Event{
		Data: `{"id":"test-123","choices":[{"index":0,"delta":{"content":"Hello"}}]}`,
	})
	transformer.Flush()

	output := buf.String()
	if !bytes.Contains([]byte(output), []byte("Hello")) {
		t.Errorf("Expected output to contain 'Hello', got: %s", output)
	}
}

func TestOpenAIAdapter_IsStreamingRequest(t *testing.T) {
	adapter := &OpenAIAdapter{}

	streamingReq := `{"model":"test","stream":true}`
	if !adapter.IsStreamingRequest([]byte(streamingReq)) {
		t.Error("Expected streaming request to be detected")
	}

	nonStreamingReq := `{"model":"test","stream":false}`
	if adapter.IsStreamingRequest([]byte(nonStreamingReq)) {
		t.Error("Expected non-streaming request to be rejected")
	}

	invalidReq := `{invalid json}`
	if adapter.IsStreamingRequest([]byte(invalidReq)) {
		t.Error("Expected invalid JSON to return false (default)")
	}
}

func TestBridgeAdapter_TransformRequest(t *testing.T) {
	adapter := &BridgeAdapter{}

	anthReq := map[string]interface{}{
		"model": "test-model",
		"messages": []interface{}{
			map[string]interface{}{
				"role":    "user",
				"content": "Hello",
			},
		},
		"stream": true,
	}

	body, err := json.Marshal(anthReq)
	if err != nil {
		t.Fatalf("Failed to marshal request: %v", err)
	}

	transformed, err := adapter.TransformRequest(body)
	if err != nil {
		t.Fatalf("Failed to transform request: %v", err)
	}

	var openReq map[string]interface{}
	if err := json.Unmarshal(transformed, &openReq); err != nil {
		t.Fatalf("Failed to unmarshal transformed request: %v", err)
	}

	if openReq["model"] != "test-model" {
		t.Errorf("Expected model to be preserved, got: %v", openReq["model"])
	}

	messages, ok := openReq["messages"].([]interface{})
	if !ok || len(messages) == 0 {
		t.Error("Expected messages to be preserved")
	}
}

func TestBridgeAdapter_TransformRequest_InvalidJSON(t *testing.T) {
	adapter := &BridgeAdapter{}
	_, err := adapter.TransformRequest([]byte(`{invalid json}`))
	if err == nil {
		t.Error("Expected error for invalid JSON")
	}
}

func TestBridgeAdapter_ValidateRequest(t *testing.T) {
	adapter := &BridgeAdapter{}
	streamingReq := []byte(`{"model":"test","stream":true}`)
	if err := adapter.ValidateRequest(streamingReq); err != nil {
		t.Errorf("Expected no error for streaming request, got: %v", err)
	}
	nonStreamingReq := []byte(`{"model":"test","stream":false}`)
	if err := adapter.ValidateRequest(nonStreamingReq); err != ErrNonStreamingNotSupported {
		t.Errorf("Expected ErrNonStreamingNotSupported, got: %v", err)
	}
}

func TestBridgeAdapter_UpstreamURL(t *testing.T) {
	adapter := &BridgeAdapter{}
	cfg := &config.Config{OpenAIUpstreamURL: "https://api.openai.com/v1/chat/completions"}
	if url := adapter.UpstreamURL(cfg); url != "https://api.openai.com/v1/chat/completions" {
		t.Errorf("Expected correct upstream URL, got: %s", url)
	}
}

func TestBridgeAdapter_UpstreamAPIKey(t *testing.T) {
	adapter := &BridgeAdapter{}
	cfg := &config.Config{OpenAIUpstreamAPIKey: "sk-bridge-key"}
	if key := adapter.UpstreamAPIKey(cfg); key != "sk-bridge-key" {
		t.Errorf("Expected correct API key, got: %s", key)
	}
}

func TestBridgeAdapter_ForwardHeaders(t *testing.T) {
	adapter := &BridgeAdapter{}
	src := http.Header{}
	dst := http.Header{}
	src.Set("X-Custom-Header", "value1")
	src.Set("Extra", "extra-value")
	src.Set("Authorization", "secret")
	src.Set("Connection", "keep-alive")
	src.Set("Keep-Alive", "timeout=5")
	src.Set("Upgrade", "websocket")
	src.Set("TE", "trailers")
	adapter.ForwardHeaders(src, dst)
	if dst.Get("X-Custom-Header") != "value1" {
		t.Error("Expected X-Custom-Header to be forwarded")
	}
	if dst.Get("Extra") != "extra-value" {
		t.Error("Expected Extra to be forwarded")
	}
	if dst.Get("Authorization") != "" {
		t.Error("Expected Authorization not to be forwarded")
	}
	if dst.Get("Connection") != "keep-alive" {
		t.Error("Expected Connection to be forwarded")
	}
	if dst.Get("Keep-Alive") != "timeout=5" {
		t.Error("Expected Keep-Alive to be forwarded")
	}
	if dst.Get("Upgrade") != "websocket" {
		t.Error("Expected Upgrade to be forwarded")
	}
	if dst.Get("TE") != "trailers" {
		t.Error("Expected TE to be forwarded")
	}
}

func TestBridgeAdapter_SendError(t *testing.T) {
	gin.SetMode(gin.TestMode)
	adapter := &BridgeAdapter{}
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	adapter.SendError(c, http.StatusBadRequest, "bridge error message")
	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected status %d, got %d", http.StatusBadRequest, w.Code)
	}
	var resp map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &resp)
	if resp["type"] != "error" {
		t.Error("Expected error type")
	}
	errorDetail, ok := resp["error"].(map[string]interface{})
	if !ok {
		t.Fatal("Expected error detail object")
	}
	if errorDetail["message"] != "bridge error message" {
		t.Errorf("Expected correct error message, got: %v", errorDetail["message"])
	}
}

func TestBridgeAdapter_IsStreamingRequest(t *testing.T) {
	adapter := &BridgeAdapter{}

	streamingReq := `{"model":"test","stream":true}`
	if !adapter.IsStreamingRequest([]byte(streamingReq)) {
		t.Error("Expected streaming request to be detected")
	}

	nonStreamingReq := `{"model":"test","stream":false}`
	if adapter.IsStreamingRequest([]byte(nonStreamingReq)) {
		t.Error("Expected non-streaming request to return false")
	}

	invalidReq := `{invalid json}`
	if adapter.IsStreamingRequest([]byte(invalidReq)) {
		t.Error("Expected invalid JSON to return false (default)")
	}
}

func TestBridgeAdapter_CreateTransformer(t *testing.T) {
	adapter := &BridgeAdapter{}
	var buf bytes.Buffer
	base := types.StreamChunk{ID: "msg-123"}

	transformer := adapter.CreateTransformer(&buf, base)
	if transformer == nil {
		t.Fatal("Expected transformer to be created")
	}
}

func TestBridgeAdapter_ConvertMessage_StringContent(t *testing.T) {
	adapter := &BridgeAdapter{}
	msg := AnthropicMessageInput{
		Role:    "user",
		Content: "Hello world",
	}
	result := adapter.convertMessage(msg)
	if result.Role != "user" {
		t.Errorf("Expected role 'user', got: %s", result.Role)
	}
	if result.Content != "Hello world" {
		t.Errorf("Expected content 'Hello world', got: %v", result.Content)
	}
}

func TestBridgeAdapter_ConvertMessage_ArrayContent_Text(t *testing.T) {
	adapter := &BridgeAdapter{}
	msg := AnthropicMessageInput{
		Role: "assistant",
		Content: []interface{}{
			map[string]interface{}{
				"type": "text",
				"text": "Hello",
			},
			map[string]interface{}{
				"type": "text",
				"text": "world",
			},
		},
	}
	result := adapter.convertMessage(msg)
	if result.Role != "assistant" {
		t.Errorf("Expected role 'assistant', got: %s", result.Role)
	}
	if result.Content != "Hello\nworld" {
		t.Errorf("Expected content 'Hello\\nworld', got: %v", result.Content)
	}
}

func TestBridgeAdapter_ConvertMessage_ArrayContent_ToolUse(t *testing.T) {
	adapter := &BridgeAdapter{}
	msg := AnthropicMessageInput{
		Role: "assistant",
		Content: []interface{}{
			map[string]interface{}{
				"type": "tool_use",
				"id":   "tool-123",
				"name": "bash",
				"input": map[string]interface{}{
					"command": "ls",
				},
			},
		},
	}
	result := adapter.convertMessage(msg)
	if len(result.ToolCalls) != 1 {
		t.Fatalf("Expected 1 tool call, got: %d", len(result.ToolCalls))
	}
	if result.ToolCalls[0].ID != "tool-123" {
		t.Errorf("Expected tool call ID 'tool-123', got: %s", result.ToolCalls[0].ID)
	}
	if result.ToolCalls[0].Function.Name != "bash" {
		t.Errorf("Expected function name 'bash', got: %s", result.ToolCalls[0].Function.Name)
	}
}

func TestBridgeAdapter_ConvertMessage_ArrayContent_ToolResult(t *testing.T) {
	adapter := &BridgeAdapter{}
	msg := AnthropicMessageInput{
		Role: "user",
		Content: []interface{}{
			map[string]interface{}{
				"type":        "tool_result",
				"tool_use_id": "tool-456",
			},
		},
	}
	result := adapter.convertMessage(msg)
	if result.ToolCallID != "tool-456" {
		t.Errorf("Expected tool call ID 'tool-456', got: %s", result.ToolCallID)
	}
}

func TestBridgeAdapter_ConvertMessage_ArrayContent_Mixed(t *testing.T) {
	adapter := &BridgeAdapter{}
	msg := AnthropicMessageInput{
		Role: "assistant",
		Content: []interface{}{
			map[string]interface{}{
				"type": "text",
				"text": "Here is the result:",
			},
			map[string]interface{}{
				"type": "tool_use",
				"id":   "tool-789",
				"name": "read_file",
				"input": map[string]interface{}{
					"path": "/tmp/test.txt",
				},
			},
		},
	}
	result := adapter.convertMessage(msg)
	if result.Content != "Here is the result:" {
		t.Errorf("Expected content 'Here is the result:', got: %v", result.Content)
	}
	if len(result.ToolCalls) != 1 {
		t.Errorf("Expected 1 tool call, got: %d", len(result.ToolCalls))
	}
}

func TestBridgeAdapter_ConvertMessage_OtherContent(t *testing.T) {
	adapter := &BridgeAdapter{}
	msg := AnthropicMessageInput{
		Role:    "user",
		Content: 123,
	}
	result := adapter.convertMessage(msg)
	if result.Role != "user" {
		t.Errorf("Expected role 'user', got: %s", result.Role)
	}
}

func TestBridgeAdapter_ConvertTool(t *testing.T) {
	adapter := &BridgeAdapter{}
	anthTool := AnthropicToolDefinition{
		Name:        "bash",
		Description: "Execute bash commands",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"command":{"type":"string"}}}`),
	}
	result := adapter.convertTool(anthTool)
	if result.Type != "function" {
		t.Errorf("Expected type 'function', got: %s", result.Type)
	}
	if result.Function.Name != "bash" {
		t.Errorf("Expected name 'bash', got: %s", result.Function.Name)
	}
	if result.Function.Description != "Execute bash commands" {
		t.Errorf("Expected description 'Execute bash commands', got: %s", result.Function.Description)
	}
	if string(result.Function.Parameters) != `{"type":"object","properties":{"command":{"type":"string"}}}` {
		t.Errorf("Expected correct parameters, got: %s", string(result.Function.Parameters))
	}
}

func TestBridgeAdapter_ExtractSystemMessage_Nil(t *testing.T) {
	adapter := &BridgeAdapter{}
	result := adapter.extractSystemMessage(nil)
	if result != "" {
		t.Errorf("Expected empty string for nil, got: %s", result)
	}
}

func TestBridgeAdapter_ExtractSystemMessage_String(t *testing.T) {
	adapter := &BridgeAdapter{}
	result := adapter.extractSystemMessage("You are a helpful assistant.")
	if result != "You are a helpful assistant." {
		t.Errorf("Expected 'You are a helpful assistant.', got: %s", result)
	}
}

func TestBridgeAdapter_ExtractSystemMessage_Array(t *testing.T) {
	adapter := &BridgeAdapter{}
	system := []interface{}{
		map[string]interface{}{
			"type": "text",
			"text": "You are helpful.",
		},
		map[string]interface{}{
			"type": "text",
			"text": "Be concise.",
		},
	}
	result := adapter.extractSystemMessage(system)
	if result != "You are helpful.Be concise." {
		t.Errorf("Expected concatenated text, got: %s", result)
	}
}

func TestBridgeAdapter_ExtractSystemMessage_Array_NonMapItem(t *testing.T) {
	adapter := &BridgeAdapter{}
	system := []interface{}{
		"not a map",
		map[string]interface{}{
			"type": "text",
			"text": "Hello",
		},
	}
	result := adapter.extractSystemMessage(system)
	if result != "Hello" {
		t.Errorf("Expected 'Hello', got: %s", result)
	}
}

func TestBridgeAdapter_ExtractSystemMessage_Array_NoTextField(t *testing.T) {
	adapter := &BridgeAdapter{}
	system := []interface{}{
		map[string]interface{}{
			"type": "image",
			"data": "base64...",
		},
	}
	result := adapter.extractSystemMessage(system)
	if result != "" {
		t.Errorf("Expected empty string, got: %s", result)
	}
}

func TestBridgeAdapter_ExtractSystemMessage_OtherType(t *testing.T) {
	adapter := &BridgeAdapter{}
	result := adapter.extractSystemMessage(123)
	if result != "" {
		t.Errorf("Expected empty string for other type, got: %s", result)
	}
}

func TestBridgeAdapter_TransformRequest_WithSystem(t *testing.T) {
	adapter := &BridgeAdapter{}
	anthReq := map[string]interface{}{
		"model": "test-model",
		"system": []interface{}{
			map[string]interface{}{
				"type": "text",
				"text": "You are helpful.",
			},
		},
		"messages": []interface{}{
			map[string]interface{}{
				"role":    "user",
				"content": "Hello",
			},
		},
		"stream": true,
	}
	body, _ := json.Marshal(anthReq)
	transformed, err := adapter.TransformRequest(body)
	if err != nil {
		t.Fatalf("Failed to transform: %v", err)
	}
	var openReq OpenAIRequest
	json.Unmarshal(transformed, &openReq)
	if openReq.System != "You are helpful." {
		t.Errorf("Expected system message, got: %s", openReq.System)
	}
}

func TestBridgeAdapter_TransformRequest_WithTools(t *testing.T) {
	adapter := &BridgeAdapter{}
	anthReq := map[string]interface{}{
		"model": "test-model",
		"messages": []interface{}{
			map[string]interface{}{
				"role":    "user",
				"content": "Hello",
			},
		},
		"stream": true,
		"tools": []interface{}{
			map[string]interface{}{
				"name":        "bash",
				"description": "Execute bash commands",
				"input_schema": map[string]interface{}{
					"type": "object",
				},
			},
		},
	}
	body, _ := json.Marshal(anthReq)
	transformed, err := adapter.TransformRequest(body)
	if err != nil {
		t.Fatalf("Failed to transform: %v", err)
	}
	var openReq OpenAIRequest
	json.Unmarshal(transformed, &openReq)
	if len(openReq.Tools) != 1 {
		t.Fatalf("Expected 1 tool, got: %d", len(openReq.Tools))
	}
	if openReq.Tools[0].Function.Name != "bash" {
		t.Errorf("Expected tool name 'bash', got: %s", openReq.Tools[0].Function.Name)
	}
}

func TestBridgeAdapter_TransformRequest_WithParameters(t *testing.T) {
	adapter := &BridgeAdapter{}
	anthReq := map[string]interface{}{
		"model":       "test-model",
		"messages":    []interface{}{},
		"stream":      true,
		"max_tokens":  1024,
		"temperature": 0.7,
		"top_p":       0.9,
	}
	body, _ := json.Marshal(anthReq)
	transformed, err := adapter.TransformRequest(body)
	if err != nil {
		t.Fatalf("Failed to transform: %v", err)
	}
	var openReq OpenAIRequest
	json.Unmarshal(transformed, &openReq)
	if openReq.MaxTokens != 1024 {
		t.Errorf("Expected max_tokens 1024, got: %d", openReq.MaxTokens)
	}
	if openReq.Temperature != 0.7 {
		t.Errorf("Expected temperature 0.7, got: %f", openReq.Temperature)
	}
	if openReq.TopP != 0.9 {
		t.Errorf("Expected top_p 0.9, got: %f", openReq.TopP)
	}
}

func TestToolCallTransformer_Transform_EmptyData(t *testing.T) {
	var buf bytes.Buffer
	base := types.StreamChunk{ID: "test"}
	output := toolcall.NewOpenAIOutput(&buf, base)
	transformer := NewToolCallTransformer(&buf, base, output)
	transformer.Transform(&sse.Event{Data: ""})
	transformer.Flush()
}

func TestToolCallTransformer_Transform_Done(t *testing.T) {
	var buf bytes.Buffer
	base := types.StreamChunk{ID: "test"}
	output := toolcall.NewOpenAIOutput(&buf, base)
	transformer := NewToolCallTransformer(&buf, base, output)
	transformer.Transform(&sse.Event{Data: "[DONE]"})
	transformer.Flush()
	if !bytes.Contains(buf.Bytes(), []byte("data: [DONE]")) {
		t.Errorf("Expected [DONE] marker, got: %s", buf.String())
	}
}

func TestToolCallTransformer_Transform_InvalidJSON(t *testing.T) {
	var buf bytes.Buffer
	base := types.StreamChunk{ID: "test"}
	output := toolcall.NewOpenAIOutput(&buf, base)
	transformer := NewToolCallTransformer(&buf, base, output)
	transformer.Transform(&sse.Event{Data: `{invalid json}`})
	transformer.Flush()
}

func TestToolCallTransformer_Transform_Reasoning(t *testing.T) {
	var buf bytes.Buffer
	base := types.StreamChunk{ID: "test"}
	output := toolcall.NewOpenAIOutput(&buf, base)
	transformer := NewToolCallTransformer(&buf, base, output)
	transformer.Transform(&sse.Event{
		Data: `{"id":"test","choices":[{"index":0,"delta":{"reasoning":"thinking..."}}]}`,
	})
	transformer.Flush()
}

func TestToolCallTransformer_Transform_ReasoningContent(t *testing.T) {
	var buf bytes.Buffer
	base := types.StreamChunk{ID: "test"}
	output := toolcall.NewOpenAIOutput(&buf, base)
	transformer := NewToolCallTransformer(&buf, base, output)
	transformer.Transform(&sse.Event{
		Data: `{"id":"test","choices":[{"index":0,"delta":{"reasoning_content":"thinking..."}}]}`,
	})
	transformer.Flush()
}

func TestToolCallTransformer_Transform_ToolCalls(t *testing.T) {
	var buf bytes.Buffer
	base := types.StreamChunk{ID: "test"}
	output := toolcall.NewOpenAIOutput(&buf, base)
	transformer := NewToolCallTransformer(&buf, base, output)
	transformer.Transform(&sse.Event{
		Data: `{"id":"test","choices":[{"index":0,"delta":{"tool_calls":[{"id":"call-1","index":0,"function":{"name":"bash","arguments":"{\"cmd\":\"ls\"}"}}]}}]}`,
	})
	transformer.Flush()
}

func TestToolCallTransformer_Close(t *testing.T) {
	var buf bytes.Buffer
	base := types.StreamChunk{ID: "test"}
	output := toolcall.NewOpenAIOutput(&buf, base)
	transformer := NewToolCallTransformer(&buf, base, output)
	transformer.Close()
}

func TestProtocolError(t *testing.T) {
	err := ErrNonStreamingNotSupported
	if err.Error() != "Non-streaming requests not supported" {
		t.Errorf("Expected correct error message, got: %s", err.Error())
	}
}

func TestProtocolError_Message(t *testing.T) {
	err := &ProtocolError{Message: "custom error"}
	if err.Error() != "custom error" {
		t.Errorf("Expected 'custom error', got: %s", err.Error())
	}
}
