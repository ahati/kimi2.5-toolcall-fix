package downstream

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"ai-proxy/config"
	"ai-proxy/downstream/protocols"
	"ai-proxy/logging"
	"ai-proxy/types"

	"github.com/gin-gonic/gin"
	"github.com/tmaxmax/go-sse"
)

type mockStreamResponseWriter struct {
	header http.Header
	body   bytes.Buffer
	status int
}

func newMockStreamResponseWriter() *mockStreamResponseWriter {
	return &mockStreamResponseWriter{
		header: make(http.Header),
	}
}

func (m *mockStreamResponseWriter) Header() http.Header {
	return m.header
}

func (m *mockStreamResponseWriter) Write(data []byte) (int, error) {
	return m.body.Write(data)
}

func (m *mockStreamResponseWriter) WriteHeader(statusCode int) {
	m.status = statusCode
}

func (m *mockStreamResponseWriter) CloseNotify() <-chan bool {
	return make(chan bool)
}

func (m *mockStreamResponseWriter) Flush() {}

func TestStreamHandler_OpenAI_ToolCalls_Transformer(t *testing.T) {
	gin.SetMode(gin.TestMode)

	var output bytes.Buffer
	adapter := protocols.NewOpenAIAdapter()
	transformer := adapter.CreateTransformer(&output, types.StreamChunk{})

	events := []string{
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":123,"model":"test","choices":[{"index":0,"delta":{"reasoning":"Let me help.<|tool_calls_section_begin|><|tool_call_begin|>bash:1<|tool_call_argument_begin|>{\"command\":\"ls\"}<|tool_call_end|><|tool_calls_section_end|>"}}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":123,"model":"test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		`[DONE]`,
	}

	for _, event := range events {
		transformer.Transform(&sse.Event{Data: event})
	}
	transformer.Flush()

	result := output.String()
	if !strings.Contains(result, "tool_calls") {
		t.Errorf("Expected tool_calls in response, got: %s", result)
	}
	if !strings.Contains(result, "bash") {
		t.Errorf("Expected bash function name in response, got: %s", result)
	}
}

func TestStreamHandler_OpenAI_TextContent_Transformer(t *testing.T) {
	gin.SetMode(gin.TestMode)

	var output bytes.Buffer
	adapter := protocols.NewOpenAIAdapter()
	transformer := adapter.CreateTransformer(&output, types.StreamChunk{})

	events := []string{
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":123,"model":"test","choices":[{"index":0,"delta":{"content":"Hello"}}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":123,"model":"test","choices":[{"index":0,"delta":{"content":" world"}}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":123,"model":"test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		`[DONE]`,
	}

	for _, event := range events {
		transformer.Transform(&sse.Event{Data: event})
	}
	transformer.Flush()

	result := output.String()
	if !strings.Contains(result, "Hello") {
		t.Errorf("Expected 'Hello' in response, got: %s", result)
	}
	if !strings.Contains(result, "world") {
		t.Errorf("Expected 'world' in response, got: %s", result)
	}
}

func TestStreamHandler_Anthropic_ToolCalls_Transformer(t *testing.T) {
	gin.SetMode(gin.TestMode)

	var output bytes.Buffer
	adapter := protocols.NewAnthropicAdapter()
	transformer := adapter.CreateTransformer(&output, types.StreamChunk{ID: "msg_123"})

	events := []string{
		`{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","model":"kimi-k2.5"}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"I'll help you.<|tool_calls_section_begin|><|tool_call_begin|>bash:1<|tool_call_argument_begin|>{\"command\":\"ls -la\"}<|tool_call_end|><|tool_calls_section_end|>"}}`,
		`{"type":"content_block_stop","index":0}`,
	}

	for _, event := range events {
		transformer.Transform(&sse.Event{Data: event})
	}
	transformer.Flush()

	result := output.String()
	if !strings.Contains(result, "tool_use") {
		t.Errorf("Expected tool_use in response, got: %s", result)
	}
	if !strings.Contains(result, "bash") {
		t.Errorf("Expected bash function name in response, got: %s", result)
	}
}

func TestStreamHandler_Bridge_ToolCalls_Transformer(t *testing.T) {
	gin.SetMode(gin.TestMode)

	var output bytes.Buffer
	adapter := protocols.NewBridgeAdapter()
	transformer := adapter.CreateTransformer(&output, types.StreamChunk{ID: "chatcmpl-1"})

	events := []string{
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":123,"model":"test","choices":[{"index":0,"delta":{"reasoning":"Thinking...<|tool_calls_section_begin|><|tool_call_begin|>get_weather:1<|tool_call_argument_begin|>{\"city\":\"NYC\"}<|tool_call_end|><|tool_calls_section_end|>"}}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":123,"model":"test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
	}

	for _, event := range events {
		transformer.Transform(&sse.Event{Data: event})
	}
	transformer.Flush()

	result := output.String()
	if !strings.Contains(result, "tool_use") {
		t.Errorf("Expected tool_use in response, got: %s", result)
	}
	if !strings.Contains(result, "get_weather") {
		t.Errorf("Expected get_weather function name, got: %s", result)
	}
}

func TestStreamHandler_Bridge_RequestTransformation(t *testing.T) {
	adapter := protocols.NewBridgeAdapter()

	anthReq := map[string]interface{}{
		"model": "claude-3-sonnet",
		"messages": []interface{}{
			map[string]interface{}{
				"role":    "user",
				"content": "Hello",
			},
		},
		"stream":     true,
		"max_tokens": 1024,
		"tools": []interface{}{
			map[string]interface{}{
				"name":        "bash",
				"description": "Run bash",
				"input_schema": map[string]interface{}{
					"type": "object",
				},
			},
		},
	}

	anthBody, _ := json.Marshal(anthReq)
	transformed, err := adapter.TransformRequest(anthBody)
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var openReq map[string]interface{}
	if err := json.Unmarshal(transformed, &openReq); err != nil {
		t.Fatalf("Failed to parse transformed request: %v", err)
	}

	if openReq["model"] != "claude-3-sonnet" {
		t.Errorf("Expected model preserved, got: %v", openReq["model"])
	}

	messages, ok := openReq["messages"].([]interface{})
	if !ok || len(messages) == 0 {
		t.Errorf("Expected messages to be transformed, got: %v", openReq)
	}

	tools, ok := openReq["tools"].([]interface{})
	if !ok || len(tools) == 0 {
		t.Errorf("Expected tools to be transformed, got: %v", openReq)
	}
}

func TestStreamHandler_NonStreamingRequest(t *testing.T) {
	gin.SetMode(gin.TestMode)

	cfg := &config.Config{
		OpenAIUpstreamURL:    "http://localhost:9999",
		OpenAIUpstreamAPIKey: "test-key",
	}

	adapter := protocols.NewOpenAIAdapter()
	handler := StreamHandler(cfg, adapter)

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	reqBody := `{"model":"test","messages":[{"role":"user","content":"hello"}],"stream":false}`
	c.Request = httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
	c.Request.Header.Set("Content-Type", "application/json")

	handler(c)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400 for non-streaming request, got %d", w.Code)
	}

	body := w.Body.String()
	if !strings.Contains(body, "Non-streaming") {
		t.Errorf("Expected error message about non-streaming, got: %s", body)
	}
}

func TestStreamHandler_UpstreamError(t *testing.T) {
	gin.SetMode(gin.TestMode)

	mockUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error": {"message": "Internal error"}}`))
	}))
	defer mockUpstream.Close()

	cfg := &config.Config{
		OpenAIUpstreamURL:    mockUpstream.URL,
		OpenAIUpstreamAPIKey: "test-key",
	}

	adapter := protocols.NewOpenAIAdapter()
	handler := StreamHandler(cfg, adapter)

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	reqBody := `{"model":"test","messages":[{"role":"user","content":"hello"}],"stream":true}`
	c.Request = httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
	c.Request.Header.Set("Content-Type", "application/json")

	handler(c)

	if w.Code != http.StatusBadGateway {
		t.Errorf("Expected status 502 for upstream error, got %d", w.Code)
	}
}

func TestStreamHandler_UpstreamConnectionError(t *testing.T) {
	gin.SetMode(gin.TestMode)

	cfg := &config.Config{
		OpenAIUpstreamURL:    "http://localhost:59999",
		OpenAIUpstreamAPIKey: "test-key",
	}

	adapter := protocols.NewOpenAIAdapter()
	handler := StreamHandler(cfg, adapter)

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	reqBody := `{"model":"test","messages":[{"role":"user","content":"hello"}],"stream":true}`
	c.Request = httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
	c.Request.Header.Set("Content-Type", "application/json")

	handler(c)

	if w.Code != http.StatusBadGateway {
		t.Errorf("Expected status 502 for connection error, got %d", w.Code)
	}
}

func TestStreamHandler_InvalidRequestBody(t *testing.T) {
	gin.SetMode(gin.TestMode)

	cfg := &config.Config{
		OpenAIUpstreamURL:    "http://localhost:9999",
		OpenAIUpstreamAPIKey: "test-key",
	}

	adapter := protocols.NewOpenAIAdapter()
	handler := StreamHandler(cfg, adapter)

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	c.Request = httptest.NewRequest("POST", "/v1/chat/completions", errReader(0))
	c.Request.Header.Set("Content-Type", "application/json")

	handler(c)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400 for read error, got %d", w.Code)
	}
}

type errReader int

func (errReader) Read(p []byte) (n int, err error) {
	return 0, io.ErrUnexpectedEOF
}

func TestStreamHandler_OpenAI_MultipleToolCalls(t *testing.T) {
	gin.SetMode(gin.TestMode)

	var output bytes.Buffer
	adapter := protocols.NewOpenAIAdapter()
	transformer := adapter.CreateTransformer(&output, types.StreamChunk{})

	events := []string{
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":123,"model":"test","choices":[{"index":0,"delta":{"reasoning":"<|tool_calls_section_begin|><|tool_call_begin|>bash:1<|tool_call_argument_begin|>{\"cmd\":\"ls\"}<|tool_call_end|><|tool_call_begin|>read:2<|tool_call_argument_begin|>{\"path\":\"/test\"}<|tool_call_end|><|tool_calls_section_end|>"}}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":123,"model":"test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		`[DONE]`,
	}

	for _, event := range events {
		transformer.Transform(&sse.Event{Data: event})
	}
	transformer.Flush()

	result := output.String()
	if strings.Count(result, `"function"`) < 2 {
		t.Errorf("Expected at least 2 tool calls in response, got: %s", result)
	}
}

func TestProtocolAdapter_IsStreamingRequest(t *testing.T) {
	tests := []struct {
		name     string
		adapter  protocols.ProtocolAdapter
		body     string
		expected bool
	}{
		{"OpenAI streaming true", protocols.NewOpenAIAdapter(), `{"stream":true}`, true},
		{"OpenAI streaming false", protocols.NewOpenAIAdapter(), `{"stream":false}`, false},
		{"OpenAI missing stream", protocols.NewOpenAIAdapter(), `{}`, false},
		{"Anthropic streaming true", protocols.NewAnthropicAdapter(), `{"stream":true}`, true},
		{"Anthropic streaming false", protocols.NewAnthropicAdapter(), `{"stream":false}`, false},
		{"Bridge streaming true", protocols.NewBridgeAdapter(), `{"stream":true}`, true},
		{"Bridge streaming false", protocols.NewBridgeAdapter(), `{"stream":false}`, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.adapter.IsStreamingRequest([]byte(tt.body))
			if result != tt.expected {
				t.Errorf("Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

func TestProtocolAdapter_TransformRequest(t *testing.T) {
	t.Run("OpenAI passthrough", func(t *testing.T) {
		adapter := protocols.NewOpenAIAdapter()
		body := []byte(`{"model":"test"}`)
		result, err := adapter.TransformRequest(body)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if string(result) != string(body) {
			t.Errorf("Expected passthrough, got: %s", result)
		}
	})

	t.Run("Anthropic passthrough", func(t *testing.T) {
		adapter := protocols.NewAnthropicAdapter()
		body := []byte(`{"model":"test"}`)
		result, err := adapter.TransformRequest(body)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if string(result) != string(body) {
			t.Errorf("Expected passthrough, got: %s", result)
		}
	})

	t.Run("Bridge transforms Anthropic to OpenAI", func(t *testing.T) {
		adapter := protocols.NewBridgeAdapter()
		anthBody := `{"model":"claude","messages":[{"role":"user","content":"hi"}],"max_tokens":100}`
		result, err := adapter.TransformRequest([]byte(anthBody))
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		var openReq map[string]interface{}
		if err := json.Unmarshal(result, &openReq); err != nil {
			t.Fatalf("Failed to parse result: %v", err)
		}

		if openReq["model"] != "claude" {
			t.Errorf("Expected model preserved, got: %v", openReq["model"])
		}
	})
}

func TestStreamHandler_CreateTransformer(t *testing.T) {
	tests := []struct {
		name    string
		adapter protocols.ProtocolAdapter
	}{
		{"OpenAI", protocols.NewOpenAIAdapter()},
		{"Anthropic", protocols.NewAnthropicAdapter()},
		{"Bridge", protocols.NewBridgeAdapter()},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			base := types.StreamChunk{ID: "test-123"}

			transformer := tt.adapter.CreateTransformer(&buf, base)
			if transformer == nil {
				t.Fatal("Expected transformer to be created")
			}

			transformer.Transform(&sse.Event{
				Data: `{"id":"test","choices":[{"delta":{"content":"Hello"}}]}`,
			})
			transformer.Flush()

			if buf.Len() == 0 {
				t.Error("Expected output from transformer")
			}
		})
	}
}

func TestStreamHandler_UpstreamError_Anthropic(t *testing.T) {
	gin.SetMode(gin.TestMode)

	mockUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		w.Write([]byte(`{"error": {"message": "Service unavailable"}}`))
	}))
	defer mockUpstream.Close()

	cfg := &config.Config{
		AnthropicUpstreamURL: mockUpstream.URL,
		AnthropicAPIKey:      "test-key",
	}

	adapter := protocols.NewAnthropicAdapter()
	handler := StreamHandler(cfg, adapter)

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	reqBody := `{"model":"test","messages":[{"role":"user","content":"hello"}],"stream":true,"max_tokens":1024}`
	c.Request = httptest.NewRequest("POST", "/v1/messages", strings.NewReader(reqBody))
	c.Request.Header.Set("Content-Type", "application/json")

	handler(c)

	if w.Code != http.StatusBadGateway {
		t.Errorf("Expected status 502 for upstream error, got %d", w.Code)
	}
}

func TestStreamHandler_NonStreamingRequest_Anthropic(t *testing.T) {
	gin.SetMode(gin.TestMode)

	cfg := &config.Config{
		AnthropicUpstreamURL: "http://localhost:9999",
		AnthropicAPIKey:      "test-key",
	}

	adapter := protocols.NewAnthropicAdapter()
	handler := StreamHandler(cfg, adapter)

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	reqBody := `{"model":"kimi-k2.5","messages":[{"role":"user","content":"hello"}],"stream":false,"max_tokens":1024}`
	c.Request = httptest.NewRequest("POST", "/v1/messages", strings.NewReader(reqBody))
	c.Request.Header.Set("Content-Type", "application/json")

	handler(c)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400 for non-streaming request, got %d", w.Code)
	}

	body := w.Body.String()
	if !strings.Contains(body, "Non-streaming") {
		t.Errorf("Expected error message about non-streaming, got: %s", body)
	}
}

func TestStreamHandler_NonStreamingRequest_Bridge(t *testing.T) {
	gin.SetMode(gin.TestMode)

	cfg := &config.Config{
		OpenAIUpstreamURL:    "http://localhost:9999",
		OpenAIUpstreamAPIKey: "test-key",
	}

	adapter := protocols.NewBridgeAdapter()
	handler := StreamHandler(cfg, adapter)

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	reqBody := `{"model":"test","messages":[{"role":"user","content":"hello"}],"stream":false,"max_tokens":1024}`
	c.Request = httptest.NewRequest("POST", "/v1/openai-to-anthropic/messages", strings.NewReader(reqBody))
	c.Request.Header.Set("Content-Type", "application/json")

	handler(c)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400 for non-streaming request, got %d", w.Code)
	}

	body := w.Body.String()
	if !strings.Contains(body, "Non-streaming") {
		t.Errorf("Expected error message about non-streaming, got: %s", body)
	}
}

func TestStreamHandler_Bridge_ToolWithNamespace(t *testing.T) {
	gin.SetMode(gin.TestMode)

	var output bytes.Buffer
	adapter := protocols.NewBridgeAdapter()
	transformer := adapter.CreateTransformer(&output, types.StreamChunk{ID: "chatcmpl-1"})

	events := []string{
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":123,"model":"test","choices":[{"index":0,"delta":{"reasoning":"<|tool_calls_section_begin|><|tool_call_begin|>functions.bash:1<|tool_call_argument_begin|>{\"cmd\":\"ls\"}<|tool_call_end|><|tool_calls_section_end|>"}}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":123,"model":"test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
	}

	for _, event := range events {
		transformer.Transform(&sse.Event{Data: event})
	}
	transformer.Flush()

	result := output.String()
	if !strings.Contains(result, `"name":"bash"`) {
		t.Errorf("Expected function name 'bash' extracted from 'functions.bash', got: %s", result)
	}
}

func TestProtocolAdapter_UpstreamURL(t *testing.T) {
	cfg := &config.Config{
		OpenAIUpstreamURL:    "http://openai.example.com",
		OpenAIUpstreamAPIKey: "openai-key",
		AnthropicUpstreamURL: "http://anthropic.example.com",
		AnthropicAPIKey:      "anthropic-key",
	}

	t.Run("OpenAI", func(t *testing.T) {
		adapter := protocols.NewOpenAIAdapter()
		if adapter.UpstreamURL(cfg) != "http://openai.example.com" {
			t.Errorf("Expected OpenAI upstream URL")
		}
		if adapter.UpstreamAPIKey(cfg) != "openai-key" {
			t.Errorf("Expected OpenAI API key")
		}
	})

	t.Run("Anthropic", func(t *testing.T) {
		adapter := protocols.NewAnthropicAdapter()
		if adapter.UpstreamURL(cfg) != "http://anthropic.example.com" {
			t.Errorf("Expected Anthropic upstream URL")
		}
		if adapter.UpstreamAPIKey(cfg) != "anthropic-key" {
			t.Errorf("Expected Anthropic API key")
		}
	})

	t.Run("Bridge uses OpenAI", func(t *testing.T) {
		adapter := protocols.NewBridgeAdapter()
		if adapter.UpstreamURL(cfg) != "http://openai.example.com" {
			t.Errorf("Expected Bridge to use OpenAI upstream URL")
		}
		if adapter.UpstreamAPIKey(cfg) != "openai-key" {
			t.Errorf("Expected Bridge to use OpenAI API key")
		}
	})
}

func TestProtocolAdapter_ForwardHeaders(t *testing.T) {
	tests := []struct {
		name        string
		adapter     protocols.ProtocolAdapter
		headers     map[string]string
		expected    []string
		notExpected []string
	}{
		{
			name:     "OpenAI forwards X-headers",
			adapter:  protocols.NewOpenAIAdapter(),
			headers:  map[string]string{"X-Custom": "value", "Authorization": "secret"},
			expected: []string{"X-Custom"},
		},
		{
			name:     "Anthropic forwards Anthropic headers",
			adapter:  protocols.NewAnthropicAdapter(),
			headers:  map[string]string{"X-Custom": "value", "Anthropic-Version": "2023-06-01"},
			expected: []string{"X-Custom", "Anthropic-Version"},
		},
		{
			name:     "Bridge forwards X-headers",
			adapter:  protocols.NewBridgeAdapter(),
			headers:  map[string]string{"X-Request-Id": "123", "Content-Type": "application/json"},
			expected: []string{"X-Request-Id"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			src := http.Header{}
			for k, v := range tt.headers {
				src.Set(k, v)
			}
			dst := http.Header{}

			tt.adapter.ForwardHeaders(src, dst)

			for _, h := range tt.expected {
				if dst.Get(h) == "" {
					t.Errorf("Expected header %s to be forwarded", h)
				}
			}
			for _, h := range tt.notExpected {
				if dst.Get(h) != "" {
					t.Errorf("Did not expect header %s to be forwarded", h)
				}
			}
		})
	}
}

func TestStreamResponseWithAdapter_Integration(t *testing.T) {
	gin.SetMode(gin.TestMode)

	cfg := &config.Config{
		OpenAIUpstreamURL:    "http://localhost:9999",
		OpenAIUpstreamAPIKey: "test-key",
	}

	mockUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("data: {\"id\":\"chatcmpl-123\",\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\ndata: [DONE]\n\n"))
	}))
	defer mockUpstream.Close()

	cfg.OpenAIUpstreamURL = mockUpstream.URL

	adapter := protocols.NewOpenAIAdapter()
	handler := StreamHandler(cfg, adapter)

	router := gin.New()
	router.POST("/v1/chat/completions", handler)

	server := httptest.NewServer(router)
	defer server.Close()

	resp, err := http.Post(server.URL+"/v1/chat/completions", "application/json", strings.NewReader(`{"model":"test","messages":[{"role":"user","content":"hello"}],"stream":true}`))
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("Expected status %d, got %d", http.StatusOK, resp.StatusCode)
	}

	contentType := resp.Header.Get("Content-Type")
	if contentType != "text/event-stream" {
		t.Errorf("Expected Content-Type text/event-stream, got %s", contentType)
	}
}

func TestStreamResponseWithAdapter_WithCapture_Integration(t *testing.T) {
	gin.SetMode(gin.TestMode)

	cfg := &config.Config{
		OpenAIUpstreamURL:    "http://localhost:9999",
		OpenAIUpstreamAPIKey: "test-key",
	}

	mockUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("data: {\"id\":\"chatcmpl-456\",\"choices\":[{\"delta\":{\"content\":\"Test\"}}]}\n\ndata: [DONE]\n\n"))
	}))
	defer mockUpstream.Close()

	cfg.OpenAIUpstreamURL = mockUpstream.URL

	adapter := protocols.NewOpenAIAdapter()
	handler := StreamHandler(cfg, adapter)

	router := gin.New()
	router.POST("/v1/chat/completions", handler)

	server := httptest.NewServer(router)
	defer server.Close()

	resp, err := http.Post(server.URL+"/v1/chat/completions", "application/json", strings.NewReader(`{"model":"test","messages":[{"role":"user","content":"hello"}],"stream":true}`))
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "Test") {
		t.Errorf("Expected streaming content, got: %s", string(body))
	}
}

func TestStreamResponseWithAdapter_Anthropic_Integration(t *testing.T) {
	gin.SetMode(gin.TestMode)

	cfg := &config.Config{
		AnthropicUpstreamURL: "http://localhost:9999",
		AnthropicAPIKey:      "test-key",
	}

	mockUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("data: {\"id\":\"msg-789\",\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\ndata: [DONE]\n\n"))
	}))
	defer mockUpstream.Close()

	cfg.AnthropicUpstreamURL = mockUpstream.URL

	adapter := protocols.NewAnthropicAdapter()
	handler := StreamHandler(cfg, adapter)

	router := gin.New()
	router.POST("/v1/messages", handler)

	server := httptest.NewServer(router)
	defer server.Close()

	resp, err := http.Post(server.URL+"/v1/messages", "application/json", strings.NewReader(`{"model":"test","messages":[{"role":"user","content":"hello"}],"stream":true,"max_tokens":1024}`))
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("Expected status %d, got %d", http.StatusOK, resp.StatusCode)
	}
}

func TestStreamResponseWithAdapter_Bridge_Integration(t *testing.T) {
	gin.SetMode(gin.TestMode)

	cfg := &config.Config{
		OpenAIUpstreamURL:    "http://localhost:9999",
		OpenAIUpstreamAPIKey: "test-key",
	}

	mockUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("data: {\"id\":\"chatcmpl-bridge\",\"choices\":[{\"delta\":{\"content\":\"Response\"}}]}\n\ndata: [DONE]\n\n"))
	}))
	defer mockUpstream.Close()

	cfg.OpenAIUpstreamURL = mockUpstream.URL

	adapter := protocols.NewBridgeAdapter()
	handler := StreamHandler(cfg, adapter)

	router := gin.New()
	router.POST("/v1/openai-to-anthropic/messages", handler)

	server := httptest.NewServer(router)
	defer server.Close()

	resp, err := http.Post(server.URL+"/v1/openai-to-anthropic/messages", "application/json", strings.NewReader(`{"model":"test","messages":[{"role":"user","content":"hello"}],"stream":true,"max_tokens":1024}`))
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("Expected status %d, got %d", http.StatusOK, resp.StatusCode)
	}
}

func TestReadBody(t *testing.T) {
	gin.SetMode(gin.TestMode)

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	body := `{"test": "value"}`
	c.Request = httptest.NewRequest("POST", "/test", strings.NewReader(body))

	result, err := readBody(c)

	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if string(result) != body {
		t.Errorf("Expected '%s', got '%s'", body, string(result))
	}
}

func TestHandleUpstreamError(t *testing.T) {
	gin.SetMode(gin.TestMode)

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	mockResp := &http.Response{
		StatusCode: http.StatusInternalServerError,
		Body:       io.NopCloser(strings.NewReader(`{"error": "internal error"}`)),
	}

	handleUpstreamError(c, mockResp)

	if w.Code != http.StatusBadGateway {
		t.Errorf("Expected status %d, got %d", http.StatusBadGateway, w.Code)
	}

	var response map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	errObj, ok := response["error"].(map[string]interface{})
	if !ok {
		t.Fatal("Expected error object in response")
	}
	if errObj["type"] != "upstream_error" {
		t.Errorf("Expected type 'upstream_error', got %v", errObj["type"])
	}
}

func TestStreamResponseWithAdapter_WithCaptureContextMiddleware(t *testing.T) {
	gin.SetMode(gin.TestMode)

	cfg := &config.Config{
		OpenAIUpstreamURL:    "http://localhost:9999",
		OpenAIUpstreamAPIKey: "test-key",
	}

	mockUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("data: {\"id\":\"chatcmpl-capture-test\",\"choices\":[{\"delta\":{\"content\":\"Captured\"}}]}\n\ndata: [DONE]\n\n"))
	}))
	defer mockUpstream.Close()

	cfg.OpenAIUpstreamURL = mockUpstream.URL

	adapter := protocols.NewOpenAIAdapter()
	handler := StreamHandler(cfg, adapter)

	router := gin.New()
	router.Use(logging.CaptureMiddleware(nil))
	router.POST("/v1/chat/completions", handler)

	server := httptest.NewServer(router)
	defer server.Close()

	resp, err := http.Post(server.URL+"/v1/chat/completions", "application/json", strings.NewReader(`{"model":"test","messages":[{"role":"user","content":"hello"}],"stream":true}`))
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("Expected status %d, got %d", http.StatusOK, resp.StatusCode)
	}
}

func TestStreamResponseWithAdapter_WithStorage(t *testing.T) {
	gin.SetMode(gin.TestMode)

	cfg := &config.Config{
		OpenAIUpstreamURL:    "http://localhost:9999",
		OpenAIUpstreamAPIKey: "test-key",
	}

	mockUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("data: {\"id\":\"chatcmpl-storage\",\"choices\":[{\"delta\":{\"content\":\"Stored\"}}]}\n\ndata: [DONE]\n\n"))
	}))
	defer mockUpstream.Close()

	cfg.OpenAIUpstreamURL = mockUpstream.URL

	adapter := protocols.NewOpenAIAdapter()
	handler := StreamHandler(cfg, adapter)

	router := gin.New()
	router.Use(logging.CaptureMiddleware(nil))
	router.POST("/v1/chat/completions", handler)

	server := httptest.NewServer(router)
	defer server.Close()

	resp, err := http.Post(server.URL+"/v1/chat/completions", "application/json", strings.NewReader(`{"model":"test","messages":[{"role":"user","content":"hello"}],"stream":true}`))
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "Stored") {
		t.Errorf("Expected content in response, got: %s", string(body))
	}
}

func TestStreamResponseWithAdapter_ToolCallsWithCapture(t *testing.T) {
	gin.SetMode(gin.TestMode)

	cfg := &config.Config{
		OpenAIUpstreamURL:    "http://localhost:9999",
		OpenAIUpstreamAPIKey: "test-key",
	}

	toolCallData := "data: {\"id\":\"chatcmpl-tools\",\"choices\":[{\"delta\":{\"reasoning\":\"<|tool_calls_section_begin|><|tool_call_begin|>bash:1<|tool_call_argument_begin|>{\\\"cmd\\\":\\\"ls\\\"}<|tool_call_end|><|tool_calls_section_end|>\"}}]}\n\ndata: [DONE]\n\n"
	mockUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(toolCallData))
	}))
	defer mockUpstream.Close()

	cfg.OpenAIUpstreamURL = mockUpstream.URL

	adapter := protocols.NewOpenAIAdapter()
	handler := StreamHandler(cfg, adapter)

	router := gin.New()
	router.Use(logging.CaptureMiddleware(nil))
	router.POST("/v1/chat/completions", handler)

	server := httptest.NewServer(router)
	defer server.Close()

	resp, err := http.Post(server.URL+"/v1/chat/completions", "application/json", strings.NewReader(`{"model":"test","messages":[{"role":"user","content":"list files"}],"stream":true}`))
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("Expected status %d, got %d", http.StatusOK, resp.StatusCode)
	}
}

func TestStreamResponseWithAdapter_MultipleEventsWithCapture(t *testing.T) {
	gin.SetMode(gin.TestMode)

	cfg := &config.Config{
		OpenAIUpstreamURL:    "http://localhost:9999",
		OpenAIUpstreamAPIKey: "test-key",
	}

	mockUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("data: {\"id\":\"chatcmpl-multi\",\"choices\":[{\"delta\":{\"content\":\"First\"}}]}\n\n"))
		w.Write([]byte("data: {\"id\":\"chatcmpl-multi\",\"choices\":[{\"delta\":{\"content\":\"Second\"}}]}\n\n"))
		w.Write([]byte("data: {\"id\":\"chatcmpl-multi\",\"choices\":[{\"delta\":{\"content\":\"Third\"}}]}\n\n"))
		w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer mockUpstream.Close()

	cfg.OpenAIUpstreamURL = mockUpstream.URL

	adapter := protocols.NewOpenAIAdapter()
	handler := StreamHandler(cfg, adapter)

	router := gin.New()
	router.Use(logging.CaptureMiddleware(nil))
	router.POST("/v1/chat/completions", handler)

	server := httptest.NewServer(router)
	defer server.Close()

	resp, err := http.Post(server.URL+"/v1/chat/completions", "application/json", strings.NewReader(`{"model":"test","messages":[{"role":"user","content":"hello"}],"stream":true}`))
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "First") {
		t.Errorf("Expected 'First' in response, got: %s", string(body))
	}
}

func TestStreamResponseWithAdapter_AnthropicWithCapture(t *testing.T) {
	gin.SetMode(gin.TestMode)

	cfg := &config.Config{
		AnthropicUpstreamURL: "http://localhost:9999",
		AnthropicAPIKey:      "test-key",
	}

	mockUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("data: {\"id\":\"msg-anthropic\",\"choices\":[{\"delta\":{\"content\":\"Anthropic\"}}]}\n\ndata: [DONE]\n\n"))
	}))
	defer mockUpstream.Close()

	cfg.AnthropicUpstreamURL = mockUpstream.URL

	adapter := protocols.NewAnthropicAdapter()
	handler := StreamHandler(cfg, adapter)

	router := gin.New()
	router.Use(logging.CaptureMiddleware(nil))
	router.POST("/v1/messages", handler)

	server := httptest.NewServer(router)
	defer server.Close()

	resp, err := http.Post(server.URL+"/v1/messages", "application/json", strings.NewReader(`{"model":"test","messages":[{"role":"user","content":"hello"}],"stream":true,"max_tokens":1024}`))
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("Expected status %d, got %d", http.StatusOK, resp.StatusCode)
	}
}

func TestStreamResponseWithAdapter_BridgeWithCapture(t *testing.T) {
	gin.SetMode(gin.TestMode)

	cfg := &config.Config{
		OpenAIUpstreamURL:    "http://localhost:9999",
		OpenAIUpstreamAPIKey: "test-key",
	}

	mockUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("data: {\"id\":\"chatcmpl-bridge-capture\",\"choices\":[{\"delta\":{\"reasoning\":\"<|tool_calls_section_begin|><|tool_call_begin|>get_weather:1<|tool_call_argument_begin|>{\\\"city\\\":\\\"NYC\\\"}<|tool_call_end|><|tool_calls_section_end|>\"}}]}\n\ndata: [DONE]\n\n"))
	}))
	defer mockUpstream.Close()

	cfg.OpenAIUpstreamURL = mockUpstream.URL

	adapter := protocols.NewBridgeAdapter()
	handler := StreamHandler(cfg, adapter)

	router := gin.New()
	router.Use(logging.CaptureMiddleware(nil))
	router.POST("/v1/openai-to-anthropic/messages", handler)

	server := httptest.NewServer(router)
	defer server.Close()

	resp, err := http.Post(server.URL+"/v1/openai-to-anthropic/messages", "application/json", strings.NewReader(`{"model":"test","messages":[{"role":"user","content":"weather"}],"stream":true,"max_tokens":1024}`))
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("Expected status %d, got %d", http.StatusOK, resp.StatusCode)
	}
}
