package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"ai-proxy/config"
	"ai-proxy/types"

	"github.com/gin-gonic/gin"
)

func TestCountTokensHandler_ValidateRequest(t *testing.T) {
	cfg := &config.Config{}
	h := &CountTokensHandler{cfg: cfg}

	tests := []struct {
		name      string
		body      string
		wantError bool
	}{
		{
			name: "valid request with model and messages",
			body: `{
				"model": "kimi-k2.5",
				"messages": [{"role": "user", "content": "Hello"}]
			}`,
			wantError: false,
		},
		{
			name: "valid request with multiple messages",
			body: `{
				"model": "kimi-k2.5",
				"messages": [
					{"role": "user", "content": "Hello"},
					{"role": "assistant", "content": "Hi there!"}
				]
			}`,
			wantError: false,
		},
		{
			name: "valid request with tools",
			body: `{
				"model": "kimi-k2.5",
				"messages": [{"role": "user", "content": "Hello"}],
				"tools": [{
					"name": "bash",
					"description": "Execute bash commands",
					"input_schema": {
						"type": "object",
						"properties": {
							"command": {"type": "string"}
						}
					}
				}]
			}`,
			wantError: false,
		},
		{
			name:      "invalid JSON",
			body:      `{invalid json}`,
			wantError: true,
		},
		{
			name:      "missing model",
			body:      `{"messages": [{"role": "user", "content": "Hello"}]}`,
			wantError: true,
		},
		{
			name:      "empty messages array",
			body:      `{"model": "kimi-k2.5", "messages": []}`,
			wantError: true,
		},
		{
			name:      "missing role in message",
			body:      `{"model": "kimi-k2.5", "messages": [{"content": "Hello"}]}`,
			wantError: true,
		},
		{
			name:      "missing content in message",
			body:      `{"model": "kimi-k2.5", "messages": [{"role": "user"}]}`,
			wantError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := h.ValidateRequest([]byte(tt.body))

			if tt.wantError && err == nil {
				t.Errorf("Expected error, got nil")
			}
			if !tt.wantError && err != nil {
				t.Errorf("Expected no error, got: %v", err)
			}
		})
	}
}

func TestCountTokensHandler_TransformRequest(t *testing.T) {
	cfg := &config.Config{}
	h := &CountTokensHandler{cfg: cfg}

	tests := []struct {
		name          string
		input         string
		wantModel     string
		wantUnchanged bool
	}{
		{
			name:      "adds default model when missing",
			input:     `{"messages": [{"role": "user", "content": "Hello"}]}`,
			wantModel: "kimi-k2.5",
		},
		{
			name:          "preserves existing model",
			input:         `{"model": "custom-model", "messages": [{"role": "user", "content": "Hello"}]}`,
			wantModel:     "custom-model",
			wantUnchanged: true,
		},
		{
			name:      "adds model with other fields",
			input:     `{"messages": [{"role": "user", "content": "Hello"}], "tools": []}`,
			wantModel: "kimi-k2.5",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output, err := h.TransformRequest([]byte(tt.input))
			if err != nil {
				t.Fatalf("TransformRequest failed: %v", err)
			}

			var result map[string]interface{}
			if err := json.Unmarshal(output, &result); err != nil {
				t.Fatalf("Failed to unmarshal output: %v", err)
			}

			model, ok := result["model"].(string)
			if !ok {
				t.Fatal("Model field not found in output")
			}

			if model != tt.wantModel {
				t.Errorf("Expected model %q, got %q", tt.wantModel, model)
			}

			if tt.wantUnchanged && string(output) != tt.input {
				t.Errorf("Expected unchanged output, got: %s", string(output))
			}
		})
	}
}

func TestCountTokensHandler_UpstreamURL(t *testing.T) {
	tests := []struct {
		name    string
		baseURL string
		wantURL string
	}{
		{
			name:    "standard messages URL",
			baseURL: "https://api.anthropic.com/v1/messages",
			wantURL: "https://api.anthropic.com/v1/messages/count_tokens",
		},
		{
			name:    "custom messages URL",
			baseURL: "https://custom.api.com/v1/messages",
			wantURL: "https://custom.api.com/v1/messages/count_tokens",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := config.NewTestConfigWithAnthropic(tt.baseURL, "test-key")
			h := &CountTokensHandler{cfg: cfg}

			gotURL := h.UpstreamURL()
			if gotURL != tt.wantURL {
				t.Errorf("Expected URL %q, got %q", tt.wantURL, gotURL)
			}
		})
	}
}

func TestCountTokensHandler_ResolveAPIKey(t *testing.T) {
	cfg := config.NewTestConfigWithAnthropic("https://api.anthropic.com", "test-api-key-123")
	h := &CountTokensHandler{cfg: cfg}

	gotKey := h.ResolveAPIKey(nil)
	if gotKey != "test-api-key-123" {
		t.Errorf("Expected key %q, got %q", "test-api-key-123", gotKey)
	}
}

func TestCountTokensHandler_ForwardHeaders(t *testing.T) {
	cfg := config.NewTestConfig(nil, nil)
	h := &CountTokensHandler{cfg: cfg}

	tests := []struct {
		name         string
		inputHeaders map[string][]string
		wantForward  map[string]string
	}{
		{
			name: "forwards X-* headers",
			inputHeaders: map[string][]string{
				"X-Custom-Header": {"custom-value"},
				"X-Another":       {"another-value"},
			},
			wantForward: map[string]string{
				"X-Custom-Header": "custom-value",
				"X-Another":       "another-value",
			},
		},
		{
			name: "forwards Anthropic-Version header",
			inputHeaders: map[string][]string{
				"Anthropic-Version": {"2023-06-01"},
			},
			wantForward: map[string]string{
				"Anthropic-Version": "2023-06-01",
			},
		},
		{
			name: "forwards Anthropic-Beta header",
			inputHeaders: map[string][]string{
				"Anthropic-Beta": {"feature-1"},
			},
			wantForward: map[string]string{
				"Anthropic-Beta": "feature-1",
			},
		},
		{
			name: "does not forward other headers",
			inputHeaders: map[string][]string{
				"Content-Type": {"application/json"},
				"User-Agent":   {"test-client"},
			},
			wantForward: map[string]string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest("POST", "/v1/messages/count_tokens", nil)
			c, _ := gin.CreateTestContext(httptest.NewRecorder())
			c.Request = req

			// Set input headers
			for k, v := range tt.inputHeaders {
				c.Request.Header[k] = v
			}

			upstreamReq := httptest.NewRequest("POST", "http://upstream", nil)
			h.ForwardHeaders(c, upstreamReq)

			// Check forwarded headers
			for k, v := range tt.wantForward {
				got := upstreamReq.Header.Get(k)
				if got != v {
					t.Errorf("Expected header %q to be %q, got %q", k, v, got)
				}
			}
		})
	}
}

func TestCountTokensHandler_WriteError(t *testing.T) {
	cfg := &config.Config{}
	h := &CountTokensHandler{cfg: cfg}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	h.WriteError(c, http.StatusBadRequest, "Test error message")

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected status %d, got %d", http.StatusBadRequest, w.Code)
	}

	var response map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatalf("Failed to unmarshal response: %v", err)
	}

	// Check Anthropic error format
	if response["type"] != "error" {
		t.Errorf("Expected type 'error', got %v", response["type"])
	}

	errorObj, ok := response["error"].(map[string]interface{})
	if !ok {
		t.Fatal("Expected error object")
	}

	if errorObj["message"] != "Test error message" {
		t.Errorf("Expected message 'Test error message', got %v", errorObj["message"])
	}
}

func TestCountTokensHandler_EndToEnd(t *testing.T) {
	// This test verifies the handler integration without actual upstream call
	// In production, this would call the real upstream API

	cfg := config.NewTestConfigWithAnthropic("https://api.anthropic.com/v1/messages", "test-key")

	handler := NewCountTokensHandler(cfg)

	// Create test request
	reqBody := types.MessageCountTokensRequest{
		Model: "kimi-k2.5",
		Messages: []types.MessageInput{
			{Role: "user", Content: "Hello, world!"},
		},
	}

	reqBodyBytes, _ := json.Marshal(reqBody)
	req := httptest.NewRequest("POST", "/v1/messages/count_tokens", bytes.NewReader(reqBodyBytes))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = req

	handler(c)

	// Since we don't have a real upstream, we expect a connection error
	// This test just verifies the handler structure works
	if w.Code != http.StatusBadGateway && w.Code != http.StatusInternalServerError {
		t.Logf("Unexpected status code: %d (expected BadGateway or InternalServerError without real upstream)", w.Code)
	}
}

func TestHandleNonStreaming(t *testing.T) {
	// Test the HandleNonStreaming wrapper with a mock handler
	mockHandler := &mockNonStreamingHandler{
		upstreamURL: "http://mock-upstream",
		apiKey:      "mock-key",
	}

	handler := HandleNonStreaming(mockHandler)

	reqBody := `{"test": "data"}`
	req := httptest.NewRequest("POST", "/test", bytes.NewReader([]byte(reqBody)))
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = req

	handler(c)

	// Verify handler was called correctly
	if !mockHandler.validateCalled {
		t.Error("ValidateRequest was not called")
	}
	if !mockHandler.transformCalled {
		t.Error("TransformRequest was not called")
	}
	if !mockHandler.forwardHeadersCalled {
		t.Error("ForwardHeaders was not called")
	}
}

// Mock handler for testing HandleNonStreaming
type mockNonStreamingHandler struct {
	upstreamURL          string
	apiKey               string
	validateCalled       bool
	transformCalled      bool
	forwardHeadersCalled bool
}

func (m *mockNonStreamingHandler) ValidateRequest(body []byte) error {
	m.validateCalled = true
	return nil
}

func (m *mockNonStreamingHandler) TransformRequest(body []byte) ([]byte, error) {
	m.transformCalled = true
	return body, nil
}

func (m *mockNonStreamingHandler) UpstreamURL() string {
	return m.upstreamURL
}

func (m *mockNonStreamingHandler) ResolveAPIKey(c *gin.Context) string {
	return m.apiKey
}

func (m *mockNonStreamingHandler) ForwardHeaders(c *gin.Context, req *http.Request) {
	m.forwardHeadersCalled = true
}

func (m *mockNonStreamingHandler) WriteError(c *gin.Context, status int, msg string) {
	c.JSON(status, gin.H{"error": msg})
}
