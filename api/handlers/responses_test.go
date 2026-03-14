package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"ai-proxy/config"
	"ai-proxy/router"
	"ai-proxy/types"

	"github.com/gin-gonic/gin"
)

type mockRouter struct {
	models    map[string]*router.ResolvedRoute
	providers map[string]config.Provider
}

func newMockRouter() *mockRouter {
	return &mockRouter{
		models:    make(map[string]*router.ResolvedRoute),
		providers: make(map[string]config.Provider),
	}
}

func (m *mockRouter) Resolve(modelName string) (*router.ResolvedRoute, error) {
	if route, ok := m.models[modelName]; ok {
		return route, nil
	}
	return nil, fmt.Errorf("unknown model: '%s'", modelName)
}

func (m *mockRouter) GetProvider(name string) (config.Provider, bool) {
	p, ok := m.providers[name]
	return p, ok
}

func (m *mockRouter) ListModels() []string {
	models := make([]string, 0, len(m.models))
	for name := range m.models {
		models = append(models, name)
	}
	return models
}

func (m *mockRouter) ListProviders() []string {
	providers := make([]string, 0, len(m.providers))
	for name := range m.providers {
		providers = append(providers, name)
	}
	return providers
}

func TestResponsesHandler_ValidateRequest_Valid(t *testing.T) {
	h := &ResponsesHandler{}

	body := []byte(`{"model":"gpt-4o","input":"Hello","stream":true}`)
	err := h.ValidateRequest(body)

	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}
}

func TestResponsesHandler_ValidateRequest_MissingModel(t *testing.T) {
	h := &ResponsesHandler{}

	body := []byte(`{"input":"Hello","stream":true}`)
	err := h.ValidateRequest(body)

	if err == nil {
		t.Error("Expected error for missing model")
	}
}

func TestResponsesHandler_ValidateRequest_InvalidJSON(t *testing.T) {
	h := &ResponsesHandler{}

	body := []byte(`not valid json`)
	err := h.ValidateRequest(body)

	if err == nil {
		t.Error("Expected error for invalid JSON")
	}
}

func TestResponsesHandler_ExtractModel(t *testing.T) {
	h := &ResponsesHandler{}

	body := []byte(`{"model":"gpt-4o","input":"Hello"}`)
	model, err := h.ExtractModel(body)

	if err != nil {
		t.Fatalf("ExtractModel failed: %v", err)
	}

	if model != "gpt-4o" {
		t.Errorf("model = %s, want gpt-4o", model)
	}
}

func TestResponsesHandler_TransformRequestWithRoute_OpenAI(t *testing.T) {
	h := &ResponsesHandler{}

	route := &router.ResolvedRoute{
		Provider: config.Provider{
			Name:    "openai",
			Type:    "openai",
			BaseURL: "https://api.openai.com/v1",
		},
		Model: "gpt-4o",
	}

	body := []byte(`{"model":"gpt-4o","input":"Hello","stream":true}`)
	transformed, err := h.TransformRequestWithRoute(body, route)

	if err != nil {
		t.Fatalf("TransformRequestWithRoute failed: %v", err)
	}

	var chatReq types.ChatCompletionRequest
	if err := json.Unmarshal(transformed, &chatReq); err != nil {
		t.Fatalf("Failed to unmarshal as ChatCompletionRequest: %v", err)
	}

	if chatReq.Model != "gpt-4o" {
		t.Errorf("Model = %s, want gpt-4o", chatReq.Model)
	}
	if len(chatReq.Messages) != 1 {
		t.Errorf("Expected 1 message, got %d", len(chatReq.Messages))
	}
}

func TestResponsesHandler_TransformRequestWithRoute_Anthropic(t *testing.T) {
	h := &ResponsesHandler{}

	route := &router.ResolvedRoute{
		Provider: config.Provider{
			Name:    "anthropic",
			Type:    "anthropic",
			BaseURL: "https://api.anthropic.com/v1/messages",
		},
		Model: "claude-3-opus-20240229",
	}

	body := []byte(`{"model":"claude-3-opus","input":"Hello","instructions":"Be helpful","stream":true}`)
	transformed, err := h.TransformRequestWithRoute(body, route)

	if err != nil {
		t.Fatalf("TransformRequestWithRoute failed: %v", err)
	}

	var anthReq types.MessageRequest
	if err := json.Unmarshal(transformed, &anthReq); err != nil {
		t.Fatalf("Failed to unmarshal as MessageRequest: %v", err)
	}

	if anthReq.Model != "claude-3-opus-20240229" {
		t.Errorf("Model = %s, want claude-3-opus-20240229", anthReq.Model)
	}
	if anthReq.System != "Be helpful" {
		t.Errorf("System = %v, want 'Be helpful'", anthReq.System)
	}
}

func TestResponsesHandler_UpstreamURLWithRoute(t *testing.T) {
	tests := []struct {
		name    string
		route   *router.ResolvedRoute
		wantURL string
	}{
		{
			name: "OpenAI provider",
			route: &router.ResolvedRoute{
				Provider: config.Provider{Type: "openai", BaseURL: "https://api.openai.com/v1"},
				Model:    "gpt-4o",
			},
			wantURL: "https://api.openai.com/v1/chat/completions",
		},
		{
			name: "OpenAI provider with trailing slash",
			route: &router.ResolvedRoute{
				Provider: config.Provider{Type: "openai", BaseURL: "https://api.openai.com/v1/"},
				Model:    "gpt-4o",
			},
			wantURL: "https://api.openai.com/v1/chat/completions",
		},
		{
			name: "OpenAI provider with full path",
			route: &router.ResolvedRoute{
				Provider: config.Provider{Type: "openai", BaseURL: "https://api.openai.com/v1/chat/completions"},
				Model:    "gpt-4o",
			},
			wantURL: "https://api.openai.com/v1/chat/completions",
		},
		{
			name: "Anthropic provider",
			route: &router.ResolvedRoute{
				Provider: config.Provider{Type: "anthropic", BaseURL: "https://api.anthropic.com/v1/messages"},
				Model:    "claude-3",
			},
			wantURL: "https://api.anthropic.com/v1/messages",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := &ResponsesHandler{}
			got := h.UpstreamURLWithRoute(tt.route)
			if got != tt.wantURL {
				t.Errorf("UpstreamURLWithRoute() = %s, want %s", got, tt.wantURL)
			}
		})
	}
}

func TestResponsesHandler_ResolveAPIKeyWithRoute(t *testing.T) {
	h := &ResponsesHandler{}

	route := &router.ResolvedRoute{
		Provider: config.Provider{
			Name:   "test-provider",
			APIKey: "test-api-key",
		},
		Model: "gpt-4o",
	}

	got := h.ResolveAPIKeyWithRoute(route)
	if got != "test-api-key" {
		t.Errorf("ResolveAPIKeyWithRoute() = %s, want test-api-key", got)
	}
}

func TestResponsesHandler_ForwardHeadersWithRoute_OpenAI(t *testing.T) {
	h := &ResponsesHandler{}
	route := &router.ResolvedRoute{
		Provider: config.Provider{Type: "openai"},
		Model:    "gpt-4o",
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/", nil)
	c.Request.Header.Set("X-Custom", "custom-value")
	c.Request.Header.Set("Extra", "extra-value")
	c.Request.Header.Set("Authorization", "Bearer token")

	upstreamReq := httptest.NewRequest(http.MethodPost, "https://upstream.example.com", nil)
	h.ForwardHeadersWithRoute(c, upstreamReq, route)

	if upstreamReq.Header.Get("X-Custom") != "custom-value" {
		t.Error("X-Custom header should be forwarded")
	}
	if upstreamReq.Header.Get("Extra") != "extra-value" {
		t.Error("Extra header should be forwarded")
	}
	if upstreamReq.Header.Get("Authorization") != "" {
		t.Error("Authorization header should not be forwarded")
	}
}

func TestResponsesHandler_ForwardHeadersWithRoute_Anthropic(t *testing.T) {
	h := &ResponsesHandler{}
	route := &router.ResolvedRoute{
		Provider: config.Provider{Type: "anthropic"},
		Model:    "claude-3",
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/", nil)
	c.Request.Header.Set("X-Custom", "custom-value")
	c.Request.Header.Set("Anthropic-Version", "2023-06-01")
	c.Request.Header.Set("Anthropic-Beta", "tools")
	c.Request.Header.Set("Authorization", "Bearer token")

	upstreamReq := httptest.NewRequest(http.MethodPost, "https://upstream.example.com", nil)
	h.ForwardHeadersWithRoute(c, upstreamReq, route)

	if upstreamReq.Header.Get("X-Custom") != "custom-value" {
		t.Error("X-Custom header should be forwarded")
	}
	if upstreamReq.Header.Get("Anthropic-Version") != "2023-06-01" {
		t.Error("Anthropic-Version header should be forwarded")
	}
	if upstreamReq.Header.Get("Anthropic-Beta") != "tools" {
		t.Error("Anthropic-Beta header should be forwarded")
	}
	if upstreamReq.Header.Get("Authorization") != "" {
		t.Error("Authorization header should not be forwarded")
	}
}

func TestResponsesHandler_CreateTransformerWithRoute_OpenAI(t *testing.T) {
	h := &ResponsesHandler{}
	route := &router.ResolvedRoute{
		Provider: config.Provider{Type: "openai"},
		Model:    "gpt-4o",
	}

	var buf bytes.Buffer
	transformer := h.CreateTransformerWithRoute(&buf, route)

	if transformer == nil {
		t.Error("CreateTransformerWithRoute returned nil")
	}
}

func TestResponsesHandler_CreateTransformerWithRoute_Anthropic(t *testing.T) {
	h := &ResponsesHandler{}
	route := &router.ResolvedRoute{
		Provider: config.Provider{Type: "anthropic"},
		Model:    "claude-3",
	}

	var buf bytes.Buffer
	transformer := h.CreateTransformerWithRoute(&buf, route)

	if transformer == nil {
		t.Error("CreateTransformerWithRoute returned nil")
	}
}

func TestResponsesHandler_WriteError(t *testing.T) {
	h := &ResponsesHandler{}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	h.WriteError(c, http.StatusBadRequest, "Test error message")

	if w.Code != http.StatusBadRequest {
		t.Errorf("Status code = %d, want %d", w.Code, http.StatusBadRequest)
	}

	body := w.Body.String()
	if !strings.Contains(body, `"type":"error"`) {
		t.Error("Response should contain error type")
	}
	if !strings.Contains(body, `"code":"invalid_request_error"`) {
		t.Error("Response should contain error code")
	}
	if !strings.Contains(body, "Test error message") {
		t.Error("Response should contain error message")
	}
}

func TestNewResponsesHandler(t *testing.T) {
	mockR := newMockRouter()
	mockR.models["gpt-4o"] = &router.ResolvedRoute{
		Provider: config.Provider{
			Name:    "openai",
			Type:    "openai",
			BaseURL: "https://api.openai.com/v1",
			APIKey:  "test-key",
		},
		Model: "gpt-4o",
	}

	handler := NewResponsesHandler(mockR)

	if handler == nil {
		t.Fatal("NewResponsesHandler returned nil")
	}
}
