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

// mockRouter implements router.Router for testing.
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

func (m *mockRouter) ResolveWithProtocol(modelName, incomingProtocol string) (*router.ResolvedRoute, error) {
	// For mock, just return the base route - protocol handling is tested in router package
	return m.Resolve(modelName)
}

// TestResponsesHandler_ValidateRequest tests request validation.
func TestResponsesHandler_ValidateRequest(t *testing.T) {
	mockR := newMockRouter()
	mockR.models["gpt-4o"] = &router.ResolvedRoute{
		Provider: config.Provider{
			Name:      "openai",
			Endpoints: map[string]string{"openai": "https://api.openai.com/v1"},
		},
		Model:          "gpt-4o",
		OutputProtocol: "openai",
	}

	tests := []struct {
		name      string
		body      string
		wantError bool
	}{
		{
			name:      "valid request with known model",
			body:      `{"model":"gpt-4o","input":"Hello","stream":true}`,
			wantError: false,
		},
		{
			name:      "invalid JSON",
			body:      `not valid json`,
			wantError: true,
		},
		{
			name:      "empty model",
			body:      `{"model":"","input":"Hello"}`,
			wantError: true,
		},
		{
			name:      "missing model field",
			body:      `{"input":"Hello"}`,
			wantError: true,
		},
		{
			name:      "unknown model",
			body:      `{"model":"unknown-model","input":"Hello"}`,
			wantError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			handler := &ResponsesHandler{
				cfg:    &config.Config{},
				router: mockR,
			}

			err := handler.ValidateRequest([]byte(tt.body))

			if tt.wantError && err == nil {
				t.Error("Expected error but got nil")
			}
			if !tt.wantError && err != nil {
				t.Errorf("Expected no error but got: %v", err)
			}
		})
	}
}

// TestResponsesHandler_TransformRequest_OpenAI tests transformation for OpenAI provider.
func TestResponsesHandler_TransformRequest_OpenAI(t *testing.T) {
	mockR := newMockRouter()
	mockR.models["gpt-4o"] = &router.ResolvedRoute{
		Provider: config.Provider{
			Name:      "openai",
			Endpoints: map[string]string{"openai": "https://api.openai.com/v1"},
		},
		Model:          "gpt-4o",
		OutputProtocol: "openai",
	}

	handler := &ResponsesHandler{
		cfg:    &config.Config{},
		router: mockR,
	}

	// First validate to set the route
	err := handler.ValidateRequest([]byte(`{"model":"gpt-4o","input":"Hello","stream":true}`))
	if err != nil {
		t.Fatalf("ValidateRequest failed: %v", err)
	}

	body := []byte(`{"model":"gpt-4o","input":"Hello","stream":true}`)
	transformed, err := handler.TransformRequest(body)

	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	// Verify it's valid JSON
	var chatReq types.ChatCompletionRequest
	if err := json.Unmarshal(transformed, &chatReq); err != nil {
		t.Fatalf("Failed to unmarshal as ChatCompletionRequest: %v", err)
	}

	// Verify model was updated
	if chatReq.Model != "gpt-4o" {
		t.Errorf("Model = %s, want gpt-4o", chatReq.Model)
	}

	// Verify messages were converted from input
	if len(chatReq.Messages) != 1 {
		t.Errorf("Expected 1 message, got %d", len(chatReq.Messages))
	}
}

// TestResponsesHandler_TransformRequest_Anthropic tests transformation for Anthropic provider.
func TestResponsesHandler_TransformRequest_Anthropic(t *testing.T) {
	mockR := newMockRouter()
	mockR.models["claude-3-opus"] = &router.ResolvedRoute{
		Provider: config.Provider{
			Name:      "anthropic",
			Endpoints: map[string]string{"anthropic": "https://api.anthropic.com/v1/messages"},
		},
		Model:          "claude-3-opus-20240229",
		OutputProtocol: "anthropic",
	}

	handler := &ResponsesHandler{
		cfg:    &config.Config{},
		router: mockR,
	}

	// First validate to set the route
	err := handler.ValidateRequest([]byte(`{"model":"claude-3-opus","input":"Hello","stream":true}`))
	if err != nil {
		t.Fatalf("ValidateRequest failed: %v", err)
	}

	body := []byte(`{"model":"claude-3-opus","input":"Hello","instructions":"Be helpful","stream":true}`)
	transformed, err := handler.TransformRequest(body)

	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	// Verify it's valid JSON
	var anthReq types.MessageRequest
	if err := json.Unmarshal(transformed, &anthReq); err != nil {
		t.Fatalf("Failed to unmarshal as MessageRequest: %v", err)
	}

	// Verify model was updated to the resolved model
	if anthReq.Model != "claude-3-opus-20240229" {
		t.Errorf("Model = %s, want claude-3-opus-20240229", anthReq.Model)
	}

	// Verify system was set from instructions
	if anthReq.System != "Be helpful" {
		t.Errorf("System = %v, want 'Be helpful'", anthReq.System)
	}
}

// TestResponsesHandler_UpstreamURL tests upstream URL generation.
// Note: OpenAI provider uses /v1/chat/completions because Responses API requests
// are converted to Chat Completions format via ResponsesToChatConverter.
// The endpoint URL is directly provided in the Endpoints map.
func TestResponsesHandler_UpstreamURL(t *testing.T) {
	tests := []struct {
		name         string
		providerType string
		endpointURL  string
		wantURL      string
	}{
		{
			name:         "OpenAI provider returns openai endpoint",
			providerType: "openai",
			endpointURL:  "https://api.openai.com/v1/chat/completions",
			wantURL:      "https://api.openai.com/v1/chat/completions",
		},
		{
			name:         "Anthropic provider returns anthropic endpoint",
			providerType: "anthropic",
			endpointURL:  "https://api.anthropic.com/v1/messages",
			wantURL:      "https://api.anthropic.com/v1/messages",
		},
		{
			name:         "Custom endpoint URL",
			providerType: "openai",
			endpointURL:  "https://custom.api.com/chat",
			wantURL:      "https://custom.api.com/chat",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			handler := &ResponsesHandler{
				cfg: &config.Config{},
				route: &router.ResolvedRoute{
					Provider: config.Provider{
						Endpoints: map[string]string{tt.providerType: tt.endpointURL},
					},
					OutputProtocol: tt.providerType,
				},
			}

			got := handler.UpstreamURL()
			if got != tt.wantURL {
				t.Errorf("UpstreamURL() = %s, want %s", got, tt.wantURL)
			}
		})
	}
}

// TestResponsesHandler_UpstreamURL_NilRoute tests upstream URL with nil route.
func TestResponsesHandler_UpstreamURL_NilRoute(t *testing.T) {
	handler := &ResponsesHandler{
		cfg:    &config.Config{},
		router: newMockRouter(),
	}

	got := handler.UpstreamURL()
	if got != "" {
		t.Errorf("UpstreamURL() = %s, want empty string", got)
	}
}

// TestResponsesHandler_ResolveAPIKey tests API key resolution.
func TestResponsesHandler_ResolveAPIKey(t *testing.T) {
	handler := &ResponsesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider: config.Provider{
				Name:   "test-provider",
				APIKey: "test-api-key",
			},
		},
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	got := handler.ResolveAPIKey(c)
	if got != "test-api-key" {
		t.Errorf("ResolveAPIKey() = %s, want test-api-key", got)
	}
}

// TestResponsesHandler_ResolveAPIKey_NilRoute tests API key with nil route.
func TestResponsesHandler_ResolveAPIKey_NilRoute(t *testing.T) {
	handler := &ResponsesHandler{
		cfg:    &config.Config{},
		router: newMockRouter(),
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	got := handler.ResolveAPIKey(c)
	if got != "" {
		t.Errorf("ResolveAPIKey() = %s, want empty string", got)
	}
}

// TestResponsesHandler_ForwardHeaders_OpenAI tests header forwarding for OpenAI.
func TestResponsesHandler_ForwardHeaders_OpenAI(t *testing.T) {
	handler := &ResponsesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider: config.Provider{
				Endpoints: map[string]string{"openai": "https://api.example.com"},
			},
			OutputProtocol: "openai",
		},
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/", nil)
	c.Request.Header.Set("X-Custom", "custom-value")
	c.Request.Header.Set("Extra", "extra-value")
	c.Request.Header.Set("Authorization", "Bearer token")

	upstreamReq := httptest.NewRequest(http.MethodPost, "https://upstream.example.com", nil)
	handler.ForwardHeaders(c, upstreamReq)

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

// TestResponsesHandler_ForwardHeaders_Anthropic tests header forwarding for Anthropic.
func TestResponsesHandler_ForwardHeaders_Anthropic(t *testing.T) {
	handler := &ResponsesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider: config.Provider{
				Endpoints: map[string]string{"anthropic": "https://api.anthropic.com"},
			},
			OutputProtocol: "anthropic",
		},
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/", nil)
	c.Request.Header.Set("X-Custom", "custom-value")
	c.Request.Header.Set("Anthropic-Version", "2023-06-01")
	c.Request.Header.Set("Anthropic-Beta", "tools")
	c.Request.Header.Set("Authorization", "Bearer token")

	upstreamReq := httptest.NewRequest(http.MethodPost, "https://upstream.example.com", nil)
	handler.ForwardHeaders(c, upstreamReq)

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

// TestResponsesHandler_CreateTransformer_OpenAI tests transformer creation for OpenAI.
func TestResponsesHandler_CreateTransformer_OpenAI(t *testing.T) {
	handler := &ResponsesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider: config.Provider{
				Endpoints: map[string]string{"openai": "https://api.example.com"},
			},
		},
	}

	var buf bytes.Buffer
	transformer := handler.CreateTransformer(&buf)

	if transformer == nil {
		t.Error("CreateTransformer returned nil")
	}
}

// TestResponsesHandler_CreateTransformer_Anthropic tests transformer creation for Anthropic.
func TestResponsesHandler_CreateTransformer_Anthropic(t *testing.T) {
	handler := &ResponsesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider: config.Provider{
				Endpoints: map[string]string{"anthropic": "https://api.anthropic.com"},
			},
		},
	}

	var buf bytes.Buffer
	transformer := handler.CreateTransformer(&buf)

	if transformer == nil {
		t.Error("CreateTransformer returned nil")
	}
}

// TestResponsesHandler_CreateTransformer_NilRoute tests transformer with nil route.
func TestResponsesHandler_CreateTransformer_NilRoute(t *testing.T) {
	handler := &ResponsesHandler{
		cfg:    &config.Config{},
		router: newMockRouter(),
	}

	var buf bytes.Buffer
	transformer := handler.CreateTransformer(&buf)

	if transformer == nil {
		t.Error("CreateTransformer returned nil")
	}
}

// TestResponsesHandler_WriteError tests error writing.
func TestResponsesHandler_WriteError(t *testing.T) {
	handler := &ResponsesHandler{
		cfg:    &config.Config{},
		router: newMockRouter(),
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	handler.WriteError(c, http.StatusBadRequest, "Test error message")

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

// TestNewResponsesHandler tests handler creation.
func TestNewResponsesHandler(t *testing.T) {
	cfg := &config.Config{}
	mockR := newMockRouter()
	mockR.models["gpt-4o"] = &router.ResolvedRoute{
		Provider: config.Provider{
			Name:      "openai",
			Endpoints: map[string]string{"openai": "https://api.openai.com/v1"},
		},
		Model:          "gpt-4o",
		OutputProtocol: "openai",
	}

	handler := NewResponsesHandler(cfg, mockR)

	if handler == nil {
		t.Fatal("NewResponsesHandler returned nil")
	}
}

// BenchmarkResponsesHandler_TransformRequest_OpenAI benchmarks OpenAI transformation.
func BenchmarkResponsesHandler_TransformRequest_OpenAI(b *testing.B) {
	mockR := newMockRouter()
	mockR.models["gpt-4o"] = &router.ResolvedRoute{
		Provider: config.Provider{
			Name:      "openai",
			Endpoints: map[string]string{"openai": "https://api.openai.com/v1"},
		},
		Model:          "gpt-4o",
		OutputProtocol: "openai",
	}

	handler := &ResponsesHandler{
		cfg:    &config.Config{},
		router: mockR,
	}
	handler.ValidateRequest([]byte(`{"model":"gpt-4o","input":"Hello","stream":true}`))

	body := []byte(`{"model":"gpt-4o","input":"Hello, how are you today? I need help with a complex problem.","stream":true}`)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := handler.TransformRequest(body)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkResponsesHandler_TransformRequest_Anthropic benchmarks Anthropic transformation.
func BenchmarkResponsesHandler_TransformRequest_Anthropic(b *testing.B) {
	mockR := newMockRouter()
	mockR.models["claude-3-opus"] = &router.ResolvedRoute{
		Provider: config.Provider{
			Name:      "anthropic",
			Endpoints: map[string]string{"anthropic": "https://api.anthropic.com/v1/messages"},
		},
		Model: "claude-3-opus-20240229",
	}

	handler := &ResponsesHandler{
		cfg:    &config.Config{},
		router: mockR,
	}
	handler.ValidateRequest([]byte(`{"model":"claude-3-opus","input":"Hello","stream":true}`))

	body := []byte(`{"model":"claude-3-opus","input":"Hello, how are you today?","instructions":"Be helpful","stream":true}`)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := handler.TransformRequest(body)
		if err != nil {
			b.Fatal(err)
		}
	}
}
