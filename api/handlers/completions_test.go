package handlers

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"

	"ai-proxy/config"
	"ai-proxy/router"

	"github.com/gin-gonic/gin"
)

func TestCompletionsHandler_ValidateRequest(t *testing.T) {
	h := &CompletionsHandler{}

	tests := []struct {
		name    string
		body    []byte
		wantErr bool
	}{
		{
			name:    "empty body",
			body:    []byte{},
			wantErr: false,
		},
		{
			name:    "valid json",
			body:    []byte(`{"model": "test", "stream": true}`),
			wantErr: false,
		},
		{
			name:    "invalid json",
			body:    []byte(`{invalid}`),
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := h.ValidateRequest(tt.body)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateRequest() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestCompletionsHandler_TransformRequest(t *testing.T) {
	h := &CompletionsHandler{}

	tests := []struct {
		name    string
		body    []byte
		want    []byte
		wantErr bool
	}{
		{
			name:    "empty body",
			body:    []byte{},
			want:    []byte{},
			wantErr: false,
		},
		{
			name:    "valid request",
			body:    []byte(`{"model": "test", "stream": true}`),
			want:    []byte(`{"model": "test", "stream": true}`),
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := h.TransformRequest(tt.body)
			if (err != nil) != tt.wantErr {
				t.Errorf("TransformRequest() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !bytes.Equal(got, tt.want) {
				t.Errorf("TransformRequest() = %s, want %s", got, tt.want)
			}
		})
	}
}

func TestCompletionsHandler_UpstreamURLWithRoute(t *testing.T) {
	h := &CompletionsHandler{}

	tests := []struct {
		name     string
		route    *router.ResolvedRoute
		expected string
	}{
		{
			name:     "openai provider",
			route:    &router.ResolvedRoute{Provider: config.Provider{Type: "openai", BaseURL: "https://api.example.com/v1"}, Model: "gpt-4"},
			expected: "https://api.example.com/v1/chat/completions",
		},
		{
			name:     "openai provider with chat/completions suffix",
			route:    &router.ResolvedRoute{Provider: config.Provider{Type: "openai", BaseURL: "https://api.example.com/v1/chat/completions"}, Model: "gpt-4"},
			expected: "https://api.example.com/v1/chat/completions",
		},
		{
			name:     "anthropic provider",
			route:    &router.ResolvedRoute{Provider: config.Provider{Type: "anthropic", BaseURL: "https://api.anthropic.com/v1/messages"}, Model: "claude-3"},
			expected: "https://api.anthropic.com/v1/messages",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := h.UpstreamURLWithRoute(tt.route); got != tt.expected {
				t.Errorf("UpstreamURLWithRoute() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestCompletionsHandler_ResolveAPIKeyWithRoute(t *testing.T) {
	h := &CompletionsHandler{}

	route := &router.ResolvedRoute{
		Provider: config.Provider{Type: "openai", BaseURL: "https://api.example.com/v1", APIKey: "test-api-key"},
		Model:    "gpt-4",
	}

	got := h.ResolveAPIKeyWithRoute(route)
	if got != "test-api-key" {
		t.Errorf("ResolveAPIKeyWithRoute() = %v, want %v", got, "test-api-key")
	}
}

func TestCompletionsHandler_ForwardHeadersWithRoute(t *testing.T) {
	h := &CompletionsHandler{}

	tests := []struct {
		name            string
		route           *router.ResolvedRoute
		requestHeaders  map[string]string
		expectedHeaders map[string]string
	}{
		{
			name:            "openai - no custom headers",
			route:           &router.ResolvedRoute{Provider: config.Provider{Type: "openai", BaseURL: "https://api.example.com/v1"}, Model: "gpt-4"},
			requestHeaders:  map[string]string{},
			expectedHeaders: map[string]string{},
		},
		{
			name:            "openai - X-Custom header forwarded",
			route:           &router.ResolvedRoute{Provider: config.Provider{Type: "openai", BaseURL: "https://api.example.com/v1"}, Model: "gpt-4"},
			requestHeaders:  map[string]string{"X-Custom": "value1"},
			expectedHeaders: map[string]string{"X-Custom": "value1"},
		},
		{
			name:            "openai - multiple X- headers forwarded",
			route:           &router.ResolvedRoute{Provider: config.Provider{Type: "openai", BaseURL: "https://api.example.com/v1"}, Model: "gpt-4"},
			requestHeaders:  map[string]string{"X-Request-Id": "123", "X-Trace-Id": "abc"},
			expectedHeaders: map[string]string{"X-Request-Id": "123", "X-Trace-Id": "abc"},
		},
		{
			name:            "openai - non-X header not forwarded",
			route:           &router.ResolvedRoute{Provider: config.Provider{Type: "openai", BaseURL: "https://api.example.com/v1"}, Model: "gpt-4"},
			requestHeaders:  map[string]string{"Authorization": "Bearer token"},
			expectedHeaders: map[string]string{},
		},
		{
			name:            "anthropic - X- and Anthropic headers forwarded",
			route:           &router.ResolvedRoute{Provider: config.Provider{Type: "anthropic", BaseURL: "https://api.anthropic.com/v1/messages"}, Model: "claude-3"},
			requestHeaders:  map[string]string{"X-Custom": "value1", "Anthropic-Version": "2023-06-01"},
			expectedHeaders: map[string]string{"X-Custom": "value1", "Anthropic-Version": "2023-06-01"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			c.Request = httptest.NewRequest(http.MethodPost, "/", nil)
			for k, v := range tt.requestHeaders {
				c.Request.Header.Set(k, v)
			}

			upstreamReq := httptest.NewRequest(http.MethodPost, "https://upstream.example.com", nil)
			h.ForwardHeadersWithRoute(c, upstreamReq, tt.route)

			for k, v := range tt.expectedHeaders {
				if upstreamReq.Header.Get(k) != v {
					t.Errorf("expected header %s = %s, got %s", k, v, upstreamReq.Header.Get(k))
				}
			}
		})
	}
}

func TestCompletionsHandler_WriteError(t *testing.T) {
	h := &CompletionsHandler{}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	h.WriteError(c, http.StatusBadRequest, "test error")

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected status %d, got %d", http.StatusBadRequest, w.Code)
	}
}

func TestCompletionsHandler_CreateTransformerWithRoute(t *testing.T) {
	h := &CompletionsHandler{}

	w := httptest.NewRecorder()
	route := &router.ResolvedRoute{
		Provider: config.Provider{Type: "openai", BaseURL: "https://api.example.com/v1"},
		Model:    "gpt-4",
	}

	transformer := h.CreateTransformerWithRoute(w, route)

	if transformer == nil {
		t.Error("expected non-nil transformer")
	}
}

func TestForwardCustomHeaders(t *testing.T) {
	tests := []struct {
		name            string
		requestHeaders  map[string]string
		prefixes        []string
		expectedHeaders map[string]string
	}{
		{
			name:            "empty headers",
			requestHeaders:  map[string]string{},
			prefixes:        []string{"X-"},
			expectedHeaders: map[string]string{},
		},
		{
			name:            "single X- header",
			requestHeaders:  map[string]string{"X-Custom": "value"},
			prefixes:        []string{"X-"},
			expectedHeaders: map[string]string{"X-Custom": "value"},
		},
		{
			name:            "multiple prefixes",
			requestHeaders:  map[string]string{"X-Custom": "x-value", "Y-Custom": "y-value", "Other": "other-value"},
			prefixes:        []string{"X-", "Y-"},
			expectedHeaders: map[string]string{"X-Custom": "x-value", "Y-Custom": "y-value"},
		},
		{
			name:            "no matching prefix",
			requestHeaders:  map[string]string{"Authorization": "Bearer token", "Content-Type": "application/json"},
			prefixes:        []string{"X-"},
			expectedHeaders: map[string]string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			c.Request = httptest.NewRequest(http.MethodPost, "/", nil)
			for k, v := range tt.requestHeaders {
				c.Request.Header.Set(k, v)
			}

			upstreamReq := httptest.NewRequest(http.MethodPost, "https://upstream.example.com", nil)
			forwardCustomHeaders(c, upstreamReq, tt.prefixes...)

			for k, v := range tt.expectedHeaders {
				if upstreamReq.Header.Get(k) != v {
					t.Errorf("expected header %s = %s, got %s", k, v, upstreamReq.Header.Get(k))
				}
			}
		})
	}
}
