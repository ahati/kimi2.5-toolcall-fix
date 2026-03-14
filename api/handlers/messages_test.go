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

func TestMessagesHandler_ValidateRequest(t *testing.T) {
	h := &MessagesHandler{}

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
			body:    []byte(`{"model": "claude-3", "stream": true}`),
			wantErr: false,
		},
		{
			name:    "invalid json passes validation",
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

func TestMessagesHandler_TransformRequest(t *testing.T) {
	h := &MessagesHandler{}

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
			body:    []byte(`{"model": "claude-3", "stream": true}`),
			want:    []byte(`{"model": "claude-3", "stream": true}`),
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

func TestMessagesHandler_UpstreamURLWithRoute(t *testing.T) {
	h := &MessagesHandler{}

	tests := []struct {
		name     string
		route    *router.ResolvedRoute
		expected string
	}{
		{
			name:     "anthropic provider",
			route:    &router.ResolvedRoute{Provider: config.Provider{Type: "anthropic", BaseURL: "https://api.anthropic.com/v1/messages"}, Model: "claude-3"},
			expected: "https://api.anthropic.com/v1/messages",
		},
		{
			name:     "openai provider",
			route:    &router.ResolvedRoute{Provider: config.Provider{Type: "openai", BaseURL: "https://api.example.com/v1"}, Model: "gpt-4"},
			expected: "https://api.example.com/v1/chat/completions",
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

func TestMessagesHandler_ResolveAPIKeyWithRoute(t *testing.T) {
	h := &MessagesHandler{}

	route := &router.ResolvedRoute{
		Provider: config.Provider{Type: "anthropic", BaseURL: "https://api.anthropic.com/v1/messages", APIKey: "anthropic-api-key"},
		Model:    "claude-3",
	}

	got := h.ResolveAPIKeyWithRoute(route)
	if got != "anthropic-api-key" {
		t.Errorf("ResolveAPIKeyWithRoute() = %v, want %v", got, "anthropic-api-key")
	}
}

func TestMessagesHandler_ForwardHeadersWithRoute(t *testing.T) {
	h := &MessagesHandler{}

	tests := []struct {
		name            string
		route           *router.ResolvedRoute
		requestHeaders  map[string]string
		expectedHeaders map[string]string
	}{
		{
			name:            "anthropic - no custom headers",
			route:           &router.ResolvedRoute{Provider: config.Provider{Type: "anthropic", BaseURL: "https://api.anthropic.com/v1/messages"}, Model: "claude-3"},
			requestHeaders:  map[string]string{},
			expectedHeaders: map[string]string{},
		},
		{
			name:            "anthropic - X- header forwarded",
			route:           &router.ResolvedRoute{Provider: config.Provider{Type: "anthropic", BaseURL: "https://api.anthropic.com/v1/messages"}, Model: "claude-3"},
			requestHeaders:  map[string]string{"X-Custom": "value1"},
			expectedHeaders: map[string]string{"X-Custom": "value1"},
		},
		{
			name:            "anthropic - Anthropic-Version header forwarded",
			route:           &router.ResolvedRoute{Provider: config.Provider{Type: "anthropic", BaseURL: "https://api.anthropic.com/v1/messages"}, Model: "claude-3"},
			requestHeaders:  map[string]string{"Anthropic-Version": "2023-06-01"},
			expectedHeaders: map[string]string{"Anthropic-Version": "2023-06-01"},
		},
		{
			name:            "anthropic - Anthropic-Beta header forwarded",
			route:           &router.ResolvedRoute{Provider: config.Provider{Type: "anthropic", BaseURL: "https://api.anthropic.com/v1/messages"}, Model: "claude-3"},
			requestHeaders:  map[string]string{"Anthropic-Beta": "some-beta"},
			expectedHeaders: map[string]string{"Anthropic-Beta": "some-beta"},
		},
		{
			name:            "anthropic - non-forwarded headers ignored",
			route:           &router.ResolvedRoute{Provider: config.Provider{Type: "anthropic", BaseURL: "https://api.anthropic.com/v1/messages"}, Model: "claude-3"},
			requestHeaders:  map[string]string{"Authorization": "Bearer token", "Content-Type": "application/json"},
			expectedHeaders: map[string]string{},
		},
		{
			name:            "anthropic - mixed headers",
			route:           &router.ResolvedRoute{Provider: config.Provider{Type: "anthropic", BaseURL: "https://api.anthropic.com/v1/messages"}, Model: "claude-3"},
			requestHeaders:  map[string]string{"X-Request-Id": "123", "Anthropic-Version": "2023-06-01", "Authorization": "Bearer token"},
			expectedHeaders: map[string]string{"X-Request-Id": "123", "Anthropic-Version": "2023-06-01"},
		},
		{
			name:            "openai - X- header forwarded",
			route:           &router.ResolvedRoute{Provider: config.Provider{Type: "openai", BaseURL: "https://api.example.com/v1"}, Model: "gpt-4"},
			requestHeaders:  map[string]string{"X-Custom": "value1"},
			expectedHeaders: map[string]string{"X-Custom": "value1"},
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

func TestMessagesHandler_WriteError(t *testing.T) {
	h := &MessagesHandler{}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	h.WriteError(c, http.StatusBadRequest, "test error")

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected status %d, got %d", http.StatusBadRequest, w.Code)
	}
}

func TestMessagesHandler_CreateTransformerWithRoute(t *testing.T) {
	h := &MessagesHandler{}

	w := httptest.NewRecorder()
	route := &router.ResolvedRoute{
		Provider: config.Provider{Type: "anthropic", BaseURL: "https://api.anthropic.com/v1/messages"},
		Model:    "claude-3",
	}

	transformer := h.CreateTransformerWithRoute(w, route)

	if transformer == nil {
		t.Error("expected non-nil transformer")
	}
}
