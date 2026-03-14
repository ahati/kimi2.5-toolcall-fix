package handlers

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"

	"ai-proxy/config"

	"github.com/gin-gonic/gin"
)

func TestCompletionsHandler_ValidateRequest(t *testing.T) {
	cfg := &config.Config{}
	h := &CompletionsHandler{cfg: cfg}

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
	cfg := &config.Config{}
	h := &CompletionsHandler{cfg: cfg}

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

func TestCompletionsHandler_UpstreamURL(t *testing.T) {
	cfg := config.LoadConfig(&config.SchemaConfig{
		Providers: []config.Provider{
			{
				Name:    "test-openai",
				Type:    "openai",
				BaseURL: "https://api.example.com/v1/chat/completions",
				APIKey:  "test-key",
			},
		},
	})
	h := &CompletionsHandler{cfg: cfg}

	want := "https://api.example.com/v1/chat/completions"
	if got := h.UpstreamURL(); got != want {
		t.Errorf("UpstreamURL() = %v, want %v", got, want)
	}
}

func TestCompletionsHandler_ResolveAPIKey(t *testing.T) {
	cfg := config.LoadConfig(&config.SchemaConfig{
		Providers: []config.Provider{
			{
				Name:    "test-openai",
				Type:    "openai",
				BaseURL: "https://api.example.com/v1/chat/completions",
				APIKey:  "test-api-key",
			},
		},
	})
	h := &CompletionsHandler{cfg: cfg}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/", nil)

	got := h.ResolveAPIKey(c)
	if got != "test-api-key" {
		t.Errorf("ResolveAPIKey() = %v, want %v", got, "test-api-key")
	}
}

func TestCompletionsHandler_ForwardHeaders(t *testing.T) {
	cfg := &config.Config{}
	h := &CompletionsHandler{cfg: cfg}

	tests := []struct {
		name            string
		requestHeaders  map[string]string
		expectedHeaders map[string]string
	}{
		{
			name:            "no custom headers",
			requestHeaders:  map[string]string{},
			expectedHeaders: map[string]string{},
		},
		{
			name: "X-Custom header forwarded",
			requestHeaders: map[string]string{
				"X-Custom": "value1",
			},
			expectedHeaders: map[string]string{
				"X-Custom": "value1",
			},
		},
		{
			name: "multiple X- headers forwarded",
			requestHeaders: map[string]string{
				"X-Request-Id": "123",
				"X-Trace-Id":   "abc",
			},
			expectedHeaders: map[string]string{
				"X-Request-Id": "123",
				"X-Trace-Id":   "abc",
			},
		},
		{
			name: "non-X header not forwarded",
			requestHeaders: map[string]string{
				"Authorization": "Bearer token",
			},
			expectedHeaders: map[string]string{},
		},
		{
			name: "Extra header forwarded",
			requestHeaders: map[string]string{
				"Extra": "extra-value",
			},
			expectedHeaders: map[string]string{
				"Extra": "extra-value",
			},
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
			h.ForwardHeaders(c, upstreamReq)

			for k, v := range tt.expectedHeaders {
				if upstreamReq.Header.Get(k) != v {
					t.Errorf("expected header %s = %s, got %s", k, v, upstreamReq.Header.Get(k))
				}
			}
		})
	}
}

func TestCompletionsHandler_WriteError(t *testing.T) {
	cfg := &config.Config{}
	h := &CompletionsHandler{cfg: cfg}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	h.WriteError(c, http.StatusBadRequest, "test error")

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected status %d, got %d", http.StatusBadRequest, w.Code)
	}
}

func TestCompletionsHandler_CreateTransformer(t *testing.T) {
	cfg := &config.Config{}
	h := &CompletionsHandler{cfg: cfg}

	w := httptest.NewRecorder()
	transformer := h.CreateTransformer(w)

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
			name: "single X- header",
			requestHeaders: map[string]string{
				"X-Custom": "value",
			},
			prefixes: []string{"X-"},
			expectedHeaders: map[string]string{
				"X-Custom": "value",
			},
		},
		{
			name: "multiple prefixes",
			requestHeaders: map[string]string{
				"X-Custom": "x-value",
				"Y-Custom": "y-value",
				"Other":    "other-value",
			},
			prefixes: []string{"X-", "Y-"},
			expectedHeaders: map[string]string{
				"X-Custom": "x-value",
				"Y-Custom": "y-value",
			},
		},
		{
			name: "no matching prefix",
			requestHeaders: map[string]string{
				"Authorization": "Bearer token",
				"Content-Type":  "application/json",
			},
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
