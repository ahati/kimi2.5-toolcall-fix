package handlers

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"

	"ai-proxy/config"

	"github.com/gin-gonic/gin"
)

func TestMessagesHandler_ValidateRequest(t *testing.T) {
	cfg := &config.Config{}
	h := &MessagesHandler{cfg: cfg}

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
	cfg := &config.Config{}
	h := &MessagesHandler{cfg: cfg}

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

func TestMessagesHandler_UpstreamURL(t *testing.T) {
	cfg := config.LoadConfig(&config.SchemaConfig{
		Providers: []config.Provider{
			{
				Name:    "test-anthropic",
				Type:    "anthropic",
				BaseURL: "https://api.anthropic.com/v1/messages",
				APIKey:  "test-key",
			},
		},
	})
	h := &MessagesHandler{cfg: cfg}

	want := "https://api.anthropic.com/v1/messages"
	if got := h.UpstreamURL(); got != want {
		t.Errorf("UpstreamURL() = %v, want %v", got, want)
	}
}

func TestMessagesHandler_ResolveAPIKey(t *testing.T) {
	cfg := config.LoadConfig(&config.SchemaConfig{
		Providers: []config.Provider{
			{
				Name:    "test-anthropic",
				Type:    "anthropic",
				BaseURL: "https://api.anthropic.com/v1/messages",
				APIKey:  "anthropic-api-key",
			},
		},
	})
	h := &MessagesHandler{cfg: cfg}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/", nil)

	got := h.ResolveAPIKey(c)
	if got != "anthropic-api-key" {
		t.Errorf("ResolveAPIKey() = %v, want %v", got, "anthropic-api-key")
	}
}

func TestMessagesHandler_ForwardHeaders(t *testing.T) {
	cfg := &config.Config{}
	h := &MessagesHandler{cfg: cfg}

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
			name: "X- header forwarded",
			requestHeaders: map[string]string{
				"X-Custom": "value1",
			},
			expectedHeaders: map[string]string{
				"X-Custom": "value1",
			},
		},
		{
			name: "Anthropic-Version header forwarded",
			requestHeaders: map[string]string{
				"Anthropic-Version": "2023-06-01",
			},
			expectedHeaders: map[string]string{
				"Anthropic-Version": "2023-06-01",
			},
		},
		{
			name: "Anthropic-Beta header forwarded",
			requestHeaders: map[string]string{
				"Anthropic-Beta": "some-beta",
			},
			expectedHeaders: map[string]string{
				"Anthropic-Beta": "some-beta",
			},
		},
		{
			name: "non-forwarded headers ignored",
			requestHeaders: map[string]string{
				"Authorization": "Bearer token",
				"Content-Type":  "application/json",
			},
			expectedHeaders: map[string]string{},
		},
		{
			name: "mixed headers",
			requestHeaders: map[string]string{
				"X-Request-Id":      "123",
				"Anthropic-Version": "2023-06-01",
				"Authorization":     "Bearer token",
			},
			expectedHeaders: map[string]string{
				"X-Request-Id":      "123",
				"Anthropic-Version": "2023-06-01",
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

			for k := range tt.requestHeaders {
				if _, expected := tt.expectedHeaders[k]; !expected {
					if upstreamReq.Header.Get(k) != "" {
						t.Errorf("unexpected header %s forwarded", k)
					}
				}
			}
		})
	}
}

func TestMessagesHandler_WriteError(t *testing.T) {
	cfg := &config.Config{}
	h := &MessagesHandler{cfg: cfg}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	h.WriteError(c, http.StatusBadRequest, "test error")

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected status %d, got %d", http.StatusBadRequest, w.Code)
	}
}

func TestMessagesHandler_CreateTransformer(t *testing.T) {
	cfg := &config.Config{}
	h := &MessagesHandler{cfg: cfg}

	w := httptest.NewRecorder()
	transformer := h.CreateTransformer(w)

	if transformer == nil {
		t.Error("expected non-nil transformer")
	}
}
