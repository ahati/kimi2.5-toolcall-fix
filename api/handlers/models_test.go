package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"ai-proxy/config"

	"github.com/gin-gonic/gin"
)

func TestModelsHandler_Handle_MissingAPIKey(t *testing.T) {
	cfg := &config.Config{
		AppConfig: &config.Schema{
			Providers: []config.Provider{
				{
					Name:   "openai",
					Type:   "openai",
					BaseURL: "https://api.example.com/v1/chat/completions",
					APIKey:  "",
				},
			},
		},
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodGet, "/v1/models", nil)

	h := &ModelsHandler{cfg: cfg}
	h.Handle(c)

	if w.Code != http.StatusUnauthorized {
		t.Errorf("expected status %d, got %d", http.StatusUnauthorized, w.Code)
	}

	var response map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatalf("failed to parse response: %v", err)
	}

	errObj, ok := response["error"].(map[string]interface{})
	if !ok {
		t.Fatal("expected error object in response")
	}

	if errObj["message"] != "Missing API key" {
		t.Errorf("expected message %q, got %q", "Missing API key", errObj["message"])
	}
}

func TestModelsHandler_ResolveAPIKey(t *testing.T) {
	cfg := &config.Config{
		AppConfig: &config.Schema{
			Providers: []config.Provider{
				{
					Name:   "openai",
					Type:   "openai",
					APIKey: "default-api-key",
				},
			},
		},
	}
	h := &ModelsHandler{cfg: cfg}

	tests := []struct {
		name           string
		authHeader     string
		expectedAPIKey string
	}{
		{
			name:           "no auth header uses default",
			authHeader:     "",
			expectedAPIKey: "default-api-key",
		},
		{
			name:           "bearer token extracted",
			authHeader:     "Bearer custom-api-key",
			expectedAPIKey: "custom-api-key",
		},
		{
			name:           "non-bearer auth uses default",
			authHeader:     "Basic dXNlcjpwYXNz",
			expectedAPIKey: "default-api-key",
		},
		{
			name:           "bearer with empty token",
			authHeader:     "Bearer ",
			expectedAPIKey: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			c.Request = httptest.NewRequest(http.MethodGet, "/v1/models", nil)
			if tt.authHeader != "" {
				c.Request.Header.Set("Authorization", tt.authHeader)
			}

			result := h.resolveAPIKey(c)
			if result != tt.expectedAPIKey {
				t.Errorf("resolveAPIKey() = %q, want %q", result, tt.expectedAPIKey)
			}
		})
	}
}

func TestModelsHandler_BuildModelsURL(t *testing.T) {
	tests := []struct {
		name        string
		upstreamURL string
		expectedURL string
	}{
		{
			name:        "standard URL",
			upstreamURL: "https://api.example.com/v1/chat/completions",
			expectedURL: "https://api.example.com/v1/models",
		},
		{
			name:        "URL without chat/completions",
			upstreamURL: "https://api.example.com/v1/",
			expectedURL: "https://api.example.com/v1/models",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &config.Config{
				AppConfig: &config.Schema{
					Providers: []config.Provider{
						{
							Name:    "openai",
							Type:    "openai",
							BaseURL: tt.upstreamURL,
						},
					},
				},
			}
			h := &ModelsHandler{cfg: cfg}

			result := h.buildModelsURL()
			if result != tt.expectedURL {
				t.Errorf("buildModelsURL() = %q, want %q", result, tt.expectedURL)
			}
		})
	}
}

func TestModelsHandler_Handle_UpstreamError(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		w.Write([]byte("upstream unavailable"))
	}))
	defer upstream.Close()

	cfg := &config.Config{
		AppConfig: &config.Schema{
			Providers: []config.Provider{
				{
					Name:    "openai",
					Type:    "openai",
					BaseURL: upstream.URL + "/v1/chat/completions",
					APIKey:  "test-api-key",
				},
			},
		},
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	c.Request.Header.Set("Authorization", "Bearer test-key")

	h := &ModelsHandler{cfg: cfg}
	h.Handle(c)

	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("expected status %d, got %d", http.StatusServiceUnavailable, w.Code)
	}
}

func TestModelsHandler_Handle_Success(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("unexpected authorization header: %s", r.Header.Get("Authorization"))
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"object": "list", "data": [{"id": "model-1"}]}`))
	}))
	defer upstream.Close()

	cfg := &config.Config{
		AppConfig: &config.Schema{
			Providers: []config.Provider{
				{
					Name:    "openai",
					Type:    "openai",
					BaseURL: upstream.URL + "/v1/chat/completions",
					APIKey:  "default-api-key",
				},
			},
		},
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	c.Request.Header.Set("Authorization", "Bearer test-key")

	h := &ModelsHandler{cfg: cfg}
	h.Handle(c)

	if w.Code != http.StatusOK {
		t.Errorf("expected status %d, got %d", http.StatusOK, w.Code)
	}

	var response map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatalf("failed to parse response: %v", err)
	}

	if response["object"] != "list" {
		t.Errorf("expected object 'list', got %v", response["object"])
	}
}
