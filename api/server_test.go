package api

import (
	"ai-proxy/config"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
)

func init() {
	gin.SetMode(gin.TestMode)
}

func TestNewServer(t *testing.T) {
	tests := []struct {
		name     string
		config   *config.Config
		wantMode string
	}{
		{
			name: "with empty port defaults to test mode",
			config: &config.Config{
				Port: "",
			},
			wantMode: gin.TestMode,
		},
		{
			name: "with non-empty port sets release mode",
			config: &config.Config{
				Port: "8080",
			},
			wantMode: gin.ReleaseMode,
		},
		{
			name: "with all config fields",
			config: &config.Config{
				AppConfig: &config.Schema{
					Providers: []config.Provider{
						{
							Name:    "openai",
							Type:    "openai",
							BaseURL: "https://api.example.com/v1",
							APIKey:  "test-key",
						},
						{
							Name:    "anthropic",
							Type:    "anthropic",
							BaseURL: "https://api.anthropic.com/v1",
							APIKey:  "anthropic-key",
						},
					},
				},
				Port:      "9090",
				SSELogDir: "/tmp/logs",
			},
			wantMode: gin.ReleaseMode,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.config.Port != "" {
				gin.SetMode(gin.DebugMode)
			}

			server := NewServer(tt.config)

			if server == nil {
				t.Fatal("NewServer returned nil")
			}

			if server.config != tt.config {
				t.Error("server config not set correctly")
			}

			if server.router == nil {
				t.Error("server router is nil")
			}
		})
	}
}

func TestServer_setupRoutes(t *testing.T) {
	cfg := &config.Config{
		Port: "",
	}

	server := NewServer(cfg)

	routes := server.router.Routes()

	expectedRoutes := []struct {
		method string
		path   string
	}{
		{method: "GET", path: "/health"},
		{method: "GET", path: "/v1/models"},
		{method: "POST", path: "/v1/chat/completions"},
		{method: "POST", path: "/v1/messages"},
		{method: "POST", path: "/v1/messages/count_tokens"},
	}

	for _, expected := range expectedRoutes {
		found := false
		for _, route := range routes {
			if route.Method == expected.method && route.Path == expected.path {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected route %s %s not found", expected.method, expected.path)
		}
	}
}

func TestServer_setupRoutes_RouteCount(t *testing.T) {
	cfg := &config.Config{}
	server := NewServer(cfg)

	routes := server.router.Routes()

	expectedCount := 5
	if len(routes) != expectedCount {
		t.Errorf("expected %d routes, got %d", expectedCount, len(routes))
	}
}

func TestServer_Use(t *testing.T) {
	gin.SetMode(gin.TestMode)

	cfg := &config.Config{}
	server := NewServer(cfg)

	var middlewareCalled bool
	server.Use(func(c *gin.Context) {
		middlewareCalled = true
		c.Next()
	})

	server.router.GET("/test-middleware", func(c *gin.Context) {
		c.Status(http.StatusOK)
	})

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/test-middleware", nil)
	server.router.ServeHTTP(w, req)

	if !middlewareCalled {
		t.Error("middleware was not called")
	}
}

func TestServer_Use_MultipleMiddlewares(t *testing.T) {
	gin.SetMode(gin.TestMode)

	cfg := &config.Config{}
	server := NewServer(cfg)

	callOrder := []int{}
	server.Use(
		func(c *gin.Context) {
			callOrder = append(callOrder, 1)
			c.Next()
		},
		func(c *gin.Context) {
			callOrder = append(callOrder, 2)
			c.Next()
		},
	)

	server.router.GET("/test-multi-middleware", func(c *gin.Context) {
		callOrder = append(callOrder, 3)
		c.Status(http.StatusOK)
	})

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/test-multi-middleware", nil)
	server.router.ServeHTTP(w, req)

	if len(callOrder) != 3 {
		t.Errorf("expected 3 calls (2 middleware + handler), got %d", len(callOrder))
	}
	if callOrder[0] != 1 || callOrder[1] != 2 || callOrder[2] != 3 {
		t.Errorf("call order wrong: got %v, expected [1 2 3]", callOrder)
	}
}

func TestServer_Routes_HealthCheck(t *testing.T) {
	cfg := &config.Config{}
	server := NewServer(cfg)

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	server.router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status %d, got %d", http.StatusOK, w.Code)
	}
}

func TestServer_Routes_Models(t *testing.T) {
	cfg := &config.Config{}
	server := NewServer(cfg)

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	server.router.ServeHTTP(w, req)

	if w.Code != http.StatusUnauthorized {
		t.Errorf("expected status %d (unauthorized without API key), got %d", http.StatusUnauthorized, w.Code)
	}
}

func TestServer_Routes_Models_WithAPIKey(t *testing.T) {
	cfg := &config.Config{
		AppConfig: &config.Schema{
			Providers: []config.Provider{
				{
					Name:   "openai",
					Type:   "openai",
					APIKey: "test-key",
				},
			},
		},
	}
	server := NewServer(cfg)

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer test-key")
	server.router.ServeHTTP(w, req)

	if w.Code == http.StatusUnauthorized {
		t.Error("should not return unauthorized when API key is provided")
	}
}

func TestServer_Routes_Completions_InvalidMethod(t *testing.T) {
	cfg := &config.Config{}
	server := NewServer(cfg)

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/v1/chat/completions", nil)
	server.router.ServeHTTP(w, req)

	if w.Code == http.StatusOK {
		t.Error("GET request to /v1/chat/completions should not return 200")
	}
}

func TestServer_Routes_Messages_InvalidMethod(t *testing.T) {
	cfg := &config.Config{}
	server := NewServer(cfg)

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/v1/messages", nil)
	server.router.ServeHTTP(w, req)

	if w.Code == http.StatusOK {
		t.Error("GET request to /v1/messages should not return 200")
	}
}

func TestServer_Routes_Bridge_InvalidMethod(t *testing.T) {
	cfg := &config.Config{}
	server := NewServer(cfg)

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/v1/openai-to-anthropic/messages", nil)
	server.router.ServeHTTP(w, req)

	if w.Code == http.StatusOK {
		t.Error("GET request to bridge endpoint should not return 200")
	}
}

func TestServer_Routes_UnknownPath(t *testing.T) {
	cfg := &config.Config{}
	server := NewServer(cfg)

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/unknown", nil)
	server.router.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("expected status %d, got %d", http.StatusNotFound, w.Code)
	}
}

func TestServer_NilConfig(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic with nil config, but did not panic")
		}
	}()

	NewServer(nil)
}

func TestServer_Run_InvalidAddress(t *testing.T) {
	cfg := &config.Config{}
	server := NewServer(cfg)

	err := server.Run(":-1")
	if err == nil {
		t.Error("expected error for invalid address")
	}
}

func TestServer_Run_ValidAddress(t *testing.T) {
	cfg := &config.Config{}
	server := NewServer(cfg)

	go func() {
		_ = server.Run(":0")
	}()

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	server.router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status %d, got %d", http.StatusOK, w.Code)
	}
}

func TestNewServer_SetsConfigCorrectly(t *testing.T) {
	cfg := &config.Config{
		AppConfig: &config.Schema{
			Providers: []config.Provider{
				{
					Name:    "openai",
					Type:    "openai",
					BaseURL: "https://upstream.example.com",
					APIKey:  "test-api-key",
				},
			},
		},
		Port: "8080",
	}

	server := NewServer(cfg)

	expectedURL := "https://upstream.example.com"
	if server.config.GetOpenAIUpstreamURL() != expectedURL {
		t.Errorf("expected OpenAIUpstreamURL %s, got %s", expectedURL, server.config.GetOpenAIUpstreamURL())
	}

	expectedKey := "test-api-key"
	if server.config.GetOpenAIUpstreamAPIKey() != expectedKey {
		t.Errorf("expected OpenAIUpstreamAPIKey %s, got %s", expectedKey, server.config.GetOpenAIUpstreamAPIKey())
	}

	if server.config.Port != cfg.Port {
		t.Errorf("expected Port %s, got %s", cfg.Port, server.config.Port)
	}
}

func TestServer_RouterNotNil(t *testing.T) {
	cfg := &config.Config{}
	server := NewServer(cfg)

	if server.router == nil {
		t.Error("router should not be nil")
	}
}
