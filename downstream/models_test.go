package downstream

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"ai-proxy/config"

	"github.com/gin-gonic/gin"
)

func TestListModels_MissingAPIKey(t *testing.T) {
	gin.SetMode(gin.TestMode)

	cfg := &config.Config{
		OpenAIUpstreamURL:    "http://localhost:9999/v1/chat/completions",
		OpenAIUpstreamAPIKey: "",
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest("GET", "/v1/models", nil)

	handler := ListModels(cfg)
	handler(c)

	if w.Code != http.StatusUnauthorized {
		t.Errorf("Expected status %d, got %d", http.StatusUnauthorized, w.Code)
	}

	var response map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	errObj, ok := response["error"].(map[string]interface{})
	if !ok {
		t.Fatal("Expected error object in response")
	}
	if errObj["code"] != "missing_api_key" {
		t.Errorf("Expected missing_api_key code, got %v", errObj["code"])
	}
}

func TestListModels_BearerToken(t *testing.T) {
	gin.SetMode(gin.TestMode)

	mockUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-token-123" {
			t.Errorf("Expected Bearer test-token-123, got %s", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"object":"list","data":[{"id":"model-1"}]}`))
	}))
	defer mockUpstream.Close()

	cfg := &config.Config{
		OpenAIUpstreamURL:    mockUpstream.URL + "/v1/chat/completions",
		OpenAIUpstreamAPIKey: "default-key",
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest("GET", "/v1/models", nil)
	c.Request.Header.Set("Authorization", "Bearer test-token-123")

	handler := ListModels(cfg)
	handler(c)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status %d, got %d", http.StatusOK, w.Code)
	}
}

func TestListModels_UsesDefaultKey(t *testing.T) {
	gin.SetMode(gin.TestMode)

	mockUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer default-key" {
			t.Errorf("Expected Bearer default-key, got %s", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"object":"list","data":[]}`))
	}))
	defer mockUpstream.Close()

	cfg := &config.Config{
		OpenAIUpstreamURL:    mockUpstream.URL + "/v1/chat/completions",
		OpenAIUpstreamAPIKey: "default-key",
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest("GET", "/v1/models", nil)

	handler := ListModels(cfg)
	handler(c)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status %d, got %d", http.StatusOK, w.Code)
	}
}

func TestListModels_UpstreamError(t *testing.T) {
	gin.SetMode(gin.TestMode)

	cfg := &config.Config{
		OpenAIUpstreamURL:    "http://localhost:59998/v1/chat/completions",
		OpenAIUpstreamAPIKey: "test-key",
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest("GET", "/v1/models", nil)
	c.Request.Header.Set("Authorization", "Bearer test-key")

	handler := ListModels(cfg)
	handler(c)

	if w.Code != http.StatusBadGateway {
		t.Errorf("Expected status %d, got %d", http.StatusBadGateway, w.Code)
	}
}

func TestListModels_PropagatesStatusCode(t *testing.T) {
	gin.SetMode(gin.TestMode)

	mockUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		w.Write([]byte(`{"error": "rate limited"}`))
	}))
	defer mockUpstream.Close()

	cfg := &config.Config{
		OpenAIUpstreamURL:    mockUpstream.URL + "/v1/chat/completions",
		OpenAIUpstreamAPIKey: "test-key",
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest("GET", "/v1/models", nil)
	c.Request.Header.Set("Authorization", "Bearer test-key")

	handler := ListModels(cfg)
	handler(c)

	if w.Code != http.StatusTooManyRequests {
		t.Errorf("Expected status %d, got %d", http.StatusTooManyRequests, w.Code)
	}
}

func TestListModels_PropagatesContentType(t *testing.T) {
	gin.SetMode(gin.TestMode)

	mockUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"object":"list"}`))
	}))
	defer mockUpstream.Close()

	cfg := &config.Config{
		OpenAIUpstreamURL:    mockUpstream.URL + "/v1/chat/completions",
		OpenAIUpstreamAPIKey: "test-key",
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest("GET", "/v1/models", nil)
	c.Request.Header.Set("Authorization", "Bearer test-key")

	handler := ListModels(cfg)
	handler(c)

	contentType := w.Header().Get("Content-Type")
	if contentType != "application/json; charset=utf-8" {
		t.Errorf("Expected content type to be propagated, got '%s'", contentType)
	}
}
