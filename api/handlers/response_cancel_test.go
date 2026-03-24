package handlers

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/gin-gonic/gin"
)

func init() {
	gin.SetMode(gin.TestMode)
}

func TestResponseCancelHandler_Handle_Success(t *testing.T) {
	// Reset the global registry for testing
	globalRegistry = nil
	globalRegistryOnce = sync.Once{}

	registry := GetGlobalRegistry()

	// Register a mock stream
	_, cancel := context.WithCancel(context.Background())
	registry.Register("resp_cancel_test", cancel)

	router := gin.New()
	router.POST("/v1/responses/:id/cancel", NewResponseCancelHandler())

	req := httptest.NewRequest(http.MethodPost, "/v1/responses/resp_cancel_test/cancel", nil)
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	// Verify response contains cancelled: true
	if !containsStr(w.Body.String(), `"cancelled":true`) {
		t.Errorf("expected response to contain cancelled:true, got: %s", w.Body.String())
	}

	// Verify stream is removed from registry
	if registry.Get("resp_cancel_test") != nil {
		t.Error("stream should be removed from registry after cancellation")
	}
}

func TestResponseCancelHandler_Handle_NotFound(t *testing.T) {
	// Reset the global registry for testing
	globalRegistry = nil
	globalRegistryOnce = sync.Once{}

	router := gin.New()
	router.POST("/v1/responses/:id/cancel", NewResponseCancelHandler())

	req := httptest.NewRequest(http.MethodPost, "/v1/responses/nonexistent/cancel", nil)
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("expected status 404, got %d", w.Code)
	}
}

func TestResponseCancelHandler_Handle_EmptyID(t *testing.T) {
	router := gin.New()
	router.POST("/v1/responses/:id/cancel", NewResponseCancelHandler())

	req := httptest.NewRequest(http.MethodPost, "/v1/responses//cancel", nil)
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	// Gin will return 404 for the route, not the handler
	if w.Code != http.StatusNotFound && w.Code != http.StatusBadRequest {
		t.Errorf("expected status 404 or 400, got %d", w.Code)
	}
}

func TestGetGlobalRegistry_Singleton(t *testing.T) {
	// Reset
	globalRegistry = nil
	globalRegistryOnce = sync.Once{}

	r1 := GetGlobalRegistry()
	r2 := GetGlobalRegistry()

	if r1 == nil {
		t.Fatal("GetGlobalRegistry() returned nil")
	}
	if r1 != r2 {
		t.Error("GetGlobalRegistry() should return the same instance")
	}
}

func containsStr(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
