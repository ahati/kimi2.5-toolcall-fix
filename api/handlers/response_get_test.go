package handlers

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"ai-proxy/conversation"
	"ai-proxy/types"

	"github.com/gin-gonic/gin"
)

func init() {
	gin.SetMode(gin.TestMode)
}

// ─────────────────────────────────────────────────────────────────────────────
// ResponseGetHandler Tests
// ─────────────────────────────────────────────────────────────────────────────

func TestResponseGetHandler_Handle_Success(t *testing.T) {
	// Initialize store with a test conversation
	conversation.InitDefaultStore(conversation.Config{
		MaxSize: 100,
		TTL:     time.Hour,
	})

	testConv := &conversation.Conversation{
		ID:        "resp_test123",
		Input:     []types.InputItem{{Type: "message", Role: "user", Content: "Hello"}},
		Output:    []types.OutputItem{{Type: "message", Role: "assistant", Content: []types.OutputContent{{Type: "output_text", Text: "Hi there!"}}}},
		CreatedAt: time.Now(),
	}
	conversation.StoreInDefault(testConv)

	router := gin.New()
	router.GET("/v1/responses/:id", NewResponseGetHandler())

	req := httptest.NewRequest(http.MethodGet, "/v1/responses/resp_test123", nil)
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	// Verify response contains the ID
	if !contains(w.Body.String(), "resp_test123") {
		t.Errorf("expected response to contain ID, got: %s", w.Body.String())
	}
}

func TestResponseGetHandler_Handle_NotFound(t *testing.T) {
	conversation.InitDefaultStore(conversation.Config{MaxSize: 100, TTL: time.Hour})
	conversation.DefaultStore.Clear()

	router := gin.New()
	router.GET("/v1/responses/:id", NewResponseGetHandler())

	req := httptest.NewRequest(http.MethodGet, "/v1/responses/nonexistent", nil)
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("expected status 404, got %d", w.Code)
	}
}

func TestResponseGetHandler_Handle_EmptyID(t *testing.T) {
	router := gin.New()
	router.GET("/v1/responses/:id", NewResponseGetHandler())

	req := httptest.NewRequest(http.MethodGet, "/v1/responses/", nil)
	w := httptest.NewRecorder()

	// This will hit the 404 route, not the handler
	router.ServeHTTP(w, req)

	// Gin returns 404 for route not found when ID is empty
	if w.Code != http.StatusNotFound && w.Code != http.StatusBadRequest {
		t.Errorf("expected status 404 or 400, got %d", w.Code)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// ResponseInputItemsHandler Tests
// ─────────────────────────────────────────────────────────────────────────────

func TestResponseInputItemsHandler_Handle_Success(t *testing.T) {
	conversation.InitDefaultStore(conversation.Config{
		MaxSize: 100,
		TTL:     time.Hour,
	})

	testInput := []types.InputItem{
		{Type: "message", Role: "user", Content: "What is AI?"},
		{Type: "message", Role: "assistant", Content: "AI stands for artificial intelligence."},
	}
	testConv := &conversation.Conversation{
		ID:        "resp_input_test",
		Input:     testInput,
		Output:    []types.OutputItem{},
		CreatedAt: time.Now(),
	}
	conversation.StoreInDefault(testConv)

	router := gin.New()
	router.GET("/v1/responses/:id/input_items", NewResponseInputItemsHandler())

	req := httptest.NewRequest(http.MethodGet, "/v1/responses/resp_input_test/input_items", nil)
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	// Verify response contains the input items
	body := w.Body.String()
	if !contains(body, "What is AI?") {
		t.Errorf("expected response to contain input content, got: %s", body)
	}
	if !contains(body, "artificial intelligence") {
		t.Errorf("expected response to contain assistant content, got: %s", body)
	}
}

func TestResponseInputItemsHandler_Handle_NotFound(t *testing.T) {
	conversation.InitDefaultStore(conversation.Config{MaxSize: 100, TTL: time.Hour})
	conversation.DefaultStore.Clear()

	router := gin.New()
	router.GET("/v1/responses/:id/input_items", NewResponseInputItemsHandler())

	req := httptest.NewRequest(http.MethodGet, "/v1/responses/nonexistent/input_items", nil)
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("expected status 404, got %d", w.Code)
	}
}

func TestResponseInputItemsHandler_Handle_ListFormat(t *testing.T) {
	conversation.InitDefaultStore(conversation.Config{MaxSize: 100, TTL: time.Hour})
	conversation.DefaultStore.Clear()

	testConv := &conversation.Conversation{
		ID:        "resp_list_test",
		Input:     []types.InputItem{},
		Output:    []types.OutputItem{},
		CreatedAt: time.Now(),
	}
	conversation.StoreInDefault(testConv)

	router := gin.New()
	router.GET("/v1/responses/:id/input_items", NewResponseInputItemsHandler())

	req := httptest.NewRequest(http.MethodGet, "/v1/responses/resp_list_test/input_items", nil)
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", w.Code)
	}

	// Verify response has "object": "list"
	body := w.Body.String()
	if !contains(body, `"object":"list"`) {
		t.Errorf("expected response to have object='list', got: %s", body)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper
// ─────────────────────────────────────────────────────────────────────────────

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsHelper(s, substr))
}

func containsHelper(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
