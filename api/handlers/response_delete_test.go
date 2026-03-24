package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"ai-proxy/conversation"
	"ai-proxy/types"

	"github.com/gin-gonic/gin"
)

func TestResponseDeleteHandler_Handle_Success(t *testing.T) {
	// Initialize the default store
	conversation.InitDefaultStore(conversation.Config{
		MaxSize: 100,
		TTL:     time.Hour,
	})
	defer conversation.DefaultStore.Clear()

	// Store a test conversation
	testID := "test-response-id-123"
	testConv := &conversation.Conversation{
		ID:        testID,
		Input:     []types.InputItem{},
		Output:    []types.OutputItem{},
		CreatedAt: time.Now(),
	}
	conversation.StoreInDefault(testConv)

	// Create the handler and request
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodDelete, "/v1/responses/"+testID, nil)
	c.AddParam("id", testID)

	h := &ResponseDeleteHandler{}
	h.Handle(c)

	// Verify response
	if w.Code != http.StatusOK {
		t.Errorf("expected status %d, got %d", http.StatusOK, w.Code)
	}

	var response ResponseDeleteResponse
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatalf("failed to parse response: %v", err)
	}

	if !response.Deleted {
		t.Error("expected deleted to be true")
	}
	if response.ID != testID {
		t.Errorf("expected ID %s, got %s", testID, response.ID)
	}

	// Verify the conversation was actually deleted
	if conversation.GetFromDefault(testID) != nil {
		t.Error("expected conversation to be deleted from store")
	}
}

func TestResponseDeleteHandler_Handle_NotFound(t *testing.T) {
	// Initialize the default store
	conversation.InitDefaultStore(conversation.Config{
		MaxSize: 100,
		TTL:     time.Hour,
	})
	defer conversation.DefaultStore.Clear()

	// Create the handler and request with non-existent ID
	nonExistentID := "non-existent-id"
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodDelete, "/v1/responses/"+nonExistentID, nil)
	c.AddParam("id", nonExistentID)

	h := &ResponseDeleteHandler{}
	h.Handle(c)

	// Verify response
	if w.Code != http.StatusNotFound {
		t.Errorf("expected status %d, got %d", http.StatusNotFound, w.Code)
	}

	var response map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatalf("failed to parse response: %v", err)
	}

	// Check for structured error format
	errObj, ok := response["error"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected error to be an object, got %v", response["error"])
	}
	if errObj["code"] != "response_not_found" {
		t.Errorf("expected error code 'response_not_found', got %v", errObj["code"])
	}
	if errObj["message"] != "Response not found" {
		t.Errorf("expected error message 'Response not found', got %v", errObj["message"])
	}
}

func TestResponseDeleteHandler_Handle_EmptyID(t *testing.T) {
	// Initialize the default store
	conversation.InitDefaultStore(conversation.Config{
		MaxSize: 100,
		TTL:     time.Hour,
	})
	defer conversation.DefaultStore.Clear()

	// Create the handler and request with empty ID
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodDelete, "/v1/responses/", nil)
	c.AddParam("id", "")

	h := &ResponseDeleteHandler{}
	h.Handle(c)

	// Verify response
	if w.Code != http.StatusBadRequest {
		t.Errorf("expected status %d, got %d", http.StatusBadRequest, w.Code)
	}

	var response map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatalf("failed to parse response: %v", err)
	}

	if response["error"] != "Response ID is required" {
		t.Errorf("expected error 'Response ID is required', got %v", response["error"])
	}
}

func TestResponseDeleteHandler_Handle_NilStore(t *testing.T) {
	// Ensure the default store is nil
	conversation.DefaultStore = nil

	// Create the handler and request
	testID := "test-id-nil-store"
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodDelete, "/v1/responses/"+testID, nil)
	c.AddParam("id", testID)

	h := &ResponseDeleteHandler{}
	h.Handle(c)

	// When store is nil, GetFromDefault returns nil, so we should get 404
	if w.Code != http.StatusNotFound {
		t.Errorf("expected status %d, got %d", http.StatusNotFound, w.Code)
	}
}

func TestNewResponseDeleteHandler_ReturnsHandler(t *testing.T) {
	handler := NewResponseDeleteHandler()
	if handler == nil {
		t.Error("expected handler to be non-nil")
	}
}

func TestResponseDeleteHandler_Handle_DeletesOnlyTarget(t *testing.T) {
	// Initialize the default store
	conversation.InitDefaultStore(conversation.Config{
		MaxSize: 100,
		TTL:     time.Hour,
	})
	defer conversation.DefaultStore.Clear()

	// Store multiple test conversations
	testID1 := "test-response-id-1"
	testID2 := "test-response-id-2"
	conversation.StoreInDefault(&conversation.Conversation{
		ID:        testID1,
		CreatedAt: time.Now(),
	})
	conversation.StoreInDefault(&conversation.Conversation{
		ID:        testID2,
		CreatedAt: time.Now(),
	})

	// Delete only the first one
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodDelete, "/v1/responses/"+testID1, nil)
	c.AddParam("id", testID1)

	h := &ResponseDeleteHandler{}
	h.Handle(c)

	// Verify the first was deleted
	if conversation.GetFromDefault(testID1) != nil {
		t.Error("expected first conversation to be deleted")
	}

	// Verify the second still exists
	if conversation.GetFromDefault(testID2) == nil {
		t.Error("expected second conversation to still exist")
	}
}
