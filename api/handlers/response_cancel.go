package handlers

import (
	"net/http"
	"sync"

	"ai-proxy/stream"

	"github.com/gin-gonic/gin"
)

// Global stream registry for tracking active streams
var (
	globalRegistry     *stream.Registry
	globalRegistryOnce sync.Once
)

// GetGlobalRegistry returns the global stream registry, initializing it if necessary.
func GetGlobalRegistry() *stream.Registry {
	globalRegistryOnce.Do(func() {
		globalRegistry = stream.NewRegistry()
	})
	return globalRegistry
}

// ResponseCancelHandler handles requests to cancel an in-progress streaming response.
// It allows clients to abort a streaming response by its ID.
//
// This handler:
//   - Accepts POST requests to cancel active streams
//   - Returns 404 if the stream is not found or already completed
//   - Returns {cancelled: true, id: "..."} on successful cancellation
//
// @note This endpoint is part of the Responses API streaming lifecycle.
type ResponseCancelHandler struct{}

// NewResponseCancelHandler creates a Gin handler for the POST /v1/responses/:id/cancel endpoint.
//
// @return Gin handler function that processes stream cancellation requests.
func NewResponseCancelHandler() gin.HandlerFunc {
	h := &ResponseCancelHandler{}
	return h.Handle
}

// ResponseCancelResponse represents the response from a successful cancellation.
type ResponseCancelResponse struct {
	Cancelled bool   `json:"cancelled"`
	ID        string `json:"id"`
}

// Handle processes the stream cancellation request.
// It extracts the ID from the URL path, looks up the active stream, and cancels it.
//
// @param c - Gin context for the HTTP request.
func (h *ResponseCancelHandler) Handle(c *gin.Context) {
	id := c.Param("id")
	if id == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Response ID is required"})
		return
	}

	// Get the global registry
	registry := GetGlobalRegistry()

	// Attempt to cancel the stream
	cancelled := registry.Cancel(id)
	if !cancelled {
		c.JSON(http.StatusNotFound, gin.H{
			"error": gin.H{
				"code":    "stream_not_found",
				"message": "Stream not found or already completed",
			},
		})
		return
	}

	// Return success response
	c.JSON(http.StatusOK, ResponseCancelResponse{
		Cancelled: true,
		ID:        id,
	})
}
