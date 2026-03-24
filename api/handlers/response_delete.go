package handlers

import (
	"net/http"

	"ai-proxy/conversation"

	"github.com/gin-gonic/gin"
)

// ResponseDeleteHandler handles requests to delete a stored response.
// It allows clients to remove a previously stored conversation by its ID.
//
// This handler:
//   - Accepts DELETE requests to remove stored conversations
//   - Returns 404 if the conversation is not found
//   - Returns {deleted: true, id: "..."} on successful deletion
//
// @note This endpoint is useful for cleanup and privacy purposes.
type ResponseDeleteHandler struct{}

// NewResponseDeleteHandler creates a Gin handler for the DELETE /v1/responses/:id endpoint.
//
// @return Gin handler function that processes response deletion requests.
func NewResponseDeleteHandler() gin.HandlerFunc {
	h := &ResponseDeleteHandler{}
	return h.Handle
}

// ResponseDeleteResponse represents the response from a successful deletion.
type ResponseDeleteResponse struct {
	Deleted bool   `json:"deleted"`
	ID      string `json:"id"`
}

// Handle processes the response deletion request.
// It extracts the ID from the URL path, validates existence, and deletes the conversation.
//
// @param c - Gin context for the HTTP request.
func (h *ResponseDeleteHandler) Handle(c *gin.Context) {
	id := c.Param("id")
	if id == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Response ID is required"})
		return
	}

	// Check if conversation exists
	conv := conversation.GetFromDefault(id)
	if conv == nil {
		c.JSON(http.StatusNotFound, gin.H{
			"error": gin.H{
				"code":    "response_not_found",
				"message": "Response not found",
			},
		})
		return
	}

	// Delete the conversation
	conversation.DefaultStore.Delete(id)

	// Return success response
	c.JSON(http.StatusOK, ResponseDeleteResponse{
		Deleted: true,
		ID:      id,
	})
}
