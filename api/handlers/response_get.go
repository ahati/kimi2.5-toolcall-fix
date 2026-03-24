package handlers

import (
	"net/http"

	"ai-proxy/conversation"
	"ai-proxy/types"

	"github.com/gin-gonic/gin"
)

// ResponseGetHandler handles requests to retrieve a stored response.
// It allows clients to fetch a previously stored conversation by its ID.
//
// This handler:
//   - Accepts GET requests to retrieve stored conversations
//   - Returns 404 if the conversation is not found
//   - Returns the full ResponsesResponse object
//
// @note This endpoint is part of the Responses API stateful session management.
type ResponseGetHandler struct{}

// NewResponseGetHandler creates a Gin handler for the GET /v1/responses/:id endpoint.
//
// @return Gin handler function that processes response retrieval requests.
func NewResponseGetHandler() gin.HandlerFunc {
	h := &ResponseGetHandler{}
	return h.Handle
}

// Handle processes the response retrieval request.
// It extracts the ID from the URL path, looks up the conversation, and returns it.
//
// @param c - Gin context for the HTTP request.
func (h *ResponseGetHandler) Handle(c *gin.Context) {
	id := c.Param("id")
	if id == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Response ID is required"})
		return
	}

	// Look up the conversation
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

	// Convert Conversation to ResponsesResponse
	response := buildResponsesResponse(conv)
	c.JSON(http.StatusOK, response)
}

// buildResponsesResponse converts a Conversation to a ResponsesResponse.
func buildResponsesResponse(conv *conversation.Conversation) *types.ResponsesResponse {
	return &types.ResponsesResponse{
		ID:        conv.ID,
		Object:    "response",
		CreatedAt: conv.CreatedAt.Unix(),
		Status:    "completed",
		Output:    conv.Output,
		Model:     "",  // Model not stored in conversation, could be added later
		Usage:     nil, // Usage not stored in conversation, could be added later
	}
}

// ResponseInputItemsHandler handles requests to list input items for a response.
// It returns the input items stored for a specific conversation.
//
// This handler:
//   - Accepts GET requests to retrieve input items
//   - Returns 404 if the conversation is not found
//   - Returns a list of input items
//
// @note This endpoint is part of the Responses API stateful session management.
type ResponseInputItemsHandler struct{}

// NewResponseInputItemsHandler creates a Gin handler for the GET /v1/responses/:id/input_items endpoint.
//
// @return Gin handler function that processes input items retrieval requests.
func NewResponseInputItemsHandler() gin.HandlerFunc {
	h := &ResponseInputItemsHandler{}
	return h.Handle
}

// InputItemsResponse represents the response from the input_items endpoint.
type InputItemsResponse struct {
	Object string            `json:"object"`
	Data   []types.InputItem `json:"data"`
}

// Handle processes the input items retrieval request.
// It extracts the ID from the URL path, looks up the conversation, and returns the input items.
//
// @param c - Gin context for the HTTP request.
func (h *ResponseInputItemsHandler) Handle(c *gin.Context) {
	id := c.Param("id")
	if id == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Response ID is required"})
		return
	}

	// Look up the conversation
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

	// Return input items
	response := InputItemsResponse{
		Object: "list",
		Data:   conv.Input,
	}
	c.JSON(http.StatusOK, response)
}
