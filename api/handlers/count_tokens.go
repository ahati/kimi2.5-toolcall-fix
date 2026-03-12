package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"ai-proxy/config"
	"ai-proxy/proxy"
	"ai-proxy/tokens"
	"ai-proxy/types"

	"github.com/gin-gonic/gin"
)

// CountTokensHandler handles Anthropic messages count_tokens API requests.
// It implements a non-streaming handler for the /v1/messages/count_tokens endpoint.
//
// This handler:
//   - Accepts requests in Anthropic Messages format
//   - Forwards to Anthropic-compatible upstream count_tokens endpoint
//   - Returns token count estimates in Anthropic format
//
// @note This is a non-streaming endpoint that returns a single JSON response.
type CountTokensHandler struct {
	// cfg contains the application configuration including upstream URL and API key.
	// Must not be nil after construction.
	cfg *config.Config
}

// NonStreamingHandler defines the interface for non-streaming API requests.
// Similar to Handler but without SSE transformation logic.
type NonStreamingHandler interface {
	// ValidateRequest checks if the request body is valid.
	ValidateRequest(body []byte) error

	// TransformRequest converts the request body to upstream format.
	TransformRequest(body []byte) ([]byte, error)

	// UpstreamURL returns the target URL for the upstream API.
	UpstreamURL() string

	// ResolveAPIKey extracts or determines the API key.
	ResolveAPIKey(c *gin.Context) string

	// ForwardHeaders copies relevant headers to the upstream request.
	ForwardHeaders(c *gin.Context, req *http.Request)

	// WriteError sends an error response in the appropriate format.
	WriteError(c *gin.Context, status int, msg string)
}

// NewCountTokensHandler creates a Gin handler for the /v1/messages/count_tokens endpoint.
//
// @param cfg - Application configuration. Must not be nil.
// @return Gin handler function that processes count_tokens requests.
//
// @pre cfg != nil
// @pre cfg.AnthropicUpstreamURL != ""
func NewCountTokensHandler(cfg *config.Config) gin.HandlerFunc {
	return HandleNonStreaming(&CountTokensHandler{cfg: cfg})
}

// ValidateRequest validates the count_tokens request format.
// Checks for required fields and valid structure.
//
// @param body - Raw request body bytes.
// @return nil if valid, error describing validation failure otherwise.
func (h *CountTokensHandler) ValidateRequest(body []byte) error {
	var req types.MessageCountTokensRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return err
	}

	// Model is required for accurate token counting
	if req.Model == "" {
		return fmt.Errorf("model is required")
	}

	// Messages array is required
	if len(req.Messages) == 0 {
		return fmt.Errorf("messages array is required and cannot be empty")
	}

	// Validate each message has required fields
	for i, msg := range req.Messages {
		if msg.Role == "" {
			return fmt.Errorf("message[%d]: role is required", i)
		}
		if msg.Content == nil {
			return fmt.Errorf("message[%d]: content is required", i)
		}
	}

	return nil
}

// TransformRequest returns the body with model field defaulted if missing.
// The upstream API requires a model field for token counting.
//
// @param body - Raw request body in Anthropic format.
// @return Transformed body with model field ensured.
// @return Error if JSON parsing fails.
func (h *CountTokensHandler) TransformRequest(body []byte) ([]byte, error) {
	// Parse request to check if model is specified
	var req map[string]interface{}
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, err
	}

	// Add default model if not specified
	if _, hasModel := req["model"]; !hasModel {
		req["model"] = "kimi-k2.5"
		return json.Marshal(req)
	}

	// Return body unchanged if model already specified
	return body, nil
}

// UpstreamURL returns the Anthropic count_tokens endpoint URL.
// Constructs the URL by replacing "/messages" with "/messages/count_tokens".
//
// @return URL string for the count_tokens endpoint.
//
// @pre h.cfg != nil
// @post URL includes the full path to count_tokens endpoint.
func (h *CountTokensHandler) UpstreamURL() string {
	// Replace /messages with /messages/count_tokens in the base URL
	baseURL := h.cfg.AnthropicUpstreamURL
	return strings.Replace(baseURL, "/messages", "/messages/count_tokens", 1)
}

// ResolveAPIKey returns the configured Anthropic API key.
// This key is used for authentication with the upstream API.
//
// @param c - Gin context (unused in this implementation).
// @return API key string from configuration.
//
// @pre h.cfg != nil
func (h *CountTokensHandler) ResolveAPIKey(c *gin.Context) string {
	return h.cfg.AnthropicAPIKey
}

// ForwardHeaders copies X-*, Anthropic-Version, and Anthropic-Beta headers
// to the upstream request.
//
// @param c - Gin context containing the original request headers.
// @param req - Upstream request to receive forwarded headers.
//
// @pre c != nil, req != nil
// @post All X-* headers are copied to upstream request.
// @post Anthropic-Version header is copied if present.
// @post Anthropic-Beta header is copied if present.
func (h *CountTokensHandler) ForwardHeaders(c *gin.Context, req *http.Request) {
	for k, v := range c.Request.Header {
		if strings.HasPrefix(k, "X-") || k == "Anthropic-Version" || k == "Anthropic-Beta" {
			req.Header[k] = v
		}
	}
}

// WriteError sends an error response in Anthropic format.
// Maintains consistency with Anthropic API error responses.
//
// @param c - Gin context for writing the response.
// @param status - HTTP status code for the error.
// @param msg - Human-readable error message.
//
// @pre c != nil and response not yet written.
// @post Anthropic-format error response is written.
func (h *CountTokensHandler) WriteError(c *gin.Context, status int, msg string) {
	sendAnthropicError(c, status, msg)
}

// HandleNonStreaming wraps a NonStreamingHandler and returns a Gin handler function.
// It orchestrates the full request pipeline for non-streaming endpoints.
//
// The processing flow is:
//  1. Read request body from client
//  2. Validate request format
//  3. Transform request to upstream format
//  4. Forward to upstream and return response
//
// @param h - NonStreamingHandler implementation.
// @return Gin handler function that processes requests.
//
// @pre h != nil
// @post Response is fully written to client on return.
func HandleNonStreaming(h NonStreamingHandler) gin.HandlerFunc {
	return func(c *gin.Context) {
		// Step 1: Read the complete request body
		body, err := io.ReadAll(c.Request.Body)
		if err != nil {
			h.WriteError(c, http.StatusBadRequest, "Failed to read request body")
			return
		}

		// Step 2: Validate request format
		if err := h.ValidateRequest(body); err != nil {
			h.WriteError(c, http.StatusBadRequest, err.Error())
			return
		}

		// Step 3: Transform request to upstream format
		transformedBody, err := h.TransformRequest(body)
		if err != nil {
			h.WriteError(c, http.StatusInternalServerError, "Failed to transform request")
			return
		}

		// Step 4: Forward to upstream and return response
		proxyNonStreamingRequest(c, h, transformedBody)
	}
}

// proxyNonStreamingRequest forwards the request to the upstream API and returns the response.
// If upstream returns 404 (endpoint not supported), falls back to local token counting.
//
// @param c - Gin context for the current request.
// @param h - Handler defining upstream URL, headers, and error handling.
// @param body - Transformed request body to send upstream.
//
// @pre body is in correct upstream format.
// @pre h.UpstreamURL() returns valid URL.
// @post Response is returned to client or error response is sent.
func proxyNonStreamingRequest(c *gin.Context, h NonStreamingHandler, body []byte) {
	// Resolve API key for upstream authentication
	apiKey := h.ResolveAPIKey(c)

	// Create HTTP client configured for upstream endpoint
	client := proxy.NewClient(h.UpstreamURL(), apiKey)
	defer client.Close()

	// Build the upstream HTTP request
	req, err := client.BuildRequest(c.Request.Context(), body)
	if err != nil {
		h.WriteError(c, http.StatusInternalServerError, "Failed to create upstream request")
		return
	}

	// Set standard headers required by upstream API
	client.SetHeaders(req)
	// Forward custom headers from original request
	h.ForwardHeaders(c, req)

	// Execute the upstream request
	resp, err := client.Do(req)
	if err != nil {
		h.WriteError(c, http.StatusBadGateway, "Upstream request failed")
		return
	}
	defer resp.Body.Close()

	// Check for non-200 responses from upstream
	if resp.StatusCode != http.StatusOK {
		// If endpoint not found, fall back to local counting
		if resp.StatusCode == http.StatusNotFound {
			handleLocalTokenCount(c, body, h)
			return
		}
		handleUpstreamError(c, resp)
		return
	}

	// Read the response body
	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		h.WriteError(c, http.StatusInternalServerError, "Failed to read upstream response")
		return
	}

	// Set content type for JSON response
	c.Header("Content-Type", "application/json")
	c.Data(resp.StatusCode, "application/json", responseBody)
}

// handleLocalTokenCount performs local token counting when upstream doesn't support the endpoint.
//
// @param c - Gin context for writing the response.
// @param body - The request body containing token counting request.
// @param h - Handler for error responses.
func handleLocalTokenCount(c *gin.Context, body []byte, h NonStreamingHandler) {
	var req types.MessageCountTokensRequest
	if err := json.Unmarshal(body, &req); err != nil {
		h.WriteError(c, http.StatusInternalServerError, "Failed to parse request for local counting")
		return
	}

	count, err := tokens.CountTokens(&req)
	if err != nil {
		h.WriteError(c, http.StatusInternalServerError, "Failed to count tokens locally")
		return
	}

	response := types.MessageCountTokensResponse{
		InputTokens: count,
	}

	responseJSON, err := json.Marshal(response)
	if err != nil {
		h.WriteError(c, http.StatusInternalServerError, "Failed to encode response")
		return
	}

	c.Header("Content-Type", "application/json")
	c.Data(http.StatusOK, "application/json", responseJSON)
}
