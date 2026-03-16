package handlers

import (
	"io"
	"net/http"
	"strings"

	"ai-proxy/config"
	"ai-proxy/proxy"

	"github.com/gin-gonic/gin"
)

// ModelsHandler handles requests to list available models from the upstream API.
// It implements a simple proxy to the upstream /v1/models endpoint.
//
// This handler:
//   - Accepts GET requests to retrieve available models
//   - Proxies the request to the upstream API's models endpoint
//   - Returns the upstream response directly to the client
//
// @note This endpoint does not implement the full Handler interface.
// @note It uses a simpler Handle method directly due to different request flow.
type ModelsHandler struct {
	// cfg contains the application configuration including upstream URL and API key.
	// Must not be nil after construction.
	cfg *config.Config
}

// NewModelsHandler creates a Gin handler for the /v1/models endpoint.
//
// @param cfg - Application configuration. Must not be nil.
// @return Gin handler function that processes models list requests.
//
// @pre cfg != nil
// @pre cfg.OpenAIUpstreamURL != ""
func NewModelsHandler(cfg *config.Config) gin.HandlerFunc {
	h := &ModelsHandler{cfg: cfg}
	return h.Handle
}

// Handle processes the models list request by proxying to the upstream API.
// This method directly handles the request without the standard Handler pipeline.
//
// @param c - Gin context for the HTTP request.
//
// @pre c.Request is a valid GET request.
// @post Response from upstream is forwarded to client.
// @post 401 Unauthorized if no API key is available.
// @post 502 Bad Gateway if upstream request fails.
func (h *ModelsHandler) Handle(c *gin.Context) {
	// Resolve API key from request header or configuration
	apiKey := h.resolveAPIKey(c)
	// Reject request if no API key is available
	// This prevents unauthorized upstream requests
	if apiKey == "" {
		sendOpenAIError(c, http.StatusUnauthorized, "Missing API key")
		return
	}

	// Build the models endpoint URL from the configuration
	modelsURL := h.buildModelsURL()

	// Create HTTP client for upstream request
	client := proxy.NewClient(modelsURL, apiKey)
	// Ensure connection resources are released
	defer client.Close()

	// Build GET request for models endpoint
	req, err := http.NewRequestWithContext(c.Request.Context(), "GET", modelsURL, nil)
	if err != nil {
		// Request build failure indicates internal error
		sendOpenAIError(c, http.StatusInternalServerError, "Failed to create request")
		return
	}

	// Set authorization header for upstream API
	req.Header.Set("Authorization", "Bearer "+apiKey)

	// Execute upstream request
	resp, err := client.Do(req)
	if err != nil {
		// Upstream connection failure indicates gateway error
		sendOpenAIError(c, http.StatusBadGateway, "Upstream request failed")
		return
	}
	// Ensure response body is closed
	defer resp.Body.Close()

	// Read entire response body
	body, _ := io.ReadAll(resp.Body)
	// Forward response with original status code and content type
	c.Data(resp.StatusCode, resp.Header.Get("Content-Type"), body)
}

// resolveAPIKey extracts the API key from the Authorization header
// or falls back to the configured key.
// This allows per-request authentication override.
//
// @param c - Gin context containing request headers.
// @return API key string, or empty string if none available.
//
// @pre c != nil
// @post Returns key from Bearer header if present.
// @post Falls back to configured key if no Bearer header.
func (h *ModelsHandler) resolveAPIKey(c *gin.Context) string {
	// Check for Bearer token in Authorization header
	auth := c.GetHeader("Authorization")
	// Extract token after "Bearer " prefix if present
	if strings.HasPrefix(auth, "Bearer ") {
		return strings.TrimPrefix(auth, "Bearer ")
	}
	// Fall back to configured API key
	return h.cfg.GetOpenAIUpstreamAPIKey()
}

// buildModelsURL constructs the models endpoint URL from the configured upstream URL.
// It replaces the chat/completions path with models.
//
// @return URL string for the models endpoint.
//
// @pre h.cfg.GetOpenAIUpstreamURL() ends with "chat/completions" or similar path.
// @post Returned URL ends with "models" path.
func (h *ModelsHandler) buildModelsURL() string {
	url := h.cfg.GetOpenAIUpstreamURL()
	// Replace chat/completions path with models
	// This assumes upstream URL is configured to the completions endpoint
	return strings.TrimSuffix(url, "chat/completions") + "models"
}
