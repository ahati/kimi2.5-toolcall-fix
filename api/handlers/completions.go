package handlers

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"ai-proxy/config"
	"ai-proxy/router"
	"ai-proxy/transform"
	"ai-proxy/transform/toolcall"

	"github.com/gin-gonic/gin"
)

// CompletionsHandler handles OpenAI-compatible chat completion requests.
// It implements the Handler interface for the /v1/chat/completions endpoint,
// forwarding requests directly to an OpenAI-compatible upstream API.
//
// This handler:
//   - Accepts requests in OpenAI ChatCompletion format
//   - Routes to the appropriate upstream based on model configuration
//   - Returns responses in OpenAI format
//   - Supports streaming responses with tool call handling
//
// @note This handler does not validate streaming flag as non-streaming is allowed.
type CompletionsHandler struct {
	// cfg contains the application configuration including upstream URL and API key.
	// Must not be nil after construction.
	cfg *config.Config
	// router resolves model names to providers. May be nil for legacy behavior.
	modelRouter router.Router
	// route is the resolved route for the current request.
	route *router.ResolvedRoute
}

// NewCompletionsHandler creates a Gin handler for the /v1/chat/completions endpoint.
//
// @param cfg - Application configuration. Must not be nil.
// @return Gin handler function that processes completion requests.
//
// @pre cfg != nil
func NewCompletionsHandler(cfg *config.Config, r router.Router) gin.HandlerFunc {
	return Handle(&CompletionsHandler{
		cfg:         cfg,
		modelRouter: r,
	})
}

// ValidateRequest validates the request and resolves the model route.
// It parses the request to extract the model name and resolves it to a provider.
//
// @param body - Raw request body bytes.
// @return Error if JSON parsing fails or model cannot be resolved.
func (h *CompletionsHandler) ValidateRequest(body []byte) error {
	// If no router, use legacy behavior
	if h.modelRouter == nil {
		return nil
	}

	// Parse to get model name
	var req struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return nil // Let upstream handle invalid JSON
	}

	if req.Model == "" {
		return nil // Let upstream handle missing model
	}

	// Resolve the model to a route
	route, err := h.modelRouter.Resolve(req.Model)
	if err != nil {
		return nil // Use fallback behavior
	}

	// Only accept OpenAI providers for this endpoint
	if route.Provider.Type != "openai" {
		return nil // Fall back to legacy behavior
	}

	h.route = route
	return nil
}

// TransformRequest updates the model name if a route was resolved.
//
// @param body - Raw request body in OpenAI ChatCompletion format.
// @return The body with updated model name if resolved.
func (h *CompletionsHandler) TransformRequest(body []byte) ([]byte, error) {
	// If no route resolved, pass through
	if h.route == nil {
		return body, nil
	}

	// Update model name in request
	var req map[string]interface{}
	if err := json.Unmarshal(body, &req); err != nil {
		return body, nil
	}
	req["model"] = h.route.Model
	return json.Marshal(req)
}

// UpstreamURL returns the upstream API URL.
// Uses the resolved provider URL if available, otherwise falls back to config.
//
// @return URL string for the upstream API endpoint.
func (h *CompletionsHandler) UpstreamURL() string {
	if h.route != nil {
		url := h.route.Provider.BaseURL
		if !strings.HasSuffix(url, "/chat/completions") {
			url = strings.TrimSuffix(url, "/") + "/chat/completions"
		}
		return url
	}
	return h.cfg.GetOpenAIUpstreamURL()
}

// ResolveAPIKey returns the API key for the upstream.
// Uses the resolved provider key if available, otherwise falls back to config.
//
// @param c - Gin context (unused in this implementation).
// @return API key string.
func (h *CompletionsHandler) ResolveAPIKey(c *gin.Context) string {
	if h.route != nil {
		return h.route.Provider.GetAPIKey()
	}
	return h.cfg.GetOpenAIUpstreamAPIKey()
}

// ForwardHeaders copies custom headers (X-*) and the Extra header to the upstream request.
// Custom headers allow clients to pass additional context to the upstream API.
//
// @param c - Gin context containing the original request headers.
// @param req - Upstream request to receive forwarded headers.
//
// @pre c != nil, req != nil
// @post All X-* headers are copied to upstream request.
// @post Extra header is copied if present.
func (h *CompletionsHandler) ForwardHeaders(c *gin.Context, req *http.Request) {
	// Forward all custom headers with X- prefix
	// These may include vendor-specific or custom metadata
	forwardCustomHeaders(c, req, "X-")
	// Forward Extra header which may contain additional parameters
	req.Header.Set("Extra", c.Request.Header.Get("Extra"))
}

// CreateTransformer builds an OpenAI SSE transformer for the response stream.
// The transformer converts embedded tool call markup to proper tool_calls format.
//
// @param w - Writer to receive transformed output.
// @return Transformer for processing SSE events with tool call conversion.
//
// @pre w != nil and ready to receive writes.
// @post Caller must call Close() on returned transformer.
//
// @note Kimi K2.5 and similar models embed tool calls in reasoning content
//
//	using proprietary markup: <|tool_calls_section_begin|>...
//	This transformer converts that markup to proper OpenAI tool_calls format.
func (h *CompletionsHandler) CreateTransformer(w io.Writer) transform.SSETransformer {
	return toolcall.NewOpenAITransformer(w)
}

// WriteError sends an error response in OpenAI format.
// Maintains consistency with OpenAI API error responses.
//
// @param c - Gin context for writing the response.
// @param status - HTTP status code for the error.
// @param msg - Human-readable error message.
//
// @pre c != nil and response not yet written.
// @post OpenAI-format error response is written.
func (h *CompletionsHandler) WriteError(c *gin.Context, status int, msg string) {
	sendOpenAIError(c, status, msg)
}

// forwardCustomHeaders copies headers matching any of the given prefixes
// from the incoming request to the upstream request.
// This allows forwarding vendor-specific or custom headers.
//
// @param c - Gin context containing the original request.
// @param req - Upstream request to receive forwarded headers.
// @param prefixes - Header prefixes to match (e.g., "X-", "Custom-").
//
// @pre c != nil, req != nil
// @post All headers matching any prefix are copied to upstream request.
func forwardCustomHeaders(c *gin.Context, req *http.Request, prefixes ...string) {
	// Iterate over all headers in the original request
	for key, values := range c.Request.Header {
		// Check if header matches any of the specified prefixes
		for _, prefix := range prefixes {
			// Case-sensitive prefix match
			if strings.HasPrefix(key, prefix) {
				// Copy all values for matching header
				req.Header[key] = values
				// Break inner loop since header matched
				break
			}
		}
	}
}
