package handlers

import (
	"io"
	"net/http"
	"strings"

	"ai-proxy/config"
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
//   - Passes through requests without transformation (upstream is OpenAI-compatible)
//   - Returns responses in OpenAI format
//   - Supports streaming responses with tool call handling
//
// @note This handler does not validate streaming flag as non-streaming is allowed.
type CompletionsHandler struct {
	// cfg contains the application configuration including upstream URL and API key.
	// Must not be nil after construction.
	cfg *config.Config
}

// NewCompletionsHandler creates a Gin handler for the /v1/chat/completions endpoint.
//
// @param cfg - Application configuration. Must not be nil.
// @return Gin handler function that processes completion requests.
//
// @pre cfg != nil
// @pre cfg.OpenAIUpstreamURL != ""
// @pre cfg.OpenAIUpstreamAPIKey != "" (or valid Authorization header expected)
func NewCompletionsHandler(cfg *config.Config) gin.HandlerFunc {
	return Handle(&CompletionsHandler{cfg: cfg})
}

// ValidateRequest performs no additional validation for completions requests.
// OpenAI format requests are passed through as-is for upstream validation.
//
// @param body - Raw request body bytes (unused).
// @return Always returns nil (no validation performed).
//
// @note This implementation trusts the client to send valid OpenAI format.
// @note Upstream API will reject invalid requests.
func (h *CompletionsHandler) ValidateRequest(body []byte) error {
	// No validation - pass through to upstream for validation
	// This allows the proxy to be transparent and let upstream handle errors
	return nil
}

// TransformRequest returns the body unchanged as OpenAI format is used directly.
// The upstream API expects OpenAI format, so no transformation is needed.
//
// @param body - Raw request body in OpenAI ChatCompletion format.
// @return The same body bytes unchanged.
// @return Always nil error (no transformation can fail).
//
// @post Returned body is identical to input body.
func (h *CompletionsHandler) TransformRequest(body []byte) ([]byte, error) {
	// Pass through without transformation - upstream is OpenAI-compatible
	return body, nil
}

// UpstreamURL returns the OpenAI-compatible upstream API URL.
//
// @return URL string from configuration for the OpenAI-compatible upstream.
//
// @pre h.cfg != nil
// @post URL includes the full path to the chat completions endpoint.
func (h *CompletionsHandler) UpstreamURL() string {
	return h.cfg.OpenAIUpstreamURL
}

// ResolveAPIKey returns the configured OpenAI upstream API key.
// This key is used for authentication with the upstream API.
//
// @param c - Gin context (unused in this implementation).
// @return API key string from configuration.
//
// @pre h.cfg != nil
// @note This implementation uses configured key, not per-request key.
func (h *CompletionsHandler) ResolveAPIKey(c *gin.Context) string {
	return h.cfg.OpenAIUpstreamAPIKey
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
// The transformer handles tool call processing and SSE event formatting.
//
// @param w - Writer to receive transformed output.
// @return Transformer for processing OpenAI-format SSE events.
//
// @pre w != nil and ready to receive writes.
// @post Caller must call Close() on returned transformer.
func (h *CompletionsHandler) CreateTransformer(w io.Writer) transform.SSETransformer {
	// Create transformer for OpenAI format output
	// Empty prefix strings indicate no additional ID prefix needed
	return toolcall.NewOpenAITransformer(w, "", "")
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
