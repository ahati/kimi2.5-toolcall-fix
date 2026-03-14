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

// MessagesHandler handles native Anthropic messages API requests.
// It implements the Handler interface for the /v1/messages endpoint,
// forwarding requests directly to an Anthropic upstream API.
//
// This handler:
//   - Accepts requests in Anthropic Messages format
//   - Passes through requests without transformation (upstream is Anthropic)
//   - Returns responses in Anthropic format
//   - Supports streaming responses with tool use handling
//
// @note This handler uses Anthropic-specific headers for API versioning.
type MessagesHandler struct {
	// cfg contains the application configuration including upstream URL and API key.
	// Must not be nil after construction.
	cfg *config.Config
}

// NewMessagesHandler creates a Gin handler for the /v1/messages endpoint.
//
// @param cfg - Application configuration. Must not be nil.
// @return Gin handler function that processes message requests.
//
// @pre cfg != nil
// @pre cfg.AnthropicUpstreamURL != ""
// @pre cfg.AnthropicAPIKey != "" (or valid X-Api-Key header expected)
func NewMessagesHandler(cfg *config.Config) gin.HandlerFunc {
	return Handle(&MessagesHandler{cfg: cfg})
}

// ValidateRequest performs no additional validation for messages requests.
// Anthropic format requests are passed through as-is for upstream validation.
//
// @param body - Raw request body bytes (unused).
// @return Always returns nil (no validation performed).
//
// @note This implementation trusts the client to send valid Anthropic format.
// @note Upstream API will reject invalid requests.
func (h *MessagesHandler) ValidateRequest(body []byte) error {
	// No validation - pass through to upstream for validation
	// This allows the proxy to be transparent and let upstream handle errors
	return nil
}

// TransformRequest returns the body unchanged as Anthropic format is used directly.
// The upstream API expects Anthropic format, so no transformation is needed.
//
// @param body - Raw request body in Anthropic Messages format.
// @return The same body bytes unchanged.
// @return Always nil error (no transformation can fail).
//
// @post Returned body is identical to input body.
func (h *MessagesHandler) TransformRequest(body []byte) ([]byte, error) {
	// Pass through without transformation - upstream is Anthropic-native
	return body, nil
}

// UpstreamURL returns the Anthropic upstream API URL.
//
// @return URL string from configuration for the Anthropic upstream.
//
// @pre h.cfg != nil
// @post URL includes the full path to the messages endpoint.
func (h *MessagesHandler) UpstreamURL() string {
	return h.cfg.AnthropicUpstreamURL()
}

// ResolveAPIKey returns the configured Anthropic API key.
// This key is used for authentication with the Anthropic API.
//
// @param c - Gin context (unused in this implementation).
// @return API key string from configuration.
//
// @pre h.cfg != nil
// @note This implementation uses configured key, not per-request key.
func (h *MessagesHandler) ResolveAPIKey(c *gin.Context) string {
	return h.cfg.AnthropicAPIKey()
}

// ForwardHeaders copies X-*, Anthropic-Version, and Anthropic-Beta headers
// to the upstream request. These headers are required for proper Anthropic API operation.
//
// The forwarded headers include:
//   - X-* headers: Custom headers for vendor-specific functionality
//   - Anthropic-Version: Required header specifying API version
//   - Anthropic-Beta: Optional header for beta features access
//
// @param c - Gin context containing the original request headers.
// @param req - Upstream request to receive forwarded headers.
//
// @pre c != nil, req != nil
// @post All X-* headers are copied to upstream request.
// @post Anthropic-Version header is copied if present.
// @post Anthropic-Beta header is copied if present.
func (h *MessagesHandler) ForwardHeaders(c *gin.Context, req *http.Request) {
	// Forward headers that are important for Anthropic API
	for k, v := range c.Request.Header {
		// Forward custom headers with X- prefix
		// Forward Anthropic-Version which is required for API versioning
		// Forward Anthropic-Beta for beta feature access
		if strings.HasPrefix(k, "X-") || k == "Anthropic-Version" || k == "Anthropic-Beta" {
			req.Header[k] = v
		}
	}
}

// CreateTransformer builds an Anthropic SSE transformer for the response stream.
// The transformer converts embedded tool call markup to proper Anthropic tool_use events.
//
// @param w - Writer to receive transformed output.
// @return Transformer for processing SSE events with tool call conversion.
//
// @pre w != nil and ready to receive writes.
// @post Caller must call Close() on returned transformer.
//
// @note Kimi K2.5 embeds tool calls in reasoning content using proprietary markup:
//
//	<|tool_calls_section_begin|><|tool_call_begin|>name<|tool_call_argument_begin|>args<|tool_call_end|>
//	This transformer converts that markup to proper Anthropic tool_use content blocks.
func (h *MessagesHandler) CreateTransformer(w io.Writer) transform.SSETransformer {
	return toolcall.NewAnthropicTransformer(w)
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
func (h *MessagesHandler) WriteError(c *gin.Context, status int, msg string) {
	sendAnthropicError(c, status, msg)
}
