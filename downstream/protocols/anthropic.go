// Package protocols provides protocol adapters for different API formats.
// Each adapter implements the ProtocolAdapter interface to handle request/response
// transformation for a specific API format (OpenAI, Anthropic, or Bridge).
package protocols

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"ai-proxy/config"
	"ai-proxy/logging"
	"ai-proxy/types"

	"github.com/gin-gonic/gin"
)

// AnthropicAdapter implements ProtocolAdapter for Anthropic API format.
// This adapter passes requests through unchanged to an Anthropic upstream.
//
// @brief Adapter for Anthropic API format.
//
// @note Requests are passed through without transformation.
// @note Forwards X-* headers, Anthropic-Version, and Anthropic-Beta headers.
type AnthropicAdapter struct{}

// NewAnthropicAdapter creates a new Anthropic adapter instance.
//
// @brief    Creates a new AnthropicAdapter instance.
// @return   Pointer to newly created AnthropicAdapter.
func NewAnthropicAdapter() *AnthropicAdapter {
	return &AnthropicAdapter{}
}

// TransformRequest returns the request body unchanged.
//
// @brief    Passes through request body without transformation.
// @param    body The original request body.
// @return   The same body slice unchanged.
// @return   Always returns nil error.
//
// @note     Anthropic format requires no transformation for Anthropic upstream.
func (a *AnthropicAdapter) TransformRequest(body []byte) ([]byte, error) {
	return body, nil
}

// ValidateRequest checks if the request is a streaming request.
//
// @brief    Validates that the request is a streaming request.
// @param    body The request body to validate.
// @return   Error if request is non-streaming, nil otherwise.
//
// @note     Non-streaming requests are not supported.
func (a *AnthropicAdapter) ValidateRequest(body []byte) error {
	if !a.IsStreamingRequest(body) {
		return ErrNonStreamingNotSupported
	}
	return nil
}

// CreateTransformer creates a tool call transformer for Anthropic output format.
//
// @brief    Creates transformer for Anthropic-format tool calls.
// @param    w Writer for transformed output.
// @param    base Base stream chunk for context.
// @return   SSETransformer configured for Anthropic output.
func (a *AnthropicAdapter) CreateTransformer(w io.Writer, base types.StreamChunk) SSETransformer {
	return NewAnthropicEventTransformer(w)
}

// UpstreamURL returns the configured Anthropic upstream URL.
//
// @brief    Gets the Anthropic upstream URL from configuration.
// @param    cfg Application configuration.
// @return   Anthropic upstream URL string.
func (a *AnthropicAdapter) UpstreamURL(cfg *config.Config) string {
	return cfg.AnthropicUpstreamURL
}

// UpstreamAPIKey returns the configured Anthropic API key.
//
// @brief    Gets the Anthropic API key from configuration.
// @param    cfg Application configuration.
// @return   Anthropic API key string.
func (a *AnthropicAdapter) UpstreamAPIKey(cfg *config.Config) string {
	return cfg.AnthropicAPIKey
}

// ForwardHeaders copies relevant headers from source to destination.
//
// @brief    Forwards protocol-specific headers to upstream.
// @param    src Source HTTP headers from client request.
// @param    dst Destination HTTP headers for upstream request.
//
// @note     Forwards headers with "X-" prefix, "Anthropic-Version", and "Anthropic-Beta".
// @note     Also forwards Connection, Keep-Alive, Upgrade, and TE headers.
func (a *AnthropicAdapter) ForwardHeaders(src, dst http.Header) {
	for k, v := range src {
		if strings.HasPrefix(k, "X-") || k == "Anthropic-Version" || k == "Anthropic-Beta" {
			dst[k] = v
		}
	}
	for _, h := range []string{"Connection", "Keep-Alive", "Upgrade", "TE"} {
		if v := src.Get(h); v != "" {
			dst.Set(h, v)
		}
	}
}

// SendError sends an Anthropic-formatted error response.
//
// @brief    Sends error response in Anthropic format.
// @param    c Gin context for the HTTP response.
// @param    status HTTP status code.
// @param    msg Error message.
//
// @note     Response format matches Anthropic error structure.
func (a *AnthropicAdapter) SendError(c *gin.Context, status int, msg string) {
	logging.ErrorMsg("Anthropic handler error: %s", msg)
	c.JSON(status, types.Error{
		Type: "error",
		Error: types.ErrorDetail{
			Type:    "invalid_request_error",
			Message: msg,
		},
	})
}

// IsStreamingRequest checks if the request body indicates streaming.
//
// @brief    Determines if request is a streaming request.
// @param    body The request body to check.
// @return   True if stream field is true, false otherwise.
//
// @note     Silently returns false if body cannot be parsed.
func (a *AnthropicAdapter) IsStreamingRequest(body []byte) bool {
	var req struct {
		Stream bool `json:"stream"`
	}
	json.Unmarshal(body, &req)
	return req.Stream
}
