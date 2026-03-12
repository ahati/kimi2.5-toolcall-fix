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
	"ai-proxy/transform/toolcall"
	"ai-proxy/types"

	"github.com/gin-gonic/gin"
)

// OpenAIAdapter implements ProtocolAdapter for OpenAI-compatible API format.
// This adapter passes requests through unchanged to an OpenAI-compatible upstream.
//
// @brief Adapter for OpenAI-compatible API format.
//
// @note Requests are passed through without transformation.
// @note Forwards X-* headers and Extra header to upstream.
type OpenAIAdapter struct{}

// NewOpenAIAdapter creates a new OpenAI adapter instance.
//
// @brief    Creates a new OpenAIAdapter instance.
// @return   Pointer to newly created OpenAIAdapter.
func NewOpenAIAdapter() *OpenAIAdapter {
	return &OpenAIAdapter{}
}

// TransformRequest returns the request body unchanged.
//
// @brief    Passes through request body without transformation.
// @param    body The original request body.
// @return   The same body slice unchanged.
// @return   Always returns nil error.
//
// @note     OpenAI format requires no transformation for OpenAI upstream.
func (a *OpenAIAdapter) TransformRequest(body []byte) ([]byte, error) {
	return body, nil
}

// ValidateRequest checks if the request is a streaming request.
//
// @brief    Validates that the request is a streaming request.
// @param    body The request body to validate.
// @return   Error if request is non-streaming, nil otherwise.
//
// @note     Non-streaming requests are not supported.
func (a *OpenAIAdapter) ValidateRequest(body []byte) error {
	if !a.IsStreamingRequest(body) {
		return ErrNonStreamingNotSupported
	}
	return nil
}

// CreateTransformer creates a tool call transformer for OpenAI output format.
//
// @brief    Creates transformer for OpenAI-format tool calls.
// @param    w Writer for transformed output.
// @param    base Base stream chunk for context.
// @return   SSETransformer configured for OpenAI output.
func (a *OpenAIAdapter) CreateTransformer(w io.Writer, base types.StreamChunk) SSETransformer {
	output := toolcall.NewOpenAIOutput(w, base)
	return NewToolCallTransformer(w, base, output)
}

// UpstreamURL returns the configured OpenAI upstream URL.
//
// @brief    Gets the OpenAI upstream URL from configuration.
// @param    cfg Application configuration.
// @return   OpenAI upstream URL string.
func (a *OpenAIAdapter) UpstreamURL(cfg *config.Config) string {
	return cfg.OpenAIUpstreamURL
}

// UpstreamAPIKey returns the configured OpenAI API key.
//
// @brief    Gets the OpenAI API key from configuration.
// @param    cfg Application configuration.
// @return   OpenAI API key string.
func (a *OpenAIAdapter) UpstreamAPIKey(cfg *config.Config) string {
	return cfg.OpenAIUpstreamAPIKey
}

// ForwardHeaders copies relevant headers from source to destination.
//
// @brief    Forwards protocol-specific headers to upstream.
// @param    src Source HTTP headers from client request.
// @param    dst Destination HTTP headers for upstream request.
//
// @note     Forwards headers with "X-" prefix and "Extra" header.
// @note     Also forwards Connection, Keep-Alive, Upgrade, and TE headers.
func (a *OpenAIAdapter) ForwardHeaders(src, dst http.Header) {
	for k, v := range src {
		if strings.HasPrefix(k, "X-") || k == "Extra" {
			dst[k] = v
		}
	}
	for _, h := range []string{"Connection", "Keep-Alive", "Upgrade", "TE"} {
		if v := src.Get(h); v != "" {
			dst.Set(h, v)
		}
	}
}

// SendError sends an OpenAI-formatted error response.
//
// @brief    Sends error response in OpenAI format.
// @param    c Gin context for the HTTP response.
// @param    status HTTP status code.
// @param    msg Error message.
//
// @note     Response format matches OpenAI error structure.
func (a *OpenAIAdapter) SendError(c *gin.Context, status int, msg string) {
	logging.ErrorMsg("OpenAI handler error: %s", msg)
	c.JSON(status, gin.H{
		"error": gin.H{
			"message": msg,
			"type":    "invalid_request_error",
			"code":    "",
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
func (a *OpenAIAdapter) IsStreamingRequest(body []byte) bool {
	var req struct {
		Stream bool `json:"stream"`
	}
	json.Unmarshal(body, &req)
	return req.Stream
}

// ErrNonStreamingNotSupported indicates non-streaming requests are not supported.
//
// @brief Error for non-streaming request attempts.
var ErrNonStreamingNotSupported = &ProtocolError{Message: "Non-streaming requests not supported"}

// ProtocolError represents a protocol-level error.
//
// @brief Error type for protocol-specific failures.
type ProtocolError struct {
	Message string
}

// Error returns the protocol error message.
//
// @brief    Returns the error message string.
// @return   The error message.
func (e *ProtocolError) Error() string {
	return e.Message
}
