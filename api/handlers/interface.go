// Package handlers provides HTTP request handlers for the AI proxy endpoints.
// This file defines the core Handler interface that all request handlers must implement.
package handlers

import (
	"io"
	"net/http"

	"ai-proxy/router"
	"ai-proxy/transform"

	"github.com/gin-gonic/gin"
)

// Handler defines the interface for processing API requests through the proxy pipeline.
// Implementations handle validation, transformation, upstream communication, and response streaming.
//
// The Handler interface follows the Strategy pattern, allowing different endpoints
// to customize their behavior while sharing common proxy logic via the Handle() function.
//
// Contract for implementations:
//   - ValidateRequest should return nil for valid requests, error for invalid ones
//   - TransformRequest may modify the request body or pass it through unchanged
//   - UpstreamURL must return a valid, absolute URL with scheme
//   - ResolveAPIKey may return empty string if no key is required
//   - ForwardHeaders should not overwrite essential headers (Content-Type, Content-Length)
//   - CreateTransformer must return a non-nil transformer that properly handles SSE format
//   - WriteError should write a complete error response and not panic
//
// @note All methods must be safe to call concurrently from multiple goroutines.
// @note Implementations should not hold state between requests.
type Handler interface {
	// ValidateRequest checks if the request body is valid for this handler's endpoint.
	// This method should perform format and semantic validation appropriate for the
	// request type (e.g., required fields, valid values, logical constraints).
	//
	// @param body - Raw request body bytes. May be empty but not nil.
	// @return nil if valid, error describing validation failure otherwise.
	//
	// @pre body is the complete request body (not partial).
	// @post If error is returned, request should be rejected with appropriate status code.
	ValidateRequest(body []byte) error

	// TransformRequest converts the request body to the upstream format.
	// For endpoints that accept the same format as upstream, this may return body unchanged.
	// For bridging endpoints, this performs format conversion (e.g., Anthropic to OpenAI).
	//
	// @param body - Raw request body bytes in downstream format.
	// @return Transformed body in upstream format, or error if transformation fails.
	//
	// @pre body has passed ValidateRequest (implementation may assume validity).
	// @post Returned body is valid JSON for the upstream API.
	TransformRequest(body []byte) ([]byte, error)

	// UpstreamURL returns the target URL for the upstream API.
	// This URL includes scheme, host, and path for the specific API endpoint.
	//
	// @return Absolute URL string including scheme (http:// or https://).
	//
	// @note Must return same URL for same configuration (no dynamic behavior).
	UpstreamURL() string

	// ResolveAPIKey extracts or determines the API key to use for the upstream request.
	// May extract from request headers or use configured credentials.
	//
	// @param c - Gin context for the current request.
	// @return API key string, may be empty if no key is required/available.
	//
	// @note Empty string may cause authentication failure with upstream.
	ResolveAPIKey(c *gin.Context) string

	// ForwardHeaders copies relevant headers from the incoming request to the upstream request.
	// Should forward custom headers, authentication overrides, and protocol-specific headers.
	// Should NOT forward hop-by-hop headers (Connection, Keep-Alive, etc.).
	//
	// @param c - Gin context for the current request (source of headers).
	// @param req - Upstream request to receive forwarded headers (destination).
	//
	// @pre c != nil, req != nil.
	// @post req.Header contains appropriate forwarded headers.
	ForwardHeaders(c *gin.Context, req *http.Request)

	// CreateTransformer builds an SSE transformer for converting upstream responses.
	// The transformer handles format conversion and SSE event processing.
	//
	// @param w - Writer to receive transformed output (typically response writer).
	// @return Transformer instance for processing SSE events.
	//
	// @pre w != nil and is ready to receive writes.
	// @post Caller must call Close() on returned transformer when done.
	CreateTransformer(w io.Writer) transform.SSETransformer

	// WriteError sends an error response to the client in the appropriate format.
	// The format should match the API style (OpenAI vs Anthropic) for consistency.
	//
	// @param c - Gin context for writing the response.
	// @param status - HTTP status code for the error response.
	// @param msg - Human-readable error message.
	//
	// @pre c != nil and response has not been written yet.
	// @post Response has been fully written; no further writes should occur.
	WriteError(c *gin.Context, status int, msg string)
}

type RoutingHandler interface {
	Handler
	ExtractModel(body []byte) (string, error)
	TransformRequestWithRoute(body []byte, route *router.ResolvedRoute) ([]byte, error)
	UpstreamURLWithRoute(route *router.ResolvedRoute) string
	ResolveAPIKeyWithRoute(route *router.ResolvedRoute) string
	ForwardHeadersWithRoute(c *gin.Context, req *http.Request, route *router.ResolvedRoute)
	CreateTransformerWithRoute(w io.Writer, route *router.ResolvedRoute) transform.SSETransformer
}
