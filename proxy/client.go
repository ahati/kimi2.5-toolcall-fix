// Package proxy provides an HTTP client for making requests to upstream LLM APIs.
// This package implements a thread-safe client with configurable timeouts and connection pooling.
package proxy

import (
	"bytes"
	"context"
	"crypto/tls"
	"fmt"
	"net"
	"net/http"
	"strings"
	"time"

	"ai-proxy/capture"
	"ai-proxy/logging"
)

// Client is an HTTP client for communicating with upstream LLM APIs.
// It provides connection pooling, timeout management, and request capture capabilities.
// The client is safe for concurrent use by multiple goroutines.
type Client struct {
	// baseURL is the target URL for upstream API requests.
	// Must be a valid URL with scheme (http or https).
	baseURL string
	// apiKey is the authentication token for the upstream API.
	// Must not be empty for authenticated requests.
	apiKey string
	// httpClient is the underlying HTTP client for making requests.
	// Configured with connection pooling and TLS settings.
	httpClient *http.Client
}

// ClientOption is a functional option for configuring a Client.
// It follows the functional options pattern for flexible client configuration.
// @param c - pointer to the Client instance to configure
type ClientOption func(*Client)

// NewClient creates a new proxy client with the given base URL and API key.
// It accepts optional ClientOption functions for additional configuration.
//
// @param baseURL - the base URL of the upstream API (must be a valid URL string)
// @param apiKey - the API key for authentication (must not be empty for authenticated requests)
// @param opts - optional ClientOption functions to customize the client
// @return *Client - a configured Client instance ready for use
// @pre baseURL is a valid URL string with scheme
// @pre apiKey is not empty for authenticated endpoints
// @post returned client is initialized with sensible defaults
// @note The client uses TLS 1.2 minimum for secure connections
func NewClient(baseURL, apiKey string, opts ...ClientOption) *Client {
	c := &Client{
		baseURL: baseURL,
		apiKey:  apiKey,
		httpClient: &http.Client{
			Transport: defaultTransport(), // Use secure transport with connection pooling
		},
	}
	// Apply functional options in order, later options override earlier ones
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// WithTimeout returns a ClientOption that sets the HTTP client timeout.
//
// @param d - the timeout duration for HTTP requests
// @return ClientOption - a functional option to set the timeout
// @pre d must be greater than zero for meaningful timeout behavior
// @post Client will timeout after the specified duration
// @note A zero or negative duration results in no timeout
func WithTimeout(d time.Duration) ClientOption {
	return func(c *Client) {
		c.httpClient.Timeout = d
	}
}

// WithTransport returns a ClientOption that sets a custom HTTP transport.
//
// @param t - the custom HTTP transport to use
// @return ClientOption - a functional option to set the transport
// @pre t must not be nil
// @post Client will use the provided transport for all requests
// @note This overrides the default transport settings
func WithTransport(t *http.Transport) ClientOption {
	return func(c *Client) {
		c.httpClient.Transport = t
	}
}

// defaultTransport creates an HTTP transport with sensible defaults for upstream connections.
// It configures connection pooling, timeouts, and TLS settings for optimal performance and security.
//
// @return *http.Transport - a configured transport ready for use
// @post Transport is configured with TLS 1.2 minimum
// @post Connection pool is configured for up to 100 idle connections
// @note Settings are optimized for long-lived connections to LLM APIs
func defaultTransport() *http.Transport {
	return &http.Transport{
		Proxy: http.ProxyFromEnvironment, // Respect HTTP_PROXY environment variables
		DialContext: (&net.Dialer{
			Timeout:   30 * time.Second, // Connection establishment timeout
			KeepAlive: 30 * time.Second, // TCP keepalive interval for connection health
		}).DialContext,
		MaxIdleConns:          100,              // Maximum idle connections across all hosts
		IdleConnTimeout:       90 * time.Second, // Idle connection timeout before closure
		TLSHandshakeTimeout:   10 * time.Second, // TLS handshake timeout
		ExpectContinueTimeout: 1 * time.Second,  // Timeout for 100-continue response
		TLSClientConfig: &tls.Config{
			MinVersion: tls.VersionTLS12, // Enforce minimum TLS 1.2 for security
		},
	}
}

// BuildRequest creates an HTTP POST request with the given context and body.
// It records the request in the capture context if one exists.
//
// @param ctx - the context for the request, used for cancellation and capture
// @param body - the request body as raw bytes (must be valid JSON for API requests)
// @return *http.Request - the constructed HTTP request
// @return error - non-nil if request creation fails
// @pre body is not nil and contains valid request data
// @post Request has POST method and correct URL
// @post Request body is set as a byte reader for streaming
// @note The body reader can only be consumed once
func (c *Client) BuildRequest(ctx context.Context, body []byte) (*http.Request, error) {
	// Create request with context for cancellation support
	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	// Capture the request for debugging if capture context is available
	if cc := capture.GetCaptureContext(ctx); cc != nil {
		// Use thread-safe method to record upstream request
		// Headers will be recorded separately in SetHeaders
		cc.Recorder.RecordUpstreamRequest(nil, body)
	}

	return req, nil
}

// SetHeaders sets the required headers on the request including Content-Type, Authorization, and Accept headers.
// It also records sanitized headers in the capture context if one exists.
//
// @param req - the HTTP request to modify
// @pre req must not be nil
// @post Request has Content-Type: application/json
// @post Request has Authorization: Bearer <apiKey>
// @post Request has Accept: text/event-stream for SSE support
// @note Authorization header value is sanitized in capture logs
func (c *Client) SetHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Accept", "text/event-stream") // Required for streaming responses

	// Record headers for debugging, sanitizing sensitive values
	if cc := capture.GetCaptureContext(req.Context()); cc != nil {
		// Re-record the request with headers now that they're set
		// This overwrites the previous recording with complete header information
		cc.Recorder.RecordUpstreamRequest(req.Header, nil)
	}
}

// Do executes the HTTP request and returns the response.
// It logs the request and captures response metadata for debugging.
//
// @param req - the HTTP request to execute
// @return *http.Response - the HTTP response from the upstream server
// @return error - non-nil if the request fails (network error, timeout, etc.)
// @pre req must be properly constructed with headers set
// @post Response body must be closed by the caller
// @post Response metadata is captured for debugging if capture context exists
// @note Does not check response status code; caller must handle non-2xx responses
func (c *Client) Do(req *http.Request) (*http.Response, error) {
	logging.InfoMsg("Sending request to upstream: %s", c.baseURL)
	resp, err := c.httpClient.Do(req)
	if err != nil {
		logging.ErrorMsg("Upstream request failed: %v", err)
		return nil, fmt.Errorf("upstream request: %w", err)
	}

	// Initialize response capture structure for SSE chunks
	if cc := capture.GetCaptureContext(req.Context()); cc != nil {
		// Use thread-safe method to record upstream response
		cc.Recorder.RecordUpstreamResponse(resp.StatusCode, resp.Header)
	}

	return resp, nil
}

// Close closes idle connections held by the HTTP client.
// This should be called when the client is no longer needed to release resources.
//
// @pre Client is no longer in use
// @post All idle connections are closed
// @note Active connections are not affected; they close when their requests complete
func (c *Client) Close() {
	c.httpClient.CloseIdleConnections()
}

// GetAPIKey returns the API key to use for the request.
// If clientAuth is provided with a Bearer prefix, it extracts and returns that token.
// Otherwise, it returns the client's configured API key.
//
// @param clientAuth - the Authorization header value from the client request
// @return string - the API key to use for authentication
// @pre If clientAuth is provided, it should be in "Bearer <token>" format
// @post Returned key is safe to use in Authorization header
// @note Allows per-request authentication override via client-provided token
func (c *Client) GetAPIKey(clientAuth string) string {
	// Check if client provided their own Bearer token
	if strings.HasPrefix(clientAuth, "Bearer ") {
		return strings.TrimPrefix(clientAuth, "Bearer ")
	}
	// Fall back to client's configured API key
	return c.apiKey
}
