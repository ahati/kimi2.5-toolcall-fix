// Package upstream provides HTTP client functionality for communicating with
// upstream LLM APIs. It defines a Client interface for testability and provides
// both a real HTTP implementation and a mock for testing.
package upstream

import (
	"bytes"
	"context"
	"fmt"
	"net/http"
	"strings"

	"ai-proxy/logging"
)

// Client defines the interface for making upstream HTTP requests.
// This interface allows for easy mocking in tests and provides a clean
// abstraction over HTTP client operations.
//
// @brief Interface for upstream HTTP operations.
type Client interface {
	// Do executes an HTTP request against the upstream API.
	//
	// @brief    Executes HTTP request to upstream API.
	// @param    ctx Context for request cancellation and timeout.
	// @param    req The HTTP request to execute.
	// @return   HTTP response from upstream.
	// @return   Error if request fails or context is cancelled.
	Do(ctx context.Context, req *http.Request) (*http.Response, error)
	// Close releases resources held by the client.
	//
	// @brief    Releases client resources.
	// @note     Should be called when client is no longer needed.
	Close()
}

// HTTPClient implements the Client interface for making real HTTP requests
// to upstream LLM APIs. It manages connection pooling and request building.
//
// @brief HTTP client for upstream API communication.
type HTTPClient struct {
	// URL is the upstream API endpoint URL.
	URL string
	// APIKey is the default authentication key for the upstream API.
	APIKey string
	// Client is the underlying HTTP client with connection pooling.
	Client *http.Client
}

// NewClient creates a new HTTPClient configured for upstream API communication.
//
// @brief    Creates a new HTTP client for upstream requests.
// @param    url The upstream API endpoint URL.
// @param    apiKey The default API key for authentication.
// @return   Pointer to newly created HTTPClient instance.
//
// @note     Client is created with no timeout to support long-running SSE streams.
// @note     Connection pooling is managed by the underlying http.Client.
func NewClient(url, apiKey string) *HTTPClient {
	return &HTTPClient{
		URL:    url,
		APIKey: apiKey,
		Client: &http.Client{Timeout: 0},
	}
}

// BuildRequest constructs an HTTP POST request for the upstream API.
//
// @brief    Constructs HTTP request with context and body.
// @param    ctx Context for request lifecycle management.
// @param    body Request body as JSON bytes.
// @return   Constructed HTTP request ready for headers.
// @return   Error if request construction fails.
//
// @note     Captures request body in logging context if capture is enabled.
// @note     Caller must set appropriate headers before sending.
//
// @pre      Body must contain valid JSON for the target API.
// @post     Request is associated with context for cancellation.
func (c *HTTPClient) BuildRequest(ctx context.Context, body []byte) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, "POST", c.URL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	if cc := logging.GetCaptureContext(ctx); cc != nil {
		cc.Recorder.UpstreamRequest = &logging.HTTPRequestCapture{
			At:      cc.StartTime,
			Body:    body,
			RawBody: body,
		}
	}

	return req, nil
}

// SetHeaders configures required HTTP headers for the upstream request.
//
// @brief    Sets authentication and content headers on request.
// @param    req The HTTP request to modify.
//
// @note     Sets Content-Type, Authorization, and Accept headers.
// @note     Captures sanitized headers in logging context if enabled.
//
// @pre      Request must have been created with BuildRequest.
// @post     Request has all headers required for upstream API.
func (c *HTTPClient) SetHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.APIKey)
	req.Header.Set("Accept", "text/event-stream")

	if cc := logging.GetCaptureContext(req.Context()); cc != nil && cc.Recorder.UpstreamRequest != nil {
		cc.Recorder.UpstreamRequest.Headers = logging.SanitizeHeaders(req.Header)
	}
}

// GetAPIKey determines the appropriate API key to use for authentication.
//
// @brief    Resolves API key from client auth header or default.
// @param    clientAuth The Authorization header value from client request.
// @return   API key to use for upstream authentication.
//
// @note     Extracts Bearer token if client provided one.
// @note     Falls back to default APIKey if no client auth provided.
//
// @pre      clientAuth may be empty or "Bearer <token>" format.
func (c *HTTPClient) GetAPIKey(clientAuth string) string {
	if strings.HasPrefix(clientAuth, "Bearer ") {
		return strings.TrimPrefix(clientAuth, "Bearer ")
	}
	return c.APIKey
}

// Do executes an HTTP request against the upstream API.
//
// @brief    Executes HTTP request to upstream API.
// @param    ctx Context for request cancellation and timeout.
// @param    req The HTTP request to execute.
// @return   HTTP response from upstream.
// @return   Error if request fails or context is cancelled.
//
// @note     Response body must be closed by caller.
// @note     Captures response in logging context if available.
//
// @pre      Request must have valid URL and headers set.
// @post     Response is captured in context for logging if capture is enabled.
func (c *HTTPClient) Do(ctx context.Context, req *http.Request) (*http.Response, error) {
	logging.InfoMsg("Sending request to upstream: %s", c.URL)
	resp, err := c.Client.Do(req.WithContext(ctx))
	if err != nil {
		logging.ErrorMsg("Upstream request failed: %v", err)
		return nil, fmt.Errorf("upstream request: %w", err)
	}

	if cc := logging.GetCaptureContext(ctx); cc != nil {
		cc.Recorder.UpstreamResponse = &logging.SSEResponseCapture{
			StatusCode: resp.StatusCode,
			Headers:    logging.SanitizeHeaders(resp.Header),
			Chunks:     []logging.SSEChunk{},
		}
	}

	return resp, nil
}

// Close releases resources held by the HTTPClient.
//
// @brief    Releases HTTP client resources.
//
// @note     Closes idle connections in the connection pool.
// @note     Should be called when client is no longer needed.
//
// @post     All idle connections are closed.
func (c *HTTPClient) Close() {
	c.Client.CloseIdleConnections()
}
