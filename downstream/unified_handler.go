// Package downstream provides HTTP handlers for the proxy's client-facing API endpoints.
// It implements a unified stream handler that works with protocol adapters to support
// multiple API formats (OpenAI, Anthropic, Bridge).
package downstream

import (
	"fmt"
	"io"
	"net/http"

	"ai-proxy/config"
	"ai-proxy/downstream/protocols"
	"ai-proxy/logging"
	"ai-proxy/types"
	"ai-proxy/upstream"

	"github.com/gin-gonic/gin"
	"github.com/tmaxmax/go-sse"
)

// readBody reads the entire request body from the Gin context.
//
// @brief    Reads and returns the raw bytes of the HTTP request body.
// @param    c Gin context containing the HTTP request.
// @return   Byte slice containing the request body content.
// @return   Error if reading the body fails.
//
// @note     The request body is consumed and cannot be read again.
// @note     Returns io.EOF error if the body is empty.
func readBody(c *gin.Context) ([]byte, error) {
	return io.ReadAll(c.Request.Body)
}

// handleUpstreamError processes and responds with an upstream API error.
//
// @brief    Logs an upstream error and sends a JSON error response to the client.
// @param    c Gin context for writing the HTTP response.
// @param    resp HTTP response received from the upstream server.
//
// @note     The response body is fully consumed and closed by this function.
// @note     Error is logged before sending the response.
// @note     Response format follows OpenAI error structure with message, type, and code.
func handleUpstreamError(c *gin.Context, resp *http.Response) {
	// Read and consume the error response body to ensure the connection can be reused
	// Ignoring read error here since we're already in an error path
	body, _ := io.ReadAll(resp.Body)
	msg := fmt.Sprintf("Upstream error: %s", string(body))
	logging.ErrorMsg("%s", msg)
	// Use BadGateway (502) to indicate the proxy received an invalid response from upstream
	// The error structure follows OpenAI's format for consistency across all protocols
	c.JSON(http.StatusBadGateway, gin.H{
		"error": gin.H{
			"message": msg,
			"type":    "upstream_error",
			"code":    fmt.Sprintf("status_%d", resp.StatusCode),
		},
	})
}

// StreamHandler creates a Gin handler function for streaming chat completions.
//
// @brief    Creates a streaming handler with protocol-specific behavior.
// @param    cfg Application configuration containing upstream settings.
// @param    adapter Protocol adapter for request/response transformation.
// @return   Gin handler function for streaming responses.
//
// @note     The handler validates streaming requests and rejects non-streaming.
// @note     Responses are transformed using the adapter's CreateTransformer method.
//
// @pre      cfg must contain valid upstream URL and API key.
// @pre      adapter must not be nil.
// @post     Response is streamed to client with tool calls transformed.
func StreamHandler(cfg *config.Config, adapter protocols.ProtocolAdapter) gin.HandlerFunc {
	return func(c *gin.Context) {
		// Read body first - must be done before any validation or transformation
		// The body stream can only be read once, so we buffer it entirely
		body, err := readBody(c)
		if err != nil {
			adapter.SendError(c, http.StatusBadRequest, "Failed to read request body")
			return
		}

		// Record the incoming request for debugging/audit before any processing
		// This captures the original client request before protocol-specific transformations
		logging.RecordDownstreamRequest(c, body)

		// Reject non-streaming requests early - the proxy is optimized for streaming
		// and does not implement buffering for synchronous responses
		if !adapter.IsStreamingRequest(body) {
			adapter.SendError(c, http.StatusBadRequest, "Non-streaming requests not supported")
			return
		}

		// Transform request format (e.g., OpenAI to internal format, or pass through)
		// Each adapter handles its own protocol-specific request structure
		transformedBody, err := adapter.TransformRequest(body)
		if err != nil {
			adapter.SendError(c, http.StatusBadRequest, "Failed to transform request")
			return
		}

		proxyAndStream(c, cfg, adapter, transformedBody)
	}
}

// proxyAndStream forwards the request to the upstream API and streams the response.
//
// @brief    Proxies a request to the upstream API and streams the transformed response.
// @param    c Gin context for the HTTP request/response.
// @param    cfg Application configuration with upstream settings.
// @param    adapter Protocol adapter for header forwarding and error handling.
// @param    body Transformed request body to send upstream.
//
// @note     Connection-related headers are forwarded to support streaming.
// @note     Non-200 responses from upstream are handled by handleUpstreamError.
//
// @pre      body must be valid JSON formatted for the upstream API.
// @pre      cfg must have valid upstream URL and API key.
// @post     Response is streamed to client through the adapter's transformer.
func proxyAndStream(c *gin.Context, cfg *config.Config, adapter protocols.ProtocolAdapter, body []byte) {
	// Create upstream HTTP client with connection pooling
	// The client is closed via defer to ensure proper resource cleanup
	client := upstream.NewClient(adapter.UpstreamURL(cfg), adapter.UpstreamAPIKey(cfg))
	defer client.Close()

	// Build the upstream request with context for cancellation propagation
	// This allows the request to be cancelled if the client disconnects
	req, err := client.BuildRequest(c.Request.Context(), body)
	if err != nil {
		adapter.SendError(c, http.StatusInternalServerError, "Failed to create upstream request")
		return
	}

	// Set standard headers (Content-Type, Authorization) for the upstream API
	client.SetHeaders(req)
	// Forward protocol-specific headers (e.g., Anthropic-Version for Anthropic API)
	// Each adapter knows which headers are relevant for its protocol
	adapter.ForwardHeaders(c.Request.Header, req.Header)

	// Forward hop-by-hop headers that affect connection handling
	// These headers control connection persistence and must be forwarded
	// to maintain proper streaming behavior through the proxy
	for _, h := range []string{"Connection", "Keep-Alive", "Upgrade", "TE"} {
		if v := c.Request.Header.Get(h); v != "" {
			req.Header.Set(h, v)
		}
	}

	// Execute the upstream request with context-aware timeout handling
	resp, err := client.Do(c.Request.Context(), req)
	if err != nil {
		// Use BadGateway (502) to indicate the proxy couldn't reach upstream
		adapter.SendError(c, http.StatusBadGateway, "Upstream request failed")
		return
	}

	// Handle non-200 responses from upstream - these are errors from the LLM API
	// We wrap them in a consistent error format before sending to the client
	if resp.StatusCode != http.StatusOK {
		handleUpstreamError(c, resp)
		return
	}

	streamResponseWithAdapter(c, resp.Body, adapter)
}

// streamResponseWithAdapter streams the upstream response to the client with transformation.
//
// @brief    Sets SSE headers and streams the response through the protocol adapter's transformer.
// @param    c Gin context for writing the HTTP response.
// @param    body Reader containing the upstream SSE stream.
// @param    adapter Protocol adapter for creating the output transformer.
//
// @note     Sets Content-Type to text/event-stream for SSE compatibility.
// @note     Sets Cache-Control to no-cache to prevent proxy buffering.
// @note     Sets X-Accel-Buffering to no to disable nginx buffering.
// @note     When capture context exists, both upstream and downstream SSE are recorded.
// @note     Request ID is extracted from SSE chunks if not already set.
//
// @pre      body must be a valid SSE stream from the upstream API.
// @pre      adapter must not be nil.
// @post     SSE events are transformed and written to the client.
// @post     All SSE chunks are flushed before returning.
func streamResponseWithAdapter(c *gin.Context, body io.Reader, adapter protocols.ProtocolAdapter) {
	// Set SSE headers required for proper streaming behavior
	// Content-Type: text/event-stream is the standard MIME type for Server-Sent Events
	c.Header("Content-Type", "text/event-stream")
	// Cache-Control: no-cache prevents proxies and browsers from buffering the response
	// which would delay delivery of real-time tokens
	c.Header("Cache-Control", "no-cache")
	// Connection: keep-alive maintains the TCP connection for the duration of the stream
	c.Header("Connection", "keep-alive")
	// X-Accel-Buffering: no disables nginx's response buffering when behind an nginx reverse proxy
	// Without this, nginx may buffer chunks and delay their delivery to the client
	c.Header("X-Accel-Buffering", "no")

	var cc *logging.CaptureContext
	// Check if request logging/capture is enabled for this request
	// CaptureContext is injected via middleware when SSELOG_DIR is configured
	if cc = logging.GetCaptureContext(c.Request.Context()); cc != nil {
		startTime := cc.StartTime

		// Create capture writers to record both sides of the stream
		// This enables full request/response debugging and auditing
		downstreamCapture := logging.NewCaptureWriter(startTime)
		upstreamCapture := logging.NewCaptureWriter(startTime)

		// Wrap the response writer to intercept downstream output
		// This captures what the proxy sends to the client after transformation
		recorder := NewResponseRecorder(c.Writer, downstreamCapture)
		// Create the protocol-specific transformer for output formatting
		// The transformer handles tool call extraction and format conversion
		transformer := adapter.CreateTransformer(recorder, types.StreamChunk{})
		defer transformer.Close()

		// Stream SSE events from upstream, transform, and write to client
		// The c.Stream function handles flushing to the client
		c.Stream(func(w io.Writer) bool {
			// Iterate over SSE events using the go-sse library
			// Each event contains Type (event name) and Data (JSON payload)
			for ev, err := range sse.Read(body, nil) {
				if err != nil {
					// Log SSE parsing errors but don't fail the entire stream
					// The client may have already received partial content
					logging.ErrorMsg("SSE read error: %v", err)
					return false
				}
				// Record raw upstream chunks for debugging before transformation
				// This shows what the LLM API sent before any modifications
				if ev.Data != "" {
					upstreamCapture.RecordChunk(ev.Type, []byte(ev.Data))
				}
				// Transform the SSE event - this may modify content or extract tool calls
				transformer.Transform(&ev)
			}
			return false
		})

		// Ensure all buffered content is written to the client
		transformer.Flush()

		// Store captured downstream response in the capture context for logging
		cc.Recorder.DownstreamResponse = &logging.SSEResponseCapture{
			Chunks: downstreamCapture.Chunks(),
		}
		// Store captured upstream response if available
		if cc.Recorder.UpstreamResponse != nil {
			cc.Recorder.UpstreamResponse.Chunks = upstreamCapture.Chunks()
		}
		// Extract request ID from SSE chunks if not already set from headers
		// The LLM API includes the request ID in SSE data for traceability
		if !cc.IDExtracted {
			for _, chunk := range downstreamCapture.Chunks() {
				if id := logging.ExtractRequestIDFromSSEChunk(chunk.Data); id != "" {
					cc.SetRequestID(id)
					break
				}
			}
		}
	} else {
		// No capture context - stream directly without recording
		// This is the fast path when logging is disabled
		transformer := adapter.CreateTransformer(c.Writer, types.StreamChunk{})
		defer transformer.Close()

		c.Stream(func(w io.Writer) bool {
			for ev, err := range sse.Read(body, nil) {
				if err != nil {
					logging.ErrorMsg("SSE read error: %v", err)
					return false
				}
				transformer.Transform(&ev)
			}
			return false
		})

		transformer.Flush()
	}
}
