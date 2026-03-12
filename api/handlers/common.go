package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"ai-proxy/capture"
	"ai-proxy/proxy"

	"github.com/gin-gonic/gin"
	"github.com/tmaxmax/go-sse"
)

// Handle wraps a Handler implementation and returns a Gin handler function.
// It orchestrates the full request pipeline: reading, validating, transforming, and proxying.
//
// The processing flow is:
//  1. Read request body from client
//  2. Record downstream request for capture
//  3. Validate request format
//  4. Transform request to upstream format
//  5. Forward to upstream and stream response back
//
// @param h - Handler implementation defining endpoint-specific behavior.
//
//	Must not be nil. Handler methods are called in sequence.
//
// @return Gin handler function that processes requests through the handler pipeline.
//
// @pre h != nil
// @post Response is fully written to client on return (success or error).
func Handle(h Handler) gin.HandlerFunc {
	return func(c *gin.Context) {
		// Step 1: Read the complete request body for processing
		body, err := readBody(c)
		if err != nil {
			// Body read failure indicates client disconnected or malformed request
			h.WriteError(c, http.StatusBadRequest, "Failed to read request body")
			return
		}

		// Step 2: Record downstream request for capture/logging
		// This captures the original request before any transformation
		capture.RecordDownstreamRequest(c.Request.Context(), c.Request, body)

		// Step 3: Validate request format and semantics
		if err := h.ValidateRequest(body); err != nil {
			// Validation failure indicates client error (400-level response)
			h.WriteError(c, http.StatusBadRequest, err.Error())
			return
		}

		// Step 4: Transform request to upstream format
		transformedBody, err := h.TransformRequest(body)
		if err != nil {
			// Transformation failure indicates internal error
			h.WriteError(c, http.StatusInternalServerError, "Failed to transform request")
			return
		}

		// Step 5: Forward to upstream and stream response
		proxyRequest(c, h, transformedBody)
	}
}

// readBody reads and returns the entire request body.
// The body is consumed and cannot be read again.
//
// @param c - Gin context containing the HTTP request.
// @return Complete request body bytes, or error if read fails.
//
// @pre c.Request.Body != nil
// @post c.Request.Body is fully consumed and closed.
// @note Returns empty slice for empty body, not nil.
func readBody(c *gin.Context) ([]byte, error) {
	return io.ReadAll(c.Request.Body)
}

// validateStreaming checks that the request has streaming enabled.
// Returns an error if the request is not configured for streaming.
//
// @param body - Raw JSON request body to parse.
// @return nil if streaming is enabled, error otherwise.
//
// @pre body is valid JSON (this is not validated here).
// @post If error is returned, request should be rejected.
// @note Non-streaming requests are not supported by this proxy.
func validateStreaming(body []byte) error {
	var req struct {
		Stream bool `json:"stream"`
	}
	// Parse only the stream field to check streaming status
	if err := json.Unmarshal(body, &req); err != nil {
		return fmt.Errorf("invalid JSON: %w", err)
	}
	// Streaming must be enabled - non-streaming requests are rejected
	// as the proxy is designed for SSE streaming responses
	if !req.Stream {
		return fmt.Errorf("non-streaming requests not supported")
	}
	return nil
}

// proxyRequest forwards the request to the upstream API and streams the response back.
// This is the core proxying logic that handles upstream communication.
//
// @param c - Gin context for the current request.
// @param h - Handler defining upstream URL, headers, and error handling.
// @param body - Transformed request body to send upstream.
//
// @pre body is in correct upstream format.
// @pre h.UpstreamURL() returns valid URL.
// @post Response is streamed to client or error response is sent.
func proxyRequest(c *gin.Context, h Handler, body []byte) {
	// Resolve API key for upstream authentication
	apiKey := h.ResolveAPIKey(c)

	// Create HTTP client configured for upstream endpoint
	client := proxy.NewClient(h.UpstreamURL(), apiKey)
	// Ensure connection resources are released when done
	defer client.Close()

	// Build the upstream HTTP request
	req, err := client.BuildRequest(c.Request.Context(), body)
	if err != nil {
		// Request build failure indicates internal error
		h.WriteError(c, http.StatusInternalServerError, "Failed to create upstream request")
		return
	}

	// Set standard headers required by upstream API
	client.SetHeaders(req)
	// Forward custom headers from original request
	h.ForwardHeaders(c, req)

	// Execute the upstream request
	resp, err := client.Do(req)
	if err != nil {
		// Upstream connection failure indicates gateway error
		h.WriteError(c, http.StatusBadGateway, "Upstream request failed")
		return
	}

	// Check for non-200 responses from upstream
	// Non-OK status indicates upstream error (auth, rate limit, etc.)
	if resp.StatusCode != http.StatusOK {
		handleUpstreamError(c, resp)
		return
	}

	// Check if capture is enabled and route to appropriate streaming method
	// Capture context is attached by CaptureMiddleware if capture is enabled
	cc := capture.GetCaptureContext(c.Request.Context())
	if cc != nil {
		// Stream with capture when capture is enabled
		streamWithCapture(c, resp.Body, h, cc)
	} else {
		// Stream without capture for lower latency
		streamWithoutCapture(c, resp.Body, h)
	}
}

// streamResponse streams the upstream SSE response to the client with transformation.
// This is the core streaming logic that processes each SSE event.
//
// @param c - Gin context for writing the response.
// @param body - Reader for upstream SSE response body.
// @param h - Handler providing the SSE transformer.
//
// @pre body is a valid SSE stream reader.
// @pre Response headers have not been written yet.
// @post All SSE events are processed and response is complete.
func streamResponse(c *gin.Context, body io.Reader, h Handler) {
	// Set headers required for SSE streaming
	setStreamHeaders(c)

	// Create transformer to convert upstream format to downstream format
	transformer := h.CreateTransformer(c.Writer)
	// Ensure transformer resources are cleaned up
	defer transformer.Close()

	// Stream SSE events to client via Gin's streaming facility
	c.Stream(func(w io.Writer) bool {
		// Iterate over all SSE events from upstream
		for ev, err := range sse.Read(body, nil) {
			if err != nil {
				// Error indicates stream termination (possibly client disconnect)
				return false
			}
			// Transform each event to downstream format
			transformer.Transform(&ev)
		}
		// Return false to signal end of stream
		return false
	})
}

// setStreamHeaders sets the appropriate HTTP headers for SSE streaming responses.
// These headers are required for proper SSE handling by clients and proxies.
//
// @param c - Gin context for setting response headers.
//
// @post Content-Type is set to text/event-stream.
// @post Cache-Control is set to no-cache to prevent buffering.
// @post Connection is set to keep-alive for persistent connection.
// @post X-Accel-Buffering is set to no to disable nginx buffering.
func setStreamHeaders(c *gin.Context) {
	// Content-Type for Server-Sent Events
	c.Header("Content-Type", "text/event-stream")
	// Prevent caching to ensure real-time delivery
	c.Header("Cache-Control", "no-cache")
	// Keep connection open for streaming
	c.Header("Connection", "keep-alive")
	// Disable nginx buffering for real-time streaming through reverse proxy
	c.Header("X-Accel-Buffering", "no")
}

// streamWithCapture streams the response while capturing both upstream and downstream
// data for logging and analysis.
//
// @param c - Gin context for writing the response.
// @param body - Reader for upstream SSE response body.
// @param h - Handler providing the SSE transformer.
// @param cc - Capture context for recording request/response data.
//
// @pre body is a valid SSE stream reader.
// @pre cc != nil and is properly initialized.
// @post All events are captured in cc.Recorder.
func streamWithCapture(c *gin.Context, body io.Reader, h Handler, cc *capture.CaptureContext) {
	startTime := cc.StartTime
	// Create capture writer for downstream (transformed) events
	downstream := capture.NewCaptureWriter(startTime)
	// Create capture writer for upstream (original) events
	upstream := capture.NewCaptureWriter(startTime)

	// Wrap the response writer to capture downstream events
	recorder := NewResponseRecorder(c.Writer, downstream)
	// Create transformer that writes through the recorder
	transformer := h.CreateTransformer(recorder)
	defer transformer.Close()

	// Stream events with capture
	c.Stream(func(w io.Writer) bool {
		// Iterate over all SSE events from upstream
		for ev, err := range sse.Read(body, nil) {
			if err != nil {
				// Error indicates stream termination
				return false
			}
			// Capture upstream events before transformation
			// Only record if event has data (skip empty keepalive events)
			if ev.Data != "" {
				recordUpstreamEvent(upstream, ev)
			}
			// Transform and send event to client
			transformer.Transform(&ev)
		}
		return false
	})

	// Finalize capture by recording all captured data
	finalizeCapture(cc, downstream, upstream)
}

// streamWithoutCapture streams the response without capturing data.
// Used when capture is disabled to minimize latency.
//
// @param c - Gin context for writing the response.
// @param body - Reader for upstream SSE response body.
// @param h - Handler providing the SSE transformer.
//
// @pre body is a valid SSE stream reader.
// @post Response is streamed to client without any capture overhead.
func streamWithoutCapture(c *gin.Context, body io.Reader, h Handler) {
	// Set headers for SSE streaming
	setStreamHeaders(c)

	// Create transformer without capture wrapper
	transformer := h.CreateTransformer(c.Writer)
	defer transformer.Close()

	// Stream events without capture overhead
	c.Stream(func(w io.Writer) bool {
		for ev, err := range sse.Read(body, nil) {
			if err != nil {
				// Error indicates stream termination
				return false
			}
			// Transform and send event directly to client
			transformer.Transform(&ev)
		}
		return false
	})
}

// recordUpstreamEvent writes an SSE event to the capture writer.
// Records the event type and data for later analysis.
//
// @param w - Capture writer to record the event to.
// @param ev - SSE event to record.
//
// @pre w != nil
// @post Event is recorded if it has non-empty data.
func recordUpstreamEvent(w capture.CaptureWriter, ev sse.Event) {
	// Only record events with data - skip empty keepalive events
	if ev.Data != "" {
		w.RecordChunk(ev.Type, []byte(ev.Data))
	}
}

// finalizeCapture completes the capture process by recording response data
// and extracting request IDs from the SSE stream.
//
// @param cc - Capture context to finalize.
// @param downstream - Writer containing captured downstream events.
// @param upstream - Writer containing captured upstream events.
//
// @pre cc != nil and has been recording the request.
// @post cc.Recorder contains complete downstream and upstream response data.
// @post cc.RequestID is set if found in SSE stream.
func finalizeCapture(cc *capture.CaptureContext, downstream, upstream capture.CaptureWriter) {
	// Record downstream response (transformed events sent to client)
	// Use thread-safe method instead of direct field access
	downstreamRecorder := cc.Recorder.RecordDownstreamResponse()
	// Transfer captured chunks to the response recorder
	for _, chunk := range downstream.Chunks() {
		downstreamRecorder.RecordChunk(chunk.Event, string(chunk.Data))
	}

	// Record upstream response (original events from upstream)
	// May be nil if upstream request failed before response
	if cc.Recorder.Data().UpstreamResponse != nil {
		upstreamRecorder := cc.Recorder.RecordUpstreamResponse(
			cc.Recorder.Data().UpstreamResponse.StatusCode,
			nil, // Headers already captured during proxy request
		)
		// Transfer captured chunks to the response recorder
		for _, chunk := range upstream.Chunks() {
			upstreamRecorder.RecordChunk(chunk.Event, string(chunk.Data))
		}
	}

	// Extract request ID from SSE stream if not already found
	// Request ID is typically in the first SSE event from LLM APIs
	if !cc.IDExtracted {
		for _, chunk := range downstream.Chunks() {
			// Attempt to extract ID from each chunk until found
			if id := capture.ExtractRequestIDFromSSEChunk(chunk.Data); id != "" {
				cc.SetRequestID(id)
				// Stop after finding the first ID
				break
			}
		}
	}
}

// handleUpstreamError processes an error response from the upstream API
// and sends it to the client.
//
// @param c - Gin context for writing the error response.
// @param resp - Error response from upstream.
//
// @pre resp != nil and resp.Body is readable.
// @post Error response is sent to client in OpenAI error format.
func handleUpstreamError(c *gin.Context, resp *http.Response) {
	// Read the error body for inclusion in client error message
	body, _ := io.ReadAll(resp.Body)
	msg := fmt.Sprintf("Upstream error: %s", string(body))
	// Send error in OpenAI format for consistency
	sendOpenAIError(c, http.StatusBadGateway, msg)
}

// sendOpenAIError sends an error response in OpenAI API format.
// OpenAI format: {"error": {"message": "...", "type": "..."}}
//
// @param c - Gin context for writing the response.
// @param status - HTTP status code for the error.
// @param msg - Human-readable error message.
//
// @pre c != nil and response has not been written.
// @post JSON error response is written and flushed.
func sendOpenAIError(c *gin.Context, status int, msg string) {
	c.JSON(status, gin.H{
		"error": gin.H{
			"message": msg,
			"type":    "invalid_request_error",
		},
	})
}

// sendAnthropicError sends an error response in Anthropic API format.
// Anthropic format: {"type": "error", "error": {"type": "...", "message": "..."}}
//
// @param c - Gin context for writing the response.
// @param status - HTTP status code for the error.
// @param msg - Human-readable error message.
//
// @pre c != nil and response has not been written.
// @post JSON error response is written and flushed.
func sendAnthropicError(c *gin.Context, status int, msg string) {
	c.JSON(status, gin.H{
		"type": "error",
		"error": gin.H{
			"type":    "invalid_request_error",
			"message": msg,
		},
	})
}
