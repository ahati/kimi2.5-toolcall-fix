package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"ai-proxy/capture"
	"ai-proxy/logging"
	"ai-proxy/proxy"
	"ai-proxy/transform"

	"github.com/gin-gonic/gin"
	"github.com/tmaxmax/go-sse"
)

type upstreamClient interface {
	BuildRequest(ctx context.Context, body []byte) (*http.Request, error)
	SetHeaders(req *http.Request)
	Do(req *http.Request) (*http.Response, error)
	Close()
}

var newUpstreamClient = func(baseURL, apiKey string) upstreamClient {
	return proxy.NewClient(baseURL, apiKey)
}

// timingCaptureWriter wraps an io.Writer and captures SSE events with accurate timing.
// It detects complete SSE events (ending with "\n\n") and records them to a CaptureWriter
// at the moment they are written, preserving correct offset_ms timing.
//
// Thread Safety: NOT thread-safe. Use from single goroutine.
type timingCaptureWriter struct {
	// underlying writer for client responses
	w io.Writer
	// capture writer for recording events with timing
	cw capture.CaptureWriter
	// buffer for accumulating partial SSE events
	buf bytes.Buffer
}

// newTimingCaptureWriter creates a writer that captures SSE events with timing.
func newTimingCaptureWriter(w io.Writer, cw capture.CaptureWriter) *timingCaptureWriter {
	return &timingCaptureWriter{
		w:  w,
		cw: cw,
	}
}

// Write implements io.Writer. It forwards data to the underlying writer and
// captures complete SSE events for timing-accurate recording.
func (tcw *timingCaptureWriter) Write(p []byte) (n int, err error) {
	// Write to underlying writer first
	n, err = tcw.w.Write(p)
	if err != nil {
		return n, err
	}

	// Accumulate data for SSE parsing
	tcw.buf.Write(p)

	// Parse and record any complete SSE events
	tcw.parseAndRecordEvents()

	return n, nil
}

// parseAndRecordEvents parses complete SSE events from the buffer and records them.
// SSE events are delimited by "\n\n". Each event may have "event:" and "data:" lines.
func (tcw *timingCaptureWriter) parseAndRecordEvents() {
	data := tcw.buf.Bytes()

	// Find complete events (ending with \n\n)
	for {
		idx := bytes.Index(data, []byte("\n\n"))
		if idx == -1 {
			break
		}

		// Extract the complete event
		event := data[:idx]
		data = data[idx+2:] // Skip past \n\n

		// Parse event type and data
		eventType, eventData := parseSSEEvent(event)
		if len(eventData) > 0 {
			tcw.cw.RecordChunk(eventType, eventData)
		}
	}

	// Keep remaining partial data in buffer
	tcw.buf.Reset()
	tcw.buf.Write(data)
}

// FlushRemaining flushes any remaining buffered data as a final chunk.
// This ensures partial events are captured when the stream ends unexpectedly.
func (tcw *timingCaptureWriter) FlushRemaining() {
	data := tcw.buf.Bytes()
	if len(data) > 0 {
		// First, try to parse any complete events
		tcw.parseAndRecordEvents()

		// If there's still data in buffer, it might be a partial event
		// Record it as raw data so it's not lost
		remaining := tcw.buf.Bytes()
		if len(remaining) > 0 {
			tcw.cw.RecordChunk("", remaining)
		}
	}
}

// parseSSEEvent extracts the event type and data from an SSE event string.
// SSE format: "event: type\ndata: {...}" or just "data: {...}"
func parseSSEEvent(event []byte) (eventType string, data []byte) {
	lines := bytes.Split(event, []byte("\n"))
	for _, line := range lines {
		if bytes.HasPrefix(line, []byte("event:")) {
			eventType = string(bytes.TrimSpace(line[6:]))
		} else if bytes.HasPrefix(line, []byte("data:")) {
			data = bytes.TrimSpace(line[5:])
		}
	}
	return eventType, data
}

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
		transformedBody, err := h.TransformRequest(c.Request.Context(), body)
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

	// Log request with model info for debugging
	downstreamModel, upstreamModel := h.ModelInfo()
	logging.InfoMsg("Sending request to upstream: %s (downstream_model=%s, upstream_model=%s)", h.UpstreamURL(), downstreamModel, upstreamModel)
	client := newUpstreamClient(h.UpstreamURL(), apiKey)
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

	// Check if capture is enabled and route to appropriate streaming method
	// Capture context is attached by CaptureMiddleware if capture is enabled
	cc := capture.GetCaptureContext(c.Request.Context())

	// Set up SSE stream headers before creating transformer and making upstream request
	setStreamHeaders(c)

	// Create and initialize transformer BEFORE upstream request
	// This ensures response.created is emitted before any upstream response
	var transformer transform.SSETransformer
	var responseID string

	if cc != nil {
		// For capture mode, we need special handling - transformer created inside c.Stream
		// We'll initialize it there before reading from upstream
		transformer = nil
	} else {
		// Create transformer without capture wrapper
		transformer = h.CreateTransformer(c.Writer)
		// Set context for cache status tracking
		setContextOnTransformer(transformer, c.Request.Context())
		// Initialize transformer and emit response.created before upstream call
		if err := transformer.Initialize(); err != nil {
			logging.ErrorMsg("Failed to initialize transformer: %v", err)
			h.WriteError(c, http.StatusInternalServerError, "Failed to initialize response stream")
			return
		}
		defer transformer.Close()

		// Get response ID for stream cancellation registration
		if getter, ok := transformer.(transform.ResponseIDGetter); ok {
			responseID = getter.GetResponseID()
		}
	}

	// Register stream for cancellation support if we have a response ID
	var cancel context.CancelFunc
	if responseID != "" {
		registry := GetGlobalRegistry()
		var streamCtx context.Context
		streamCtx, cancel = context.WithCancel(c.Request.Context())
		c.Request = c.Request.WithContext(streamCtx)
		registry.Register(responseID, cancel, transformer)
		defer func() {
			registry.Remove(responseID)
			if cancel != nil {
				cancel()
			}
		}()
	}

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

	if cc != nil {
		// Stream with capture when capture is enabled
		// Transformer is created and initialized inside streamWithCapture
		streamWithCapture(c, resp.Body, h, cc)
	} else {
		// Stream without capture for lower latency
		// Transformer is already initialized, just stream events
		streamWithInitializedTransformer(c, resp.Body, transformer)
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
	// Set context for cache status tracking
	setContextOnTransformer(transformer, c.Request.Context())
	// Ensure transformer resources are cleaned up
	defer transformer.Close()

	// Stream SSE events to client via Gin's streaming facility
	c.Stream(func(w io.Writer) bool {
		for ev, err := range sse.Read(body, nil) {
			if err != nil {
				// Context canceled means client disconnected - can't send response.failed
				if errors.Is(err, context.Canceled) {
					logging.DebugMsg("Stream completed, client disconnected")
					return false
				}
				logging.ErrorMsg("SSE stream error: %v", err)
				emitStreamError(transformer, err)
				return false
			}
			// Transform each event to downstream format
			if err := transformer.Transform(&ev); err != nil {
				logging.ErrorMsg("Transform error: %v", err)
				emitStreamError(transformer, err)
				return false
			}
		}
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

	// Stream events with capture
	// Get flusher from Gin response writer for immediate delivery
	flusher, canFlush := c.Writer.(http.Flusher)

	c.Stream(func(w io.Writer) bool {
		// Create timing-aware writer that captures downstream events with correct timing
		timingWriter := newTimingCaptureWriter(w, downstream)
		// Create transformer that writes to our timing-aware writer
		transformer := h.CreateTransformer(timingWriter)
		// Set context for cache status tracking
		setContextOnTransformer(transformer, c.Request.Context())
		defer func() {
			timingWriter.FlushRemaining()
			transformer.Close()
		}()

		// Initialize transformer and emit response.created BEFORE reading from upstream
		if err := transformer.Initialize(); err != nil {
			logging.ErrorMsg("Failed to initialize transformer in capture mode: %v", err)
			emitStreamError(transformer, err)
			return false
		}

		// Iterate over all SSE events from upstream
		for ev, err := range sse.Read(body, nil) {
			if err != nil {
				// Context canceled means client disconnected - can't send response.failed
				if errors.Is(err, context.Canceled) {
					logging.DebugMsg("Stream completed, client disconnected")
					return false
				}
				logging.ErrorMsg("SSE stream error (capture): %v", err)
				emitStreamError(transformer, err)
				return false
			}
			// Capture upstream events before transformation
			// Only record if event has data (skip empty keepalive events)
			if ev.Data != "" {
				recordUpstreamEvent(upstream, ev)
			}
			// Transform and send event to client (timing captured by timingWriter)
			if err := transformer.Transform(&ev); err != nil {
				logging.ErrorMsg("Transform error (capture): %v", err)
				emitStreamError(transformer, err)
				return false
			}

			// Flush after each event to ensure immediate delivery
			// This prevents buffering that causes clients to timeout
			if canFlush {
				flusher.Flush()
			}
		}
		return false
	})

	// Finalize capture by recording all captured data
	finalizeCapture(cc, downstream, upstream)
}

// streamWithInitializedTransformer streams events using an already-initialized transformer.
// This is used in non-capture mode where Initialize() was called before the upstream request.
func streamWithInitializedTransformer(c *gin.Context, body io.Reader, transformer transform.SSETransformer) {
	// Stream events without capture overhead
	// Get flusher from Gin response writer for immediate delivery
	flusher, canFlush := c.Writer.(http.Flusher)

	c.Stream(func(w io.Writer) bool {
		for ev, err := range sse.Read(body, nil) {
			if err != nil {
				// Context canceled means client disconnected - can't send response.failed
				if errors.Is(err, context.Canceled) {
					logging.DebugMsg("Stream completed, client disconnected")
					return false
				}
				logging.ErrorMsg("SSE stream error (no-capture): %v", err)
				emitStreamError(transformer, err)
				return false
			}
			// Transform and send event directly to client
			if err := transformer.Transform(&ev); err != nil {
				logging.ErrorMsg("Transform error (no-capture): %v", err)
				emitStreamError(transformer, err)
				return false
			}

			// Flush after each event to ensure immediate delivery
			if canFlush {
				flusher.Flush()
			}
		}
		return false
	})
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
	// Set context for cache status tracking
	setContextOnTransformer(transformer, c.Request.Context())
	defer transformer.Close()

	// Stream events without capture overhead
	// Get flusher from Gin response writer for immediate delivery
	flusher, canFlush := c.Writer.(http.Flusher)

	c.Stream(func(w io.Writer) bool {
		for ev, err := range sse.Read(body, nil) {
			if err != nil {
				// Context canceled means client disconnected - can't send response.failed
				if errors.Is(err, context.Canceled) {
					logging.DebugMsg("Stream completed, client disconnected")
					return false
				}
				logging.ErrorMsg("SSE stream error (no-capture): %v", err)
				emitStreamError(transformer, err)
				return false
			}
			// Transform and send event directly to client
			if err := transformer.Transform(&ev); err != nil {
				logging.ErrorMsg("Transform error (no-capture): %v", err)
				emitStreamError(transformer, err)
				return false
			}

			// Flush after each event to ensure immediate delivery
			if canFlush {
				flusher.Flush()
			}
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

// emitStreamError sends a response.failed event to the client before closing.
// This ensures clients receive proper notification of stream failures.
func emitStreamError(transformer transform.SSETransformer, err error) {
	// Type assert to check if transformer supports error emission
	if et, ok := transformer.(interface{ EmitError(error) error }); ok {
		if emitErr := et.EmitError(err); emitErr != nil {
			logging.ErrorMsg("Failed to emit error event: %v", emitErr)
		}
	}
}

// setContextOnTransformer sets the context on transformers that support it.
// This enables cache status tracking during response transformation.
func setContextOnTransformer(transformer transform.SSETransformer, ctx context.Context) {
	if ct, ok := transformer.(interface{ SetContext(context.Context) }); ok {
		ct.SetContext(ctx)
	}
}

// finalizeCapture completes the capture process by recording response data
// and extracting request IDs from the SSE stream.
//
// @param cc - Capture context to finalize.
// @param downstream - Writer containing captured downstream events.
// @param upstream - Writer containing captured upstream events.
// @pre cc != nil and has been recording the request.
// @post cc.Recorder contains complete downstream and upstream response data.
// @post cc.RequestID is set if found in SSE stream.
func finalizeCapture(cc *capture.CaptureContext, downstream, upstream capture.CaptureWriter) {
	// Record downstream response (transformed events sent to client)
	// Use thread-safe method instead of direct field access
	downstreamRecorder := cc.Recorder.RecordDownstreamResponse()
	// Transfer captured chunks directly, preserving their original timing
	// The chunks already have correct OffsetMS from when they were recorded during streaming
	for _, chunk := range downstream.Chunks() {
		downstreamRecorder.RecordChunkPreservingTiming(chunk)
	}

	// Record upstream response (original events from upstream)
	// May be nil if upstream request failed before response
	if cc.Recorder.Data().UpstreamResponse != nil {
		upstreamRecorder := cc.Recorder.RecordUpstreamResponse(
			cc.Recorder.Data().UpstreamResponse.StatusCode,
			nil, // Headers already captured during proxy request
		)
		// Transfer captured chunks directly, preserving their original timing
		for _, chunk := range upstream.Chunks() {
			upstreamRecorder.RecordChunkPreservingTiming(chunk)
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

	// Extract and log token usage from captured chunks
	// This provides a compact summary of request costs in a powerline-style format
	upstreamUsage := capture.ExtractTokenUsageFromChunks(upstream.Chunks())
	downstreamUsage := capture.ExtractTokenUsageFromChunks(downstream.Chunks())

	// Extract finish reasons from both upstream and downstream chunks
	// Upstream: reason from LLM API, Downstream: reason sent to client (may differ after transformation)
	upstreamReason := capture.ExtractFinishReasonFromChunks(upstream.Chunks())
	downstreamReason := capture.ExtractFinishReasonFromChunks(downstream.Chunks())

	// Build cache status indicators (separate items)
	var cacheParts []string
	if cc.CacheHit {
		cacheParts = append(cacheParts, "🗄️ cache-hit")
	}
	if cc.CacheCreated {
		cacheParts = append(cacheParts, "🗃️ cache-created")
	}
	cacheStatus := strings.Join(cacheParts, " ")

	// Compact one-line log with emojis:
	// 📤 = upstream (to LLM), 📥 = downstream (to client)
	// ⬆️ = input tokens, ⬇️ = output tokens, 📖 = cache read, 💾  = cache creation
	if cacheStatus != "" {
		logging.InfoMsg("|📤 ⬆️ %d ⬇️ %d 📖 %d 💾 %d r=%s|  |📥 ⬆️ %d ⬇️ %d 📖 %d 💾 %d r=%s| %s [%s] [%s]",
			upstreamUsage.InputTokens,
			upstreamUsage.OutputTokens,
			upstreamUsage.CacheReadTokens,
			upstreamUsage.CacheCreationTokens,
			upstreamReason,
			downstreamUsage.InputTokens,
			downstreamUsage.OutputTokens,
			downstreamUsage.CacheReadTokens,
			downstreamUsage.CacheCreationTokens,
			downstreamReason,
			cacheStatus,
			cc.SessionID,
			cc.RequestID,
		)
	} else {
		logging.InfoMsg("|📤 ⬆️ %d ⬇️ %d 📖 %d 💾 %d r=%s|  |📥 ⬆️ %d ⬇️ %d 📖 %d 💾 %d r=%s| [%s] [%s]",
			upstreamUsage.InputTokens,
			upstreamUsage.OutputTokens,
			upstreamUsage.CacheReadTokens,
			upstreamUsage.CacheCreationTokens,
			upstreamReason,
			downstreamUsage.InputTokens,
			downstreamUsage.OutputTokens,
			downstreamUsage.CacheReadTokens,
			downstreamUsage.CacheCreationTokens,
			downstreamReason,
			cc.SessionID,
			cc.RequestID,
		)
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
	msg := string(body)

	// Record the upstream error for capture
	c.Set("upstream_error_body", msg)
	c.Set("upstream_error_status", resp.StatusCode)

	// Send error in OpenAI format with the original upstream status code
	// This preserves error details like 400 (bad request), 401 (auth), 429 (rate limit)
	sendOpenAIError(c, resp.StatusCode, msg)
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

// sendOpenAIResponsesError sends an error response in OpenAI Responses API format.
// OpenAI Responses API uses SSE format for errors.
//
// @param c - Gin context for writing the response.
// @param status - HTTP status code for the error.
// @param msg - Human-readable error message.
//
// @pre c != nil and response has not been written.
// @post SSE error response is written and flushed.
func sendOpenAIResponsesError(c *gin.Context, status int, msg string) {
	event := map[string]interface{}{
		"type": "error",
		"error": map[string]interface{}{
			"code":    "invalid_request_error",
			"message": msg,
		},
	}
	c.Header("Content-Type", "text/event-stream")
	data, _ := json.Marshal(event)
	c.String(status, "data: "+string(data)+"\n\n")
}
