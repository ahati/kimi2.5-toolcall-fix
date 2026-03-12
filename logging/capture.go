// Package logging provides structured logging and request/response capture functionality.
// It captures bidirectional traffic between client and upstream LLM API for debugging
// and analysis purposes.
package logging

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"time"
)

// contextKey is the type for context keys used in this package.
type contextKey string

// captureContextKey is the context key for storing CaptureContext values.
const captureContextKey contextKey = "capture_context"

// CaptureContext holds the capture state for a single request lifecycle.
// It is attached to request contexts and tracks request ID extraction status.
type CaptureContext struct {
	// RequestID is the unique identifier for the request, extracted from SSE response.
	RequestID string
	// StartTime records when the capture context was created.
	StartTime time.Time
	// Recorder accumulates all captured data for the request.
	Recorder *RequestRecorder
	// IDExtracted indicates whether RequestID has been populated from SSE.
	IDExtracted bool
}

// HTTPRequestCapture represents a captured HTTP request for logging purposes.
type HTTPRequestCapture struct {
	// At is the timestamp when the request was captured.
	At time.Time `json:"at,omitempty"`
	// Headers contains sanitized HTTP headers (sensitive values masked).
	Headers map[string]string `json:"headers,omitempty"`
	// Body contains the parsed JSON request body.
	Body json.RawMessage `json:"body,omitempty"`
	// RawBody contains the raw bytes of the request body.
	RawBody []byte `json:"-"`
}

// SSEResponseCapture represents a captured SSE streaming response.
type SSEResponseCapture struct {
	// StatusCode is the HTTP status code of the response.
	StatusCode int `json:"status_code,omitempty"`
	// Headers contains sanitized response headers.
	Headers map[string]string `json:"headers,omitempty"`
	// Chunks is the ordered list of received SSE events.
	Chunks []SSEChunk `json:"chunks,omitempty"`
	// RawBody contains the complete raw response body.
	RawBody []byte `json:"-"`
}

// SSEChunk represents a single Server-Sent Event in a streaming response.
type SSEChunk struct {
	// OffsetMS is milliseconds elapsed since capture start.
	OffsetMS int64 `json:"offset_ms"`
	// Event is the SSE event type (e.g., "message", "error").
	Event string `json:"event,omitempty"`
	// Data contains the parsed JSON data payload.
	Data json.RawMessage `json:"data,omitempty"`
	// Raw contains the raw data string if JSON parsing failed.
	Raw string `json:"raw,omitempty"`
}

// NewSSEChunk creates a new SSEChunk from event data.
//
// @brief    Creates SSEChunk with timestamp and parsed data.
// @param    offsetMS Milliseconds since capture started.
// @param    event    SSE event type name.
// @param    data     Raw event data bytes.
// @return   SSEChunk with OffsetMS, Event, and Data or Raw populated.
//
// @note     If data is valid JSON, it's stored in Data field.
// @note     If data is not valid JSON, it's stored as string in Raw field.
func NewSSEChunk(offsetMS int64, event string, data []byte) SSEChunk {
	chunk := SSEChunk{
		OffsetMS: offsetMS,
		Event:    event,
	}

	// Try to parse as JSON first - SSE data payloads are typically JSON objects
	// Storing as json.RawMessage preserves the original structure for later inspection
	var jsonData json.RawMessage
	if err := json.Unmarshal(data, &jsonData); err == nil {
		// Copy the parsed JSON to avoid holding references to the original buffer
		// This prevents memory aliasing issues when the original buffer is reused
		chunk.Data = make(json.RawMessage, len(jsonData))
		copy(chunk.Data, jsonData)
	} else {
		// Fallback to raw string for non-JSON data (e.g., error messages, plain text)
		chunk.Raw = string(data)
	}

	return chunk
}

// RequestRecorder accumulates all captured data for a single request.
// It records both downstream (client-proxy) and upstream (proxy-LLM) traffic.
type RequestRecorder struct {
	// RequestID is the unique identifier for this request.
	RequestID string
	// StartedAt is when the request was first received.
	StartedAt time.Time
	// Method is the HTTP method (GET, POST, etc.).
	Method string
	// Path is the request URL path.
	Path string
	// ClientIP is the client's IP address.
	ClientIP string
	// DownstreamRequest captures the client-to-proxy request.
	DownstreamRequest *HTTPRequestCapture
	// UpstreamRequest captures the proxy-to-LLM request.
	UpstreamRequest *HTTPRequestCapture
	// UpstreamResponse captures the LLM-to-proxy response.
	UpstreamResponse *SSEResponseCapture
	// DownstreamResponse captures the proxy-to-client response.
	DownstreamResponse *SSEResponseCapture
}

// NewCaptureContext creates a new CaptureContext for an HTTP request.
//
// @brief    Creates capture context with initialized recorder.
// @param    r HTTP request to capture.
// @return   Pointer to new CaptureContext instance.
//
// @pre      r must not be nil.
// @post     StartTime is set to current time.
// @post     Recorder is initialized with request metadata.
// @post     IDExtracted is set to false.
func NewCaptureContext(r *http.Request) *CaptureContext {
	return &CaptureContext{
		StartTime: time.Now(),
		Recorder: &RequestRecorder{
			StartedAt: time.Now(),
			Method:    r.Method,
			Path:      r.URL.Path,
			ClientIP:  r.RemoteAddr,
		},
		IDExtracted: false,
	}
}

// SetRequestID sets the request ID in the capture context.
//
// @brief    Updates RequestID and marks it as extracted.
// @param    id Request identifier string.
//
// @pre      None.
// @post     RequestID is set in both CaptureContext and Recorder.
// @post     IDExtracted is set to true.
func (cc *CaptureContext) SetRequestID(id string) {
	cc.RequestID = id
	cc.Recorder.RequestID = id
	cc.IDExtracted = true
}

// WithCaptureContext attaches a CaptureContext to a context.
//
// @brief    Stores CaptureContext in context for later retrieval.
// @param    ctx Parent context.
// @param    cc  CaptureContext to attach.
// @return   New context with CaptureContext attached.
//
// @pre      ctx must not be nil.
// @pre      cc must not be nil.
// @post     Context contains CaptureContext retrievable via GetCaptureContext.
func WithCaptureContext(ctx context.Context, cc *CaptureContext) context.Context {
	return context.WithValue(ctx, captureContextKey, cc)
}

// GetCaptureContext retrieves CaptureContext from a context.
//
// @brief    Extracts CaptureContext stored via WithCaptureContext.
// @param    ctx Context to search.
// @return   CaptureContext pointer, or nil if not found.
//
// @pre      None.
// @note     Returns nil if context is nil or no CaptureContext exists.
func GetCaptureContext(ctx context.Context) *CaptureContext {
	if cc, ok := ctx.Value(captureContextKey).(*CaptureContext); ok {
		return cc
	}
	return nil
}

// extractRequestID extracts request ID from HTTP headers.
//
// @brief    Attempts to find request ID in standard headers.
// @param    r HTTP request to examine.
// @return   Request ID string, or empty string if not found.
//
// @note     Checks X-Request-ID header (case-sensitive variants).
// @note     POST request body parsing is not implemented here.
func extractRequestID(r *http.Request) string {
	if id := r.Header.Get("X-Request-ID"); id != "" {
		return id
	}
	if id := r.Header.Get("x-request-id"); id != "" {
		return id
	}

	if r.Method == "POST" && r.Body != nil {
		return ""
	}

	return ""
}

// ExtractRequestIDFromSSEChunk parses an SSE data chunk to extract request ID.
//
// @brief    Extracts request ID from OpenAI or Anthropic SSE response format.
// @param    body Raw SSE data bytes to parse.
// @return   Request ID string, or empty string if not found.
//
// @note     Supports OpenAI format: {"id": "..."}
// @note     Supports Anthropic format: {"type": "...", "message": {"id": "..."}}
// @note     Returns empty string if JSON parsing fails or ID not present.
func ExtractRequestIDFromSSEChunk(body []byte) string {
	// OpenAI format: first chunk contains {"id": "chatcmpl-xxx", ...}
	// The ID is at the top level of the JSON object
	var openAIChunk struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(body, &openAIChunk); err == nil && openAIChunk.ID != "" {
		return openAIChunk.ID
	}

	// Anthropic format: message_start event contains {"type": "message_start", "message": {"id": "msg_xxx", ...}}
	// The ID is nested inside the "message" object, requiring a different struct
	var anthropicChunk struct {
		Type    string `json:"type"`
		Message struct {
			ID string `json:"id"`
		} `json:"message"`
	}
	if err := json.Unmarshal(body, &anthropicChunk); err == nil && anthropicChunk.Message.ID != "" {
		return anthropicChunk.Message.ID
	}

	return ""
}

// SanitizeHeaders removes sensitive values from HTTP headers.
//
// @brief    Masks sensitive header values for safe logging.
// @param    headers HTTP headers to sanitize.
// @return   Map of header names to sanitized values.
//
// @note     Sensitive headers are replaced with "***".
// @note     Case-insensitive matching for header names.
// @note     Sensitive headers: authorization, x-api-key, cookie, set-cookie, x-auth-token.
func SanitizeHeaders(headers http.Header) map[string]string {
	// Sensitive headers that contain credentials or authentication tokens
	// These must be masked to prevent credential leakage in log files
	sensitive := map[string]bool{
		"authorization": true, // Bearer tokens, Basic auth credentials
		"x-api-key":     true, // API keys (common in LLM APIs)
		"cookie":        true, // Session cookies, auth tokens
		"set-cookie":    true, // Server-set cookies with session info
		"x-auth-token":  true, // Custom auth tokens
	}

	result := make(map[string]string)
	for k, v := range headers {
		// HTTP headers are case-insensitive, so normalize to lowercase for lookup
		// This ensures "Authorization" and "authorization" are both matched
		keyLower := strings.ToLower(k)
		if sensitive[keyLower] {
			// Mask sensitive values with asterisks to hide credentials in logs
			result[k] = "***"
		} else if len(v) > 0 {
			// HTTP headers can have multiple values, but we only capture the first
			// This is sufficient for most logging purposes and keeps output simple
			result[k] = v[0]
		}
	}
	return result
}

// OffsetMS returns milliseconds elapsed since the given start time.
//
// @brief    Calculates elapsed time in milliseconds.
// @param    start Start time for calculation.
// @return   Milliseconds elapsed since start.
//
// @pre      start must be a valid time.Time.
// @note     Uses time.Since for duration calculation.
func OffsetMS(start time.Time) int64 {
	return time.Since(start).Milliseconds()
}

// CaptureWriter provides an interface for recording SSE chunks.
type CaptureWriter interface {
	// RecordChunk records an SSE event with its data.
	RecordChunk(event string, data []byte)
	// Chunks returns all recorded chunks.
	Chunks() []SSEChunk
}

// captureWriter implements CaptureWriter for in-memory chunk storage.
type captureWriter struct {
	// start is the reference time for offset calculations.
	start time.Time
	// chunks stores all recorded SSE chunks.
	chunks []SSEChunk
}

// NewCaptureWriter creates a new CaptureWriter instance.
//
// @brief    Creates CaptureWriter with start time reference.
// @param    start Reference time for offset calculations.
// @return   CaptureWriter interface instance.
//
// @pre      start should be the capture session start time.
// @post     Chunks slice is initialized to empty.
func NewCaptureWriter(start time.Time) CaptureWriter {
	return &captureWriter{
		start:  start,
		chunks: []SSEChunk{},
	}
}

// RecordChunk records an SSE event with its data.
//
// @brief    Appends new SSEChunk to the chunks slice.
// @param    event SSE event type name.
// @param    data  Raw event data bytes.
//
// @pre      None.
// @post     New chunk appended if data is non-empty.
// @note     Silently ignores empty data chunks.
func (cw *captureWriter) RecordChunk(event string, data []byte) {
	if len(data) == 0 {
		return
	}
	chunk := NewSSEChunk(OffsetMS(cw.start), event, data)
	cw.chunks = append(cw.chunks, chunk)
}

// Chunks returns all recorded SSE chunks.
//
// @brief    Returns the slice of recorded chunks.
// @return   Slice of SSEChunk instances.
//
// @pre      None.
// @note     Returns the actual slice, not a copy.
func (cw *captureWriter) Chunks() []SSEChunk {
	return cw.chunks
}
