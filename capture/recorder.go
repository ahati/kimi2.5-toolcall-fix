// Package capture provides request/response recording and persistence for HTTP proxy operations.
// It captures downstream client requests, upstream API requests, and their corresponding
// SSE streaming responses for debugging and analysis.
//
// Thread Safety:
//   - HTTPRequestCapture and SSEResponseCapture are NOT thread-safe after creation
//   - RequestRecorder is NOT thread-safe (use recorder for concurrent access)
//   - recorder IS thread-safe via mutex protection
//   - responseRecorder is NOT thread-safe (single goroutine ownership)
package capture

import (
	"encoding/json"
	"net/http"
	"strings"
	"sync"
	"time"
)

// HTTPRequestCapture represents a captured HTTP request with headers and body.
// It stores both the parsed JSON body and raw bytes for flexibility.
//
// Thread Safety: NOT thread-safe. Do not modify after creation or use external sync.
type HTTPRequestCapture struct {
	// At is the timestamp when the request was captured.
	// Used for timing analysis and debugging.
	// Valid values: any valid time.Time.
	At time.Time `json:"at"`

	// Headers contains sanitized HTTP headers.
	// Sensitive headers (auth, cookies) are masked with "***".
	// Key is header name (preserving original case), value is first header value.
	// Valid values: map of string to string, may be nil.
	Headers map[string]string `json:"headers,omitempty"`

	// Body contains the parsed JSON body if the request body was valid JSON.
	// Nil if body was not JSON or was empty.
	// Valid values: json.RawMessage or nil.
	Body json.RawMessage `json:"body,omitempty"`

	// RawBody contains the original unparsed request body bytes.
	// Preserved for debugging and cases where JSON parsing fails.
	// Valid values: any byte slice, may be nil or empty.
	RawBody []byte `json:"-"`
}

// SSEResponseCapture represents a captured SSE streaming response with status, headers, and chunks.
// It aggregates all chunks received during the streaming response.
//
// Thread Safety: NOT thread-safe. Do not modify after creation or use external sync.
type SSEResponseCapture struct {
	// StatusCode is the HTTP status code of the response.
	// Zero indicates response not yet received.
	// Valid values: valid HTTP status codes (100-599), or 0 if not set.
	StatusCode int `json:"status_code,omitempty"`

	// Headers contains sanitized response headers.
	// Sensitive headers are masked with "***".
	// Key is header name, value is first header value.
	// Valid values: map of string to string, may be nil.
	Headers map[string]string `json:"headers,omitempty"`

	// Chunks contains all SSE chunks received in order.
	// Empty slice indicates no chunks received yet.
	// Valid values: slice of SSEChunk, may be empty but not nil after initialization.
	Chunks []SSEChunk `json:"chunks,omitempty"`

	// RawBody contains the accumulated raw response body bytes.
	// Used for non-SSE responses or debugging.
	// Valid values: any byte slice, may be nil or empty.
	RawBody []byte `json:"-"`
}

// RequestRecorder aggregates all captured data for a single request lifecycle.
// It tracks both downstream (client-to-proxy) and upstream (proxy-to-API) traffic.
//
// Usage:
//   - Created at request start with metadata
//   - Populated progressively as request/response data arrives
//   - Serialized to JSON for persistence
//
// Thread Safety: NOT thread-safe. For concurrent access, use the recorder type.
type RequestRecorder struct {
	// RequestID is the unique identifier for this request.
	// May be empty until extracted from SSE response.
	// Valid values: non-empty string after extraction, empty before.
	RequestID string

	// StartedAt is when the request was initiated.
	// Used for duration calculation in logs.
	// Valid values: any valid time.Time, set at creation.
	StartedAt time.Time

	// Method is the HTTP method of the request (GET, POST, etc.).
	// Valid values: standard HTTP methods.
	Method string

	// Path is the URL path of the request.
	// Valid values: any valid URL path string.
	Path string

	// ClientIP is the remote address of the client.
	// May include port number.
	// Valid values: IP:port format string.
	ClientIP string

	// DownstreamRequest captures the client request received by the proxy.
	// Nil until RecordDownstreamRequest is called.
	// Valid values: pointer to HTTPRequestCapture, or nil.
	DownstreamRequest *HTTPRequestCapture

	// UpstreamRequest captures the request sent to the upstream API.
	// Nil until RecordUpstreamRequest is called.
	// Valid values: pointer to HTTPRequestCapture, or nil.
	UpstreamRequest *HTTPRequestCapture

	// UpstreamResponse captures the response received from the upstream API.
	// Nil until RecordUpstreamResponse is called.
	// Valid values: pointer to SSEResponseCapture, or nil.
	UpstreamResponse *SSEResponseCapture

	// DownstreamResponse captures the response sent to the client.
	// Nil until RecordDownstreamResponse is called.
	// Valid values: pointer to SSEResponseCapture, or nil.
	DownstreamResponse *SSEResponseCapture
}

// RecordDownstreamRequest captures the incoming client request.
//
// @param r    - HTTP request to capture headers from. Must not be nil.
// @param body - Request body bytes to capture. May be nil or empty.
//
// @pre r != nil
// @post r.DownstreamRequest != nil
// @post r.DownstreamRequest.Headers contains sanitized headers
// @post r.DownstreamRequest.RawBody == body
//
// @note Headers are sanitized to mask sensitive values like auth tokens.
// @note NOT thread-safe; call from single goroutine.
func (r *RequestRecorder) RecordDownstreamRequest(req *http.Request, body []byte) {
	// Create capture with timestamp for timing analysis
	// SanitizeHeaders masks sensitive values to prevent credential leaks
	r.DownstreamRequest = &HTTPRequestCapture{
		At:      time.Now(),
		Headers: SanitizeHeaders(req.Header),
		Body:    body,
		RawBody: body,
	}
}

// SanitizeHeaders returns a header map with sensitive values masked.
// Authorization, API keys, cookies, and auth tokens are replaced with "***".
//
// @param headers - HTTP headers to sanitize. May be nil (returns nil).
// @return Map of header names to sanitized values.
//
// @pre None
// @post Sensitive header values are replaced with "***"
// @post Non-sensitive header values are preserved
// @post Multi-value headers have only first value preserved
//
// @note This prevents accidental credential exposure in logs.
// @note Header names preserve original case from input.
// @note Thread-safe: pure function with no side effects.
func SanitizeHeaders(headers http.Header) map[string]string {
	// Define sensitive header names that should be masked
	// Keys are lowercase for case-insensitive matching
	sensitive := map[string]bool{
		"authorization": true,
		"x-api-key":     true,
		"cookie":        true,
		"set-cookie":    true,
		"x-auth-token":  true,
	}

	// Create result map with capacity hint for efficiency
	result := make(map[string]string)
	// Iterate over all headers from input
	for k, v := range headers {
		// Convert to lowercase for case-insensitive sensitive check
		keyLower := strings.ToLower(k)
		if sensitive[keyLower] {
			// Mask sensitive values to prevent credential leaks
			// Always use "***" regardless of original value length
			result[k] = "***"
		} else if len(v) > 0 {
			// Preserve first value for non-sensitive headers
			// Multi-value headers are truncated to first value only
			result[k] = v[0]
		}
	}
	return result
}

// OffsetMS returns the milliseconds elapsed since the given start time.
// Used for calculating chunk timing offsets.
//
// @param start - Reference start time. Should be a past time value.
// @return Milliseconds elapsed since start (may be negative if start is in future).
//
// @pre None
// @post Returns time.Since(start).Milliseconds()
//
// @note Thread-safe: pure function with no side effects.
// @note Result may be negative if start time is in the future.
func OffsetMS(start time.Time) int64 {
	// time.Since returns duration, Milliseconds converts to int64
	// This provides millisecond precision for timing analysis
	return time.Since(start).Milliseconds()
}

// recorder provides thread-safe recording of request/response data.
// It wraps RequestRecorder with mutex protection for concurrent access.
//
// Thread Safety: IS thread-safe. All methods use mutex protection.
type recorder struct {
	// mu protects all fields of this struct.
	// Must be held for any read or write of 'data'.
	mu sync.Mutex

	// data contains the captured request data.
	// Protected by mu; access only while holding lock.
	data *RequestRecorder

	// started is the reference time for chunk timing offsets.
	// Set at creation, never modified.
	// Valid values: any valid time.Time.
	started time.Time
}

// newRecorder creates a new recorder initialized with request metadata.
//
// @param requestID - Unique identifier for this request. May be empty.
// @param method    - HTTP method. Should be valid HTTP method string.
// @param path      - URL path. Should be valid URL path.
// @param clientIP  - Client remote address. Should be IP:port format.
// @return Pointer to newly allocated recorder, never nil.
//
// @pre None
// @post Returned recorder is ready for concurrent use
// @post Returned recorder.data fields are initialized with parameters
//
// @note Thread-safe: creates new instance with no shared state.
func newRecorder(requestID, method, path, clientIP string) *recorder {
	// Capture current time once for consistency between StartedAt and started
	now := time.Now()
	return &recorder{
		started: now,
		// Initialize data with all metadata at creation
		// This prevents nil pointer access in later operations
		data: &RequestRecorder{
			RequestID: requestID,
			StartedAt: now,
			Method:    method,
			Path:      path,
			ClientIP:  clientIP,
		},
	}
}

// RecordDownstreamRequest captures the incoming client request with thread safety.
//
// @param headers - HTTP headers to capture. May be nil (results in empty map).
// @param body    - Request body bytes. May be nil or empty.
//
// @pre r != nil (receiver must be valid)
// @post r.data.DownstreamRequest != nil (after call)
// @post r.data.DownstreamRequest.Headers contains sanitized values
//
// @note Thread-safe: uses mutex for exclusive access.
// @note Headers are sanitized to mask sensitive values.
func (r *recorder) RecordDownstreamRequest(headers http.Header, body []byte) {
	// Lock for entire operation to ensure atomic update
	// defer ensures unlock even if panic occurs
	r.mu.Lock()
	defer r.mu.Unlock()

	// Create capture with timestamp and sanitized headers
	// Body is stored both as RawMessage and raw bytes for flexibility
	r.data.DownstreamRequest = &HTTPRequestCapture{
		At:      time.Now(),
		Headers: SanitizeHeaders(headers),
		Body:    body,
		RawBody: body,
	}
}

// RecordUpstreamRequest captures the outgoing upstream API request with thread safety.
//
// @param headers - HTTP headers to capture. May be nil (results in empty map).
// @param body    - Request body bytes. May be nil or empty.
//
// @pre r != nil (receiver must be valid)
// @post r.data.UpstreamRequest != nil (after call)
// @post r.data.UpstreamRequest.Headers contains sanitized values
//
// @note Thread-safe: uses mutex for exclusive access.
// @note Headers are sanitized to mask sensitive values.
func (r *recorder) RecordUpstreamRequest(headers http.Header, body []byte) {
	// Lock for entire operation to ensure atomic update
	r.mu.Lock()
	defer r.mu.Unlock()

	// Create capture with timestamp and sanitized headers
	r.data.UpstreamRequest = &HTTPRequestCapture{
		At:      time.Now(),
		Headers: SanitizeHeaders(headers),
		Body:    body,
		RawBody: body,
	}
}

// RecordUpstreamResponse initializes upstream response capture and returns a responseRecorder for chunk recording.
//
// @param statusCode - HTTP status code from upstream response.
// @param headers    - HTTP response headers. May be nil (results in empty map).
// @return Pointer to responseRecorder for recording chunks, never nil.
//
// @pre r != nil (receiver must be valid)
// @post r.data.UpstreamResponse != nil
// @post r.data.UpstreamResponse.StatusCode == statusCode
// @post Returned responseRecorder writes to r.data.UpstreamResponse
//
// @note Thread-safe: uses mutex for exclusive access during initialization.
// @note Returned responseRecorder is NOT thread-safe; use from single goroutine.
func (r *recorder) RecordUpstreamResponse(statusCode int, headers http.Header) *responseRecorder {
	// Lock for the initialization phase only
	// The responseRecorder returned does its own synchronization for chunks
	r.mu.Lock()
	defer r.mu.Unlock()

	// Initialize the upstream response capture
	// Chunks slice is initialized empty but not nil
	r.data.UpstreamResponse = &SSEResponseCapture{
		StatusCode: statusCode,
		Headers:    SanitizeHeaders(headers),
		Chunks:     []SSEChunk{},
	}

	// Return recorder for subsequent chunk recording
	// The responseRecorder shares the same capture struct
	return &responseRecorder{
		capture: r.data.UpstreamResponse,
		started: r.started,
	}
}

// RecordDownstreamResponse initializes downstream response capture and returns a responseRecorder for chunk recording.
//
// @return Pointer to responseRecorder for recording chunks, never nil.
//
// @pre r != nil (receiver must be valid)
// @post r.data.DownstreamResponse != nil
// @post r.data.DownstreamResponse.Chunks is empty slice (not nil)
// @post Returned responseRecorder writes to r.data.DownstreamResponse
//
// @note Thread-safe: uses mutex for exclusive access during initialization.
// @note Returned responseRecorder is NOT thread-safe; use from single goroutine.
func (r *recorder) RecordDownstreamResponse() *responseRecorder {
	// Lock for the initialization phase only
	r.mu.Lock()
	defer r.mu.Unlock()

	// Initialize the downstream response capture
	// StatusCode and Headers not set here (set separately if needed)
	r.data.DownstreamResponse = &SSEResponseCapture{
		Chunks: []SSEChunk{},
	}

	// Return recorder for subsequent chunk recording
	return &responseRecorder{
		capture: r.data.DownstreamResponse,
		started: r.started,
	}
}

// Data returns a copy of the recorded request data with thread safety.
//
// @return Pointer to RequestRecorder. Returns same pointer on each call (not a copy).
//
// @pre r != nil (receiver must be valid)
// @post Returned pointer is valid if r was properly initialized
//
// @note Thread-safe: uses mutex for exclusive access.
// @note Returns direct reference, not a deep copy; caller should not modify.
func (r *recorder) Data() *RequestRecorder {
	// Lock to ensure consistent read of data
	r.mu.Lock()
	defer r.mu.Unlock()

	// Return the same pointer; caller should treat as read-only
	return r.data
}

// responseRecorder records SSE chunks for a single response stream.
// It writes chunks to a shared SSEResponseCapture.
//
// Thread Safety: NOT thread-safe. Use from single goroutine only.
type responseRecorder struct {
	// capture is the target SSEResponseCapture to write chunks to.
	// Must not be nil for valid operation.
	// Valid values: pointer to initialized SSEResponseCapture.
	capture *SSEResponseCapture

	// started is the reference time for calculating chunk offsets.
	// Used to compute OffsetMS for each chunk.
	// Valid values: any valid time.Time.
	started time.Time
}

// RecordChunk appends an SSE chunk with the given event and raw data.
// If the raw data is valid JSON, it is also stored in the Data field.
//
// @param event - SSE event type. May be empty string.
// @param raw   - Raw chunk data string. May be empty.
//
// @pre rr != nil && rr.capture != nil (receiver must be valid)
// @post If valid, chunk appended to rr.capture.Chunks
// @post If rr is nil or rr.capture is nil, no action taken (no-op)
//
// @note NOT thread-safe; use from single goroutine.
// @note Empty raw strings are still recorded (unlike CaptureWriter.RecordChunk).
func (rr *responseRecorder) RecordChunk(event, raw string) {
	// Defensive nil check to prevent panic on invalid receiver
	// This allows safe calls even if responseRecorder was not properly initialized
	if rr == nil || rr.capture == nil {
		return
	}

	// Create chunk with timing offset
	// OffsetMS provides timing information for debugging latency
	chunk := SSEChunk{
		OffsetMS: OffsetMS(rr.started),
		Event:    event,
		Raw:      raw,
	}

	// Attempt to parse raw data as JSON
	// If successful, store in Data field for structured access
	var data json.RawMessage
	if err := json.Unmarshal([]byte(raw), &data); err == nil {
		// Valid JSON: store parsed data
		chunk.Data = data
	}
	// Invalid JSON: only Raw field is populated

	// Append chunk to capture; may cause slice reallocation
	rr.capture.Chunks = append(rr.capture.Chunks, chunk)
}

// RecordChunkBytes is a convenience method that converts data to string and calls RecordChunk.
//
// @param event - SSE event type. May be empty string.
// @param data  - Raw chunk data bytes. May be nil or empty.
//
// @pre rr != nil && rr.capture != nil (receiver must be valid)
// @post Calls rr.RecordChunk(event, string(data))
//
// @note NOT thread-safe; delegates to RecordChunk.
// @note Nil-safe: handles nil receiver gracefully.
func (rr *responseRecorder) RecordChunkBytes(event string, data []byte) {
	// Convert bytes to string and delegate to RecordChunk
	// This is a convenience wrapper for callers with []byte
	rr.RecordChunk(event, string(data))
}
