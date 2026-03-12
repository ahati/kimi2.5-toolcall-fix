// Package logging provides structured logging and request/response capture functionality.
// It captures bidirectional traffic between client and upstream LLM API for debugging
// and analysis purposes.
package logging

import (
	"encoding/json"
	"net/http"
	"sync"
	"time"
)

// recorder provides thread-safe recording of request/response data.
type recorder struct {
	// mu protects concurrent access to data.
	mu sync.Mutex
	// data holds all captured request/response information.
	data *RequestRecorder
	// started is the reference time for offset calculations.
	started time.Time
}

// newRecorder creates a new recorder instance.
//
// @brief    Creates recorder with initial request metadata.
// @param    requestID Unique identifier for the request.
// @param    method    HTTP method (GET, POST, etc.).
// @param    path      Request URL path.
// @param    clientIP  Client's IP address.
// @return   Pointer to new recorder instance.
//
// @pre      None.
// @post     Recorder is initialized with timestamp and metadata.
// @note     Thread-safe for concurrent use.
func newRecorder(requestID, method, path, clientIP string) *recorder {
	now := time.Now()
	return &recorder{
		started: now,
		data: &RequestRecorder{
			RequestID: requestID,
			StartedAt: now,
			Method:    method,
			Path:      path,
			ClientIP:  clientIP,
		},
	}
}

// RecordDownstreamRequest captures the client-to-proxy request.
//
// @brief    Records downstream request headers and body.
// @param    headers HTTP headers from the request.
// @param    body    Raw request body bytes.
//
// @pre      Recorder must be initialized.
// @post     DownstreamRequest field is populated.
// @note     Headers are sanitized before storage.
// @note     Thread-safe via mutex.
func (r *recorder) RecordDownstreamRequest(headers http.Header, body []byte) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.data.DownstreamRequest = &HTTPRequestCapture{
		At:      time.Now(),
		Headers: SanitizeHeaders(headers),
		Body:    body,
		RawBody: body,
	}
}

// RecordUpstreamRequest captures the proxy-to-LLM request.
//
// @brief    Records upstream request headers and body.
// @param    headers HTTP headers for the upstream request.
// @param    body    Raw request body bytes.
//
// @pre      Recorder must be initialized.
// @post     UpstreamRequest field is populated.
// @note     Headers are sanitized before storage.
// @note     Thread-safe via mutex.
func (r *recorder) RecordUpstreamRequest(headers http.Header, body []byte) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.data.UpstreamRequest = &HTTPRequestCapture{
		At:      time.Now(),
		Headers: SanitizeHeaders(headers),
		Body:    body,
		RawBody: body,
	}
}

// RecordUpstreamResponse initializes capture for the LLM-to-proxy response.
//
// @brief    Creates response recorder for upstream SSE capture.
// @param    statusCode HTTP status code from upstream.
// @param    headers    HTTP headers from upstream response.
// @return   Pointer to responseRecorder for chunk recording.
//
// @pre      Recorder must be initialized.
// @post     UpstreamResponse field is initialized.
// @note     Headers are sanitized before storage.
// @note     Thread-safe via mutex.
func (r *recorder) RecordUpstreamResponse(statusCode int, headers http.Header) *responseRecorder {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.data.UpstreamResponse = &SSEResponseCapture{
		StatusCode: statusCode,
		Headers:    SanitizeHeaders(headers),
		Chunks:     []SSEChunk{},
	}

	return &responseRecorder{
		capture: r.data.UpstreamResponse,
		started: r.started,
	}
}

// RecordDownstreamResponse initializes capture for the proxy-to-client response.
//
// @brief    Creates response recorder for downstream SSE capture.
// @return   Pointer to responseRecorder for chunk recording.
//
// @pre      Recorder must be initialized.
// @post     DownstreamResponse field is initialized.
// @note     Thread-safe via mutex.
func (r *recorder) RecordDownstreamResponse() *responseRecorder {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.data.DownstreamResponse = &SSEResponseCapture{
		Chunks: []SSEChunk{},
	}

	return &responseRecorder{
		capture: r.data.DownstreamResponse,
		started: r.started,
	}
}

// Data returns the accumulated request recorder data.
//
// @brief    Returns captured data for storage.
// @return   Pointer to RequestRecorder with all captured data.
//
// @pre      None.
// @note     Thread-safe via mutex.
// @note     Returns pointer to internal data, not a copy.
func (r *recorder) Data() *RequestRecorder {
	r.mu.Lock()
	defer r.mu.Unlock()

	return r.data
}

// responseRecorder captures individual SSE chunks for a response.
type responseRecorder struct {
	// capture is the SSEResponseCapture to append chunks to.
	capture *SSEResponseCapture
	// started is the reference time for offset calculations.
	started time.Time
}

// RecordChunk records an SSE event with raw string data.
//
// @brief    Appends SSE chunk from raw string data.
// @param    event SSE event type name.
// @param    raw   Raw event data string.
//
// @pre      None.
// @post     New chunk appended to capture.Chunks.
// @note     Attempts to parse raw as JSON for Data field.
// @note     Silently ignores nil receiver or capture.
func (rr *responseRecorder) RecordChunk(event, raw string) {
	if rr == nil || rr.capture == nil {
		return
	}

	chunk := SSEChunk{
		OffsetMS: OffsetMS(rr.started),
		Event:    event,
		Raw:      raw,
	}

	var data json.RawMessage
	if err := json.Unmarshal([]byte(raw), &data); err == nil {
		chunk.Data = data
	}

	rr.capture.Chunks = append(rr.capture.Chunks, chunk)
}

// RecordChunkBytes records an SSE event with byte data.
//
// @brief    Appends SSE chunk from byte data.
// @param    event SSE event type name.
// @param    data  Raw event data bytes.
//
// @pre      None.
// @post     Delegates to RecordChunk with string conversion.
// @note     Convenience wrapper for byte input.
func (rr *responseRecorder) RecordChunkBytes(event string, data []byte) {
	rr.RecordChunk(event, string(data))
}
