package handlers

import (
	"bytes"
	"net/http"

	"ai-proxy/capture"
)

// ResponseRecorder wraps an http.ResponseWriter and captures SSE events for logging.
// It implements the http.ResponseWriter interface to intercept all writes to the
// underlying response writer and capture SSE event data.
//
// This is used to record downstream responses for debugging and analysis purposes.
// The captured data can be written to storage for later inspection.
//
// @invariant r.writer != nil after construction
// @invariant r.capture may be nil (capture disabled)
type ResponseRecorder struct {
	// writer is the underlying http.ResponseWriter that receives the actual response.
	// All Write calls are delegated to this writer after capturing.
	// Must not be nil.
	writer http.ResponseWriter

	// capture is the CaptureWriter that records SSE events.
	// May be nil to disable capture. When nil, writes are only delegated.
	capture capture.CaptureWriter
}

// NewResponseRecorder creates a new recorder that wraps the given writer
// and captures events to the provided CaptureWriter.
//
// @param writer - The underlying http.ResponseWriter to wrap.
//
//	Must not be nil. All writes are delegated to this writer.
//
// @param capture - The CaptureWriter for recording SSE events.
//
//	May be nil to disable capture.
//
// @return Pointer to newly allocated ResponseRecorder. Never returns nil.
//
// @pre writer != nil
// @post All writes go through the recorder to the underlying writer.
// @post Events are captured to the CaptureWriter if non-nil.
func NewResponseRecorder(writer http.ResponseWriter, capture capture.CaptureWriter) *ResponseRecorder {
	return &ResponseRecorder{
		writer:  writer,
		capture: capture,
	}
}

// Write captures SSE event data before delegating to the underlying writer.
// This method implements the io.Writer interface and intercepts all response
// data to extract and record SSE events.
//
// @param data - The byte slice to write. Expected to contain SSE-formatted data.
// @return Number of bytes written (from underlying writer).
// @return Error if the underlying writer returns an error.
//
// @pre r.writer != nil
// @post If r.capture != nil, SSE events are extracted and recorded.
// @post Data is always written to the underlying writer regardless of capture.
// @note SSE events are extracted by parsing "data: " and "event: " lines.
func (r *ResponseRecorder) Write(data []byte) (int, error) {
	// Capture SSE event data if capture is enabled
	// Only capture when capture writer is configured (non-nil)
	if r.capture != nil {
		// Extract event type from SSE data (e.g., "message", "content_block_delta")
		eventType := extractEventType(data)
		// Extract data portion from SSE data (content after "data: " prefix)
		dataPart := extractDataPart(data)
		// Only record if there is actual data to capture
		// Empty data indicates a keepalive or empty event
		if len(dataPart) > 0 {
			r.capture.RecordChunk(eventType, dataPart)
		}
	}
	// Always delegate write to underlying writer
	// This ensures the actual response is still sent to the client
	return r.writer.Write(data)
}

// Header returns the underlying writer's header map.
// This method implements the http.ResponseWriter interface.
//
// @return Header map from the underlying response writer.
//
// @pre r.writer != nil
// @post Returned header can be modified to set response headers.
func (r *ResponseRecorder) Header() http.Header {
	return r.writer.Header()
}

// WriteHeader writes the status code to the underlying writer.
// This method implements the http.ResponseWriter interface.
//
// @param statusCode - HTTP status code to set (e.g., 200, 404, 500).
//
// @pre r.writer != nil
// @post Status code is set on the underlying writer.
// @note This should be called before any Write calls.
func (r *ResponseRecorder) WriteHeader(statusCode int) {
	r.writer.WriteHeader(statusCode)
}

// extractDataPart parses SSE data and returns the content after the "data: " prefix.
// SSE format has lines like "data: {json_content}".
//
// @param data - Raw SSE data bytes to parse.
// @return Content after "data: " prefix, or nil if not found.
//
// @pre data may contain multiple lines separated by \n.
// @post Returns only the first "data: " line's content.
// @note Returns nil if no "data: " line is found.
func extractDataPart(data []byte) []byte {
	// Split data into lines to find the data line
	lines := bytes.Split(data, []byte("\n"))
	for _, line := range lines {
		// Check if this line starts with "data: " prefix
		if bytes.HasPrefix(line, []byte("data: ")) {
			// Return content after the prefix
			return bytes.TrimPrefix(line, []byte("data: "))
		}
	}
	// No data line found - return nil
	return nil
}

// extractEventType parses SSE data and returns the event type if present.
// SSE format has lines like "event: message_start".
//
// @param data - Raw SSE data bytes to parse.
// @return Event type string, or empty string if not found.
//
// @pre data may contain multiple lines separated by \n.
// @post Returns only the first "event: " line's content.
// @note Returns empty string if no "event: " line is found.
func extractEventType(data []byte) string {
	// Split data into lines to find the event type line
	lines := bytes.Split(data, []byte("\n"))
	for _, line := range lines {
		// Check if this line starts with "event: " prefix
		if bytes.HasPrefix(line, []byte("event: ")) {
			// Return event type after the prefix
			return string(bytes.TrimPrefix(line, []byte("event: ")))
		}
	}
	// No event type line found - return empty string
	return ""
}
