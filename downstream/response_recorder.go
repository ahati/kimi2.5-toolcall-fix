// Package downstream provides HTTP handlers for the proxy's client-facing API endpoints.
// It implements a unified stream handler that works with protocol adapters to support
// multiple API formats (OpenAI, Anthropic, Bridge).
package downstream

import (
	"net/http"
	"regexp"

	"ai-proxy/logging"
)

// eventTypeRegex matches the "event:" line in SSE format to extract the event type.
var eventTypeRegex = regexp.MustCompile(`^event:\s*(\S+)`)

// ResponseRecorder wraps an http.ResponseWriter to capture SSE chunks.
// It implements the http.ResponseWriter interface for transparent proxying
// and the CaptureWriter interface for homogeneous logging.
//
// @brief    Wraps an HTTP response writer to intercept and log SSE events.
// @field    writer Underlying HTTP response writer to delegate writes to.
// @field    capture CaptureWriter instance for recording SSE chunks.
//
// @note     Implements http.ResponseWriter interface for drop-in replacement.
// @note     Each Write call extracts and logs the SSE data portion.
type ResponseRecorder struct {
	writer  http.ResponseWriter
	capture logging.CaptureWriter
}

// NewResponseRecorder creates a new ResponseRecorder that wraps the given writer.
//
// @brief    Creates a ResponseRecorder that captures SSE chunks while proxying data.
// @param    writer HTTP response writer to wrap for data forwarding.
// @param    capture CaptureWriter for recording SSE chunks for logging.
// @return   Pointer to a new ResponseRecorder instance.
//
// @note     The capture parameter receives extracted SSE data without "data: " prefix.
//
// @pre      writer must not be nil.
// @pre      capture must not be nil.
// @post     All writes go through both capture and the underlying writer.
func NewResponseRecorder(writer http.ResponseWriter, capture logging.CaptureWriter) *ResponseRecorder {
	return &ResponseRecorder{
		writer:  writer,
		capture: capture,
	}
}

// Write intercepts data written to the response and logs SSE chunks.
//
// @brief    Writes data to the underlying writer while capturing SSE events.
// @param    data Byte slice containing SSE formatted data to write.
// @return   Number of bytes written to the underlying writer.
// @return   Error if the underlying write fails.
//
// @note     Extracts the JSON data portion from SSE format before logging.
// @note     Extracts the event type from SSE format, defaults to "message".
// @note     Empty data slices are not captured but still written.
//
// @post     If data is non-empty, it is recorded with the capture writer.
func (r *ResponseRecorder) Write(data []byte) (int, error) {
	if len(data) > 0 {
		dataForLogging := extractDataPart(data)
		event := extractEventType(data)
		r.capture.RecordChunk(event, dataForLogging)
	}
	return r.writer.Write(data)
}

// extractDataPart extracts the JSON data portion from an SSE formatted message.
//
// @brief    Parses SSE format to extract the content after "data: " prefix.
// @param    data Raw SSE formatted bytes (e.g., "data: {...}\n\n").
// @return   Byte slice containing just the JSON data, or original data if not SSE format.
//
// @note     Handles both "data: {...}\n\n" and "event: xxx\ndata: {...}\n\n" formats.
// @note     Returns the original data if "data: " prefix is not found.
func extractDataPart(data []byte) []byte {
	s := string(data)

	if idx := findDataPrefix(s); idx >= 0 {
		start := idx + 6
		end := len(s)
		if nlIdx := indexOfDoubleNewline(s[start:]); nlIdx >= 0 {
			end = start + nlIdx
		}
		return []byte(s[start:end])
	}
	return data
}

// findDataPrefix locates the "data: " prefix in an SSE message.
//
// @brief    Searches for the position of "data: " in the given string.
// @param    s String to search for the data prefix.
// @return   Index of the start of "data: " prefix, or -1 if not found.
//
// @note     Uses linear search to find exact prefix match.
func findDataPrefix(s string) int {
	for i := 0; i < len(s)-5; i++ {
		if s[i:i+6] == "data: " {
			return i
		}
	}
	return -1
}

// indexOfDoubleNewline finds the position of a double newline sequence.
//
// @brief    Locates the first occurrence of "\n\n" in the given string.
// @param    s String to search for double newline.
// @return   Index of the first newline in the pair, or -1 if not found.
//
// @note     Double newline marks the end of an SSE event.
func indexOfDoubleNewline(s string) int {
	for i := 0; i < len(s)-1; i++ {
		if s[i] == '\n' && s[i+1] == '\n' {
			return i
		}
	}
	return -1
}

// Header returns the HTTP headers from the underlying response writer.
//
// @brief    Provides access to the response headers for modification.
// @return   HTTP header map that can be modified before writing.
//
// @note     Implements http.ResponseWriter interface.
func (r *ResponseRecorder) Header() http.Header {
	return r.writer.Header()
}

// WriteHeader writes the HTTP status code to the underlying response writer.
//
// @brief    Sets the HTTP response status code.
// @param    statusCode HTTP status code to send (e.g., 200, 404, 500).
//
// @note     Implements http.ResponseWriter interface.
// @note     Must be called before any Write calls.
func (r *ResponseRecorder) WriteHeader(statusCode int) {
	r.writer.WriteHeader(statusCode)
}

// extractEventType extracts the SSE event type from raw SSE data.
//
// @brief    Parses the "event:" line from SSE formatted data.
// @param    data Raw SSE bytes potentially containing an event type.
// @return   The extracted event type string, or "message" if not found.
//
// @note     Uses regex to match "event: <type>" pattern at line start.
// @note     Default event type is "message" per SSE specification.
func extractEventType(data []byte) string {
	matches := eventTypeRegex.FindSubmatch(data)
	if len(matches) > 1 {
		return string(matches[1])
	}
	return "message"
}
