// Package downstream provides HTTP handlers for the proxy's client-facing API endpoints.
// It implements a unified stream handler that works with protocol adapters to support
// multiple API formats (OpenAI, Anthropic, Bridge).
package downstream

import (
	"io"

	"ai-proxy/logging"

	"github.com/tmaxmax/go-sse"
)

// SSECaptureReader wraps an io.Reader to capture SSE events during iteration.
// It provides a convenient way to process SSE events while logging them
// to a CaptureWriter for debugging and monitoring purposes.
//
// @brief    Wraps an SSE stream reader to capture events during processing.
// @field    reader Underlying io.Reader containing the SSE stream.
// @field    capture CaptureWriter for recording SSE events.
//
// @note     Events with empty data fields are not captured but still processed.
// @note     The reader is consumed in full during ForEach iteration.
type SSECaptureReader struct {
	reader  io.Reader
	capture logging.CaptureWriter
}

// NewSSECaptureReader creates a new SSECaptureReader with the given reader and capture.
//
// @brief    Creates an SSE reader that captures events while iterating.
// @param    reader io.Reader providing the SSE stream data.
// @param    capture CaptureWriter for recording SSE events.
// @return   Pointer to a new SSECaptureReader instance.
//
// @pre      reader must not be nil and must contain valid SSE data.
// @pre      capture must not be nil.
// @post     SSE events are captured during ForEach iteration.
func NewSSECaptureReader(reader io.Reader, capture logging.CaptureWriter) *SSECaptureReader {
	return &SSECaptureReader{
		reader:  reader,
		capture: capture,
	}
}

// ForEach iterates over all SSE events, capturing and processing each one.
//
// @brief    Iterates through SSE events, capturing each to the CaptureWriter.
// @param    fn Callback function for each event; return false to stop iteration.
// @return   Error if reading the SSE stream fails.
//
// @note     Events with non-empty data are recorded with their event type.
// @note     Iteration stops if the callback returns false or an error occurs.
// @note     Returns nil on successful completion of iteration.
//
// @post     All processed events are recorded to the capture writer.
func (scr *SSECaptureReader) ForEach(fn func(sse.Event) bool) error {
	for ev, err := range sse.Read(scr.reader, nil) {
		if err != nil {
			return err
		}
		if ev.Data != "" {
			scr.capture.RecordChunk(ev.Type, []byte(ev.Data))
		}
		if !fn(ev) {
			break
		}
	}
	return nil
}
