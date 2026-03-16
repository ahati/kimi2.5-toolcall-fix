// Package transform provides interfaces and utilities for transforming SSE events.
// This file provides common SSE writing functionality to reduce code duplication.
package transform

import (
	"fmt"
	"io"
)

// SSEWriter provides common Server-Sent Events writing functionality.
// It consolidates the repeated SSE formatting patterns used across transformers.
//
// SSE Format:
//   - Data only: "data: <json>\n\n"
//   - With event type: "event: <type>\ndata: <json>\n\n"
//   - Done marker: "data: [DONE]\n\n"
type SSEWriter struct {
	w io.Writer
}

// NewSSEWriter creates a new SSEWriter wrapping the provided writer.
func NewSSEWriter(w io.Writer) *SSEWriter {
	return &SSEWriter{w: w}
}

// WriteData writes a data-only SSE event.
// Format: "data: <data>\n\n"
func (w *SSEWriter) WriteData(data []byte) error {
	if len(data) == 0 {
		return nil
	}
	if _, err := w.w.Write([]byte("data: ")); err != nil {
		return err
	}
	if _, err := w.w.Write(data); err != nil {
		return err
	}
	_, err := w.w.Write([]byte("\n\n"))
	return err
}

// WriteEvent writes an SSE event with a type.
// Format: "event: <type>\ndata: <data>\n\n"
func (w *SSEWriter) WriteEvent(eventType string, data []byte) error {
	if _, err := fmt.Fprintf(w.w, "event: %s\n", eventType); err != nil {
		return err
	}
	return w.WriteData(data)
}

// WriteDone writes the SSE stream termination marker.
// Format: "data: [DONE]\n\n"
func (w *SSEWriter) WriteDone() error {
	_, err := w.w.Write([]byte("data: [DONE]\n\n"))
	return err
}

// WriteRaw writes raw bytes directly to the underlying writer.
// Use for pre-formatted content that doesn't need SSE framing.
func (w *SSEWriter) WriteRaw(data []byte) (int, error) {
	return w.w.Write(data)
}

// Writer returns the underlying io.Writer.
func (w *SSEWriter) Writer() io.Writer {
	return w.w
}
