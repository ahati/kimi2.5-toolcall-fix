// Package transform provides interfaces for transforming SSE events.
// This package defines the core abstractions for Server-Sent Events (SSE)
// transformation in the AI proxy system. Implementations process streaming
// data from LLM APIs and convert it to client-compatible formats.
package transform

import (
	"io"

	"github.com/tmaxmax/go-sse"
)

// SSETransformer defines the interface for transforming server-sent events.
// Implementations process SSE events and write transformed output.
//
// @brief Interface for transforming server-sent events from upstream LLM APIs.
//
// @note Implementations must be safe for concurrent use if handlers are shared
//
//	across multiple requests. Each request should use its own transformer
//	instance unless explicitly documented as thread-safe.
//
// @note The transformation pipeline follows: Transform (multiple calls) -> Flush -> Close.
//
//	Callers must ensure Close() is called to release resources.
//
// @pre The output writer must be properly initialized before creating implementations.
// @post After Close(), the transformer must not be used for further transformations.
type SSETransformer interface {
	// Transform processes a single SSE event.
	//
	// @brief Processes a single SSE event and writes transformed output.
	//
	// @param event The SSE event to process. Must not be nil.
	//              The event may contain data in various formats depending on
	//              the upstream API (OpenAI, Anthropic, proprietary formats).
	//
	// @return error Returns nil on success.
	//               Returns error if:
	//               - Event data cannot be parsed
	//               - Output write fails
	//               - Internal state is corrupted
	//
	// @pre event must not be nil.
	// @pre The transformer must not be in a closed state.
	// @post Output is written to the configured writer.
	// @post Internal parser state is updated for subsequent events.
	//
	// @note Implementations may buffer partial data until complete tokens
	//       are recognized. Callers should invoke Flush() after the last event.
	Transform(event *sse.Event) error

	// Flush writes any buffered data to the output.
	//
	// @brief Flushes all buffered transformation output to the underlying writer.
	//
	// @return error Returns nil on success.
	//               Returns error if:
	//               - Output write fails
	//               - Buffered data is malformed
	//
	// @pre Transform() must have been called for events requiring processing.
	// @pre The transformer must not be in a closed state.
	// @post All buffered data is written to output.
	// @post Internal buffers are cleared.
	//
	// @note Must be called after the last Transform() call to ensure all
	//       pending output is written. Not calling Flush() may result in
	//       truncated output at the client.
	Flush() error

	// Close flushes and releases resources.
	//
	// @brief Releases all resources held by the transformer.
	//
	// @return error Returns nil on success.
	//               Returns error if:
	//               - Flush() fails during close
	//               - Resources cannot be released cleanly
	//
	// @pre None (safe to call even if never used).
	// @post The transformer is in a closed state and must not be used.
	// @post All resources are released.
	//
	// @note Close() calls Flush() internally before releasing resources.
	//       After Close(), subsequent calls to Transform(), Flush(), or Close()
	//       may panic or return errors depending on implementation.
	Close() error
}

// FlushingWriter combines io.Writer with a Flush method for buffered output.
// This interface is used to ensure output destinations can be explicitly flushed.
//
// @brief Interface combining io.Writer with explicit flush capability.
//
// @note Implementations typically wrap buffered writers (e.g., bufio.Writer)
//
//	to provide explicit control over when data is written to the underlying
//	destination. This is critical for streaming responses where data must
//	be sent immediately rather than buffered.
//
// @pre The underlying writer must be properly initialized and not closed.
// @post After Flush(), all buffered data is written to the underlying destination.
type FlushingWriter interface {
	io.Writer

	// Flush writes buffered data to the underlying writer.
	//
	// @brief Flushes all buffered data to the underlying writer.
	//
	// @return error Returns nil on success.
	//               Returns error if:
	//               - Underlying write fails
	//               - Writer is closed
	//
	// @pre The writer must be open and accepting writes.
	// @post All buffered data is written to the underlying destination.
	//
	// @note For HTTP response writers, Flush() ensures data is sent to the
	//       client immediately rather than held in OS buffers. This is
	//       essential for real-time streaming of LLM responses.
	Flush() error
}
