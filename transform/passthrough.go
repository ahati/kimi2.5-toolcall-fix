package transform

import (
	"io"

	"github.com/tmaxmax/go-sse"
)

// PassthroughTransformer is an SSE transformer that passes events through unchanged.
// It is used when the upstream already returns data in the desired format.
//
// @brief SSE transformer that passes events through without modification.
//
// @note Use this transformer when upstream format matches downstream format.
// @note This transformer is lightweight and has no internal state.
// @note The transformer is thread-safe for concurrent Transform calls.
//
// Use cases:
//   - Anthropic upstream → Anthropic downstream (no transformation needed)
//   - OpenAI upstream → OpenAI downstream (no transformation needed)
type PassthroughTransformer struct {
	// output is the destination writer for SSE data.
	output io.Writer
}

// NewPassthroughTransformer creates a transformer that passes events through unchanged.
//
// @brief Creates a new PassthroughTransformer.
//
// @param output The destination writer for SSE data.
//
//	Must be non-nil and writable.
//	Should support flushing for streaming responses.
//
// @return *PassthroughTransformer A new transformer ready for Transform calls.
//
// @pre output must be non-nil and writable.
// @post Transformer is ready to process SSE events.
//
// @note This is the simplest transformer - it just writes data as-is.
func NewPassthroughTransformer(output io.Writer) *PassthroughTransformer {
	return &PassthroughTransformer{output: output}
}

// Transform writes the SSE event data to the output unchanged.
//
// @brief Passes the event data through without modification.
//
// @param event The SSE event to pass through.
//
//	Must not be nil.
//	Data field is written as-is.
//
// @return error Returns nil on success.
//
//	Returns error if output write fails.
//
// @pre event must not be nil.
// @pre output writer must be writable.
// @post Event data is written to output unchanged.
//
// @note Event.Type is ignored - only Data is written.
// @note The "data: " prefix and newlines are added to maintain SSE format.
func (t *PassthroughTransformer) Transform(event *sse.Event) error {
	// Skip empty events
	if event.Data == "" {
		return nil
	}

	// Write the event in SSE format: "data: <content>\n\n"
	// This maintains proper SSE protocol format.
	_, err := t.output.Write([]byte("data: " + event.Data + "\n\n"))
	return err
}

// Flush is a no-op for passthrough transformer.
// There is no buffered content to flush.
//
// @brief No-op flush for passthrough transformer.
//
// @return Always returns nil.
//
// @note Included for interface compatibility.
func (t *PassthroughTransformer) Flush() error {
	return nil
}

// Close is a no-op for passthrough transformer.
// There are no resources to release.
//
// @brief No-op close for passthrough transformer.
//
// @return Always returns nil.
//
// @note Included for interface compatibility.
func (t *PassthroughTransformer) Close() error {
	return nil
}
