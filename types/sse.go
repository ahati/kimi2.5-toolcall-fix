// Package types defines data structures for OpenAI and Anthropic API formats.
// This file contains types for Server-Sent Events (SSE) capture and logging.
package types

import "encoding/json"

// SSEChunk represents a captured Server-Sent Event chunk with timing and raw data.
// Used for debugging, logging, and replaying SSE responses.
type SSEChunk struct {
	// OffsetMS is the time offset in milliseconds from request start.
	// Used to reconstruct timing of the original SSE stream.
	OffsetMS int64 `json:"offset_ms"`
	// Event is the SSE event type, if specified.
	// Common values: empty (default), "ping", "error".
	Event string `json:"event,omitempty"`
	// Data is the parsed JSON data from the SSE event.
	// May be null if parsing failed or data field was empty.
	Data json.RawMessage `json:"data,omitempty"`
	// Raw is the original raw data string from the SSE event.
	// Preserved for cases where JSON parsing fails or for exact replay.
	Raw string `json:"raw,omitempty"`
}
