// Package capture provides request/response recording and persistence for HTTP proxy operations.
// It captures downstream client requests, upstream API requests, and their corresponding
// SSE streaming responses for debugging and analysis.
//
// Thread Safety:
//   - SSEChunk is a value type and is safe for concurrent read access
//   - captureWriter is NOT thread-safe; use external synchronization if sharing
//   - All functions in this file are pure or operate on single-goroutine data
package capture

import (
	"encoding/json"
	"time"
)

// SSEChunk represents a single Server-Sent Events chunk with timing metadata.
// It captures both the parsed JSON data and raw content for malformed chunks.
//
// Design Rationale:
//   - Data field stores valid JSON for structured analysis
//   - Raw field preserves original content for debugging malformed responses
//   - OffsetMS enables timeline reconstruction of streaming responses
//
// Thread Safety: Value type; safe for concurrent read access after creation.
type SSEChunk struct {
	// OffsetMS is milliseconds elapsed since stream start.
	// Used for timing analysis and debugging latency issues.
	// Valid values: non-negative integers, 0 for first chunk.
	OffsetMS int64 `json:"offset_ms"`

	// Event is the SSE event type (e.g., "message", "error", "ping").
	// Empty string indicates no event type was specified.
	// Valid values: any SSE event type string, or empty.
	Event string `json:"event,omitempty"`

	// Data contains the parsed JSON payload if the chunk data was valid JSON.
	// Nil if data was not valid JSON or chunk had no data.
	// Valid values: json.RawMessage (may be nil, empty, or contain valid JSON).
	Data json.RawMessage `json:"data,omitempty"`

	// Raw contains the original unparsed data string.
	// Populated when data is not valid JSON to preserve debugging information.
	// Empty if data was valid JSON (Data field is used instead).
	// Valid values: any string, including malformed JSON.
	Raw string `json:"raw,omitempty"`
}

// NewSSEChunk creates an SSEChunk with the given offset, event type, and data.
// If the data is valid JSON, it is stored in the Data field; otherwise it is stored as Raw.
//
// @param offsetMS - Milliseconds since stream start. Should be non-negative.
// @param event    - SSE event type. May be empty string.
// @param data     - Raw chunk data bytes. May be nil or empty.
// @return SSEChunk populated with timing, event, and appropriately stored data.
//
// @pre None
// @post If data is valid JSON, returned chunk.Data != nil
// @post If data is invalid JSON, returned chunk.Raw == string(data)
// @post If data is empty, both Data and Raw are empty
// @post Returned chunk.OffsetMS == offsetMS
// @post Returned chunk.Event == event
//
// @note This function does not validate that offsetMS is non-negative.
// @note JSON validation is performed; invalid JSON is stored in Raw field.
// @note Thread-safe: pure function with no side effects.
func NewSSEChunk(offsetMS int64, event string, data []byte) SSEChunk {
	// Initialize chunk with required fields; Data and Raw set conditionally below
	chunk := SSEChunk{
		OffsetMS: offsetMS,
		Event:    event,
	}

	// Early return for empty data prevents unnecessary JSON parsing
	// Empty chunks are valid in SSE (e.g., keep-alive pings)
	if len(data) == 0 {
		return chunk
	}

	// Attempt to parse as JSON to determine storage location
	// json.RawMessage is used to preserve the exact JSON structure without full parsing
	var jsonData json.RawMessage
	if err := json.Unmarshal(data, &jsonData); err == nil {
		// Valid JSON: store in Data field for structured analysis
		// Copy is made to avoid aliasing the input slice which may be reused
		chunk.Data = make(json.RawMessage, len(jsonData))
		copy(chunk.Data, jsonData)
	} else {
		// Invalid JSON: store as raw string for debugging
		// This preserves malformed data that might indicate API issues
		chunk.Raw = string(data)
	}

	return chunk
}

// CaptureWriter defines an interface for recording SSE chunks.
// Implementations must support sequential chunk recording and retrieval.
//
// Thread Safety: Implementations may or may not be thread-safe; check implementation docs.
type CaptureWriter interface {
	// RecordChunk appends an SSE chunk with the given event and data.
	// Empty data chunks may be ignored by implementations.
	// @param event - SSE event type
	// @param data  - Raw chunk data bytes
	RecordChunk(event string, data []byte)

	// Chunks returns all recorded chunks in order of recording.
	// @return Slice of recorded SSEChunk values; may be empty but not nil.
	Chunks() []SSEChunk
}

// captureWriter implements CaptureWriter with offset-based timing.
// It records chunks relative to a start time for timeline reconstruction.
//
// Thread Safety: NOT thread-safe. External synchronization required for concurrent access.
type captureWriter struct {
	// start is the reference time for calculating chunk offsets.
	// All chunks record their offset relative to this timestamp.
	// Valid values: any time.Time, typically time.Now() at creation.
	start time.Time

	// chunks stores all recorded SSE chunks in order.
	// Initialized as empty slice, never nil after NewCaptureWriter.
	// Valid values: slice of SSEChunk, may be empty.
	chunks []SSEChunk
}

// NewCaptureWriter creates a CaptureWriter that records chunks relative to the given start time.
// Chunks will have their OffsetMS calculated as time since start.
//
// @param start - Reference time for offset calculations. Should be time.Now() or similar.
// @return CaptureWriter interface backed by captureWriter implementation.
//
// @pre None
// @post Returned writer has empty chunk slice
// @post Returned writer uses start for all offset calculations
//
// @note The start time should be captured at stream start for accurate timing.
// @note Thread-safe: creates new instance; no shared state.
func NewCaptureWriter(start time.Time) CaptureWriter {
	// Initialize with empty slice (not nil) to distinguish from uninitialized
	// This allows callers to safely iterate over Chunks() without nil checks
	return &captureWriter{
		start:  start,
		chunks: []SSEChunk{},
	}
}

// RecordChunk appends a new SSE chunk to the writer.
// Empty data chunks are ignored to avoid recording meaningless entries.
//
// @param event - SSE event type (e.g., "message", "error"). May be empty.
// @param data  - Raw chunk data bytes. Empty slices are ignored.
//
// @pre cw != nil (receiver must be valid)
// @post If len(data) > 0, new chunk appended to cw.chunks
// @post If len(data) == 0, no changes made (early return)
//
// @note Empty data chunks are ignored as they provide no useful information.
// @note NOT thread-safe; do not call concurrently from multiple goroutines.
func (cw *captureWriter) RecordChunk(event string, data []byte) {
	// Ignore empty chunks to avoid polluting the recording with meaningless entries
	// SSE keep-alive chunks typically have empty data and don't need recording
	if len(data) == 0 {
		return
	}
	// Create new chunk with current offset from start time
	// OffsetMS calculates elapsed time for timeline reconstruction
	chunk := NewSSEChunk(OffsetMS(cw.start), event, data)
	// Append to slice; this may cause reallocation but is acceptable for recording
	cw.chunks = append(cw.chunks, chunk)
}

// Chunks returns all recorded SSE chunks.
// Returns a reference to the internal slice; modifications affect the writer.
//
// @return Slice of recorded SSEChunk values. Never nil, may be empty.
//
// @pre cw != nil (receiver must be valid)
// @post Returned slice is the same reference as internal storage
//
// @note Returned slice shares storage with writer; modifications affect both.
// @note NOT thread-safe; do not call concurrently with RecordChunk.
// @note Callers should copy the slice if they need to modify it independently.
func (cw *captureWriter) Chunks() []SSEChunk {
	// Return direct reference to avoid allocation
	// Callers are responsible for copying if they need to modify
	return cw.chunks
}

// ExtractRequestIDFromSSEChunk attempts to extract a request ID from SSE chunk JSON data.
// It returns an empty string if the data is invalid or does not contain an ID field.
//
// @param data - JSON data to parse for ID field. May be nil or empty.
// @return Extracted ID string, or empty string if not found or invalid.
//
// @pre None
// @post Returns empty string if data is nil, empty, or invalid JSON
// @post Returns empty string if JSON does not contain an ID field
// @post Returns non-empty string if ID field exists and is non-empty
//
// @note This function is used to extract request IDs from SSE response chunks.
// @note Supports multiple formats:
//   - OpenAI Chat/Responses: {"id": "..."}
//   - Anthropic message_start: {"type": "message_start", "message": {"id": "..."}}
//
// @note Thread-safe: pure function with no side effects.
func ExtractRequestIDFromSSEChunk(data json.RawMessage) string {
	if len(data) == 0 {
		return ""
	}

	var topLevel struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(data, &topLevel); err == nil && topLevel.ID != "" {
		return topLevel.ID
	}

	var msgStart struct {
		Type    string `json:"type"`
		Message struct {
			ID string `json:"id"`
		} `json:"message"`
	}
	if err := json.Unmarshal(data, &msgStart); err == nil && msgStart.Message.ID != "" {
		return msgStart.Message.ID
	}

	return ""
}

// TokenUsage holds extracted token counts from SSE stream
type TokenUsage struct {
	// InputTokens is the number of input/prompt tokens
	InputTokens int
	// OutputTokens is the number of output/completion tokens
	OutputTokens int
	// CacheReadTokens is the number of tokens read from cache
	CacheReadTokens int
	// CacheCreationTokens is the number of tokens used to create cache
	CacheCreationTokens int
}

// ExtractTokenUsageFromChunks extracts token usage from SSE chunks.
// It scans through all chunks looking for usage data in various API formats.
//
// Supports:
//   - OpenAI: {"usage":{"prompt_tokens":X,"completion_tokens":Y,"prompt_tokens_details":{"cached_tokens":Z}}}
//   - Anthropic: {"usage":{"input_tokens":X,"output_tokens":Y,"cache_read_input_tokens":Z,"cache_creation_input_tokens":W}}
//   - Responses: {"usage":{"input_tokens":X,"output_tokens":Y,"input_tokens_details":{"cached_tokens":Z}}}
//
// @param chunks - SSE chunks to extract usage from
// @return TokenUsage with aggregated token counts
//
// @pre None
// @post Returns zero values if no usage found
// @post Returns last usage values found if multiple usages present
//
// @note Thread-safe: pure function with no side effects
func ExtractTokenUsageFromChunks(chunks []SSEChunk) TokenUsage {
	var usage TokenUsage
	for _, chunk := range chunks {
		if len(chunk.Data) == 0 {
			continue
		}
		// Parse usage from this chunk
		extractUsageFromJSON(chunk.Data, &usage)
	}
	return usage
}

// extractUsageFromJSON parses JSON data and extracts token usage.
// Handles multiple API formats by checking various field names.
//
// @param data - JSON data to parse
// @param usage - TokenUsage to populate
//
// @pre data is valid JSON
// @post usage fields updated if usage data found
func extractUsageFromJSON(data json.RawMessage, usage *TokenUsage) {
	// Try to parse as generic map to handle different formats
	var root map[string]interface{}
	if err := json.Unmarshal(data, &root); err != nil {
		return
	}

	// Find usage object - could be top-level or nested
	var usageObj map[string]interface{}

	if u, ok := root["usage"].(map[string]interface{}); ok {
		usageObj = u
	} else {
		// Check for nested in message (Anthropic message_start)
		if msg, ok := root["message"].(map[string]interface{}); ok {
			if u, ok := msg["usage"].(map[string]interface{}); ok {
				usageObj = u
			}
		}
		// Check for nested in response (Responses API)
		if resp, ok := root["response"].(map[string]interface{}); ok {
			if u, ok := resp["usage"].(map[string]interface{}); ok {
				usageObj = u
			}
		}
		// Check for nested in delta (Anthropic message_delta)
		if delta, ok := root["delta"].(map[string]interface{}); ok {
			if u, ok := delta["usage"].(map[string]interface{}); ok {
				usageObj = u
			}
		}
	}

	if usageObj == nil {
		return
	}

	// Extract input tokens - try multiple field names
	if v, ok := usageObj["input_tokens"].(float64); ok {
		usage.InputTokens = int(v)
	} else if v, ok := usageObj["prompt_tokens"].(float64); ok {
		usage.InputTokens = int(v)
	}

	// Extract output tokens - try multiple field names
	if v, ok := usageObj["output_tokens"].(float64); ok {
		usage.OutputTokens = int(v)
	} else if v, ok := usageObj["completion_tokens"].(float64); ok {
		usage.OutputTokens = int(v)
	}

	// Extract cache read tokens - check both Anthropic and OpenAI formats
	if v, ok := usageObj["cache_read_input_tokens"].(float64); ok {
		usage.CacheReadTokens = int(v)
	}
	if v, ok := usageObj["cached_tokens"].(float64); ok {
		usage.CacheReadTokens = int(v)
	}

	// Extract cache creation tokens
	if v, ok := usageObj["cache_creation_input_tokens"].(float64); ok {
		usage.CacheCreationTokens = int(v)
	}

	// Check for nested cache details in OpenAI format
	if details, ok := usageObj["prompt_tokens_details"].(map[string]interface{}); ok {
		if v, ok := details["cached_tokens"].(float64); ok {
			usage.CacheReadTokens = int(v)
		}
		if v, ok := details["cache_creation_input_tokens"].(float64); ok {
			usage.CacheCreationTokens = int(v)
		}
	}

	// Check for nested cache details in Responses API format
	// For Responses API: calculate cache creation as input - cached
	// This represents tokens that were NOT served from cache (fresh tokens)
	// Anthropic provides cache_creation_input_tokens directly, but Responses API doesn't
	if details, ok := usageObj["input_tokens_details"].(map[string]interface{}); ok {
		if v, ok := details["cached_tokens"].(float64); ok {
			usage.CacheReadTokens = int(v)
		}
		// Calculate cache creation for Responses API format
		if usage.CacheReadTokens > 0 && usage.CacheCreationTokens == 0 && usage.InputTokens > usage.CacheReadTokens {
			usage.CacheCreationTokens = usage.InputTokens - usage.CacheReadTokens
		}
	}
}

// ExtractFinishReasonFromChunks extracts the finish reason from SSE chunks.
// It scans through all chunks looking for finish/stop reasons in various API formats.
//
// Supports:
//   - OpenAI: {"choices":[{"finish_reason":"stop"}]}
//   - Anthropic: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}
//   - Responses API: {"status":"completed"} (maps "completed" to "stop")
//
// @param chunks - SSE chunks to extract finish reason from
// @return The finish reason string, or "unknown" if not found
//
// @pre None
// @post Returns "unknown" if no finish reason found
// @post Returns last finish reason found if multiple present
//
// @note Thread-safe: pure function with no side effects
func ExtractFinishReasonFromChunks(chunks []SSEChunk) string {
	reason := "unknown"
	for _, chunk := range chunks {
		if len(chunk.Data) == 0 {
			continue
		}
		// Parse finish reason from this chunk
		if r := extractFinishReasonFromJSON(chunk.Data); r != "" {
			reason = r
		}
	}
	return reason
}

// extractFinishReasonFromJSON parses JSON data and extracts the finish/stop reason.
// Handles multiple API formats by checking various field paths.
//
// @param data - JSON data to parse
// @return The finish reason string, or empty string if not found
//
// @pre data is valid JSON
// @post Returns non-empty string if finish reason found
func extractFinishReasonFromJSON(data json.RawMessage) string {
	var root map[string]interface{}
	if err := json.Unmarshal(data, &root); err != nil {
		return ""
	}

	// Check for OpenAI Chat format: choices[0].finish_reason
	if choices, ok := root["choices"].([]interface{}); ok && len(choices) > 0 {
		if choice, ok := choices[0].(map[string]interface{}); ok {
			if fr, ok := choice["finish_reason"].(string); ok && fr != "" {
				return fr
			}
		}
	}

	// Check for Anthropic message_delta format: delta.stop_reason
	if delta, ok := root["delta"].(map[string]interface{}); ok {
		if sr, ok := delta["stop_reason"].(string); ok && sr != "" {
			return sr
		}
	}

	// Check for Anthropic message_stop format: stop_reason
	if sr, ok := root["stop_reason"].(string); ok && sr != "" {
		return sr
	}

	// Check for Responses API format: status at root level
	if status, ok := root["status"].(string); ok && status != "" {
		// Map Responses API status to finish reason
		switch status {
		case "completed":
			return "stop"
		case "incomplete":
			return "length"
		default:
			return status
		}
	}

	// Check for Responses API event format: response.status (nested in events like response.completed)
	if resp, ok := root["response"].(map[string]interface{}); ok {
		if status, ok := resp["status"].(string); ok && status != "" {
			switch status {
			case "completed":
				return "stop"
			case "incomplete":
				return "length"
			default:
				return status
			}
		}
	}

	return ""
}
