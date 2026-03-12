// Package toolcall provides parsing and formatting for LLM tool call tokens.
// It transforms proprietary tool call markup into OpenAI and Anthropic streaming formats.
//
// This package implements a state-machine based parser that identifies tool call
// markers in streaming LLM output and converts them to API-specific formats.
// The parser handles partial tokens that may span multiple chunks, ensuring
// correct reconstruction of tool calls regardless of how the input is chunked.
//
// Safety Critical Notes:
// - The parser must handle arbitrarily chunked input without data loss.
// - State transitions must be deterministic based on token recognition.
// - All buffers must be processed or accounted for to prevent memory leaks.
// - Generated tool call IDs must be unique within a session.
package toolcall

import "strings"

// Tokens defines the delimiter strings used to mark tool call sections in LLM output.
// These tokens are emitted by the upstream LLM to indicate the structure of tool calls.
//
// @brief Configuration struct for tool call delimiter tokens used in parsing.
//
// @note Token values must not contain other token values as substrings, as this
//
//	could cause incorrect parsing behavior. Each token should be uniquely
//	identifiable within the input stream.
//
// @note Changes to token values must be synchronized with the upstream LLM
//
//	configuration. Mismatched tokens will result in parsing failures.
//
// @pre All token fields must be non-empty strings.
// @post Parser behavior is determined by the configured token values.
type Tokens struct {
	// SectionBegin marks the start of a tool calls section.
	// Must be unique and not a substring of other tokens.
	// Example: "<|tool_calls_section_begin|>"
	SectionBegin string

	// CallBegin marks the start of an individual tool call.
	// Must be unique and not a substring of other tokens.
	// Example: "<|tool_call_begin|>"
	CallBegin string

	// ArgBegin marks the start of tool call arguments.
	// Must be unique and not a substring of other tokens.
	// Example: "<|tool_call_argument_begin|>"
	ArgBegin string

	// CallEnd marks the end of an individual tool call.
	// Must be unique and not a substring of other tokens.
	// Example: "<|tool_call_end|>"
	CallEnd string

	// SectionEnd marks the end of a tool calls section.
	// Must be unique and not a substring of other tokens.
	// Example: "<|tool_calls_section_end|>"
	SectionEnd string
}

// DefaultTokens contains the standard delimiter tokens used by Kimi models.
// These tokens are specific to the Moonshot AI / Kimi model family.
//
// @brief Pre-configured Tokens instance with Kimi model delimiters.
//
// @note These values are model-specific and may change with model updates.
//
//	Use DefaultTokens for Kimi models; create custom Tokens for other models.
//
// @pre None (constant configuration).
// @post Contains valid, non-overlapping token values.
var DefaultTokens = Tokens{
	SectionBegin: "<|tool_calls_section_begin|>",
	CallBegin:    "<|tool_call_begin|>",
	ArgBegin:     "<|tool_call_argument_begin|>",
	CallEnd:      "<|tool_call_end|>",
	SectionEnd:   "<|tool_calls_section_end|>",
}

// ContainsAny reports whether s contains any tool call marker tokens.
// This is a fast pre-check to determine if parsing is needed.
//
// @brief Performs a quick check for tool call markers in a string.
//
// @param s The string to check for tool call markers.
//
//	Can be empty (returns false).
//	Must be valid UTF-8 for correct behavior.
//
// @return bool Returns true if any tool call marker is present.
//
//	Returns false if no markers are found or string is empty.
//
// @pre s must be valid UTF-8 for correct substring matching.
// @post No state is modified; this is a pure query function.
//
// @note This function uses a simplified check for "<|tool_call" prefix,
//
//	which covers all tool call markers. This is more efficient than
//	checking each token individually.
//
// @note A true result does not guarantee valid tool call structure;
//
//	it only indicates that parsing should be attempted.
func (t Tokens) ContainsAny(s string) bool {
	// Check for the common prefix of all tool call markers.
	// This is more efficient than multiple Contains calls.
	return strings.Contains(s, "<|tool_call")
}
