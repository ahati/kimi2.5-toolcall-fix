// Package types provides shared type definitions for OpenAI and Anthropic streaming formats.
// This package centralizes all type definitions to avoid duplication across the codebase.
package types

import (
	"fmt"
	"strings"
	"time"
)

// ToolCall represents a normalized tool call used across OpenAI and Anthropic formats.
// It provides a common structure for tool call handling during streaming.
//
// Fields:
//   - ID: Unique identifier for the tool call
//   - Type: Type of tool call (typically "function")
//   - Index: Index of this tool call in the sequence
//   - Function: Function call details with name and arguments
type ToolCall struct {
	ID       string       `json:"id,omitempty"`
	Type     string       `json:"type,omitempty"`
	Index    int          `json:"index"`
	Function ToolFunction `json:"function"`
}

// ToolFunction represents function call details in a normalized tool call.
// It contains the function name and JSON-encoded arguments.
//
// Fields:
//   - Name: Name of the function being called
//   - Arguments: JSON-encoded string of function arguments
type ToolFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// NormalizeToolID normalizes a raw tool call identifier into a standard format.
//
// @brief    Normalizes tool call ID for consistent handling across formats.
// @param    raw   The raw tool call identifier from the upstream response.
// @param    index The index of the tool call in the sequence.
// @return   A normalized tool call identifier in standard format.
//
// @note     If the raw ID is already in standard format (starts with "call_"),
//
//	it is returned unchanged after whitespace trimming.
//
// @note     If the raw ID is not in standard format, a new ID is generated using
//
//	the format "call_{index}_{timestamp}".
//
// @pre      raw may be empty or contain whitespace (will be trimmed).
// @post     Returned ID is guaranteed to be non-empty and in standard format.
func NormalizeToolID(raw string, index int) string {
	raw = strings.TrimSpace(raw)
	if strings.HasPrefix(raw, "call_") {
		return raw
	}
	return fmt.Sprintf("call_%d_%d", index, time.Now().UnixMilli())
}

// ParseFunctionName extracts the base function name from a raw identifier.
//
// @brief    Parses and simplifies function names from various formats.
// @param    raw The raw function name that may contain package or type prefixes.
// @return   The base function name without package or type qualifiers.
//
// @note     Handles names with package prefixes (e.g., "pkg.Function" -> "Function").
// @note     Handles names with type qualifiers (e.g., "Type:method" -> "Type").
// @note     Whitespace is trimmed from the input before processing.
//
// @pre      raw may contain package prefixes, type qualifiers, or whitespace.
// @post     Returned string contains only the base function name.
func ParseFunctionName(raw string) string {
	raw = strings.TrimSpace(raw)
	if i := strings.LastIndex(raw, ":"); i >= 0 {
		raw = raw[:i]
	}
	if i := strings.LastIndex(raw, "."); i >= 0 {
		raw = raw[i+1:]
	}
	return raw
}
