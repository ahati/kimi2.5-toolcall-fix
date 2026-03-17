// Package convert provides converters between different API formats.
// This file provides bidirectional finish reason mapping between OpenAI and Anthropic.
package convert

// FinishReasonMapper provides bidirectional mapping between OpenAI and Anthropic finish reasons.
// This consolidates the stop reason mapping logic that was duplicated across transformers.
//
// OpenAI finish reasons:
//   - stop: Natural end of response
//   - length: Max tokens reached
//   - tool_calls: Model called a tool
//   - content_filter: Content was filtered
//
// Anthropic stop reasons:
//   - end_turn: Natural end of response
//   - max_tokens: Max tokens reached
//   - tool_use: Model called a tool
//   - stop_sequence: Custom stop sequence triggered
type FinishReasonMapper struct{}

// Global instance for convenience
var FinishReason = &FinishReasonMapper{}

// openAIToAnthropic maps OpenAI finish reasons to Anthropic stop reasons.
var openAIToAnthropic = map[string]string{
	"stop":           "end_turn",
	"length":         "max_tokens",
	"tool_calls":     "tool_use",
	"content_filter": "end_turn", // No direct Anthropic equivalent - mapped to end_turn for compatibility
}

// anthropicToOpenAI maps Anthropic stop reasons to OpenAI finish reasons.
var anthropicToOpenAI = map[string]string{
	"end_turn":      "stop",
	"max_tokens":    "length",
	"tool_use":      "tool_calls",
	"stop_sequence": "stop",
}

// OpenAIToAnthropic converts an OpenAI finish reason to Anthropic stop reason.
// Returns "end_turn" for unknown reasons.
func (m *FinishReasonMapper) OpenAIToAnthropic(reason string) string {
	if mapped, ok := openAIToAnthropic[reason]; ok {
		return mapped
	}
	return "end_turn"
}

// AnthropicToOpenAI converts an Anthropic stop reason to OpenAI finish reason.
// Returns "stop" for unknown reasons.
func (m *FinishReasonMapper) AnthropicToOpenAI(reason string) string {
	if mapped, ok := anthropicToOpenAI[reason]; ok {
		return mapped
	}
	return "stop"
}

// MapOpenAIToAnthropic is a convenience function using the global mapper.
func MapOpenAIToAnthropic(reason string) string {
	return FinishReason.OpenAIToAnthropic(reason)
}

// MapAnthropicToOpenAI is a convenience function using the global mapper.
func MapAnthropicToOpenAI(reason string) string {
	return FinishReason.AnthropicToOpenAI(reason)
}
