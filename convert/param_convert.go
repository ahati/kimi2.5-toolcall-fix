// Package convert provides converters between different API formats.
// This file provides conversion functions for request parameters like
// response_format, tool_choice, stop sequences, etc.
package convert

import (
	"ai-proxy/types"
	"encoding/json"
)

// ResponseFormatConverter handles conversion between OpenAI response_format
// and Anthropic equivalents.
type ResponseFormatConverter struct{}

// NewResponseFormatConverter creates a new converter for response format.
func NewResponseFormatConverter() *ResponseFormatConverter {
	return &ResponseFormatConverter{}
}

// ConvertOpenAIToAnthropic converts OpenAI response_format to Anthropic equivalent.
// Anthropic doesn't have a direct response_format field, but structured outputs
// can be achieved through tools with input_schema.
//
// For json_object: Returns a hint that tools should be used.
// For json_schema: Returns a tool definition that matches the schema.
func (c *ResponseFormatConverter) ConvertOpenAIToAnthropic(format *types.ResponseFormat) (map[string]interface{}, error) {
	if format == nil {
		return nil, nil
	}

	switch format.Type {
	case "json_object":
		// Anthropic doesn't have json_object mode directly
		// Return a marker that indicates JSON mode should be requested
		return map[string]interface{}{
			"_json_mode": true,
		}, nil

	case "json_schema":
		if format.JSONSchema == nil {
			return nil, nil
		}
		// Convert JSON schema to an Anthropic tool
		// This allows structured output via tool calling
		return map[string]interface{}{
			"_structured_output": true,
			"_schema_name":       format.JSONSchema.Name,
			"_schema":            format.JSONSchema.Schema,
		}, nil

	default:
		return nil, nil
	}
}

// ShouldUseJSONMode returns true if the response format requires JSON output.
func (c *ResponseFormatConverter) ShouldUseJSONMode(format *types.ResponseFormat) bool {
	if format == nil {
		return false
	}
	return format.Type == "json_object" || format.Type == "json_schema"
}

// ToolChoiceConverter handles conversion between OpenAI and Anthropic tool_choice formats.
type ToolChoiceConverter struct{}

// NewToolChoiceConverter creates a new converter for tool choice.
func NewToolChoiceConverter() *ToolChoiceConverter {
	return &ToolChoiceConverter{}
}

// ConvertOpenAIToAnthropic converts OpenAI tool_choice to Anthropic format.
// OpenAI values: "none", "auto", "required", or {"type": "function", "function": {"name": "..."}}
// Anthropic values: {"type": "auto"}, {"type": "any"}, {"type": "tool", "name": "..."}
func (c *ToolChoiceConverter) ConvertOpenAIToAnthropic(toolChoice interface{}) *types.ToolChoice {
	if toolChoice == nil {
		return nil
	}

	switch tc := toolChoice.(type) {
	case string:
		switch tc {
		case "none":
			// Anthropic doesn't have "none" - return nil and caller should clear tools
			return nil
		case "auto":
			return &types.ToolChoice{Type: "auto"}
		case "required":
			return &types.ToolChoice{Type: "any"}
		default:
			return &types.ToolChoice{Type: "auto"}
		}

	case map[string]interface{}:
		return c.convertOpenAIObjectToAnthropic(tc)
	}

	return nil
}

func (c *ToolChoiceConverter) convertOpenAIObjectToAnthropic(tc map[string]interface{}) *types.ToolChoice {
	objType, ok := tc["type"].(string)
	if !ok {
		return nil
	}

	switch objType {
	case "function":
		if fn, ok := tc["function"].(map[string]interface{}); ok {
			if name, ok := fn["name"].(string); ok {
				return &types.ToolChoice{
					Type: "tool",
					Name: name,
				}
			}
		}
	}

	return nil
}

// ConvertAnthropicToOpenAI converts Anthropic tool_choice to OpenAI format.
func (c *ToolChoiceConverter) ConvertAnthropicToOpenAI(toolChoice *types.ToolChoice) interface{} {
	if toolChoice == nil {
		return "auto"
	}

	switch toolChoice.Type {
	case "auto":
		return "auto"
	case "any":
		return "required"
	case "tool":
		return map[string]interface{}{
			"type": "function",
			"function": map[string]interface{}{
				"name": toolChoice.Name,
			},
		}
	default:
		return "auto"
	}
}

// StopConverter handles conversion between OpenAI stop and Anthropic stop_sequences.
type StopConverter struct{}

// NewStopConverter creates a new converter for stop sequences.
func NewStopConverter() *StopConverter {
	return &StopConverter{}
}

// ConvertOpenAIToAnthropic converts OpenAI stop to Anthropic stop_sequences.
// OpenAI stop can be: string, []string, or nil
// Anthropic stop_sequences is always []string
func (c *StopConverter) ConvertOpenAIToAnthropic(stop interface{}) []string {
	if stop == nil {
		return nil
	}

	switch s := stop.(type) {
	case string:
		if s == "" {
			return nil
		}
		return []string{s}

	case []interface{}:
		result := make([]string, 0, len(s))
		for _, v := range s {
			if str, ok := v.(string); ok && str != "" {
				result = append(result, str)
			}
		}
		if len(result) == 0 {
			return nil
		}
		return result

	case []string:
		// Filter empty strings
		result := make([]string, 0, len(s))
		for _, str := range s {
			if str != "" {
				result = append(result, str)
			}
		}
		if len(result) == 0 {
			return nil
		}
		return result

	default:
		return nil
	}
}

// ConvertAnthropicToOpenAI converts Anthropic stop_sequences to OpenAI stop.
func (c *StopConverter) ConvertAnthropicToOpenAI(stopSequences []string) interface{} {
	if len(stopSequences) == 0 {
		return nil
	}
	if len(stopSequences) == 1 {
		return stopSequences[0]
	}
	return stopSequences
}

// MaxTokensConverter handles max_tokens conversion and defaults.
type MaxTokensConverter struct {
	defaultTokens int
}

// NewMaxTokensConverter creates a new converter with the specified default.
func NewMaxTokensConverter(defaultTokens int) *MaxTokensConverter {
	return &MaxTokensConverter{defaultTokens: defaultTokens}
}

// ResolveMaxTokens returns the max_tokens value, applying defaults if necessary.
func (c *MaxTokensConverter) ResolveMaxTokens(maxTokens int) int {
	if maxTokens <= 0 {
		return c.defaultTokens
	}
	return maxTokens
}

// Global converters for convenience
var (
	DefaultToolChoiceConverter = NewToolChoiceConverter()
	DefaultStopConverter       = NewStopConverter()
	DefaultResponseFormatConverter = NewResponseFormatConverter()
)

// ConvertToolChoiceOpenAIToAnthropic is a convenience function using the global converter.
func ConvertToolChoiceOpenAIToAnthropic(toolChoice interface{}) *types.ToolChoice {
	return DefaultToolChoiceConverter.ConvertOpenAIToAnthropic(toolChoice)
}

// ConvertToolChoiceAnthropicToOpenAI is a convenience function using the global converter.
func ConvertToolChoiceAnthropicToOpenAI(toolChoice *types.ToolChoice) interface{} {
	return DefaultToolChoiceConverter.ConvertAnthropicToOpenAI(toolChoice)
}

// ConvertStopOpenAIToAnthropic is a convenience function using the global converter.
func ConvertStopOpenAIToAnthropic(stop interface{}) []string {
	return DefaultStopConverter.ConvertOpenAIToAnthropic(stop)
}

// ConvertStopAnthropicToOpenAI is a convenience function using the global converter.
func ConvertStopAnthropicToOpenAI(stopSequences []string) interface{} {
	return DefaultStopConverter.ConvertAnthropicToOpenAI(stopSequences)
}

// UnmarshalToolChoice parses a raw JSON tool_choice field.
func UnmarshalToolChoice(data json.RawMessage) (interface{}, error) {
	if len(data) == 0 {
		return nil, nil
	}

	// Try string first
	var strVal string
	if err := json.Unmarshal(data, &strVal); err == nil {
		return strVal, nil
	}

	// Try object
	var objVal map[string]interface{}
	if err := json.Unmarshal(data, &objVal); err == nil {
		return objVal, nil
	}

	return nil, nil
}