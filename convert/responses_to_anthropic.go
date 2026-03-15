// Package convert provides converters between different API formats.
// This file implements OpenAI Responses API to Anthropic Messages conversion.
package convert

import (
	"encoding/json"
	"strings"

	"ai-proxy/types"
)

// ResponsesToAnthropicConverter converts OpenAI ResponsesRequest to Anthropic MessageRequest.
type ResponsesToAnthropicConverter struct{}

// NewResponsesToAnthropicConverter creates a new converter for Responses to Anthropic format.
func NewResponsesToAnthropicConverter() *ResponsesToAnthropicConverter {
	return &ResponsesToAnthropicConverter{}
}

// Convert transforms a ResponsesRequest body to Anthropic MessageRequest format.
func (c *ResponsesToAnthropicConverter) Convert(body []byte) ([]byte, error) {
	return TransformResponsesToAnthropic(body)
}

// TransformResponsesToAnthropic converts an OpenAI ResponsesRequest to an Anthropic MessageRequest.
// This handles the translation of all request fields including input, tools, and parameters.
func TransformResponsesToAnthropic(body []byte) ([]byte, error) {
	// Parse the OpenAI Responses API format request
	var openReq types.ResponsesRequest
	if err := json.Unmarshal(body, &openReq); err != nil {
		return nil, err
	}

	// Build the Anthropic format request using the client-provided model name
	// Anthropic requires max_tokens, so use a default if not provided
	maxTokens := openReq.MaxOutputTokens
	if maxTokens == 0 {
		maxTokens = 16384 // Default max tokens (16k) for Anthropic API
	}
	anthReq := types.MessageRequest{
		Model:     openReq.Model,
		MaxTokens: maxTokens,
		// Force streaming mode - this proxy only supports SSE streaming
		Stream:      openReq.Stream || true,
		Temperature: openReq.Temperature,
		TopP:        openReq.TopP,
	}

	// Combine instructions with developer messages from input
	systemParts := []string{}
	if openReq.Instructions != "" {
		systemParts = append(systemParts, openReq.Instructions)
	}

	// Extract developer messages from input array
	if arr, ok := openReq.Input.([]interface{}); ok {
		for _, item := range arr {
			if msg, ok := item.(map[string]interface{}); ok {
				if role, ok := msg["role"].(string); ok && (role == "developer" || role == "system") {
					content := extractContentFromInput(msg["content"])
					if content != "" {
						systemParts = append(systemParts, content)
					}
				}
			}
		}
	}

	// Set system message if we have any parts
	if len(systemParts) > 0 {
		anthReq.System = strings.Join(systemParts, "\n\n")
	}

	// Convert input to messages (excluding developer/system)
	anthReq.Messages = convertInputToAnthropicMessages(openReq.Input)

	// Convert tools
	anthReq.Tools = convertResponsesToolsToAnthropic(openReq.Tools)

	// Convert tool_choice
	anthReq.ToolChoice = ConvertToolChoiceOpenAIToAnthropic(openReq.ToolChoice)

	return json.Marshal(anthReq)
}

// convertInputToAnthropicMessages converts OpenAI Responses API input to Anthropic messages.
func convertInputToAnthropicMessages(input interface{}) []types.MessageInput {
	if input == nil {
		return []types.MessageInput{}
	}

	if s, ok := input.(string); ok {
		return []types.MessageInput{
			{Role: "user", Content: s},
		}
	}

	if arr, ok := input.([]interface{}); ok {
		messages := make([]types.MessageInput, 0, len(arr))
		for _, item := range arr {
			if msg, ok := item.(map[string]interface{}); ok {
				itemType, _ := msg["type"].(string)

				switch itemType {
				case "message":
					role := "user"
					if r, ok := msg["role"].(string); ok {
						switch r {
						case "developer", "system":
							continue
						case "assistant":
							role = "assistant"
						default:
							role = "user"
						}
					}

					content := extractContentFromInput(msg["content"])
					if content == "" {
						continue
					}

					messages = append(messages, types.MessageInput{
						Role:    role,
						Content: content,
					})

				case "function_call":
					callID, _ := msg["call_id"].(string)
					name, _ := msg["name"].(string)
					args, _ := msg["arguments"].(string)

					var inputObj map[string]interface{}
					if args != "" {
						json.Unmarshal([]byte(args), &inputObj)
					}

					messages = append(messages, types.MessageInput{
						Role: "assistant",
						Content: []map[string]interface{}{
							{
								"type":  "tool_use",
								"id":    callID,
								"name":  name,
								"input": inputObj,
							},
						},
					})

				case "function_call_output":
					callID, _ := msg["call_id"].(string)
					output, _ := msg["output"].(string)

					messages = append(messages, types.MessageInput{
						Role: "user",
						Content: []map[string]interface{}{
							{
								"type":        "tool_result",
								"tool_use_id": callID,
								"content":     output,
							},
						},
					})

				case "reasoning":
					// Skip reasoning items - not needed for conversation history
				}
			}
		}
		return messages
	}

	return []types.MessageInput{}
}

// extractContentFromInput extracts text content from various formats.
func extractContentFromInput(content interface{}) string {
	if content == nil {
		return ""
	}

	// Handle string content directly
	if s, ok := content.(string); ok {
		return s
	}

	// Handle array of content parts
	if arr, ok := content.([]interface{}); ok {
		var result strings.Builder
		for _, part := range arr {
			if partMap, ok := part.(map[string]interface{}); ok {
				// Extract text from input_text or input_image parts
				partType, _ := partMap["type"].(string)
				switch partType {
				case "input_text":
					if text, ok := partMap["text"].(string); ok {
						if result.Len() > 0 {
							result.WriteString("\n")
						}
						result.WriteString(text)
					}
				case "input_image":
					// Images not supported by Anthropic text API - add placeholder
					if result.Len() > 0 {
						result.WriteString("\n")
					}
					result.WriteString("[Image attached]")
				}
			}
		}
		return result.String()
	}

	return ""
}

// convertResponsesToolsToAnthropic converts OpenAI Responses API tools to Anthropic tools.
func convertResponsesToolsToAnthropic(openTools []types.ResponsesTool) []types.ToolDef {
	if len(openTools) == 0 {
		return nil
	}

	anthTools := make([]types.ToolDef, 0, len(openTools))
	for _, openTool := range openTools {
		// Only handle function type tools
		if openTool.Type == "function" {
			// Handle nested format (Function object)
			if openTool.Function != nil {
				anthTools = append(anthTools, types.ToolDef{
					Name:        openTool.Function.Name,
					Description: openTool.Function.Description,
					InputSchema: openTool.Function.Parameters,
				})
			} else if openTool.Name != "" {
				// Handle flat format (fields at top level)
				anthTools = append(anthTools, types.ToolDef{
					Name:        openTool.Name,
					Description: openTool.Description,
					InputSchema: openTool.Parameters,
				})
			}
		}
	}
	return anthTools
}

