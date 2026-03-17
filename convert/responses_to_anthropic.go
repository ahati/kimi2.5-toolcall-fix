// Package convert provides converters between different API formats.
// This file implements OpenAI Responses API to Anthropic Messages conversion.
package convert

import (
	"encoding/json"
	"strings"

	"ai-proxy/conversation"
	"ai-proxy/logging"
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
// When previous_response_id is provided, it fetches the conversation history from the store
// and prepends it to the current input.
func TransformResponsesToAnthropic(body []byte) ([]byte, error) {
	// Parse the OpenAI Responses API format request
	var openReq types.ResponsesRequest
	if err := json.Unmarshal(body, &openReq); err != nil {
		return nil, err
	}

	// Fetch conversation history if previous_response_id is provided
	if openReq.PreviousResponseID != "" {
		if hist := conversation.GetFromDefault(openReq.PreviousResponseID); hist != nil {
			openReq.Input = prependHistoryToInput(hist, openReq.Input)
		} else {
			logging.InfoMsg("Warning: Previous response ID not found in conversation store: %s", openReq.PreviousResponseID)
		}
	}

	// Build the Anthropic format request using the client-provided model name
	// Anthropic requires max_tokens, so use a default if not provided
	maxTokens := openReq.MaxOutputTokens
	if maxTokens == 0 {
		maxTokens = 65536 // Default max tokens (64k) for Anthropic API
	}
	anthReq := types.MessageRequest{
		Model:     openReq.Model,
		MaxTokens: maxTokens,
		// Force streaming mode - this proxy only supports SSE streaming
		Stream:      true,
		Temperature: openReq.Temperature,
		TopP:        openReq.TopP,
	}

	// Combine instructions with developer messages from input
	systemParts := []string{}
	seenContent := make(map[string]bool) // Track content to avoid duplicates

	if openReq.Instructions != "" {
		systemParts = append(systemParts, openReq.Instructions)
		seenContent[openReq.Instructions] = true
	}

	// Extract developer messages from input array
	if arr, ok := openReq.Input.([]interface{}); ok {
		for _, item := range arr {
			if msg, ok := item.(map[string]interface{}); ok {
				if role, ok := msg["role"].(string); ok && (role == "developer" || role == "system") {
					content := extractContentFromInput(msg["content"])
					// Normalize content for deduplication
					normalizedContent := strings.TrimSpace(content)
					if content != "" && !seenContent[normalizedContent] {
						systemParts = append(systemParts, content)
						seenContent[normalizedContent] = true
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

	// If tool_choice is "none", strip tools from request
	// Anthropic doesn't have a "none" option, so we must clear tools to prevent tool calls
	if openReq.ToolChoice == "none" {
		anthReq.Tools = nil
		anthReq.ToolChoice = nil
	}

	// Convert reasoning to thinking configuration
	if openReq.Reasoning != nil {
		anthReq.Thinking = convertReasoningToThinking(openReq.Reasoning, maxTokens)
	}

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
					isError, _ := msg["is_error"].(bool)

					// Generate unique ID for tool_result if not provided
					resultID := callID
					if id, ok := msg["id"].(string); ok && id != "" {
						resultID = id
					}

					messages = append(messages, types.MessageInput{
						Role: "user",
						Content: []map[string]interface{}{
							{
								"type":        "tool_result",
								"id":          resultID,
								"tool_use_id": callID,
								"content":     output,
								"is_error":    isError,
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
				// Extract text from various content part types
				partType, _ := partMap["type"].(string)
				switch partType {
				case "input_text", "output_text":
					// Both input_text and output_text contain text content
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
				case "input_file":
					// File attachments not supported by Anthropic text API - add placeholder
					if fileData, ok := partMap["file_data"].(map[string]interface{}); ok {
						if filename, ok := fileData["filename"].(string); ok {
							if result.Len() > 0 {
								result.WriteString("\n")
							}
							result.WriteString("[File attached: " + filename + "]")
						}
					}
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

// convertReasoningToThinking converts OpenAI reasoning configuration to Anthropic thinking configuration.
// Budget calculation:
//   - "concise": ~20-30% of max_output_tokens (min 1024)
//   - "detailed": ~40-50% of max_output_tokens (min 2048)
//   - Cap at 32000 for safety
func convertReasoningToThinking(reasoning *types.ReasoningConfig, maxTokens int) *types.ThinkingConfig {
	if reasoning == nil || (reasoning.Summary == "" && reasoning.Effort == "") {
		return nil
	}

	var budgetTokens int

	switch {
	case reasoning.Summary == "detailed" || reasoning.Effort == "high":
		budgetTokens = int(float64(maxTokens) * 0.45)
		if budgetTokens < 2048 {
			budgetTokens = 2048
		}
	case reasoning.Summary == "concise" || reasoning.Effort == "medium":
		budgetTokens = int(float64(maxTokens) * 0.25)
		if budgetTokens < 1024 {
			budgetTokens = 1024
		}
	case reasoning.Effort == "low":
		budgetTokens = int(float64(maxTokens) * 0.10)
		if budgetTokens < 1024 {
			budgetTokens = 1024
		}
	default:
		budgetTokens = int(float64(maxTokens) * 0.25)
		if budgetTokens < 1024 {
			budgetTokens = 1024
		}
	}

	// Cap at 32000 for safety
	if budgetTokens > 32000 {
		budgetTokens = 32000
	}

	return &types.ThinkingConfig{
		Type:         "enabled",
		BudgetTokens: budgetTokens,
	}
}

// prependHistoryToInput prepends conversation history to the current input.
// It converts the stored conversation (input/output items) into the input format
// expected by the Responses API, then appends the current input.
func prependHistoryToInput(hist *conversation.Conversation, currentInput interface{}) interface{} {
	// Build a slice to hold all input items
	var items []interface{}

	// First, add all input items from history
	for _, item := range hist.Input {
		// Convert to map[string]interface{} for consistent handling
		itemMap := map[string]interface{}{
			"type": item.Type,
		}
		if item.Role != "" {
			itemMap["role"] = item.Role
		}
		if item.Content != nil {
			itemMap["content"] = item.Content
		}
		if item.CallID != "" {
			itemMap["call_id"] = item.CallID
		}
		if item.Name != "" {
			itemMap["name"] = item.Name
		}
		if item.Arguments != "" {
			itemMap["arguments"] = item.Arguments
		}
		if item.Output != "" {
			itemMap["output"] = item.Output
		}
		items = append(items, itemMap)
	}

	// Then, convert output items to input format for the assistant's responses
	for _, output := range hist.Output {
		switch output.Type {
		case "message":
			// Convert assistant message output to input format
			if output.Role == "assistant" {
				itemMap := map[string]interface{}{
					"type": "message",
					"role": "assistant",
				}
				if len(output.Content) > 0 {
					// Convert OutputContent to content parts
					contentParts := make([]interface{}, len(output.Content))
					for i, c := range output.Content {
						contentParts[i] = map[string]interface{}{
							"type": c.Type,
							"text": c.Text,
						}
					}
					itemMap["content"] = contentParts
				}
				items = append(items, itemMap)
			}
		case "function_call":
			// Include function_call output as function_call input
			items = append(items, map[string]interface{}{
				"type":      "function_call",
				"call_id":   output.CallID,
				"name":      output.Name,
				"arguments": output.Arguments,
			})
		}
	}

	// Finally, append the current input
	switch v := currentInput.(type) {
	case string:
		// String input becomes a user message
		items = append(items, map[string]interface{}{
			"type":    "message",
			"role":    "user",
			"content": v,
		})
	case []interface{}:
		// Already an array of items
		items = append(items, v...)
	default:
		// Try to handle other formats by wrapping as user message
		items = append(items, map[string]interface{}{
			"type":    "message",
			"role":    "user",
			"content": currentInput,
		})
	}

	return items
}
