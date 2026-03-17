// Package convert provides shared helper functions for converting between
// OpenAI and Anthropic API formats.
package convert

import (
	"ai-proxy/logging"
	"ai-proxy/types"
	"encoding/json"
	"strings"
)

// ConvertAnthropicMessagesToOpenAI converts Anthropic messages to OpenAI format.
// Each message's content is converted from Anthropic's content blocks to OpenAI's format.
// Handles content as either a string or array of content blocks.
//
// For assistant messages with tool_use blocks, ToolCalls are populated.
// For user messages with tool_result blocks, ToolCallID is populated.
func ConvertAnthropicMessagesToOpenAI(anthMsgs []types.MessageInput) []types.Message {
	if len(anthMsgs) == 0 {
		return []types.Message{}
	}

	openMsgs := make([]types.Message, 0, len(anthMsgs))
	for _, anthMsg := range anthMsgs {
		openMsgs = append(openMsgs, convertAnthropicMessage(anthMsg))
	}
	return openMsgs
}

// convertAnthropicMessage transforms a single Anthropic message to OpenAI format.
func convertAnthropicMessage(anthMsg types.MessageInput) types.Message {
	openMsg := types.Message{Role: anthMsg.Role}

	switch content := anthMsg.Content.(type) {
	case string:
		openMsg.Content = content
	case []interface{}:
		openMsg.Content, openMsg.ToolCalls, openMsg.ToolCallID = ConvertContentBlocks(content)
	}

	return openMsg
}

// ConvertOpenAIMessagesToAnthropic converts OpenAI messages to Anthropic format.
// Each message's content is converted to Anthropic's content blocks format.
// Handles tool calls and tool results appropriately.
func ConvertOpenAIMessagesToAnthropic(openMsgs []types.Message) []types.MessageInput {
	if len(openMsgs) == 0 {
		return []types.MessageInput{}
	}

	anthMsgs := make([]types.MessageInput, 0, len(openMsgs))
	for _, openMsg := range openMsgs {
		anthMsgs = append(anthMsgs, convertOpenAIMessage(openMsg))
	}
	return anthMsgs
}

// convertOpenAIMessage transforms a single OpenAI message to Anthropic format.
func convertOpenAIMessage(openMsg types.Message) types.MessageInput {
	anthMsg := types.MessageInput{Role: openMsg.Role}

	// Handle tool response messages
	if openMsg.Role == "tool" && openMsg.ToolCallID != "" {
		anthMsg.Role = "user"
		anthMsg.Content = []map[string]interface{}{
			{
				"type":        "tool_result",
				"tool_use_id": openMsg.ToolCallID,
				"content":     openMsg.Content,
			},
		}
		return anthMsg
	}

	// Handle assistant messages with tool calls
	if len(openMsg.ToolCalls) > 0 {
		blocks := []interface{}{}

		// Add text content first if present
		if text := ExtractTextFromContent(openMsg.Content); text != "" {
			blocks = append(blocks, map[string]interface{}{
				"type": "text",
				"text": text,
			})
		}

		// Add tool_use blocks
		for _, tc := range openMsg.ToolCalls {
			blocks = append(blocks, map[string]interface{}{
				"type":  "tool_use",
				"id":    tc.ID,
				"name":  tc.Function.Name,
				"input": unmarshalArgs(tc.Function.Arguments),
			})
		}
		anthMsg.Content = blocks
		return anthMsg
	}

	// Handle regular content
	switch c := openMsg.Content.(type) {
	case string:
		anthMsg.Content = c
	case []interface{}:
		// Convert OpenAI content parts to Anthropic format
		blocks := []interface{}{}
		for _, part := range c {
			if partMap, ok := part.(map[string]interface{}); ok {
				partType, _ := partMap["type"].(string)
				switch partType {
				case "text":
					if text, ok := partMap["text"].(string); ok {
						blocks = append(blocks, map[string]interface{}{
							"type": "text",
							"text": text,
						})
					}
				case "image_url":
					// Convert OpenAI image_url to Anthropic image format
					if imageURL, ok := partMap["image_url"].(map[string]interface{}); ok {
						if url, ok := imageURL["url"].(string); ok {
							// Handle base64 data URLs
							if strings.HasPrefix(url, "data:") {
								blocks = append(blocks, map[string]interface{}{
									"type": "image",
									"source": map[string]interface{}{
										"type":       "base64",
										"media_type": extractMediaType(url),
										"data":       extractBase64Data(url),
									},
								})
							}
						}
					}
				}
			}
		}
		if len(blocks) > 0 {
			anthMsg.Content = blocks
		} else {
			anthMsg.Content = ""
		}
	default:
		anthMsg.Content = ""
	}

	return anthMsg
}

// unmarshalArgs parses a JSON arguments string into a map.
func unmarshalArgs(args string) map[string]interface{} {
	var result map[string]interface{}
	if args == "" {
		return result
	}
	if err := json.Unmarshal([]byte(args), &result); err != nil {
		return map[string]interface{}{}
	}
	return result
}

// ConvertAnthropicToolsToOpenAI converts Anthropic tool definitions to OpenAI format.
// Anthropic uses "input_schema" while OpenAI uses "parameters".
func ConvertAnthropicToolsToOpenAI(anthTools []types.ToolDef) []types.Tool {
	if len(anthTools) == 0 {
		return []types.Tool{}
	}

	openTools := make([]types.Tool, 0, len(anthTools))
	for _, anthTool := range anthTools {
		openTools = append(openTools, types.Tool{
			Type: "function",
			Function: types.ToolFunction{
				Name:        anthTool.Name,
				Description: anthTool.Description,
				Parameters:  anthTool.InputSchema,
			},
		})
	}
	return openTools
}

// ConvertOpenAIToolsToAnthropic converts OpenAI tool definitions to Anthropic format.
// OpenAI uses "parameters" while Anthropic uses "input_schema".
func ConvertOpenAIToolsToAnthropic(openTools []types.Tool) []types.ToolDef {
	if len(openTools) == 0 {
		return []types.ToolDef{}
	}

	anthTools := make([]types.ToolDef, 0, len(openTools))
	for _, openTool := range openTools {
		if openTool.Type == "function" {
			anthTools = append(anthTools, types.ToolDef{
				Name:        openTool.Function.Name,
				Description: openTool.Function.Description,
				InputSchema: openTool.Function.Parameters,
			})
		}
	}
	return anthTools
}

// ExtractTextFromContent extracts text from various content formats.
// This is the SINGLE source of truth for text extraction across the codebase.
// All other implementations should call this function.
//
// Content can be:
//   - string: returned directly
//   - []interface{}: array of content blocks (Anthropic/OpenAI format)
//   - []map[string]interface{}: typed content blocks
//   - nil: returns empty string
//
// Supported content block types:
//   - "text", "input_text", "output_text": extracts "text" field
//   - "thinking": extracts "thinking" field
func ExtractTextFromContent(content interface{}) string {
	if content == nil {
		return ""
	}

	switch c := content.(type) {
	case string:
		return c
	case []interface{}:
		return extractTextFromInterfaceSlice(c)
	case []map[string]interface{}:
		return extractTextFromMapSlice(c)
	default:
		return ""
	}
}

// extractTextFromInterfaceSlice extracts text from a slice of interface{} (untyped content blocks).
func extractTextFromInterfaceSlice(blocks []interface{}) string {
	var result strings.Builder
	for _, part := range blocks {
		if partMap, ok := part.(map[string]interface{}); ok {
			text := extractTextFromBlock(partMap)
			if text != "" {
				if result.Len() > 0 {
					result.WriteString("\n")
				}
				result.WriteString(text)
			}
		}
	}
	return result.String()
}

// extractTextFromMapSlice extracts text from a slice of map[string]interface{} (typed content blocks).
func extractTextFromMapSlice(blocks []map[string]interface{}) string {
	var result strings.Builder
	for _, part := range blocks {
		text := extractTextFromBlock(part)
		if text != "" {
			if result.Len() > 0 {
				result.WriteString("\n")
			}
			result.WriteString(text)
		}
	}
	return result.String()
}

// extractTextFromBlock extracts text from a single content block based on its type.
func extractTextFromBlock(block map[string]interface{}) string {
	partType, _ := block["type"].(string)
	switch partType {
	case "text", "input_text", "output_text":
		if text, ok := block["text"].(string); ok {
			return text
		}
	case "thinking":
		if thinking, ok := block["thinking"].(string); ok {
			return thinking
		}
	default:
		if partType != "" {
			logging.DebugMsg("Unknown content block type: %s", partType)
		}
	}
	return ""
}

// ConvertContentBlocks converts Anthropic content blocks to OpenAI format.
// Returns: text content, tool calls, and tool_call_id.
//
// Anthropic content blocks can be:
//   - text: Simple text content
//   - tool_use: Tool call request (converted to OpenAI tool_calls)
//   - tool_result: Tool execution result (converted to OpenAI tool_call_id)
func ConvertContentBlocks(blocks []interface{}) (string, []types.ToolCall, string) {
	var textContent strings.Builder
	var toolCalls []types.ToolCall
	var toolCallID string

	for _, item := range blocks {
		m, ok := item.(map[string]interface{})
		if !ok {
			continue
		}

		switch m["type"] {
		case "text":
			if text, ok := m["text"].(string); ok {
				if textContent.Len() > 0 {
					textContent.WriteString("\n")
				}
				textContent.WriteString(text)
			}
		case "thinking":
			// Include thinking content as part of text (with prefix)
			if thinking, ok := m["thinking"].(string); ok && thinking != "" {
				if textContent.Len() > 0 {
					textContent.WriteString("\n")
				}
				textContent.WriteString("[Thinking: " + thinking + "]")
			}
		case "tool_use":
			if id, ok := m["id"].(string); ok {
				if name, ok := m["name"].(string); ok {
					input, _ := json.Marshal(m["input"])
					toolCalls = append(toolCalls, types.ToolCall{
						ID:   id,
						Type: "function",
						Function: types.Function{
							Name:      name,
							Arguments: string(input),
						},
					})
				}
			}
		case "tool_result":
			if id, ok := m["tool_use_id"].(string); ok {
				toolCallID = id
			}
		}
	}

	return textContent.String(), toolCalls, toolCallID
}

// ExtractSystemMessage extracts a system message string from various formats.
// System can be: string, []interface{} (content blocks), or nil.
// Returns an empty string if no system content is found.
func ExtractSystemMessage(system interface{}) string {
	if system == nil {
		return ""
	}

	switch s := system.(type) {
	case string:
		return s
	case []interface{}:
		var content strings.Builder
		for _, item := range s {
			if m, ok := item.(map[string]interface{}); ok {
				if text, ok := m["text"].(string); ok {
					content.WriteString(text)
				}
			}
		}
		return content.String()
	default:
		return ""
	}
}

// extractMediaType extracts the media type from a data URL.
// e.g., "data:image/png;base64,abc" -> "image/png"
func extractMediaType(dataURL string) string {
	if !strings.HasPrefix(dataURL, "data:") {
		return ""
	}
	// Remove "data:" prefix
	rest := dataURL[5:]
	// Find the semicolon or comma that ends the media type
	idx := strings.Index(rest, ";")
	if idx == -1 {
		idx = strings.Index(rest, ",")
	}
	if idx == -1 {
		return ""
	}
	return rest[:idx]
}

// extractBase64Data extracts the base64 data from a data URL.
// e.g., "data:image/png;base64,abc" -> "abc"
func extractBase64Data(dataURL string) string {
	idx := strings.Index(dataURL, ",")
	if idx == -1 {
		return ""
	}
	return dataURL[idx+1:]
}
