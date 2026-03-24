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
		maxTokens = defaultAnthropicMaxTokens
	}
	stream := true
	if openReq.Stream != nil {
		stream = *openReq.Stream
	}
	anthReq := types.MessageRequest{
		Model:       openReq.Model,
		MaxTokens:   maxTokens,
		Stream:      stream,
		Temperature: ClampTemperatureToAnthropic(openReq.Temperature),
		TopP:        openReq.TopP,
	}

	anthReq.Messages, anthReq.System = convertResponsesInputToAnthropicMessages(openReq.Instructions, openReq.Input)
	anthReq.Messages = normalizeResponsesAnthropicMessages(anthReq.Messages)

	// Convert tools
	anthReq.Tools = convertResponsesToolsToAnthropic(openReq.Tools)

	// Convert tool_choice
	anthReq.ToolChoice = convertResponsesToolChoiceToAnthropic(openReq.ToolChoice)
	if isResponsesToolChoiceNone(openReq.ToolChoice) {
		anthReq.Tools = nil
		anthReq.ToolChoice = nil
	}

	// Convert reasoning to thinking configuration
	if openReq.Reasoning != nil {
		anthReq.Thinking = convertReasoningToThinking(openReq.Reasoning, maxTokens)
	}

	// Convert metadata.user_id
	if openReq.Metadata != nil {
		if userID, ok := openReq.Metadata["user_id"]; ok {
			if s, ok := userID.(string); ok && s != "" {
				anthReq.Metadata = &types.AnthropicMetadata{
					UserID: s,
				}
			}
		}
	}

	return json.Marshal(anthReq)
}

func convertResponsesInputToAnthropicMessages(instructions string, input interface{}) ([]types.MessageInput, interface{}) {
	systemParts := make([]string, 0, 4)
	seen := make(map[string]struct{})

	addSystemPart := func(text string) {
		trimmed := strings.TrimSpace(text)
		if trimmed == "" {
			return
		}
		if _, ok := seen[trimmed]; ok {
			return
		}
		seen[trimmed] = struct{}{}
		systemParts = append(systemParts, trimmed)
	}

	addSystemPart(instructions)

	if arr, ok := input.([]interface{}); ok {
		system := strings.Join(systemParts, "\n\n")
		if system == "" {
			return convertResponsesInputItemsToAnthropicMessages(arr, addSystemPart), nil
		}
		return convertResponsesInputItemsToAnthropicMessages(arr, addSystemPart), system
	}

	if input == nil {
		system := strings.Join(systemParts, "\n\n")
		if system == "" {
			return nil, nil
		}
		return nil, system
	}

	if s, ok := input.(string); ok {
		system := strings.Join(systemParts, "\n\n")
		if strings.TrimSpace(s) != "" {
			if system == "" {
				return []types.MessageInput{{Role: "user", Content: s}}, nil
			}
			return []types.MessageInput{{Role: "user", Content: s}}, system
		}
		if system == "" {
			return nil, nil
		}
		return nil, system
	}

	data, err := json.Marshal(input)
	if err != nil {
		system := strings.Join(systemParts, "\n\n")
		if system == "" {
			return nil, nil
		}
		return nil, system
	}

	var items []interface{}
	if err := json.Unmarshal(data, &items); err != nil {
		system := strings.Join(systemParts, "\n\n")
		if system == "" {
			return nil, nil
		}
		return nil, system
	}

	system := strings.Join(systemParts, "\n\n")
	if system == "" {
		return convertResponsesInputItemsToAnthropicMessages(items, addSystemPart), nil
	}
	return convertResponsesInputItemsToAnthropicMessages(items, addSystemPart), system
}

func convertResponsesInputItemsToAnthropicMessages(items []interface{}, addSystemPart func(string)) []types.MessageInput {
	messages := make([]types.MessageInput, 0, len(items))

	var currentRole string
	var currentBlocks []interface{}
	flush := func() {
		if currentRole == "" {
			return
		}
		content := finalizeResponsesAnthropicContent(currentBlocks)
		if content == nil {
			currentRole = ""
			currentBlocks = nil
			return
		}
		messages = append(messages, types.MessageInput{
			Role:    currentRole,
			Content: content,
		})
		currentRole = ""
		currentBlocks = nil
	}

	appendMessage := func(role string, blocks []interface{}) {
		if len(blocks) == 0 {
			return
		}
		if currentRole == "" {
			currentRole = role
			currentBlocks = append([]interface{}(nil), blocks...)
			return
		}
		if currentRole == role {
			currentBlocks = append(currentBlocks, blocks...)
			return
		}
		flush()
		currentRole = role
		currentBlocks = append([]interface{}(nil), blocks...)
	}

	for _, item := range items {
		msg, ok := item.(map[string]interface{})
		if !ok {
			continue
		}

		itemType, _ := msg["type"].(string)
		switch itemType {
		case "message":
			role := "user"
			if r, ok := msg["role"].(string); ok {
				switch r {
				case "assistant":
					role = "assistant"
				case "developer", "system":
					addSystemPart(extractContentFromInput(msg["content"]))
					continue
				default:
					role = "user"
				}
			}

			blocks := convertResponsesContentToAnthropicBlocks(msg["content"])

			// Handle embedded tool_calls in the combined message format
			if toolCallsRaw, ok := msg["tool_calls"]; ok && role == "assistant" {
				toolUseBlocks := convertResponsesToolCallsToAnthropicBlocks(toolCallsRaw)
				blocks = append(blocks, toolUseBlocks...)
			}

			appendMessage(role, blocks)

		case "function_call":
			block := convertResponsesFunctionCallToAnthropicBlock(msg)
			if block != nil {
				appendMessage("assistant", []interface{}{block})
			}

		case "function_call_output":
			block := convertResponsesFunctionCallOutputToAnthropicBlock(msg)
			if block != nil {
				appendMessage("user", []interface{}{block})
			}

		case "reasoning":
			// Dropped when converting into an Anthropic request.
		}
	}

	flush()

	return messages
}

func normalizeResponsesAnthropicMessages(messages []types.MessageInput) []types.MessageInput {
	if len(messages) == 0 {
		return messages
	}

	normalized := make([]types.MessageInput, 0, len(messages)+1)
	for _, msg := range messages {
		if len(normalized) > 0 && normalized[len(normalized)-1].Role == msg.Role {
			normalized[len(normalized)-1].Content = mergeResponsesAnthropicContent(
				normalized[len(normalized)-1].Content,
				msg.Content,
			)
			continue
		}
		normalized = append(normalized, msg)
	}

	return normalized
}

func mergeResponsesAnthropicContent(existing, next interface{}) interface{} {
	blocks := append(responsesAnthropicContentToBlocks(existing), responsesAnthropicContentToBlocks(next)...)
	return finalizeResponsesAnthropicContent(blocks)
}

func finalizeResponsesAnthropicContent(blocks []interface{}) interface{} {
	if len(blocks) == 0 {
		return nil
	}

	if text, ok := responsesAnthropicBlocksToText(blocks); ok {
		return text
	}

	return blocks
}

func responsesAnthropicContentToBlocks(content interface{}) []interface{} {
	switch v := content.(type) {
	case nil:
		return nil
	case string:
		if v == "" {
			return nil
		}
		return []interface{}{
			map[string]interface{}{
				"type": "text",
				"text": v,
			},
		}
	case []interface{}:
		blocks := make([]interface{}, 0, len(v))
		for _, item := range v {
			switch b := item.(type) {
			case map[string]interface{}:
				blocks = append(blocks, b)
			case string:
				if b != "" {
					blocks = append(blocks, map[string]interface{}{
						"type": "text",
						"text": b,
					})
				}
			}
		}
		return blocks
	default:
		data, err := json.Marshal(v)
		if err != nil {
			return nil
		}

		var arr []interface{}
		if err := json.Unmarshal(data, &arr); err == nil {
			return responsesAnthropicContentToBlocks(arr)
		}

		var s string
		if err := json.Unmarshal(data, &s); err == nil && s != "" {
			return []interface{}{
				map[string]interface{}{
					"type": "text",
					"text": s,
				},
			}
		}
		return nil
	}
}

func responsesAnthropicBlocksToText(blocks []interface{}) (string, bool) {
	texts := make([]string, 0, len(blocks))
	for _, item := range blocks {
		block, ok := item.(map[string]interface{})
		if !ok {
			return "", false
		}

		blockType, _ := block["type"].(string)
		if blockType != "text" {
			return "", false
		}

		if text, ok := block["text"].(string); ok {
			texts = append(texts, text)
		}
	}

	return strings.Join(texts, "\n"), true
}

func convertResponsesContentToAnthropicBlocks(content interface{}) []interface{} {
	switch v := content.(type) {
	case nil:
		return nil
	case string:
		if strings.TrimSpace(v) == "" {
			return nil
		}
		return []interface{}{
			map[string]interface{}{
				"type": "text",
				"text": v,
			},
		}
	case []interface{}:
		return convertResponsesContentPartsToAnthropicBlocks(v)
	default:
		data, err := json.Marshal(v)
		if err != nil {
			return nil
		}
		var parts []interface{}
		if err := json.Unmarshal(data, &parts); err != nil {
			return nil
		}
		return convertResponsesContentPartsToAnthropicBlocks(parts)
	}
}

func convertResponsesContentPartsToAnthropicBlocks(parts []interface{}) []interface{} {
	blocks := make([]interface{}, 0, len(parts))
	for _, part := range parts {
		partMap, ok := part.(map[string]interface{})
		if !ok {
			continue
		}

		block := convertResponsesContentPartToAnthropicBlock(partMap)
		if block != nil {
			blocks = append(blocks, block)
		}
	}
	return blocks
}

func convertResponsesContentPartToAnthropicBlock(partMap map[string]interface{}) map[string]interface{} {
	partType, _ := partMap["type"].(string)

	switch partType {
	case "input_text", "output_text", "text", "refusal":
		text, _ := partMap["text"].(string)
		if text == "" {
			return nil
		}
		return map[string]interface{}{
			"type": "text",
			"text": text,
		}

	case "input_image":
		return convertResponsesImagePartToAnthropicBlock(partMap)

	case "input_file":
		filename := ""
		if fileData, ok := partMap["file_data"].(map[string]interface{}); ok {
			filename, _ = fileData["filename"].(string)
		}
		if filename == "" {
			filename = "file"
		}
		return map[string]interface{}{
			"type": "text",
			"text": "[File attached: " + filename + "]",
		}
	}

	return nil
}

func convertResponsesImagePartToAnthropicBlock(partMap map[string]interface{}) map[string]interface{} {
	imageURL, _ := partMap["image_url"].(string)
	if imageURL == "" {
		if source, ok := partMap["source"].(map[string]interface{}); ok {
			if srcType, ok := source["type"].(string); ok {
				switch srcType {
				case "base64":
					mediaType, _ := source["media_type"].(string)
					data, _ := source["data"].(string)
					if data == "" {
						return nil
					}
					return map[string]interface{}{
						"type": "image",
						"source": map[string]interface{}{
							"type":       "base64",
							"media_type": mediaType,
							"data":       data,
						},
					}
				case "url":
					imageURL, _ = source["url"].(string)
				}
			}
		}
	}

	if imageURL == "" {
		return nil
	}

	source := map[string]interface{}{
		"type": "url",
		"url":  imageURL,
	}

	if mediaType, data, err := ParseDataURI(imageURL); err == nil {
		source = map[string]interface{}{
			"type":       "base64",
			"media_type": mediaType,
			"data":       data,
		}
	}

	return map[string]interface{}{
		"type":   "image",
		"source": source,
	}
}

func convertResponsesFunctionCallToAnthropicBlock(item map[string]interface{}) map[string]interface{} {
	callID, _ := item["call_id"].(string)
	if callID == "" {
		callID, _ = item["id"].(string)
	}
	name, _ := item["name"].(string)
	if name == "" {
		return nil
	}

	input := map[string]interface{}{}
	switch args := item["arguments"].(type) {
	case string:
		if strings.TrimSpace(args) != "" {
			_ = json.Unmarshal([]byte(args), &input)
		}
	case map[string]interface{}:
		input = args
	default:
		if data, err := json.Marshal(args); err == nil {
			_ = json.Unmarshal(data, &input)
		}
	}

	return map[string]interface{}{
		"type":  "tool_use",
		"id":    callID,
		"name":  name,
		"input": input,
	}
}

// convertResponsesToolCallsToAnthropicBlocks converts an array of tool_calls
// to Anthropic tool_use blocks. This handles the combined message format where
// tool_calls are embedded in the message item.
func convertResponsesToolCallsToAnthropicBlocks(toolCallsRaw interface{}) []interface{} {
	if toolCallsRaw == nil {
		return nil
	}

	var blocks []interface{}

	switch v := toolCallsRaw.(type) {
	case []interface{}:
		for _, tc := range v {
			if tcMap, ok := tc.(map[string]interface{}); ok {
				block := convertResponsesFunctionCallToAnthropicBlock(tcMap)
				if block != nil {
					blocks = append(blocks, block)
				}
			}
		}
	}

	return blocks
}

func convertResponsesFunctionCallOutputToAnthropicBlock(item map[string]interface{}) map[string]interface{} {
	callID, _ := item["call_id"].(string)
	if callID == "" {
		callID, _ = item["tool_call_id"].(string)
	}
	if callID == "" {
		return nil
	}

	output, _ := item["output"].(string)
	isError, _ := item["is_error"].(bool)

	return map[string]interface{}{
		"type":        "tool_result",
		"tool_use_id": callID,
		"content":     output,
		"is_error":    isError,
	}
}

func convertResponsesToolChoiceToAnthropic(toolChoice interface{}) *types.ToolChoice {
	switch tc := toolChoice.(type) {
	case nil:
		return nil
	case string:
		switch tc {
		case "auto":
			return &types.ToolChoice{Type: "auto"}
		case "required":
			return &types.ToolChoice{Type: "any"}
		case "none":
			return nil
		default:
			return &types.ToolChoice{Type: "auto"}
		}
	case map[string]interface{}:
		if choiceType, ok := tc["type"].(string); ok && choiceType == "function" {
			if fn, ok := tc["function"].(map[string]interface{}); ok {
				if name, ok := fn["name"].(string); ok && name != "" {
					return &types.ToolChoice{Type: "tool", Name: name}
				}
			}
			if name, ok := tc["name"].(string); ok && name != "" {
				return &types.ToolChoice{Type: "tool", Name: name}
			}
		}
	default:
		data, err := json.Marshal(tc)
		if err == nil {
			var raw map[string]interface{}
			if err := json.Unmarshal(data, &raw); err == nil {
				return convertResponsesToolChoiceToAnthropic(raw)
			}
		}
	}

	return nil
}

func isResponsesToolChoiceNone(toolChoice interface{}) bool {
	s, ok := toolChoice.(string)
	return ok && s == "none"
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
				case "refusal":
					// Per documentation, treat refusal as text content
					if text, ok := partMap["text"].(string); ok && text != "" {
						if result.Len() > 0 {
							result.WriteString("\n")
						}
						result.WriteString(text)
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

	// Then, convert output items to input format for the assistant's responses.
	// When an assistant message is followed by function_calls, combine them
	// into a single input item with both content and tool_calls.
	items = append(items, combineOutputItems(hist.Output)...)

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

// combineOutputItems converts output items to input format, combining assistant messages
// with their corresponding function_calls into a single input item.
//
// In the Responses API, an assistant response can have multiple output items:
// a message (text) and function_calls. When these are prepended to the next request's
// input, they need to be combined so that the Chat Completions API sees them as a single
// assistant message with both content and tool_calls.
//
// Without this combination, the input would have:
//   - message (assistant text)
//   - function_call
//   - function_call_output
//
// Which converts to two assistant messages (wrong). With combination:
//   - message with tool_calls (combined)
//   - function_call_output
//
// Which correctly converts to one assistant message with tool_calls followed by tool message.
func combineOutputItems(outputs []types.OutputItem) []interface{} {
	var result []interface{}

	// First pass: collect assistant messages and function_calls
	var assistantMsg *types.OutputItem
	var functionCalls []types.OutputItem

	for _, output := range outputs {
		switch output.Type {
		case "message":
			if output.Role == "assistant" {
				assistantMsg = &output
			}
		case "function_call":
			functionCalls = append(functionCalls, output)
		}
	}

	// If we have both an assistant message and function_calls, combine them
	if assistantMsg != nil && len(functionCalls) > 0 {
		// Create a combined input item with both content and tool_calls
		combinedItem := map[string]interface{}{
			"type": "message",
			"role": "assistant",
		}

		// Add content if present
		if len(assistantMsg.Content) > 0 {
			contentParts := make([]interface{}, len(assistantMsg.Content))
			for i, c := range assistantMsg.Content {
				contentParts[i] = map[string]interface{}{
					"type": c.Type,
					"text": c.Text,
				}
			}
			combinedItem["content"] = contentParts
		}

		// Add tool_calls as a nested array
		toolCalls := make([]interface{}, len(functionCalls))
		for i, fc := range functionCalls {
			toolCalls[i] = map[string]interface{}{
				"type":      "function_call",
				"call_id":   fc.CallID,
				"name":      fc.Name,
				"arguments": fc.Arguments,
			}
		}
		combinedItem["tool_calls"] = toolCalls

		result = append(result, combinedItem)
	} else {
		// No combination needed, add items as-is
		for _, output := range outputs {
			switch output.Type {
			case "message":
				if output.Role == "assistant" {
					itemMap := map[string]interface{}{
						"type": "message",
						"role": "assistant",
					}
					if len(output.Content) > 0 {
						contentParts := make([]interface{}, len(output.Content))
						for i, c := range output.Content {
							contentParts[i] = map[string]interface{}{
								"type": c.Type,
								"text": c.Text,
							}
						}
						itemMap["content"] = contentParts
					}
					result = append(result, itemMap)
				}
			case "function_call":
				result = append(result, map[string]interface{}{
					"type":      "function_call",
					"call_id":   output.CallID,
					"name":      output.Name,
					"arguments": output.Arguments,
				})
			}
		}
	}

	return result
}
