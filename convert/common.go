package convert

import (
	"encoding/json"
	"strings"

	"ai-proxy/types"
)

func AnthropicMessagesToOpenAI(anthMsgs []types.MessageInput) []types.Message {
	openMsgs := make([]types.Message, 0, len(anthMsgs))
	for _, anthMsg := range anthMsgs {
		openMsgs = append(openMsgs, anthropicMessageToOpenAI(anthMsg))
	}
	return openMsgs
}

func anthropicMessageToOpenAI(anthMsg types.MessageInput) types.Message {
	openMsg := types.Message{Role: anthMsg.Role}

	switch content := anthMsg.Content.(type) {
	case string:
		openMsg.Content = content
	case []interface{}:
		openMsg.Content, openMsg.ToolCalls, openMsg.ToolCallID = ConvertAnthropicContentBlocks(content)
	}

	return openMsg
}

func OpenAIMessagesToAnthropic(openMsgs []types.Message) []types.MessageInput {
	anthMsgs := make([]types.MessageInput, 0, len(openMsgs))
	for _, openMsg := range openMsgs {
		anthMsgs = append(anthMsgs, openAIMessageToAnthropic(openMsg))
	}
	return anthMsgs
}

func openAIMessageToAnthropic(openMsg types.Message) types.MessageInput {
	anthMsg := types.MessageInput{Role: openMsg.Role}

	if openMsg.Role == "tool" {
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

	if len(openMsg.ToolCalls) > 0 {
		blocks := make([]map[string]interface{}, 0, len(openMsg.ToolCalls)+1)
		if openMsg.Content != nil {
			if str, ok := openMsg.Content.(string); ok && str != "" {
				blocks = append(blocks, map[string]interface{}{
					"type": "text",
					"text": str,
				})
			}
		}
		for _, tc := range openMsg.ToolCalls {
			var inputObj map[string]interface{}
			if tc.Function.Arguments != "" {
				json.Unmarshal([]byte(tc.Function.Arguments), &inputObj)
			}
			blocks = append(blocks, map[string]interface{}{
				"type":  "tool_use",
				"id":    tc.ID,
				"name":  tc.Function.Name,
				"input": inputObj,
			})
		}
		anthMsg.Content = blocks
		return anthMsg
	}

	switch c := openMsg.Content.(type) {
	case string:
		anthMsg.Content = c
	default:
		if data, err := json.Marshal(c); err == nil {
			anthMsg.Content = string(data)
		}
	}

	return anthMsg
}

func AnthropicToolsToOpenAI(anthTools []types.ToolDef) []types.Tool {
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

func OpenAIToolsToAnthropic(openTools []types.Tool) []types.ToolDef {
	anthTools := make([]types.ToolDef, 0, len(openTools))
	for _, openTool := range openTools {
		anthTools = append(anthTools, types.ToolDef{
			Name:        openTool.Function.Name,
			Description: openTool.Function.Description,
			InputSchema: openTool.Function.Parameters,
		})
	}
	return anthTools
}

func ExtractTextFromContent(content interface{}) string {
	if content == nil {
		return ""
	}

	if s, ok := content.(string); ok {
		return s
	}

	if arr, ok := content.([]interface{}); ok {
		var result strings.Builder
		for _, item := range arr {
			if m, ok := item.(map[string]interface{}); ok {
				if text, ok := m["text"].(string); ok {
					if result.Len() > 0 {
						result.WriteString("\n")
					}
					result.WriteString(text)
				}
			}
		}
		return result.String()
	}

	return ""
}

func ConvertAnthropicContentBlocks(blocks []interface{}) (string, []types.ToolCall, string) {
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

func ExtractSystemMessage(system interface{}) string {
	if system == nil {
		return ""
	}

	if s, ok := system.(string); ok {
		return s
	}

	if arr, ok := system.([]interface{}); ok {
		var content strings.Builder
		for _, item := range arr {
			if m, ok := item.(map[string]interface{}); ok {
				if text, ok := m["text"].(string); ok {
					content.WriteString(text)
				}
			}
		}
		return content.String()
	}

	return ""
}

func ResponsesInputToAnthropicMessages(input interface{}) []types.MessageInput {
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

					content := extractResponsesContent(msg["content"])
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
				}
			}
		}
		return messages
	}

	return []types.MessageInput{}
}

func extractResponsesContent(content interface{}) string {
	if content == nil {
		return ""
	}

	if s, ok := content.(string); ok {
		return s
	}

	if arr, ok := content.([]interface{}); ok {
		var result strings.Builder
		for _, part := range arr {
			if partMap, ok := part.(map[string]interface{}); ok {
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

func ResponsesToolsToAnthropic(openTools []types.ResponsesTool) []types.ToolDef {
	if len(openTools) == 0 {
		return nil
	}

	anthTools := make([]types.ToolDef, 0, len(openTools))
	for _, openTool := range openTools {
		if openTool.Type == "function" {
			if openTool.Function != nil {
				anthTools = append(anthTools, types.ToolDef{
					Name:        openTool.Function.Name,
					Description: openTool.Function.Description,
					InputSchema: openTool.Function.Parameters,
				})
			} else if openTool.Name != "" {
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

func ResponsesToolChoiceToAnthropic(toolChoice interface{}) *types.ToolChoice {
	if toolChoice == nil {
		return nil
	}

	if s, ok := toolChoice.(string); ok {
		switch s {
		case "none":
			return nil
		case "auto":
			return &types.ToolChoice{Type: "auto"}
		case "required":
			return &types.ToolChoice{Type: "any"}
		default:
			return &types.ToolChoice{Type: "auto"}
		}
	}

	if obj, ok := toolChoice.(map[string]interface{}); ok {
		objType, _ := obj["type"].(string)
		if objType == "function" {
			if fn, ok := obj["function"].(map[string]interface{}); ok {
				if name, ok := fn["name"].(string); ok {
					return &types.ToolChoice{
						Type: "tool",
						Name: name,
					}
				}
			}
		}
	}

	return &types.ToolChoice{Type: "auto"}
}
