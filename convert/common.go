package convert

import (
	"ai-proxy/types"
)

func ConvertAnthropicMessagesToOpenAI(anthMsgs []types.MessageInput) []types.Message {
	if anthMsgs == nil {
		return nil
	}
	result := make([]types.Message, len(anthMsgs))
	for i, m := range anthMsgs {
		result[i] = types.Message{
			Role:    m.Role,
			Content: m.Content,
		}
	}
	return result
}

func ConvertOpenAIMessagesToAnthropic(openMsgs []types.Message) []types.MessageInput {
	if openMsgs == nil {
		return nil
	}
	result := make([]types.MessageInput, len(openMsgs))
	for i, m := range openMsgs {
		result[i] = types.MessageInput{
			Role:    m.Role,
			Content: m.Content,
		}
	}
	return result
}

func ConvertAnthropicToolsToOpenAI(anthTools []types.ToolDef) []types.Tool {
	if anthTools == nil {
		return nil
	}
	result := make([]types.Tool, len(anthTools))
	for i, t := range anthTools {
		result[i] = types.Tool{
			Type: "function",
			Function: types.ToolFunction{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.InputSchema,
			},
		}
	}
	return result
}

func ConvertOpenAIToolsToAnthropic(openTools []types.Tool) []types.ToolDef {
	if openTools == nil {
		return nil
	}
	result := make([]types.ToolDef, len(openTools))
	for i, t := range openTools {
		result[i] = types.ToolDef{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			InputSchema: t.Function.Parameters,
		}
	}
	return result
}

func ExtractTextFromContent(content interface{}) string {
	if content == nil {
		return ""
	}
	switch v := content.(type) {
	case string:
		return v
	case []types.ContentBlock:
		var text string
		for _, block := range v {
			if block.Type == "text" {
				text += block.Text
			}
		}
		return text
	case []interface{}:
		var text string
		for _, item := range v {
			if block, ok := item.(types.ContentBlock); ok && block.Type == "text" {
				text += block.Text
			}
		}
		return text
	}
	return ""
}

func ExtractSystemMessage(system interface{}) string {
	if system == nil {
		return ""
	}
	switch v := system.(type) {
	case string:
		return v
	case []types.ContentBlock:
		var text string
		for _, block := range v {
			if block.Type == "text" {
				text += block.Text
			}
		}
		return text
	case []interface{}:
		var text string
		for _, item := range v {
			if block, ok := item.(types.ContentBlock); ok && block.Type == "text" {
				text += block.Text
			}
		}
		return text
	}
	return ""
}
