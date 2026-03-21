package convert

import (
	"encoding/json"
	"testing"

	"ai-proxy/conversation"
	"ai-proxy/types"
)

func TestResponsesToAnthropicConverter_Convert(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		validate func(t *testing.T, output []byte)
	}{
		{
			name: "simple string input",
			input: `{
				"model": "claude-3-opus",
				"input": "Hello, world!"
			}`,
			validate: func(t *testing.T, output []byte) {
				var req types.MessageRequest
				if err := json.Unmarshal(output, &req); err != nil {
					t.Fatalf("Failed to parse output: %v", err)
				}
				if req.Model != "claude-3-opus" {
					t.Errorf("Expected model claude-3-opus, got %s", req.Model)
				}
				if len(req.Messages) != 1 {
					t.Errorf("Expected 1 message, got %d", len(req.Messages))
				}
				if req.Messages[0].Role != "user" {
					t.Errorf("Expected user role, got %s", req.Messages[0].Role)
				}
			},
		},
		{
			name: "input with instructions",
			input: `{
				"model": "claude-3-opus",
				"input": "Hello",
				"instructions": "You are a helpful assistant."
			}`,
			validate: func(t *testing.T, output []byte) {
				var req types.MessageRequest
				if err := json.Unmarshal(output, &req); err != nil {
					t.Fatalf("Failed to parse output: %v", err)
				}
				if req.System != "You are a helpful assistant." {
					t.Errorf("Expected system message, got %s", req.System)
				}
			},
		},
		{
			name: "input with max_output_tokens",
			input: `{
				"model": "claude-3-opus",
				"input": "Hello",
				"max_output_tokens": 1000
			}`,
			validate: func(t *testing.T, output []byte) {
				var req types.MessageRequest
				if err := json.Unmarshal(output, &req); err != nil {
					t.Fatalf("Failed to parse output: %v", err)
				}
				if req.MaxTokens != 1000 {
					t.Errorf("Expected max_tokens 1000, got %d", req.MaxTokens)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output, err := TransformResponsesToAnthropic([]byte(tt.input))
			if err != nil {
				t.Fatalf("TransformResponsesToAnthropic returned error: %v", err)
			}
			tt.validate(t, output)
		})
	}
}

func TestResponsesToAnthropicConverter_StructuredContentAndToolChoice(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		validate func(t *testing.T, output []byte)
	}{
		{
			name: "input image preserves URL structure",
			input: `{
				"model": "claude-3-opus",
				"input": [
					{
						"type": "message",
						"role": "user",
						"content": [
							{"type": "input_text", "text": "Describe this image."},
							{"type": "input_image", "image_url": "https://example.com/image.png"}
						]
					}
				]
			}`,
			validate: func(t *testing.T, output []byte) {
				var req types.MessageRequest
				if err := json.Unmarshal(output, &req); err != nil {
					t.Fatalf("Failed to parse output: %v", err)
				}
				if len(req.Messages) != 1 {
					t.Fatalf("Expected 1 message, got %d", len(req.Messages))
				}
				content, ok := req.Messages[0].Content.([]interface{})
				if !ok {
					t.Fatalf("Expected structured content, got %T", req.Messages[0].Content)
				}
				if len(content) != 2 {
					t.Fatalf("Expected 2 content blocks, got %d", len(content))
				}
				imageBlock, ok := content[1].(map[string]interface{})
				if !ok {
					t.Fatalf("Expected image block, got %T", content[1])
				}
				if imageBlock["type"] != "image" {
					t.Fatalf("Expected image block, got %v", imageBlock["type"])
				}
				source, ok := imageBlock["source"].(map[string]interface{})
				if !ok {
					t.Fatalf("Expected image source object, got %T", imageBlock["source"])
				}
				if source["type"] != "url" {
					t.Fatalf("Expected url image source, got %v", source["type"])
				}
				if source["url"] != "https://example.com/image.png" {
					t.Fatalf("Expected image url preserved, got %v", source["url"])
				}
			},
		},
		{
			name: "input image data uri becomes base64 source",
			input: `{
				"model": "claude-3-opus",
				"input": [
					{
						"type": "message",
						"role": "user",
						"content": [
							{"type": "input_image", "image_url": "data:image/png;base64,aGVsbG8="}
						]
					}
				]
			}`,
			validate: func(t *testing.T, output []byte) {
				var req types.MessageRequest
				if err := json.Unmarshal(output, &req); err != nil {
					t.Fatalf("Failed to parse output: %v", err)
				}
				if len(req.Messages) != 1 {
					t.Fatalf("Expected 1 message, got %d", len(req.Messages))
				}
				content, ok := req.Messages[0].Content.([]interface{})
				if !ok {
					t.Fatalf("Expected structured content, got %T", req.Messages[0].Content)
				}
				imageBlock, ok := content[0].(map[string]interface{})
				if !ok {
					t.Fatalf("Expected image block, got %T", content[0])
				}
				source, ok := imageBlock["source"].(map[string]interface{})
				if !ok {
					t.Fatalf("Expected image source object, got %T", imageBlock["source"])
				}
				if source["type"] != "base64" {
					t.Fatalf("Expected base64 image source, got %v", source["type"])
				}
				if source["media_type"] != "image/png" {
					t.Fatalf("Expected media_type image/png, got %v", source["media_type"])
				}
				if source["data"] != "aGVsbG8=" {
					t.Fatalf("Expected base64 image data preserved, got %v", source["data"])
				}
			},
		},
		{
			name: "flat tool_choice object",
			input: `{
				"model": "claude-3-opus",
				"input": "Hello",
				"tool_choice": {"type": "function", "name": "lookup"}
			}`,
			validate: func(t *testing.T, output []byte) {
				var req types.MessageRequest
				if err := json.Unmarshal(output, &req); err != nil {
					t.Fatalf("Failed to parse output: %v", err)
				}
				if req.ToolChoice == nil {
					t.Fatal("Expected tool_choice to be set")
				}
				if req.ToolChoice.Type != "tool" {
					t.Fatalf("Expected tool_choice type tool, got %s", req.ToolChoice.Type)
				}
				if req.ToolChoice.Name != "lookup" {
					t.Fatalf("Expected tool_choice name lookup, got %s", req.ToolChoice.Name)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output, err := TransformResponsesToAnthropic([]byte(tt.input))
			if err != nil {
				t.Fatalf("TransformResponsesToAnthropic returned error: %v", err)
			}
			tt.validate(t, output)
		})
	}
}

func TestResponsesToAnthropicConverter_MergeAssistantAndToolOutputs(t *testing.T) {
	input := `{
		"model": "claude-3-opus",
		"input": [
			{"type": "message", "role": "user", "content": "Tell me something."},
			{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Let me check."}]},
			{"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{\"q\":\"weather\"}"},
			{"type": "function_call_output", "call_id": "call_1", "output": "Sunny"},
			{"type": "function_call_output", "call_id": "call_2", "output": "Warm"}
		]
	}`

	output, err := TransformResponsesToAnthropic([]byte(input))
	if err != nil {
		t.Fatalf("TransformResponsesToAnthropic returned error: %v", err)
	}

	var req types.MessageRequest
	if err := json.Unmarshal(output, &req); err != nil {
		t.Fatalf("Failed to parse output: %v", err)
	}

	if len(req.Messages) != 3 {
		t.Fatalf("Expected 3 messages, got %d", len(req.Messages))
	}
	if req.Messages[0].Role != "user" {
		t.Fatalf("Expected first message role user, got %s", req.Messages[0].Role)
	}
	if req.Messages[1].Role != "assistant" {
		t.Fatalf("Expected second message role assistant, got %s", req.Messages[1].Role)
	}
	assistantContent, ok := req.Messages[1].Content.([]interface{})
	if !ok {
		t.Fatalf("Expected assistant content array, got %T", req.Messages[1].Content)
	}
	if len(assistantContent) != 2 {
		t.Fatalf("Expected assistant content to contain text and tool_use, got %d blocks", len(assistantContent))
	}
	toolUse, ok := assistantContent[1].(map[string]interface{})
	if !ok {
		t.Fatalf("Expected tool_use block, got %T", assistantContent[1])
	}
	if toolUse["type"] != "tool_use" {
		t.Fatalf("Expected tool_use block, got %v", toolUse["type"])
	}
	if toolUse["id"] != "call_1" {
		t.Fatalf("Expected tool_use id call_1, got %v", toolUse["id"])
	}

	if req.Messages[2].Role != "user" {
		t.Fatalf("Expected third message role user, got %s", req.Messages[2].Role)
	}
	toolResults, ok := req.Messages[2].Content.([]interface{})
	if !ok {
		t.Fatalf("Expected tool_result array, got %T", req.Messages[2].Content)
	}
	if len(toolResults) != 2 {
		t.Fatalf("Expected two tool_result blocks, got %d", len(toolResults))
	}
	firstResult, ok := toolResults[0].(map[string]interface{})
	if !ok {
		t.Fatalf("Expected tool_result block, got %T", toolResults[0])
	}
	if firstResult["type"] != "tool_result" {
		t.Fatalf("Expected tool_result block, got %v", firstResult["type"])
	}
	if firstResult["tool_use_id"] != "call_1" {
		t.Fatalf("Expected first tool_result call_1, got %v", firstResult["tool_use_id"])
	}
	secondResult, ok := toolResults[1].(map[string]interface{})
	if !ok {
		t.Fatalf("Expected tool_result block, got %T", toolResults[1])
	}
	if secondResult["tool_use_id"] != "call_2" {
		t.Fatalf("Expected second tool_result call_2, got %v", secondResult["tool_use_id"])
	}
}

func TestResponsesToAnthropicConverter_PreviousResponseID(t *testing.T) {
	oldStore := conversation.DefaultStore
	conversation.DefaultStore = conversation.NewStore(conversation.Config{})
	t.Cleanup(func() {
		conversation.DefaultStore = oldStore
	})

	conversation.StoreInDefault(&conversation.Conversation{
		ID: "resp_prev123",
		Input: []types.InputItem{
			{Type: "message", Role: "user", Content: "Earlier question?"},
		},
		Output: []types.OutputItem{
			{
				Type: "message",
				Role: "assistant",
				Content: []types.OutputContent{
					{Type: "output_text", Text: "Earlier answer."},
				},
			},
		},
	})

	input := `{
		"model": "claude-3-opus",
		"input": "What about tomorrow?",
		"previous_response_id": "resp_prev123"
	}`

	output, err := TransformResponsesToAnthropic([]byte(input))
	if err != nil {
		t.Fatalf("TransformResponsesToAnthropic returned error: %v", err)
	}

	var req types.MessageRequest
	if err := json.Unmarshal(output, &req); err != nil {
		t.Fatalf("Failed to parse output: %v", err)
	}

	if len(req.Messages) != 3 {
		t.Fatalf("Expected 3 messages, got %d", len(req.Messages))
	}
	if req.Messages[0].Role != "user" || req.Messages[0].Content != "Earlier question?" {
		t.Fatalf("Expected previous user message first, got %+v", req.Messages[0])
	}
	if req.Messages[1].Role != "assistant" || req.Messages[1].Content != "Earlier answer." {
		t.Fatalf("Expected previous assistant message second, got %+v", req.Messages[1])
	}
	if req.Messages[2].Role != "user" || req.Messages[2].Content != "What about tomorrow?" {
		t.Fatalf("Expected current user message last, got %+v", req.Messages[2])
	}
}

func TestConvertReasoningToThinking_Effort(t *testing.T) {
	tests := []struct {
		name       string
		reasoning  *types.ReasoningConfig
		maxTokens  int
		wantNil    bool
		wantBudget int
		budgetMin  int
		budgetMax  int
	}{
		{
			name:      "effort high",
			reasoning: &types.ReasoningConfig{Effort: "high"},
			maxTokens: 10000,
			wantNil:   false,
			budgetMin: 2048,
			budgetMax: 4500,
		},
		{
			name:      "effort medium",
			reasoning: &types.ReasoningConfig{Effort: "medium"},
			maxTokens: 10000,
			wantNil:   false,
			budgetMin: 1024,
			budgetMax: 2500,
		},
		{
			name:      "effort low",
			reasoning: &types.ReasoningConfig{Effort: "low"},
			maxTokens: 10000,
			wantNil:   false,
			budgetMin: 1024,
			budgetMax: 1024,
		},
		{
			name:      "summary detailed",
			reasoning: &types.ReasoningConfig{Summary: "detailed"},
			maxTokens: 10000,
			wantNil:   false,
			budgetMin: 2048,
			budgetMax: 4500,
		},
		{
			name:      "summary concise",
			reasoning: &types.ReasoningConfig{Summary: "concise"},
			maxTokens: 10000,
			wantNil:   false,
			budgetMin: 1024,
			budgetMax: 2500,
		},
		{
			name:      "effort high with summary detailed",
			reasoning: &types.ReasoningConfig{Effort: "high", Summary: "detailed"},
			maxTokens: 10000,
			wantNil:   false,
			budgetMin: 2048,
			budgetMax: 4500,
		},
		{
			name:      "effort high overrides summary concise",
			reasoning: &types.ReasoningConfig{Effort: "high", Summary: "concise"},
			maxTokens: 10000,
			wantNil:   false,
			budgetMin: 2048,
			budgetMax: 4500,
		},
		{
			name:      "nil reasoning",
			reasoning: nil,
			maxTokens: 10000,
			wantNil:   true,
		},
		{
			name:      "empty reasoning",
			reasoning: &types.ReasoningConfig{},
			maxTokens: 10000,
			wantNil:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := convertReasoningToThinking(tt.reasoning, tt.maxTokens)

			if tt.wantNil {
				if result != nil {
					t.Errorf("Expected nil result, got %+v", result)
				}
				return
			}

			if result == nil {
				t.Fatalf("Expected non-nil result, got nil")
			}

			if result.Type != "enabled" {
				t.Errorf("Expected Type 'enabled', got %s", result.Type)
			}

			if result.BudgetTokens < tt.budgetMin {
				t.Errorf("Expected BudgetTokens >= %d, got %d", tt.budgetMin, result.BudgetTokens)
			}
			if result.BudgetTokens > tt.budgetMax {
				t.Errorf("Expected BudgetTokens <= %d, got %d", tt.budgetMax, result.BudgetTokens)
			}
		})
	}
}

func TestConvertReasoningToThinking_EffortOnly(t *testing.T) {
	tests := []struct {
		name       string
		reasoning  *types.ReasoningConfig
		maxTokens  int
		wantNil    bool
		wantBudget int
	}{
		{
			name:       "effort low without summary",
			reasoning:  &types.ReasoningConfig{Effort: "low"},
			maxTokens:  10000,
			wantNil:    false,
			wantBudget: 1000,
		},
		{
			name:       "effort medium without summary",
			reasoning:  &types.ReasoningConfig{Effort: "medium"},
			maxTokens:  10000,
			wantNil:    false,
			wantBudget: 2500,
		},
		{
			name:       "effort high without summary",
			reasoning:  &types.ReasoningConfig{Effort: "high"},
			maxTokens:  10000,
			wantNil:    false,
			wantBudget: 4500,
		},
		{
			name:      "no effort no summary",
			reasoning: &types.ReasoningConfig{},
			maxTokens: 10000,
			wantNil:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := convertReasoningToThinking(tt.reasoning, tt.maxTokens)

			if tt.wantNil {
				if result != nil {
					t.Errorf("Expected nil result, got %+v", result)
				}
				return
			}

			if result == nil {
				t.Fatalf("Expected non-nil result, got nil")
			}

			if result.Type != "enabled" {
				t.Errorf("Expected Type 'enabled', got %s", result.Type)
			}

			expectedBudget := int(float64(tt.maxTokens) * float64(tt.wantBudget) / 10000)
			if result.BudgetTokens < expectedBudget-100 || result.BudgetTokens > expectedBudget+100 {
				t.Errorf("Expected BudgetTokens around %d, got %d", expectedBudget, result.BudgetTokens)
			}
		})
	}
}

func TestConvertReasoningToThinking_BudgetCapping(t *testing.T) {
	reasoning := &types.ReasoningConfig{Effort: "high"}
	maxTokens := 100000

	result := convertReasoningToThinking(reasoning, maxTokens)

	if result == nil {
		t.Fatalf("Expected non-nil result, got nil")
	}

	if result.BudgetTokens > 32000 {
		t.Errorf("Expected BudgetTokens <= 32000 (capped), got %d", result.BudgetTokens)
	}
}

func TestConvertReasoningToThinking_MinBudget(t *testing.T) {
	tests := []struct {
		name      string
		reasoning *types.ReasoningConfig
		maxTokens int
		minBudget int
	}{
		{
			name:      "effort high with small max_tokens",
			reasoning: &types.ReasoningConfig{Effort: "high"},
			maxTokens: 1000,
			minBudget: 2048,
		},
		{
			name:      "effort medium with small max_tokens",
			reasoning: &types.ReasoningConfig{Effort: "medium"},
			maxTokens: 1000,
			minBudget: 1024,
		},
		{
			name:      "effort low with small max_tokens",
			reasoning: &types.ReasoningConfig{Effort: "low"},
			maxTokens: 500,
			minBudget: 1024,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := convertReasoningToThinking(tt.reasoning, tt.maxTokens)

			if result == nil {
				t.Fatalf("Expected non-nil result, got nil")
			}

			if result.BudgetTokens < tt.minBudget {
				t.Errorf("Expected BudgetTokens >= %d (minimum), got %d", tt.minBudget, result.BudgetTokens)
			}
		})
	}
}

func TestTransformResponsesToAnthropic_ReasoningToThinking(t *testing.T) {
	tests := []struct {
		name         string
		input        string
		wantThinking bool
		wantBudget   int
	}{
		{
			name: "effort high enables thinking",
			input: `{
				"model": "claude-3-opus",
				"input": "Think about this",
				"reasoning": {"effort": "high"}
			}`,
			wantThinking: true,
		},
		{
			name: "effort low enables thinking",
			input: `{
				"model": "claude-3-opus",
				"input": "Quick thought",
				"reasoning": {"effort": "low"}
			}`,
			wantThinking: true,
		},
		{
			name: "summary detailed enables thinking",
			input: `{
				"model": "claude-3-opus",
				"input": "Think deeply",
				"reasoning": {"summary": "detailed"}
			}`,
			wantThinking: true,
		},
		{
			name: "no reasoning no thinking",
			input: `{
				"model": "claude-3-opus",
				"input": "Hello"
			}`,
			wantThinking: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output, err := TransformResponsesToAnthropic([]byte(tt.input))
			if err != nil {
				t.Fatalf("TransformResponsesToAnthropic returned error: %v", err)
			}

			var req types.MessageRequest
			if err := json.Unmarshal(output, &req); err != nil {
				t.Fatalf("Failed to parse output: %v", err)
			}

			if tt.wantThinking {
				if req.Thinking == nil {
					t.Error("Expected Thinking to be set")
					return
				}
				if req.Thinking.Type != "enabled" {
					t.Errorf("Expected Thinking.Type 'enabled', got %s", req.Thinking.Type)
				}
			} else {
				if req.Thinking != nil {
					t.Errorf("Expected Thinking to be nil, got %+v", req.Thinking)
				}
			}
		})
	}
}

// TestExtractContentFromInput_Refusal tests refusal content type handling.
func TestExtractContentFromInput_Refusal(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		wantText string
	}{
		{
			name: "refusal content treated as text",
			input: `{
				"model": "claude-3-opus",
				"input": [
					{
						"type": "message",
						"role": "assistant",
						"content": [
							{"type": "refusal", "text": "I cannot help with that request."}
						]
					}
				]
			}`,
			wantText: "I cannot help with that request.",
		},
		{
			name: "mixed refusal and output_text",
			input: `{
				"model": "claude-3-opus",
				"input": [
					{
						"type": "message",
						"role": "assistant",
						"content": [
							{"type": "output_text", "text": "Here is some info."},
							{"type": "refusal", "text": "But I cannot do more."}
						]
					}
				]
			}`,
			wantText: "Here is some info.\nBut I cannot do more.",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output, err := TransformResponsesToAnthropic([]byte(tt.input))
			if err != nil {
				t.Fatalf("TransformResponsesToAnthropic returned error: %v", err)
			}

			var req types.MessageRequest
			if err := json.Unmarshal(output, &req); err != nil {
				t.Fatalf("Failed to parse output: %v", err)
			}

			if len(req.Messages) == 0 {
				t.Fatal("Expected at least one message")
			}

			content, ok := req.Messages[0].Content.(string)
			if !ok {
				t.Fatalf("Expected content to be string, got %T", req.Messages[0].Content)
			}

			if content != tt.wantText {
				t.Errorf("Expected content %q, got %q", tt.wantText, content)
			}
		})
	}
}

// TestExtractContentFromInput_InputFile tests input_file content type handling.
func TestExtractContentFromInput_InputFile(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		wantText string
	}{
		{
			name: "input_file with filename",
			input: `{
				"model": "claude-3-opus",
				"input": [
					{
						"type": "message",
						"role": "user",
						"content": [
							{"type": "input_text", "text": "Check this file:"},
							{"type": "input_file", "file_data": {"filename": "document.pdf"}}
						]
					}
				]
			}`,
			wantText: "Check this file:\n[File attached: document.pdf]",
		},
		{
			name: "input_file only",
			input: `{
				"model": "claude-3-opus",
				"input": [
					{
						"type": "message",
						"role": "user",
						"content": [
							{"type": "input_file", "file_data": {"filename": "data.csv"}}
						]
					}
				]
			}`,
			wantText: "[File attached: data.csv]",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output, err := TransformResponsesToAnthropic([]byte(tt.input))
			if err != nil {
				t.Fatalf("TransformResponsesToAnthropic returned error: %v", err)
			}

			var req types.MessageRequest
			if err := json.Unmarshal(output, &req); err != nil {
				t.Fatalf("Failed to parse output: %v", err)
			}

			if len(req.Messages) == 0 {
				t.Fatal("Expected at least one message")
			}

			content, ok := req.Messages[0].Content.(string)
			if !ok {
				t.Fatalf("Expected content to be string, got %T", req.Messages[0].Content)
			}

			if content != tt.wantText {
				t.Errorf("Expected content %q, got %q", tt.wantText, content)
			}
		})
	}
}
