package convert

import (
	"encoding/json"
	"testing"

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
