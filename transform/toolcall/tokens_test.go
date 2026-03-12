package toolcall

import (
	"testing"
)

func TestDefaultTokens_Values(t *testing.T) {
	tests := []struct {
		name     string
		actual   string
		expected string
	}{
		{"SectionBegin", DefaultTokens.SectionBegin, "<|tool_calls_section_begin|>"},
		{"CallBegin", DefaultTokens.CallBegin, "<|tool_call_begin|>"},
		{"ArgBegin", DefaultTokens.ArgBegin, "<|tool_call_argument_begin|>"},
		{"CallEnd", DefaultTokens.CallEnd, "<|tool_call_end|>"},
		{"SectionEnd", DefaultTokens.SectionEnd, "<|tool_calls_section_end|>"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.actual != tt.expected {
				t.Errorf("expected %q, got %q", tt.expected, tt.actual)
			}
		})
	}
}

func TestTokens_ContainsAny(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected bool
	}{
		{
			name:     "contains tool_call_begin",
			input:    "text<|tool_call_begin|>more",
			expected: true,
		},
		{
			name:     "contains tool_call_end",
			input:    "text<|tool_call_end|>more",
			expected: true,
		},
		{
			name:     "contains tool_call_argument_begin",
			input:    "text<|tool_call_argument_begin|>more",
			expected: true,
		},
		{
			name:     "contains partial tool_call",
			input:    "text<|tool_call_more",
			expected: true,
		},
		{
			name:     "no tool_call token",
			input:    "plain text without tokens",
			expected: false,
		},
		{
			name:     "empty string",
			input:    "",
			expected: false,
		},
		{
			name:     "similar but not matching",
			input:    "<|tool_calls_section_begin|>",
			expected: true,
		},
		{
			name:     "tool_calls (plural) should not match",
			input:    "<|tool_calls_begin|>",
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := DefaultTokens.ContainsAny(tt.input)
			if result != tt.expected {
				t.Errorf("ContainsAny(%q) = %v, expected %v", tt.input, result, tt.expected)
			}
		})
	}
}

func TestTokens_CustomTokens(t *testing.T) {
	custom := Tokens{
		SectionBegin: "[START]",
		CallBegin:    "[CALL]",
		ArgBegin:     "[ARGS]",
		CallEnd:      "[ENDCALL]",
		SectionEnd:   "[END]",
	}

	if custom.SectionBegin != "[START]" {
		t.Errorf("expected SectionBegin to be [START], got %q", custom.SectionBegin)
	}
	if custom.CallBegin != "[CALL]" {
		t.Errorf("expected CallBegin to be [CALL], got %q", custom.CallBegin)
	}
	if custom.ArgBegin != "[ARGS]" {
		t.Errorf("expected ArgBegin to be [ARGS], got %q", custom.ArgBegin)
	}
	if custom.CallEnd != "[ENDCALL]" {
		t.Errorf("expected CallEnd to be [ENDCALL], got %q", custom.CallEnd)
	}
	if custom.SectionEnd != "[END]" {
		t.Errorf("expected SectionEnd to be [END], got %q", custom.SectionEnd)
	}
}

func TestTokens_ContainsAnyWithCustomTokens(t *testing.T) {
	custom := Tokens{
		SectionBegin: "[SECTION]",
		CallBegin:    "[CALL]",
	}

	tests := []struct {
		name     string
		input    string
		expected bool
	}{
		{
			name:     "custom tokens still use default ContainsAny pattern",
			input:    "[CALL]test",
			expected: false,
		},
		{
			name:     "default tool_call pattern matches",
			input:    "<|tool_call_begin|>",
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := custom.ContainsAny(tt.input)
			if result != tt.expected {
				t.Errorf("ContainsAny(%q) = %v, expected %v", tt.input, result, tt.expected)
			}
		})
	}
}

func TestDefaultTokens_AllTokensAreDistinct(t *testing.T) {
	tokens := []string{
		DefaultTokens.SectionBegin,
		DefaultTokens.CallBegin,
		DefaultTokens.ArgBegin,
		DefaultTokens.CallEnd,
		DefaultTokens.SectionEnd,
	}

	seen := make(map[string]int)
	for i, token := range tokens {
		if j, exists := seen[token]; exists {
			t.Errorf("token %q appears at both index %d and %d", token, j, i)
		}
		seen[token] = i
	}
}

func TestDefaultTokens_TokenLengths(t *testing.T) {
	if len(DefaultTokens.SectionBegin) == 0 {
		t.Error("SectionBegin should not be empty")
	}
	if len(DefaultTokens.CallBegin) == 0 {
		t.Error("CallBegin should not be empty")
	}
	if len(DefaultTokens.ArgBegin) == 0 {
		t.Error("ArgBegin should not be empty")
	}
	if len(DefaultTokens.CallEnd) == 0 {
		t.Error("CallEnd should not be empty")
	}
	if len(DefaultTokens.SectionEnd) == 0 {
		t.Error("SectionEnd should not be empty")
	}
}

func TestTokens_EmptyTokens(t *testing.T) {
	empty := Tokens{}

	if empty.SectionBegin != "" {
		t.Errorf("expected empty SectionBegin, got %q", empty.SectionBegin)
	}
	if empty.CallBegin != "" {
		t.Errorf("expected empty CallBegin, got %q", empty.CallBegin)
	}
	if empty.ArgBegin != "" {
		t.Errorf("expected empty ArgBegin, got %q", empty.ArgBegin)
	}
	if empty.CallEnd != "" {
		t.Errorf("expected empty CallEnd, got %q", empty.CallEnd)
	}
	if empty.SectionEnd != "" {
		t.Errorf("expected empty SectionEnd, got %q", empty.SectionEnd)
	}
}
