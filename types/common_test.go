package types

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestNormalizeToolID(t *testing.T) {
	tests := []struct {
		name     string
		raw      string
		index    int
		contains string
		exact    string
	}{
		{
			name:  "preserves_call_prefix",
			raw:   "call_abc123",
			index: 0,
			exact: "call_abc123",
		},
		{
			name:     "generates_from_index",
			raw:      "bash:1",
			index:    5,
			contains: "call_5_",
		},
		{
			name:     "generates_from_simple_name",
			raw:      "bash",
			index:    0,
			contains: "call_0_",
		},
		{
			name:  "trims_whitespace",
			raw:   "  call_xyz  ",
			index: 0,
			exact: "call_xyz",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := NormalizeToolID(tt.raw, tt.index)

			if tt.exact != "" && result != tt.exact {
				t.Errorf("Expected %q, got %q", tt.exact, result)
			}
			if tt.contains != "" && !strings.Contains(result, tt.contains) {
				t.Errorf("Expected result to contain %q, got %q", tt.contains, result)
			}
		})
	}
}

func TestParseFunctionName(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{"simple_name", "bash", "bash"},
		{"with_namespace", "functions.bash", "bash"},
		{"with_colon", "bash:1", "bash"},
		{"with_namespace_and_colon", "functions.bash:1", "bash"},
		{"complex_namespace", "tools.utils.bash:456", "bash"},
		{"with_colon_only", "bash:", "bash"},
		{"trims_whitespace", "  bash  ", "bash"},
		{"namespace_trailing_colon", "tools.read:", "read"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ParseFunctionName(tt.input)
			if result != tt.expected {
				t.Errorf("Expected %q, got %q", tt.expected, result)
			}
		})
	}
}

func TestToolCall_JSON(t *testing.T) {
	tc := ToolCall{
		ID:    "call_123",
		Type:  "function",
		Index: 0,
		Function: ToolFunction{
			Name:      "bash",
			Arguments: `{"cmd":"ls"}`,
		},
	}

	data, err := json.Marshal(tc)
	if err != nil {
		t.Fatalf("Failed to marshal: %v", err)
	}

	var parsed ToolCall
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("Failed to unmarshal: %v", err)
	}

	if parsed.ID != tc.ID {
		t.Errorf("Expected ID %q, got %q", tc.ID, parsed.ID)
	}
	if parsed.Function.Name != tc.Function.Name {
		t.Errorf("Expected function name %q, got %q", tc.Function.Name, parsed.Function.Name)
	}
}
