package downstream

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmaxmax/go-sse"
)

func TestToolCallTransformer_AllLogs(t *testing.T) {
	logDir := "../sse_logs"
	files, err := filepath.Glob(filepath.Join(logDir, "*.log"))
	if err != nil {
		t.Skipf("Log directory not found: %v", err)
	}

	if len(files) == 0 {
		t.Skip("No log files found")
	}

	t.Logf("Testing %d log files", len(files))

	toolCallCount := 0
	malformedCount := 0

	for _, logFile := range files {
		t.Run(filepath.Base(logFile), func(t *testing.T) {
			malformed, hasToolCalls := testSingleLogVerbose(t, logFile)
			if hasToolCalls {
				toolCallCount++
			}
			if malformed {
				malformedCount++
			}
		})
	}

	t.Logf("Summary: %d files with tool calls, %d files with malformed output", toolCallCount, malformedCount)
}

func testSingleLogVerbose(t *testing.T, logFile string) (malformed, hasToolCalls bool) {
	data, err := os.ReadFile(logFile)
	if err != nil {
		t.Fatalf("Failed to read log file: %v", err)
	}

	lines := strings.Split(string(data), "\n")
	var jsonLines []string
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "data: ") {
			jsonLines = append(jsonLines, strings.TrimPrefix(line, "data: "))
		}
	}

	if len(jsonLines) == 0 {
		t.Fatal("No SSE data lines found")
	}

	var output bytes.Buffer
	transformer := NewToolCallTransformer(&output)

	for _, jsonStr := range jsonLines {
		if jsonStr == "" || jsonStr == "[DONE]" {
			continue
		}
		event := &sse.Event{Data: jsonStr}
		transformer.Transform(event)
	}

	transformer.Flush()

	result := output.String()

	// Check for malformed tool call tokens in output
	lines = strings.Split(result, "\n")
	for i, line := range lines {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		jsonStr := strings.TrimPrefix(line, "data: ")
		if jsonStr == "" || jsonStr == "[DONE]" {
			continue
		}

		// Check for malformed tool call tokens in reasoning-related fields
		malformedPatterns := []string{
			"<|tool_call",
			"<|tool_call_end",
			"<|tool_calls_section",
			"<|tool_calls_section_end",
		}

		for _, pattern := range malformedPatterns {
			if containsInReasoningField(jsonStr, pattern) {
				t.Errorf("Line %d: Found malformed token '%s' in reasoning field:\n%s", i, pattern, jsonStr)
				malformed = true
			}
		}
	}

	// Check if output has actual tool_calls (in delta field)
	hasToolCalls = hasToolCallsInOutput(result)

	if hasToolCalls {
		t.Logf("Output contains tool_calls")
		printToolCalls(t, result)
	}

	return
}

func hasToolCallsInOutput(output string) bool {
	lines := strings.Split(output, "\n")
	for _, line := range lines {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		jsonStr := strings.TrimPrefix(line, "data: ")
		if jsonStr == "" || jsonStr == "[DONE]" {
			continue
		}

		var chunk map[string]interface{}
		if err := json.Unmarshal([]byte(jsonStr), &chunk); err != nil {
			continue
		}

		choices, ok := chunk["choices"].([]interface{})
		if !ok || len(choices) == 0 {
			continue
		}

		choice, ok := choices[0].(map[string]interface{})
		if !ok {
			continue
		}

		delta, ok := choice["delta"].(map[string]interface{})
		if !ok {
			continue
		}

		_, ok = delta["tool_calls"]
		if ok {
			return true
		}
	}
	return false
}

func containsInReasoningField(jsonStr, pattern string) bool {
	var chunk map[string]interface{}
	if err := json.Unmarshal([]byte(jsonStr), &chunk); err != nil {
		return false
	}

	choices, ok := chunk["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return false
	}

	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		return false
	}

	delta, ok := choice["delta"].(map[string]interface{})
	if !ok {
		return false
	}

	if reasoning, ok := delta["reasoning"].(string); ok {
		if strings.Contains(reasoning, pattern) {
			return true
		}
	}

	if reasoningContent, ok := delta["reasoning_content"].(string); ok {
		if strings.Contains(reasoningContent, pattern) {
			return true
		}
	}

	return false
}

func printToolCalls(t *testing.T, output string) {
	lines := strings.Split(output, "\n")
	for _, line := range lines {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		jsonStr := strings.TrimPrefix(line, "data: ")
		if jsonStr == "" || jsonStr == "[DONE]" {
			continue
		}

		var chunk map[string]interface{}
		if err := json.Unmarshal([]byte(jsonStr), &chunk); err != nil {
			continue
		}

		choices, ok := chunk["choices"].([]interface{})
		if !ok || len(choices) == 0 {
			continue
		}

		choice, ok := choices[0].(map[string]interface{})
		if !ok {
			continue
		}

		delta, ok := choice["delta"].(map[string]interface{})
		if !ok {
			continue
		}

		toolCalls, ok := delta["tool_calls"].([]interface{})
		if !ok || len(toolCalls) == 0 {
			continue
		}

		t.Logf("Found %d tool calls in chunk:", len(toolCalls))
		for i, tc := range toolCalls {
			tcMap, ok := tc.(map[string]interface{})
			if !ok {
				continue
			}
			id, _ := tcMap["id"].(string)
			ttyp, _ := tcMap["type"].(string)
			funcMap, _ := tcMap["function"].(map[string]interface{})
			name, _ := funcMap["name"].(string)
			args, _ := funcMap["arguments"].(string)
			t.Logf("  [%d] id=%q type=%q name=%q args=%q", i, id, ttyp, name, args)
		}
	}
}

func TestToolCallTransformer_SSELog(t *testing.T) {
	// Test case with content before and after tool call tokens
	testCases := []struct {
		name     string
		input    []string
		expected func(string) bool // returns true if output is valid
	}{
		{
			name: "content_before_and_after_toolcall",
			input: []string{
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":123,"model":"test","choices":[{"index":0,"delta":{"reasoning":"Hello <|tool_calls_section_begin|>world"}}]}`,
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":123,"model":"test","choices":[{"index":0,"delta":{"reasoning":" <|tool_call_begin|>bash<|tool_call_argument_begin|>{\"cmd\":\"ls\"}<|tool_call_end|> <|tool_calls_section_end|> end"}}]}`,
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":123,"model":"test","choices":[{"index":0,"delta":{"finish_reason":"stop"}}]}`,
			},
			expected: func(output string) bool {
				// Should have content "Hello " separate from tool_calls
				// Should have tool_calls
				// Should have content " end" separate
				hasHello := strings.Contains(output, `"content":"Hello "`)
				hasToolCalls := strings.Contains(output, `"tool_calls"`)
				hasEnd := strings.Contains(output, `"content":" end"`)
				return hasHello && hasToolCalls && hasEnd
			},
		},
		{
			name: "spaces_preserved",
			input: []string{
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":123,"model":"test","choices":[{"index":0,"delta":{"reasoning":"Here is the result: "}}]}`,
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":123,"model":"test","choices":[{"index":0,"delta":{"finish_reason":"stop"}}]}`,
			},
			expected: func(output string) bool {
				// Spaces should be preserved in reasoning field
				return strings.Contains(output, `"reasoning":"Here is the result: "`)
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var output bytes.Buffer
			transformer := NewToolCallTransformer(&output)

			for _, jsonStr := range tc.input {
				event := &sse.Event{Data: jsonStr}
				transformer.Transform(event)
			}

			transformer.Flush()

			result := output.String()
			t.Logf("Output:\n%s", result)
			if !tc.expected(result) {
				t.Errorf("Output did not match expected pattern")
			}
		})
	}
}
