package downstream

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/tmaxmax/go-sse"
)

type OpenAIChunk struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index int `json:"index"`
		Delta struct {
			Role             string `json:"role,omitempty"`
			Content          string `json:"content,omitempty"`
			Reasoning        string `json:"reasoning,omitempty"`
			ReasoningContent string `json:"reasoning_content,omitempty"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason,omitempty"`
	} `json:"choices"`
}

// TestAnthropicFromRealOpenAILogs tests conversion from real OpenAI-format Kimi logs
// to Anthropic format using actual log files from old_sse_logs directory.
func TestAnthropicFromRealOpenAILogs(t *testing.T) {
	logDir := "../old_sse_logs"
	files, err := os.ReadDir(logDir)
	if err != nil {
		t.Skipf("Log directory not found: %v", err)
	}

	var toolCallLogs []string
	for _, file := range files {
		if !strings.HasSuffix(file.Name(), ".log") {
			continue
		}
		data, err := os.ReadFile(logDir + "/" + file.Name())
		if err != nil {
			continue
		}
		if strings.Contains(string(data), "<|tool_calls_section_begin|>") {
			toolCallLogs = append(toolCallLogs, file.Name())
		}
	}

	if len(toolCallLogs) == 0 {
		t.Skip("No log files with tool calls found")
	}

	t.Logf("Found %d log files with tool calls", len(toolCallLogs))

	maxFiles := 5
	if len(toolCallLogs) < maxFiles {
		maxFiles = len(toolCallLogs)
	}

	for i := 0; i < maxFiles; i++ {
		logFile := toolCallLogs[i]
		t.Run(logFile, func(t *testing.T) {
			testSingleLogToAnthropic(t, logDir+"/"+logFile)
		})
	}
}

func testSingleLogToAnthropic(t *testing.T, logFile string) {
	data, err := os.ReadFile(logFile)
	if err != nil {
		t.Fatalf("Failed to read log file: %v", err)
	}

	// Convert OpenAI events to Anthropic format
	anthropicEvents := convertOpenAIToAnthropic(string(data))
	if len(anthropicEvents) == 0 {
		t.Fatal("No Anthropic events generated")
	}

	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	for _, event := range anthropicEvents {
		transformer.Transform(&sse.Event{Data: event})
	}

	result := output.String()

	if !strings.Contains(result, "event: message_start") {
		t.Error("Expected message_start event in Anthropic format")
	}

	toolUseCount := strings.Count(result, "\"type\":\"tool_use\"")
	if toolUseCount == 0 && strings.Contains(string(data), "<|tool_calls_section_begin|>") {
		t.Error("Expected tool_use blocks in output for Kimi tool calls")
	}

	t.Logf("Converted %d events, output has %d tool_use blocks", len(anthropicEvents), toolUseCount)
}

func convertOpenAIToAnthropic(data string) []string {
	var anthropicEvents []string
	var messageID string
	var model string
	var reasoning strings.Builder

	lines := strings.Split(data, "\n")
	for _, line := range lines {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		jsonStr := strings.TrimPrefix(line, "data: ")
		if jsonStr == "" || jsonStr == "[DONE]" {
			continue
		}

		var chunk OpenAIChunk
		if err := json.Unmarshal([]byte(jsonStr), &chunk); err != nil {
			continue
		}

		if messageID == "" {
			messageID = chunk.ID
			model = chunk.Model
		}

		if len(chunk.Choices) > 0 {
			reasoning.WriteString(chunk.Choices[0].Delta.Reasoning)
		}
	}

	// Generate minimal Anthropic events for transformer
	msgStart := map[string]interface{}{
		"type": "message_start",
		"message": map[string]interface{}{
			"id":      messageID,
			"type":    "message",
			"role":    "assistant",
			"model":   model,
			"content": []interface{}{},
			"usage": map[string]int{
				"input_tokens":  100,
				"output_tokens": 1,
			},
		},
	}
	startJSON, _ := json.Marshal(msgStart)
	anthropicEvents = append(anthropicEvents, string(startJSON))

	blockStart := map[string]interface{}{
		"type":  "content_block_start",
		"index": 0,
		"content_block": map[string]string{
			"type": "thinking",
		},
	}
	blockStartJSON, _ := json.Marshal(blockStart)
	anthropicEvents = append(anthropicEvents, string(blockStartJSON))

	thinkingDelta := map[string]interface{}{
		"type":  "content_block_delta",
		"index": 0,
		"delta": map[string]string{
			"type":     "thinking_delta",
			"thinking": reasoning.String(),
		},
	}
	deltaJSON, _ := json.Marshal(thinkingDelta)
	anthropicEvents = append(anthropicEvents, string(deltaJSON))

	blockStop := map[string]interface{}{
		"type":  "content_block_stop",
		"index": 0,
	}
	stopJSON, _ := json.Marshal(blockStop)
	anthropicEvents = append(anthropicEvents, string(stopJSON))

	msgStop := map[string]string{"type": "message_stop"}
	stopMsgJSON, _ := json.Marshal(msgStop)
	anthropicEvents = append(anthropicEvents, string(stopMsgJSON))

	return anthropicEvents
}

// TestAnthropicFromSpecificLog_sse_2026_02_28_11_16_22 tests a specific log file
// with 3 bash tool calls for searching library directories
func TestAnthropicFromSpecificLog_sse_2026_02_28_11_16_22(t *testing.T) {
	logFile := "../old_sse_logs/sse_2026-02-28_11-16-22.log"
	data, err := os.ReadFile(logFile)
	if err != nil {
		t.Skipf("Log file not found: %v", err)
	}

	anthropicEvents := convertOpenAIToAnthropic(string(data))
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	for _, event := range anthropicEvents {
		transformer.Transform(&sse.Event{Data: event})
	}

	result := output.String()

	if !strings.Contains(result, "event: message_start") {
		t.Error("Missing message_start")
	}
	if !strings.Contains(result, "event: message_stop") {
		t.Error("Missing message_stop")
	}

	toolUseCount := strings.Count(result, "\"type\":\"tool_use\"")
	if toolUseCount != 3 {
		t.Errorf("Expected 3 tool_use blocks, got %d", toolUseCount)
	}

	if !strings.Contains(result, "bash") {
		t.Error("Expected bash function name")
	}
	if !strings.Contains(result, "ls -la") {
		t.Error("Expected ls command in arguments")
	}
	if !strings.Contains(result, "find /usr/include") {
		t.Error("Expected find command in arguments")
	}
	if !strings.Contains(result, "glob searches returned no results") {
		t.Error("Expected reasoning content before tool calls")
	}

	t.Logf("Successfully converted log with %d tool calls", toolUseCount)
}

// TestAnthropicFromSpecificLog_sse_2026_02_28_11_16_31 tests another log file
func TestAnthropicFromSpecificLog_sse_2026_02_28_11_16_31(t *testing.T) {
	logFile := "../old_sse_logs/sse_2026-02-28_11-16-31.log"
	data, err := os.ReadFile(logFile)
	if err != nil {
		t.Skipf("Log file not found: %v", err)
	}

	anthropicEvents := convertOpenAIToAnthropic(string(data))
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	for _, event := range anthropicEvents {
		transformer.Transform(&sse.Event{Data: event})
	}

	result := output.String()

	if !strings.Contains(result, "event: message_start") {
		t.Error("Missing message_start")
	}

	toolUseCount := strings.Count(result, "\"type\":\"tool_use\"")
	t.Logf("Log file converted with %d tool_use blocks", toolUseCount)
}

// TestAnthropicFromSpecificLog_sse_2026_02_28_11_17_04 tests log with multiple tool types
func TestAnthropicFromSpecificLog_sse_2026_02_28_11_17_04(t *testing.T) {
	logFile := "../old_sse_logs/sse_2026-02-28_11-17-04.log"
	data, err := os.ReadFile(logFile)
	if err != nil {
		t.Skipf("Log file not found: %v", err)
	}

	anthropicEvents := convertOpenAIToAnthropic(string(data))
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	for _, event := range anthropicEvents {
		transformer.Transform(&sse.Event{Data: event})
	}

	result := output.String()

	toolUseCount := strings.Count(result, "\"type\":\"tool_use\"")
	t.Logf("Log file converted with %d tool_use blocks", toolUseCount)

	if !strings.Contains(result, "event: content_block_start") {
		t.Error("Missing content_block_start")
	}
	if !strings.Contains(result, "event: content_block_delta") {
		t.Error("Missing content_block_delta")
	}
	if !strings.Contains(result, "event: content_block_stop") {
		t.Error("Missing content_block_stop")
	}
}

// TestAnthropicFromSpecificLog_sse_2026_02_28_11_17_56 tests another log file
func TestAnthropicFromSpecificLog_sse_2026_02_28_11_17_56(t *testing.T) {
	logFile := "../old_sse_logs/sse_2026-02-28_11-17-56.log"
	data, err := os.ReadFile(logFile)
	if err != nil {
		t.Skipf("Log file not found: %v", err)
	}

	anthropicEvents := convertOpenAIToAnthropic(string(data))
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	for _, event := range anthropicEvents {
		transformer.Transform(&sse.Event{Data: event})
	}

	result := output.String()
	toolUseCount := strings.Count(result, "\"type\":\"tool_use\"")
	t.Logf("Log file converted with %d tool_use blocks", toolUseCount)
}

// TestAnthropicFromSpecificLog_sse_2026_02_28_11_24_39 tests log file from later session
func TestAnthropicFromSpecificLog_sse_2026_02_28_11_24_39(t *testing.T) {
	logFile := "../old_sse_logs/sse_2026-02-28_11-24-39.log"
	data, err := os.ReadFile(logFile)
	if err != nil {
		t.Skipf("Log file not found: %v", err)
	}

	anthropicEvents := convertOpenAIToAnthropic(string(data))
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	for _, event := range anthropicEvents {
		transformer.Transform(&sse.Event{Data: event})
	}

	result := output.String()
	toolUseCount := strings.Count(result, "\"type\":\"tool_use\"")
	t.Logf("Log file converted with %d tool_use blocks", toolUseCount)
}

// TestAnthropicFormat_ToolArgumentsIntegrity verifies that tool arguments
// are preserved correctly during conversion
func TestAnthropicFormat_ToolArgumentsIntegrity(t *testing.T) {
	logFile := "../old_sse_logs/sse_2026-02-28_11-16-22.log"
	data, err := os.ReadFile(logFile)
	if err != nil {
		t.Skipf("Log file not found: %v", err)
	}

	anthropicEvents := convertOpenAIToAnthropic(string(data))
	var output bytes.Buffer
	transformer := NewAnthropicToolCallTransformer(&output)

	for _, event := range anthropicEvents {
		transformer.Transform(&sse.Event{Data: event})
	}

	result := output.String()

	expectedArgs := []string{
		"ls -la /usr/include/",
		"find /usr/include",
		"maxdepth 3",
		"grep -E",
	}

	for _, expected := range expectedArgs {
		if !strings.Contains(result, expected) {
			t.Errorf("Expected argument content not found: %s", expected)
		}
	}

	t.Logf("All %d expected argument patterns found", len(expectedArgs))
}
