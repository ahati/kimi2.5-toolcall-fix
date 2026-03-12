package toolcall

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"ai-proxy/types"
)

func TestOpenAIOutput_ContentBeforeAndAfterToolCall(t *testing.T) {
	var buf bytes.Buffer
	base := types.StreamChunk{
		ID:      "chatcmpl-123",
		Model:   "test-model",
		Created: 1234567890,
		Choices: []types.StreamChoice{{Index: 0}},
	}
	output := NewOpenAIOutput(&buf, base)
	parser := NewParser(DefaultTokenSet(), output)

	parser.Feed("Hello ")
	parser.Feed("<|tool_calls_section_begin|>")
	parser.Feed("<|tool_call_begin|>")
	parser.Feed("bash")
	parser.Feed("<|tool_call_argument_begin|>")
	parser.Feed(`{"cmd":"ls"}`)
	parser.Feed("<|tool_call_end|>")
	parser.Feed("<|tool_calls_section_end|>")
	parser.Feed(" end")
	parser.Flush()

	result := buf.String()
	t.Logf("Output:\n%s", result)

	if !strings.Contains(result, `"content":"Hello "`) {
		t.Error("Missing 'Hello ' content before tool call")
	}
	if !strings.Contains(result, `"content":" end"`) {
		t.Error("Missing ' end' content after tool call")
	}
	if !strings.Contains(result, `"tool_calls"`) {
		t.Error("Missing tool_calls in output")
	}
	if !strings.Contains(result, `"name":"bash"`) {
		t.Error("Missing function name 'bash'")
	}
	if !strings.Contains(result, `"arguments":"{\"cmd\":\"ls\"}"`) {
		t.Error("Missing function arguments")
	}
}

func TestOpenAIOutput_SpacesPreserved(t *testing.T) {
	var buf bytes.Buffer
	base := types.StreamChunk{
		ID:      "chatcmpl-123",
		Model:   "test-model",
		Created: 1234567890,
		Choices: []types.StreamChoice{{Index: 0}},
	}
	output := NewOpenAIOutput(&buf, base)
	parser := NewParser(DefaultTokenSet(), output)

	parser.Feed("Here is the result: ")
	parser.Flush()

	result := buf.String()
	t.Logf("Output:\n%s", result)

	if !strings.Contains(result, `"content":"Here is the result: "`) {
		t.Errorf("Spaces not preserved in output, got: %s", result)
	}
}

func TestOpenAIOutput_MultipleToolCalls(t *testing.T) {
	var buf bytes.Buffer
	base := types.StreamChunk{
		ID:      "chatcmpl-123",
		Model:   "test-model",
		Created: 1234567890,
		Choices: []types.StreamChoice{{Index: 0}},
	}
	output := NewOpenAIOutput(&buf, base)
	parser := NewParser(DefaultTokenSet(), output)

	parser.Feed("<|tool_calls_section_begin|>")
	parser.Feed("<|tool_call_begin|>bash<|tool_call_argument_begin|>{\"cmd\":\"ls\"}<|tool_call_end|>")
	parser.Feed("<|tool_call_begin|>read<|tool_call_argument_begin|>{\"file\":\"test.go\"}<|tool_call_end|>")
	parser.Feed("<|tool_calls_section_end|>")
	parser.Flush()

	result := buf.String()
	t.Logf("Output:\n%s", result)

	if !strings.Contains(result, `"index":0`) {
		t.Error("Missing index 0 for first tool call")
	}
	if !strings.Contains(result, `"index":1`) {
		t.Error("Missing index 1 for second tool call")
	}
	if !strings.Contains(result, `"name":"bash"`) {
		t.Error("Missing bash function name")
	}
	if !strings.Contains(result, `"name":"read"`) {
		t.Error("Missing read function name")
	}
}

func TestOpenAIOutput_ComplexArguments(t *testing.T) {
	var buf bytes.Buffer
	base := types.StreamChunk{
		ID:      "chatcmpl-123",
		Model:   "test-model",
		Created: 1234567890,
		Choices: []types.StreamChoice{{Index: 0}},
	}
	output := NewOpenAIOutput(&buf, base)
	parser := NewParser(DefaultTokenSet(), output)

	complexArgs := `{"nested":{"key":"value"},"array":[1,2,3],"string":"with \"quotes\""}`
	parser.Feed("<|tool_calls_section_begin|>")
	parser.Feed("<|tool_call_begin|>complex<|tool_call_argument_begin|>")
	parser.Feed(complexArgs)
	parser.Feed("<|tool_call_end|>")
	parser.Feed("<|tool_calls_section_end|>")
	parser.Flush()

	result := buf.String()
	t.Logf("Output:\n%s", result)

	if !strings.Contains(result, `"name":"complex"`) {
		t.Error("Missing complex function name")
	}

	lines := strings.Split(result, "\n")
	for _, line := range lines {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		jsonStr := strings.TrimPrefix(line, "data: ")
		if jsonStr == "" {
			continue
		}
		var chunk types.StreamChunk
		if err := json.Unmarshal([]byte(jsonStr), &chunk); err != nil {
			t.Errorf("Invalid JSON: %v", err)
			continue
		}
		if len(chunk.Choices) > 0 && len(chunk.Choices[0].Delta.ToolCalls) > 0 {
			tc := chunk.Choices[0].Delta.ToolCalls[0]
			if tc.Function.Arguments != "" {
				var args map[string]interface{}
				if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
					t.Errorf("Invalid arguments JSON: %v", err)
				}
			}
		}
	}
}

func TestOpenAIOutput_MetadataPreserved(t *testing.T) {
	var buf bytes.Buffer
	base := types.StreamChunk{
		ID:      "chatcmpl-xyz",
		Model:   "gpt-4",
		Created: 1700000000,
		Choices: []types.StreamChoice{{Index: 0}},
	}
	output := NewOpenAIOutput(&buf, base)
	parser := NewParser(DefaultTokenSet(), output)

	parser.Feed("test content")
	parser.Flush()

	result := buf.String()
	t.Logf("Output:\n%s", result)

	if !strings.Contains(result, `"id":"chatcmpl-xyz"`) {
		t.Error("ID not preserved")
	}
	if !strings.Contains(result, `"model":"gpt-4"`) {
		t.Error("Model not preserved")
	}
	if !strings.Contains(result, `"created":1700000000`) {
		t.Error("Created timestamp not preserved")
	}
}
