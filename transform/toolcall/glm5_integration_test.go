package toolcall

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"ai-proxy/types"

	"github.com/tmaxmax/go-sse"
)

// TestGLM5ToolCallsInThinkingContent tests GLM-5 XML tool call extraction from thinking content.
// This verifies that when GLM-5 models emit tool calls inside thinking blocks using XML markup,
// they are correctly extracted and converted to function_call output items.
func TestGLM5ToolCallsInThinkingContent(t *testing.T) {
	tests := []struct {
		name     string
		events   []types.Event
		validate func(t *testing.T, output string)
	}{
		{
			name: "single GLM-5 tool call in thinking block",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_abc",
						Model: "glm-5",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "thinking"}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: `<tool_call>exec_command<arg_key>cmd</arg_key><arg_value>ls -la</arg_value></tool_call>`}),
				},
				{
					Type:  "content_block_stop",
					Index: intPtr(0),
				},
				{
					Type: "message_stop",
				},
			},
			validate: func(t *testing.T, output string) {
				// Should contain function_call output item
				if !strings.Contains(output, `"type":"function_call"`) {
					t.Error("Expected function_call in output")
				}
				// Should contain function name
				if !strings.Contains(output, `"name":"exec_command"`) {
					t.Errorf("Expected function name 'exec_command' in output, got: %s", output)
				}
				// Should contain function_call_arguments.delta for the args
				if !strings.Contains(output, `"type":"response.function_call_arguments.delta"`) {
					t.Error("Expected response.function_call_arguments.delta")
				}
				// Should contain the arguments (JSON-escaped in the output)
				if !strings.Contains(output, `"cmd":"ls -la"`) && !strings.Contains(output, `\"cmd\":\"ls -la\"`) {
					t.Errorf("Expected arguments in output, got: %s", output)
				}
			},
		},
		{
			name: "GLM-5 tool call split across multiple chunks",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_abc",
						Model: "glm-5",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "thinking"}),
				},
				// First chunk: partial opening
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: `<tool_`}),
				},
				// Second chunk: rest of opening and function name
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: `call>search`}),
				},
				// Third chunk: first arg key
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: `<arg_key>query</arg_key>`}),
				},
				// Fourth chunk: first arg value
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: `<arg_value>hello world</arg_value>`}),
				},
				// Fifth chunk: closing tag
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: `</tool_call>`}),
				},
				{
					Type:  "content_block_stop",
					Index: intPtr(0),
				},
				{
					Type: "message_stop",
				},
			},
			validate: func(t *testing.T, output string) {
				// Should contain function_call output item
				if !strings.Contains(output, `"type":"function_call"`) {
					t.Error("Expected function_call in output")
				}
				// Should contain function name
				if !strings.Contains(output, `"name":"search"`) {
					t.Errorf("Expected function name 'search' in output, got: %s", output)
				}
				// Should contain the arguments (may be JSON-escaped in output)
				if !strings.Contains(output, `"query":"hello world"`) && !strings.Contains(output, `\"query\":\"hello world\"`) {
					t.Error("Expected arguments in output")
				}
			},
		},
		{
			name: "GLM-5 tool call with prefix text",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_abc",
						Model: "glm-5",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "thinking"}),
				},
				// Prefix text + tool call
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: `I'll run a search now. <tool_call>search<arg_key>q</arg_key><arg_value>test</arg_value></tool_call>`}),
				},
				{
					Type:  "content_block_stop",
					Index: intPtr(0),
				},
				{
					Type: "message_stop",
				},
			},
			validate: func(t *testing.T, output string) {
				// Should contain reasoning with prefix text
				if !strings.Contains(output, `"delta":"I'll run a search now. "`) {
					t.Error("Expected reasoning prefix in output")
				}
				// Should contain function_call
				if !strings.Contains(output, `"type":"function_call"`) {
					t.Error("Expected function_call in output")
				}
				// Should contain function name
				if !strings.Contains(output, `"name":"search"`) {
					t.Error("Expected function name 'search' in output")
				}
			},
		},
		{
			name: "multiple GLM-5 tool calls in thinking",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_multi",
						Model: "glm-5",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "thinking"}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: `<tool_call>read<arg_key>file</arg_key><arg_value>a.txt</arg_value></tool_call> and <tool_call>write<arg_key>file</arg_key><arg_value>b.txt</arg_value></tool_call>`}),
				},
				{
					Type:  "content_block_stop",
					Index: intPtr(0),
				},
				{
					Type: "message_stop",
				},
			},
			validate: func(t *testing.T, output string) {
				// Should contain both function calls
				if !strings.Contains(output, `"name":"read"`) {
					t.Error("Expected function name 'read' in output")
				}
				if !strings.Contains(output, `"name":"write"`) {
					t.Error("Expected function name 'write' in output")
				}
				// Should contain middle text
				if !strings.Contains(output, `"delta":" and "`) {
					t.Error("Expected middle text in output")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			transformer := NewResponsesTransformer(&buf)
			// Enable GLM-5 tool call extraction for these tests
			transformer.SetGLM5ToolCallTransform(true)

			for _, e := range tt.events {
				data, _ := json.Marshal(e)
				err := transformer.Transform(&sse.Event{Data: string(data)})
				if err != nil {
					t.Errorf("Transform returned error: %v", err)
				}
			}

			output := buf.String()
			t.Logf("Output:\n%s", output)

			if tt.validate != nil {
				tt.validate(t, output)
			}
		})
	}
}

// TestGLM5ChunkedDetectionEdgeCase reproduces the issue where <tool_call> is split across chunks
// and the initial chunks don't contain the full tag, causing detection to fail.
func TestGLM5ChunkedDetectionEdgeCase(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesTransformer(&buf)
	transformer.SetGLM5ToolCallTransform(true)

	events := []types.Event{
		{
			Type: "message_start",
			Message: &types.MessageInfo{
				ID:    "msg_test123",
				Model: "glm-5",
			},
		},
		{
			Type:         "content_block_start",
			Index:        intPtr(0),
			ContentBlock: mustMarshal(types.ContentBlock{Type: "thinking"}),
		},
		// CRITICAL: These chunks split the <tool_call> tag
		// Chunk 1: "<tool" - does NOT contain "<tool_call>"
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "<tool"}),
		},
		// Chunk 2: "_call>exec" - still does NOT contain "<tool_call>" alone
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "_call>exec"}),
		},
		// Chunk 3: "_command<arg_key>cmd</arg_key><arg_value>ls</arg_value></tool_call>"
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "_command<arg_key>cmd</arg_key><arg_value>ls</arg_value></tool_call>"}),
		},
		{
			Type:  "content_block_stop",
			Index: intPtr(0),
		},
		{
			Type: "message_stop",
		},
	}

	for _, event := range events {
		data, _ := json.Marshal(event)
		if err := transformer.Transform(&sse.Event{Data: string(data)}); err != nil {
			t.Fatalf("Transform error: %v", err)
		}
	}

	output := buf.String()
	t.Logf("Output:\n%s", output)

	// The bug: if detection requires "<tool_call>" in the chunk, the first two chunks
	// won't trigger GLM-5 parsing, and the third chunk will be too late.
	// The result would be the tool call appearing as regular reasoning text.

	// Check if tool call was properly extracted
	if !strings.Contains(output, `"type":"function_call"`) {
		t.Error("BUG: GLM-5 tool call was not extracted - likely because <tool_call> tag was split across chunks")
	}
	if !strings.Contains(output, `"name":"exec_command"`) {
		t.Error("BUG: function name was not extracted properly")
	}
}

// TestGLM5ToolCallsInAnthropicFormat tests GLM-5 XML tool call extraction when using
// the Anthropic Messages API path. Verifies that GLM-5 tool calls are converted to
// proper Anthropic tool_use content blocks.
func TestGLM5ToolCallsInAnthropicFormat(t *testing.T) {
	tests := []struct {
		name     string
		events   []types.Event
		validate func(t *testing.T, output string)
	}{
		{
			name: "single GLM-5 tool call in thinking block - Anthropic format",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_abc",
						Model: "glm-5",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "thinking"}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: `<tool_call>exec_command<arg_key>cmd</arg_key><arg_value>ls -la</arg_value></tool_call>`}),
				},
				{
					Type:  "content_block_stop",
					Index: intPtr(0),
				},
				{
					Type: "message_stop",
				},
			},
			validate: func(t *testing.T, output string) {
				// Should contain tool_use block
				if !strings.Contains(output, `"type":"tool_use"`) {
					t.Error("Expected tool_use in output")
				}
				// Should contain function name
				if !strings.Contains(output, `"name":"exec_command"`) {
					t.Errorf("Expected function name 'exec_command' in output, got: %s", output)
				}
				// Should contain input_json_delta for the args
				if !strings.Contains(output, `"type":"input_json_delta"`) {
					t.Error("Expected input_json_delta")
				}
				// Should contain the arguments
				if !strings.Contains(output, `"cmd":"ls -la"`) && !strings.Contains(output, `\"cmd\":\"ls -la\"`) {
					t.Errorf("Expected arguments in output, got: %s", output)
				}
			},
		},
		{
			name: "GLM-5 tool call split across multiple chunks - Anthropic format",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_abc",
						Model: "glm-5",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "thinking"}),
				},
				// First chunk: partial opening
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: `<tool_`}),
				},
				// Second chunk: rest of opening and function name
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: `call>search`}),
				},
				// Third chunk: first arg key
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: `<arg_key>query</arg_key>`}),
				},
				// Fourth chunk: first arg value
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: `<arg_value>hello world</arg_value>`}),
				},
				// Fifth chunk: closing tag
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: `</tool_call>`}),
				},
				{
					Type:  "content_block_stop",
					Index: intPtr(0),
				},
				{
					Type: "message_stop",
				},
			},
			validate: func(t *testing.T, output string) {
				// Should contain tool_use block
				if !strings.Contains(output, `"type":"tool_use"`) {
					t.Error("Expected tool_use in output")
				}
				// Should contain function name
				if !strings.Contains(output, `"name":"search"`) {
					t.Errorf("Expected function name 'search' in output, got: %s", output)
				}
				// Should contain the arguments
				if !strings.Contains(output, `"query":"hello world"`) && !strings.Contains(output, `\"query\":\"hello world\"`) {
					t.Error("Expected arguments in output")
				}
			},
		},
		{
			name: "multiple GLM-5 tool calls in thinking - Anthropic format",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_multi",
						Model: "glm-5",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "thinking"}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: `<tool_call>read<arg_key>file</arg_key><arg_value>a.txt</arg_value></tool_call> and <tool_call>write<arg_key>file</arg_key><arg_value>b.txt</arg_value></tool_call>`}),
				},
				{
					Type:  "content_block_stop",
					Index: intPtr(0),
				},
				{
					Type: "message_stop",
				},
			},
			validate: func(t *testing.T, output string) {
				// Should contain both tool_use blocks
				if !strings.Contains(output, `"name":"read"`) {
					t.Error("Expected function name 'read' in output")
				}
				if !strings.Contains(output, `"name":"write"`) {
					t.Error("Expected function name 'write' in output")
				}
				// Should contain middle text as thinking
				if !strings.Contains(output, `"thinking":" and "`) {
					t.Error("Expected middle text as thinking in output")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			transformer := NewAnthropicTransformer(&buf)
			// Enable GLM-5 tool call transformation for these tests
			transformer.SetGLM5ToolCallTransform(true)

			for _, e := range tt.events {
				data, _ := json.Marshal(e)
				err := transformer.Transform(&sse.Event{Data: string(data)})
				if err != nil {
					t.Errorf("Transform returned error: %v", err)
				}
			}

			output := buf.String()
			t.Logf("Output:\n%s", output)

			if tt.validate != nil {
				tt.validate(t, output)
			}
		})
	}
}
