package toolcall

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"ai-proxy/types"

	"github.com/tmaxmax/go-sse"
)

// TestResponsesFormatter_FormatResponseCreated tests response.created event formatting.
func TestResponsesFormatter_FormatResponseCreated(t *testing.T) {
	formatter := NewResponsesFormatter("resp_123", "gpt-4o")

	result := formatter.FormatResponseCreated(1)
	resultStr := string(result)

	if !strings.HasPrefix(resultStr, "data: ") {
		t.Error("Result should start with 'data: '")
	}

	if !strings.Contains(resultStr, `"type":"response.created"`) {
		t.Error("Result should contain response.created type")
	}

	if !strings.Contains(resultStr, `"id":"resp_123"`) {
		t.Error("Result should contain response ID")
	}

	if !strings.Contains(resultStr, `"model":"gpt-4o"`) {
		t.Error("Result should contain model")
	}

	if !strings.Contains(resultStr, `"sequence_number":1`) {
		t.Error("Result should contain sequence_number")
	}
}

// TestResponsesFormatter_FormatContentPartAdded tests content part added event.
func TestResponsesFormatter_FormatContentPartAdded(t *testing.T) {
	formatter := NewResponsesFormatter("resp_123", "gpt-4o")
	formatter.SetResponseID("resp_123")

	result := formatter.FormatContentPartAdded("msg_123", 0, "output_text", 1, 2)
	resultStr := string(result)

	if !strings.Contains(resultStr, `"type":"response.content_part.added"`) {
		t.Error("Result should contain response.content_part.added type")
	}

	if !strings.Contains(resultStr, `"content_index":0`) {
		t.Error("Result should contain content_index")
	}

	if !strings.Contains(resultStr, `"type":"output_text"`) {
		t.Error("Result should contain output_text type")
	}

	if !strings.Contains(resultStr, `"output_index":1`) {
		t.Error("Result should contain output_index")
	}

	if !strings.Contains(resultStr, `"sequence_number":2`) {
		t.Error("Result should contain sequence_number")
	}
}

// TestResponsesFormatter_FormatOutputTextDelta tests text delta formatting.
func TestResponsesFormatter_FormatOutputTextDelta(t *testing.T) {
	formatter := NewResponsesFormatter("resp_123", "gpt-4o")
	formatter.SetResponseID("resp_123")

	result := formatter.FormatOutputTextDelta("msg_123", 0, "Hello world", 1, 2)
	resultStr := string(result)

	if !strings.Contains(resultStr, `"type":"response.output_text.delta"`) {
		t.Error("Result should contain response.output_text.delta type")
	}

	if !strings.Contains(resultStr, `"delta":"Hello world"`) {
		t.Error("Result should contain delta with text")
	}

	if !strings.Contains(resultStr, `"output_index":1`) {
		t.Error("Result should contain output_index")
	}

	if !strings.Contains(resultStr, `"sequence_number":2`) {
		t.Error("Result should contain sequence_number")
	}
}

// TestResponsesFormatter_FormatFunctionCallItemAdded tests function call item added event.
func TestResponsesFormatter_FormatFunctionCallItemAdded(t *testing.T) {
	formatter := NewResponsesFormatter("resp_123", "gpt-4o")
	formatter.SetResponseID("resp_123")

	result := formatter.FormatFunctionCallItemAdded("toolu_abc", "get_weather", 1, 2)
	resultStr := string(result)

	if !strings.Contains(resultStr, `"type":"response.output_item.added"`) {
		t.Error("Result should contain response.output_item.added type")
	}

	if !strings.Contains(resultStr, `"type":"function_call"`) {
		t.Error("Result should contain function_call item type")
	}

	if !strings.Contains(resultStr, `"id":"toolu_abc"`) {
		t.Error("Result should contain call ID")
	}

	if !strings.Contains(resultStr, `"name":"get_weather"`) {
		t.Error("Result should contain function name")
	}

	if !strings.Contains(resultStr, `"output_index":1`) {
		t.Error("Result should contain output_index")
	}

	if !strings.Contains(resultStr, `"sequence_number":2`) {
		t.Error("Result should contain sequence_number")
	}
}

// TestResponsesFormatter_FormatFunctionCallArgsDelta tests function args delta.
func TestResponsesFormatter_FormatFunctionCallArgsDelta(t *testing.T) {
	formatter := NewResponsesFormatter("resp_123", "gpt-4o")
	formatter.SetResponseID("resp_123")

	result := formatter.FormatFunctionCallArgsDelta("toolu_abc", "toolu_abc", `{"locat`, 1, 2)
	resultStr := string(result)

	if !strings.Contains(resultStr, `"type":"response.function_call_arguments.delta"`) {
		t.Error("Result should contain response.function_call_arguments.delta type")
	}

	if !strings.Contains(resultStr, `"call_id":"toolu_abc"`) {
		t.Error("Result should contain call_id")
	}

	// The delta is JSON-escaped in the output
	if !strings.Contains(resultStr, `"delta":"{\"locat"`) {
		t.Errorf("Result should contain escaped delta - got: %s", resultStr)
	}

	if !strings.Contains(resultStr, `"output_index":1`) {
		t.Error("Result should contain output_index")
	}

	if !strings.Contains(resultStr, `"sequence_number":2`) {
		t.Error("Result should contain sequence_number")
	}
}

// TestResponsesFormatter_FormatContentPartDone tests content part done event.
func TestResponsesFormatter_FormatContentPartDone(t *testing.T) {
	formatter := NewResponsesFormatter("resp_123", "gpt-4o")
	formatter.SetResponseID("resp_123")

	// Test with output_text type
	result := formatter.FormatContentPartDone("msg_123", 0, "output_text", "Hello world", 1, 2)
	resultStr := string(result)

	if !strings.Contains(resultStr, `"type":"response.content_part.done"`) {
		t.Error("Result should contain response.content_part.done type")
	}

	if !strings.Contains(resultStr, `"text":"Hello world"`) {
		t.Error("Result should contain text content")
	}

	if !strings.Contains(resultStr, `"output_index":1`) {
		t.Error("Result should contain output_index")
	}

	if !strings.Contains(resultStr, `"sequence_number":2`) {
		t.Error("Result should contain sequence_number")
	}

	// Test with other type (no content)
	result2 := formatter.FormatContentPartDone("msg_123", 1, "function_call", "", 1, 3)
	resultStr2 := string(result2)

	if !strings.Contains(resultStr2, `"type":"function_call"`) {
		t.Error("Result should contain function_call type")
	}
}

// TestResponsesFormatter_FormatOutputItemDone tests output item done event.
func TestResponsesFormatter_FormatOutputItemDone(t *testing.T) {
	formatter := NewResponsesFormatter("resp_123", "gpt-4o")
	formatter.SetResponseID("resp_123")

	item := map[string]interface{}{
		"type":   "message",
		"id":     "msg_456",
		"status": "completed",
		"role":   "assistant",
		"content": []map[string]interface{}{{
			"type": "output_text",
			"text": "Hello",
		}},
	}

	result := formatter.FormatOutputItemDone("msg_456", item, 0, 1)
	resultStr := string(result)

	if !strings.Contains(resultStr, `"type":"response.output_item.done"`) {
		t.Error("Result should contain response.output_item.done type")
	}

	if !strings.Contains(resultStr, `"id":"msg_456"`) {
		t.Error("Result should contain item ID")
	}

	if !strings.Contains(resultStr, `"status":"completed"`) {
		t.Error("Result should contain completed status")
	}

	if !strings.Contains(resultStr, `"sequence_number":1`) {
		t.Error("Result should contain sequence_number")
	}
}

// TestResponsesFormatter_FormatResponseCompleted tests response completed event.
func TestResponsesFormatter_FormatResponseCompleted(t *testing.T) {
	formatter := NewResponsesFormatter("resp_123", "gpt-4o")
	formatter.SetResponseID("resp_123")
	formatter.SetModel("gpt-4o")

	outputItems := []map[string]interface{}{{
		"type":   "message",
		"id":     "msg_456",
		"status": "completed",
		"role":   "assistant",
		"content": []map[string]interface{}{{
			"type": "output_text",
			"text": "Hello",
		}},
	}}

	result := formatter.FormatResponseCompleted(outputItems, nil, 1)
	resultStr := string(result)

	if !strings.Contains(resultStr, `"type":"response.completed"`) {
		t.Error("Result should contain response.completed type")
	}

	if !strings.Contains(resultStr, `"status":"completed"`) {
		t.Error("Result should contain completed status")
	}

	if !strings.Contains(resultStr, `"sequence_number":1`) {
		t.Error("Result should contain sequence_number")
	}
}

// TestResponsesFormatter_FormatReasoningItemAdded tests reasoning item added event.
func TestResponsesFormatter_FormatReasoningItemAdded(t *testing.T) {
	formatter := NewResponsesFormatter("resp_123", "gpt-4o")

	result := formatter.FormatReasoningItemAdded("rs_abc", 0, 1)
	resultStr := string(result)

	if !strings.Contains(resultStr, `"type":"response.output_item.added"`) {
		t.Error("Result should contain response.output_item.added type")
	}

	if !strings.Contains(resultStr, `"type":"reasoning"`) {
		t.Error("Result should contain reasoning item type")
	}

	if !strings.Contains(resultStr, `"id":"rs_abc"`) {
		t.Error("Result should contain reasoning ID")
	}

	if !strings.Contains(resultStr, `"output_index":0`) {
		t.Error("Result should contain output_index")
	}

	if !strings.Contains(resultStr, `"sequence_number":1`) {
		t.Error("Result should contain sequence_number")
	}
}

// TestResponsesFormatter_FormatReasoningSummaryPartAdded tests reasoning summary part added event.
func TestResponsesFormatter_FormatReasoningSummaryPartAdded(t *testing.T) {
	formatter := NewResponsesFormatter("resp_123", "gpt-4o")

	result := formatter.FormatReasoningSummaryPartAdded("rs_abc", 0, 0, 1)
	resultStr := string(result)

	if !strings.Contains(resultStr, `"type":"response.reasoning_summary_part.added"`) {
		t.Error("Result should contain response.reasoning_summary_part.added type")
	}

	if !strings.Contains(resultStr, `"item_id":"rs_abc"`) {
		t.Error("Result should contain item_id")
	}

	if !strings.Contains(resultStr, `"output_index":0`) {
		t.Error("Result should contain output_index")
	}

	if !strings.Contains(resultStr, `"summary_index":0`) {
		t.Error("Result should contain summary_index")
	}

	if !strings.Contains(resultStr, `"sequence_number":1`) {
		t.Error("Result should contain sequence_number")
	}
}

// TestResponsesFormatter_FormatReasoningSummaryDelta tests reasoning summary delta.
func TestResponsesFormatter_FormatReasoningSummaryDelta(t *testing.T) {
	formatter := NewResponsesFormatter("resp_123", "gpt-4o")

	result := formatter.FormatReasoningSummaryDelta("rs_abc", "Analyzing...", 0, 0, 1)
	resultStr := string(result)

	if !strings.Contains(resultStr, `"type":"response.reasoning_summary_text.delta"`) {
		t.Error("Result should contain response.reasoning_summary_text.delta type")
	}

	if !strings.Contains(resultStr, `"delta":"Analyzing..."`) {
		t.Error("Result should contain delta")
	}

	if !strings.Contains(resultStr, `"output_index":0`) {
		t.Error("Result should contain output_index")
	}

	if !strings.Contains(resultStr, `"summary_index":0`) {
		t.Error("Result should contain summary_index")
	}

	if !strings.Contains(resultStr, `"sequence_number":1`) {
		t.Error("Result should contain sequence_number")
	}
}

// TestResponsesFormatter_FormatReasoningSummaryTextDone tests reasoning summary text done event.
func TestResponsesFormatter_FormatReasoningSummaryTextDone(t *testing.T) {
	formatter := NewResponsesFormatter("resp_123", "gpt-4o")

	result := formatter.FormatReasoningSummaryTextDone("rs_abc", "Full reasoning text", 0, 0, 1)
	resultStr := string(result)

	if !strings.Contains(resultStr, `"type":"response.reasoning_summary_text.done"`) {
		t.Error("Result should contain response.reasoning_summary_text.done type")
	}

	if !strings.Contains(resultStr, `"text":"Full reasoning text"`) {
		t.Error("Result should contain text")
	}

	if !strings.Contains(resultStr, `"output_index":0`) {
		t.Error("Result should contain output_index")
	}

	if !strings.Contains(resultStr, `"summary_index":0`) {
		t.Error("Result should contain summary_index")
	}

	if !strings.Contains(resultStr, `"sequence_number":1`) {
		t.Error("Result should contain sequence_number")
	}
}

// TestResponsesFormatter_FormatReasoningSummaryPartDone tests reasoning summary part done event.
func TestResponsesFormatter_FormatReasoningSummaryPartDone(t *testing.T) {
	formatter := NewResponsesFormatter("resp_123", "gpt-4o")

	result := formatter.FormatReasoningSummaryPartDone("rs_abc", "Full reasoning text", 0, 0, 1)
	resultStr := string(result)

	if !strings.Contains(resultStr, `"type":"response.reasoning_summary_part.done"`) {
		t.Error("Result should contain response.reasoning_summary_part.done type")
	}

	if !strings.Contains(resultStr, `"type":"summary_text"`) {
		t.Error("Result should contain summary_text type")
	}

	if !strings.Contains(resultStr, `"text":"Full reasoning text"`) {
		t.Error("Result should contain summary text")
	}

	if !strings.Contains(resultStr, `"output_index":0`) {
		t.Error("Result should contain output_index")
	}

	if !strings.Contains(resultStr, `"summary_index":0`) {
		t.Error("Result should contain summary_index")
	}

	if !strings.Contains(resultStr, `"sequence_number":1`) {
		t.Error("Result should contain sequence_number")
	}
}

// TestResponsesFormatter_FormatReasoningItemDone tests reasoning item done event.
func TestResponsesFormatter_FormatReasoningItemDone(t *testing.T) {
	formatter := NewResponsesFormatter("resp_123", "gpt-4o")

	result := formatter.FormatReasoningItemDone("rs_abc", "Full reasoning text", 0, 1)
	resultStr := string(result)

	if !strings.Contains(resultStr, `"type":"response.output_item.done"`) {
		t.Error("Result should contain response.output_item.done type")
	}

	if !strings.Contains(resultStr, `"type":"summary_text"`) {
		t.Error("Result should contain summary_text type")
	}

	if !strings.Contains(resultStr, `"text":"Full reasoning text"`) {
		t.Error("Result should contain summary text")
	}

	if !strings.Contains(resultStr, `"output_index":0`) {
		t.Error("Result should contain output_index")
	}

	if !strings.Contains(resultStr, `"sequence_number":1`) {
		t.Error("Result should contain sequence_number")
	}
}

// TestNewResponsesTransformer tests transformer creation.
func TestNewResponsesTransformer(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesTransformer(&buf)

	if transformer == nil {
		t.Fatal("NewResponsesTransformer returned nil")
	}

	if transformer.sseWriter == nil {
		t.Error("Transformer sseWriter should not be nil")
	}

	if transformer.formatter == nil {
		t.Error("Transformer formatter should not be nil")
	}

	if transformer.parser == nil {
		t.Error("Transformer parser should not be nil")
	}
}

// TestResponsesTransformer_Transform_EmptyData tests transform with empty data.
func TestResponsesTransformer_Transform_EmptyData(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesTransformer(&buf)

	event := &sse.Event{Data: ""}
	err := transformer.Transform(event)

	if err != nil {
		t.Errorf("Transform returned error: %v", err)
	}

	if buf.Len() != 0 {
		t.Error("Buffer should be empty for empty event data")
	}
}

// TestResponsesTransformer_Transform_Done tests transform with [DONE].
func TestResponsesTransformer_Transform_Done(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesTransformer(&buf)

	event := &sse.Event{Data: "[DONE]"}
	err := transformer.Transform(event)

	if err != nil {
		t.Errorf("Transform returned error: %v", err)
	}

	result := buf.String()
	if !strings.Contains(result, "data: [DONE]") {
		t.Error("Result should contain data: [DONE]")
	}
}

// TestResponsesTransformer_Transform_InvalidJSON tests transform with invalid JSON.
func TestResponsesTransformer_Transform_InvalidJSON(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesTransformer(&buf)

	event := &sse.Event{Data: "not valid json"}
	err := transformer.Transform(event)

	if err != nil {
		t.Errorf("Transform returned error: %v", err)
	}

	result := buf.String()
	if !strings.Contains(result, "data: not valid json") {
		t.Error("Result should pass through invalid JSON")
	}
}

// TestResponsesTransformer_HandleMessageStart tests message_start handling.
func TestResponsesTransformer_HandleMessageStart(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesTransformer(&buf)

	anthropicEvent := types.Event{
		Type: "message_start",
		Message: &types.MessageInfo{
			ID:    "msg_abc123",
			Type:  "message",
			Role:  "assistant",
			Model: "claude-3-opus",
		},
	}

	data, _ := json.Marshal(anthropicEvent)
	event := &sse.Event{Data: string(data)}
	err := transformer.Transform(event)

	if err != nil {
		t.Errorf("Transform returned error: %v", err)
	}

	result := buf.String()
	if !strings.Contains(result, `"type":"response.created"`) {
		t.Error("Result should contain response.created event")
	}

	if !strings.Contains(result, `"id":"resp_abc123"`) {
		t.Error("Result should contain converted response ID")
	}

	if !strings.Contains(result, `"model":"claude-3-opus"`) {
		t.Error("Result should contain model")
	}
}

// TestResponsesTransformer_HandleContentBlockStart_Text tests text block start.
func TestResponsesTransformer_HandleContentBlockStart_Text(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesTransformer(&buf)

	// First send message_start
	msgStart := types.Event{
		Type: "message_start",
		Message: &types.MessageInfo{
			ID:    "msg_abc",
			Model: "claude-3",
		},
	}
	data, _ := json.Marshal(msgStart)
	transformer.Transform(&sse.Event{Data: string(data)})
	buf.Reset()

	// Then send content_block_start for text
	contentBlock := types.ContentBlock{Type: "text"}
	blockData, _ := json.Marshal(contentBlock)
	blockStart := types.Event{
		Type:         "content_block_start",
		Index:        intPtr(0),
		ContentBlock: blockData,
	}
	data, _ = json.Marshal(blockStart)
	err := transformer.Transform(&sse.Event{Data: string(data)})

	if err != nil {
		t.Errorf("Transform returned error: %v", err)
	}

	result := buf.String()
	if !strings.Contains(result, `"type":"response.content_part.added"`) {
		t.Error("Result should contain content_part.added event")
	}

	if !strings.Contains(result, `"type":"output_text"`) {
		t.Error("Result should contain output_text type")
	}
}

// TestResponsesTransformer_HandleContentBlockStart_Thinking tests thinking block start.
func TestResponsesTransformer_HandleContentBlockStart_Thinking(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesTransformer(&buf)

	// First send message_start
	msgStart := types.Event{
		Type: "message_start",
		Message: &types.MessageInfo{
			ID:    "msg_abc",
			Model: "claude-3",
		},
	}
	data, _ := json.Marshal(msgStart)
	transformer.Transform(&sse.Event{Data: string(data)})
	buf.Reset()

	// Then send content_block_start for thinking
	contentBlock := types.ContentBlock{Type: "thinking"}
	blockData, _ := json.Marshal(contentBlock)
	blockStart := types.Event{
		Type:         "content_block_start",
		Index:        intPtr(0),
		ContentBlock: blockData,
	}
	data, _ = json.Marshal(blockStart)
	err := transformer.Transform(&sse.Event{Data: string(data)})

	if err != nil {
		t.Errorf("Transform returned error: %v", err)
	}

	result := buf.String()
	// Should contain response.output_item.added for reasoning
	if !strings.Contains(result, `"type":"response.output_item.added"`) {
		t.Error("Result should contain response.output_item.added type")
	}

	if !strings.Contains(result, `"type":"reasoning"`) {
		t.Error("Result should contain reasoning item type")
	}

	if !strings.Contains(result, `"id":"rs_abc"`) {
		t.Error("Result should contain reasoning ID")
	}

	// Should also contain response.reasoning_summary_part.added
	if !strings.Contains(result, `"type":"response.reasoning_summary_part.added"`) {
		t.Error("Result should contain response.reasoning_summary_part.added type")
	}
}

// TestResponsesTransformer_HandleContentBlockStart_ToolUse tests tool_use block start.
func TestResponsesTransformer_HandleContentBlockStart_ToolUse(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesTransformer(&buf)

	// First send message_start
	msgStart := types.Event{
		Type: "message_start",
		Message: &types.MessageInfo{
			ID:    "msg_abc",
			Model: "claude-3",
		},
	}
	data, _ := json.Marshal(msgStart)
	transformer.Transform(&sse.Event{Data: string(data)})
	buf.Reset()

	// Then send content_block_start for tool_use
	contentBlock := types.ContentBlock{
		Type: "tool_use",
		ID:   "toolu_123",
		Name: "get_weather",
	}
	blockData, _ := json.Marshal(contentBlock)
	blockStart := types.Event{
		Type:         "content_block_start",
		Index:        intPtr(0),
		ContentBlock: blockData,
	}
	data, _ = json.Marshal(blockStart)
	err := transformer.Transform(&sse.Event{Data: string(data)})

	if err != nil {
		t.Errorf("Transform returned error: %v", err)
	}

	result := buf.String()
	if !strings.Contains(result, `"type":"function_call"`) {
		t.Error("Result should contain function_call type")
	}

	if !strings.Contains(result, `"id":"toolu_123"`) {
		t.Error("Result should contain tool ID")
	}

	if !strings.Contains(result, `"name":"get_weather"`) {
		t.Error("Result should contain function name")
	}
}

// TestResponsesTransformer_HandleContentBlockDelta_Text tests text delta.
func TestResponsesTransformer_HandleContentBlockDelta_Text(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesTransformer(&buf)

	// Setup
	msgStart := types.Event{
		Type: "message_start",
		Message: &types.MessageInfo{
			ID:    "msg_abc",
			Model: "claude-3",
		},
	}
	data, _ := json.Marshal(msgStart)
	transformer.Transform(&sse.Event{Data: string(data)})

	contentBlock := types.ContentBlock{Type: "text"}
	blockData, _ := json.Marshal(contentBlock)
	blockStart := types.Event{
		Type:         "content_block_start",
		Index:        intPtr(0),
		ContentBlock: blockData,
	}
	data, _ = json.Marshal(blockStart)
	transformer.Transform(&sse.Event{Data: string(data)})
	buf.Reset()

	// Send text delta
	delta := types.TextDelta{Type: "text_delta", Text: "Hello"}
	deltaData, _ := json.Marshal(delta)
	blockDelta := types.Event{
		Type:  "content_block_delta",
		Index: intPtr(0),
		Delta: deltaData,
	}
	data, _ = json.Marshal(blockDelta)
	err := transformer.Transform(&sse.Event{Data: string(data)})

	if err != nil {
		t.Errorf("Transform returned error: %v", err)
	}

	result := buf.String()
	if !strings.Contains(result, `"type":"response.output_text.delta"`) {
		t.Error("Result should contain output_text.delta event")
	}

	if !strings.Contains(result, `"delta":"Hello"`) {
		t.Error("Result should contain delta text")
	}
}

// TestResponsesTransformer_HandleContentBlockDelta_Thinking tests thinking delta.
func TestResponsesTransformer_HandleContentBlockDelta_Thinking(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesTransformer(&buf)

	// Setup
	msgStart := types.Event{
		Type: "message_start",
		Message: &types.MessageInfo{
			ID:    "msg_abc",
			Model: "claude-3",
		},
	}
	data, _ := json.Marshal(msgStart)
	transformer.Transform(&sse.Event{Data: string(data)})

	contentBlock := types.ContentBlock{Type: "thinking"}
	blockData, _ := json.Marshal(contentBlock)
	blockStart := types.Event{
		Type:         "content_block_start",
		Index:        intPtr(0),
		ContentBlock: blockData,
	}
	data, _ = json.Marshal(blockStart)
	transformer.Transform(&sse.Event{Data: string(data)})
	buf.Reset()

	// Send thinking delta
	delta := types.ThinkingDelta{Type: "thinking_delta", Thinking: "Analyzing..."}
	deltaData, _ := json.Marshal(delta)
	blockDelta := types.Event{
		Type:  "content_block_delta",
		Index: intPtr(0),
		Delta: deltaData,
	}
	data, _ = json.Marshal(blockDelta)
	err := transformer.Transform(&sse.Event{Data: string(data)})

	if err != nil {
		t.Errorf("Transform returned error: %v", err)
	}

	result := buf.String()
	if !strings.Contains(result, `"type":"response.reasoning_summary_text.delta"`) {
		t.Error("Result should contain response.reasoning_summary_text.delta type")
	}

	if !strings.Contains(result, `"delta":"Analyzing..."`) {
		t.Error("Result should contain thinking delta")
	}

	// Check for required fields
	if !strings.Contains(result, `"output_index":0`) {
		t.Error("Result should contain output_index")
	}

	if !strings.Contains(result, `"summary_index":0`) {
		t.Error("Result should contain summary_index")
	}

	if !strings.Contains(result, `"sequence_number"`) {
		t.Error("Result should contain sequence_number")
	}
}

// TestResponsesTransformer_HandleContentBlockDelta_ToolInput tests tool input delta.
func TestResponsesTransformer_HandleContentBlockDelta_ToolInput(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesTransformer(&buf)

	// Setup
	msgStart := types.Event{
		Type: "message_start",
		Message: &types.MessageInfo{
			ID:    "msg_abc",
			Model: "claude-3",
		},
	}
	data, _ := json.Marshal(msgStart)
	transformer.Transform(&sse.Event{Data: string(data)})

	contentBlock := types.ContentBlock{
		Type: "tool_use",
		ID:   "toolu_123",
		Name: "get_weather",
	}
	blockData, _ := json.Marshal(contentBlock)
	blockStart := types.Event{
		Type:         "content_block_start",
		Index:        intPtr(0),
		ContentBlock: blockData,
	}
	data, _ = json.Marshal(blockStart)
	transformer.Transform(&sse.Event{Data: string(data)})
	buf.Reset()

	// Send input_json delta
	delta := types.InputJSONDelta{Type: "input_json_delta", PartialJSON: `{"loc`}
	deltaData, _ := json.Marshal(delta)
	blockDelta := types.Event{
		Type:  "content_block_delta",
		Index: intPtr(0),
		Delta: deltaData,
	}
	data, _ = json.Marshal(blockDelta)
	err := transformer.Transform(&sse.Event{Data: string(data)})

	if err != nil {
		t.Errorf("Transform returned error: %v", err)
	}

	result := buf.String()
	if !strings.Contains(result, `"type":"response.function_call_arguments.delta"`) {
		t.Error("Result should contain function_call_arguments.delta event")
	}
}

// TestResponsesTransformer_HandleContentBlockStop tests block stop handling.
func TestResponsesTransformer_HandleContentBlockStop(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesTransformer(&buf)

	// Setup
	msgStart := types.Event{
		Type: "message_start",
		Message: &types.MessageInfo{
			ID:    "msg_abc",
			Model: "claude-3",
		},
	}
	data, _ := json.Marshal(msgStart)
	transformer.Transform(&sse.Event{Data: string(data)})

	contentBlock := types.ContentBlock{Type: "text"}
	blockData, _ := json.Marshal(contentBlock)
	blockStart := types.Event{
		Type:         "content_block_start",
		Index:        intPtr(0),
		ContentBlock: blockData,
	}
	data, _ = json.Marshal(blockStart)
	transformer.Transform(&sse.Event{Data: string(data)})

	delta := types.TextDelta{Type: "text_delta", Text: "Hello world"}
	deltaData, _ := json.Marshal(delta)
	blockDelta := types.Event{
		Type:  "content_block_delta",
		Index: intPtr(0),
		Delta: deltaData,
	}
	data, _ = json.Marshal(blockDelta)
	transformer.Transform(&sse.Event{Data: string(data)})
	buf.Reset()

	// Send block stop
	blockStop := types.Event{
		Type:  "content_block_stop",
		Index: intPtr(0),
	}
	data, _ = json.Marshal(blockStop)
	err := transformer.Transform(&sse.Event{Data: string(data)})

	if err != nil {
		t.Errorf("Transform returned error: %v", err)
	}

	result := buf.String()
	if !strings.Contains(result, `"type":"response.content_part.done"`) {
		t.Error("Result should contain content_part.done event")
	}

	if !strings.Contains(result, `"text":"Hello world"`) {
		t.Error("Result should contain accumulated text")
	}
}

// TestResponsesTransformer_HandleMessageStop tests message stop.
func TestResponsesTransformer_HandleMessageStop(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesTransformer(&buf)

	// Setup
	msgStart := types.Event{
		Type: "message_start",
		Message: &types.MessageInfo{
			ID:    "msg_abc",
			Model: "claude-3",
		},
	}
	data, _ := json.Marshal(msgStart)
	transformer.Transform(&sse.Event{Data: string(data)})
	buf.Reset()

	// Send message_stop
	msgStop := types.Event{Type: "message_stop"}
	data, _ = json.Marshal(msgStop)
	err := transformer.Transform(&sse.Event{Data: string(data)})

	if err != nil {
		t.Errorf("Transform returned error: %v", err)
	}

	result := buf.String()
	if !strings.Contains(result, `"type":"response.completed"`) {
		t.Error("Result should contain response.completed event")
	}
}

// TestResponsesTransformer_HandleMessageStop_OnlyToolCalls tests message stop with only tool calls (no text).
// This verifies that assistant messages are properly emitted even when the model only returns tool calls.
func TestResponsesTransformer_HandleMessageStop_OnlyToolCalls(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesTransformer(&buf)

	// Setup: message_start
	msgStart := types.Event{
		Type: "message_start",
		Message: &types.MessageInfo{
			ID:    "msg_abc",
			Model: "claude-3",
		},
	}
	data, _ := json.Marshal(msgStart)
	transformer.Transform(&sse.Event{Data: string(data)})
	buf.Reset()

	// Send tool_use content block (no text)
	cbStart := types.Event{
		Type:         "content_block_start",
		Index:        intPtr(0),
		ContentBlock: json.RawMessage(`{"type":"tool_use","id":"tool_123","name":"exec_command"}`),
	}
	data, _ = json.Marshal(cbStart)
	transformer.Transform(&sse.Event{Data: string(data)})

	// Send tool input delta
	cbDelta := types.Event{
		Type:  "content_block_delta",
		Index: intPtr(0),
		Delta: json.RawMessage(`{"type":"input_json_delta","partial_json":"{\"cmd\":\"ls\"}"}`),
	}
	data, _ = json.Marshal(cbDelta)
	transformer.Transform(&sse.Event{Data: string(data)})

	// End content block
	cbStop := types.Event{
		Type:  "content_block_stop",
		Index: intPtr(0),
	}
	data, _ = json.Marshal(cbStop)
	transformer.Transform(&sse.Event{Data: string(data)})

	buf.Reset()

	// Send message_stop
	msgStop := types.Event{Type: "message_stop"}
	data, _ = json.Marshal(msgStop)
	err := transformer.Transform(&sse.Event{Data: string(data)})

	if err != nil {
		t.Errorf("Transform returned error: %v", err)
	}

	result := buf.String()

	// Should contain the assistant message output_item.done
	if !strings.Contains(result, `"type":"response.output_item.done"`) {
		t.Error("Result should contain response.output_item.done event for assistant message")
	}

	// Should contain the function_call output_item.done
	if !strings.Contains(result, `"type":"function_call"`) {
		t.Error("Result should contain function_call in output")
	}

	// Should contain response.completed
	if !strings.Contains(result, `"type":"response.completed"`) {
		t.Error("Result should contain response.completed event")
	}

	// Verify the assistant message is included in response.completed output
	if !strings.Contains(result, `"role":"assistant"`) {
		t.Error("Result should contain assistant role in output")
	}
}

// TestResponsesTransformer_HandlePing tests ping handling.
func TestResponsesTransformer_HandlePing(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesTransformer(&buf)

	ping := types.Event{Type: "ping"}
	data, _ := json.Marshal(ping)
	err := transformer.Transform(&sse.Event{Data: string(data)})

	if err != nil {
		t.Errorf("Transform returned error: %v", err)
	}

	// Ping should produce no output
	if buf.Len() != 0 {
		t.Error("Buffer should be empty for ping events")
	}
}

// TestResponsesTransformer_Flush tests flush operation.
func TestResponsesTransformer_Flush(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesTransformer(&buf)

	err := transformer.Flush()
	if err != nil {
		t.Errorf("Flush returned error: %v", err)
	}
}

// TestResponsesTransformer_Close tests close operation.
func TestResponsesTransformer_Close(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesTransformer(&buf)

	err := transformer.Close()
	if err != nil {
		t.Errorf("Close returned error: %v", err)
	}
}

// TestResponsesTransformer_FullFlow tests a complete streaming flow.
func TestResponsesTransformer_FullFlow(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesTransformer(&buf)

	events := []types.Event{
		{
			Type: "message_start",
			Message: &types.MessageInfo{
				ID:    "msg_abc123",
				Type:  "message",
				Role:  "assistant",
				Model: "claude-3-opus",
			},
		},
		{
			Type:         "content_block_start",
			Index:        intPtr(0),
			ContentBlock: mustMarshal(types.ContentBlock{Type: "text"}),
		},
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.TextDelta{Type: "text_delta", Text: "Hello"}),
		},
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.TextDelta{Type: "text_delta", Text: " world"}),
		},
		{
			Type:  "content_block_stop",
			Index: intPtr(0),
		},
		{
			Type: "message_stop",
		},
	}

	for i, e := range events {
		data, _ := json.Marshal(e)
		err := transformer.Transform(&sse.Event{Data: string(data)})
		if err != nil {
			t.Errorf("Transform event %d returned error: %v", i, err)
		}
	}

	result := buf.String()

	// Verify all expected events are present
	expectedEvents := []string{
		`"type":"response.created"`,
		`"type":"response.content_part.added"`,
		`"type":"response.output_text.delta"`,
		`"type":"response.content_part.done"`,
		`"type":"response.completed"`,
	}

	for _, expected := range expectedEvents {
		if !strings.Contains(result, expected) {
			t.Errorf("Result should contain %s", expected)
		}
	}
}

// TestResponsesTransformer_FullFlowWithTool tests complete flow with tool call.
func TestResponsesTransformer_FullFlowWithTool(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesTransformer(&buf)

	events := []types.Event{
		{
			Type: "message_start",
			Message: &types.MessageInfo{
				ID:    "msg_abc",
				Model: "claude-3",
			},
		},
		{
			Type:         "content_block_start",
			Index:        intPtr(0),
			ContentBlock: mustMarshal(types.ContentBlock{Type: "tool_use", ID: "toolu_123", Name: "get_weather"}),
		},
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.InputJSONDelta{Type: "input_json_delta", PartialJSON: `{"location":`}),
		},
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.InputJSONDelta{Type: "input_json_delta", PartialJSON: `"San Francisco"}`}),
		},
		{
			Type:  "content_block_stop",
			Index: intPtr(0),
		},
		{
			Type: "message_stop",
		},
	}

	for _, e := range events {
		data, _ := json.Marshal(e)
		transformer.Transform(&sse.Event{Data: string(data)})
	}

	result := buf.String()

	// Verify tool call events
	if !strings.Contains(result, `"type":"function_call"`) {
		t.Error("Result should contain function_call type")
	}

	if !strings.Contains(result, `"type":"response.function_call_arguments.delta"`) {
		t.Error("Result should contain function_call_arguments.delta")
	}
}

// BenchmarkResponsesTransformer_Transform benchmarks the transformer.
func BenchmarkResponsesTransformer_Transform(b *testing.B) {
	var buf bytes.Buffer
	transformer := NewResponsesTransformer(&buf)

	event := types.Event{
		Type: "message_start",
		Message: &types.MessageInfo{
			ID:    "msg_abc",
			Model: "claude-3",
		},
	}
	data, _ := json.Marshal(event)
	sseEvent := &sse.Event{Data: string(data)}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf.Reset()
		transformer.Transform(sseEvent)
	}
}

// Helper function to marshal to RawMessage
func mustMarshal(v interface{}) json.RawMessage {
	data, _ := json.Marshal(v)
	return json.RawMessage(data)
}

// TestResponsesTransformer_MultipleToolCalls tests multiple parallel tool calls.
func TestResponsesTransformer_MultipleToolCalls(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesTransformer(&buf)

	events := []types.Event{
		{
			Type: "message_start",
			Message: &types.MessageInfo{
				ID:    "msg_abc",
				Model: "claude-3",
			},
		},
		{
			Type:         "content_block_start",
			Index:        intPtr(0),
			ContentBlock: mustMarshal(types.ContentBlock{Type: "tool_use", ID: "toolu_1", Name: "get_weather"}),
		},
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.InputJSONDelta{Type: "input_json_delta", PartialJSON: `{"city": "SF"}`}),
		},
		{
			Type:  "content_block_stop",
			Index: intPtr(0),
		},
		{
			Type:         "content_block_start",
			Index:        intPtr(1),
			ContentBlock: mustMarshal(types.ContentBlock{Type: "tool_use", ID: "toolu_2", Name: "get_time"}),
		},
		{
			Type:  "content_block_delta",
			Index: intPtr(1),
			Delta: mustMarshal(types.InputJSONDelta{Type: "input_json_delta", PartialJSON: `{"timezone": "PST"}`}),
		},
		{
			Type:  "content_block_stop",
			Index: intPtr(1),
		},
		{
			Type: "message_stop",
		},
	}

	for _, e := range events {
		data, _ := json.Marshal(e)
		err := transformer.Transform(&sse.Event{Data: string(data)})
		if err != nil {
			t.Errorf("Transform returned error: %v", err)
		}
	}

	result := buf.String()

	// Should contain both function_call items
	if !strings.Contains(result, `"name":"get_weather"`) {
		t.Error("Expected get_weather in output")
	}
	if !strings.Contains(result, `"name":"get_time"`) {
		t.Error("Expected get_time in output")
	}

	// Should contain both tool IDs
	if !strings.Contains(result, `"id":"toolu_1"`) {
		t.Error("Expected toolu_1 in output")
	}
	if !strings.Contains(result, `"id":"toolu_2"`) {
		t.Error("Expected toolu_2 in output")
	}
}

// TestResponsesTransformer_NestedJSONArgs tests nested JSON in tool arguments.
func TestResponsesTransformer_NestedJSONArgs(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesTransformer(&buf)

	events := []types.Event{
		{
			Type: "message_start",
			Message: &types.MessageInfo{
				ID:    "msg_abc",
				Model: "claude-3",
			},
		},
		{
			Type:         "content_block_start",
			Index:        intPtr(0),
			ContentBlock: mustMarshal(types.ContentBlock{Type: "tool_use", ID: "toolu_123", Name: "search"}),
		},
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.InputJSONDelta{Type: "input_json_delta", PartialJSON: `{"query": "test", "filters": {"date": "2024-01"}}`}),
		},
		{
			Type:  "content_block_stop",
			Index: intPtr(0),
		},
		{
			Type: "message_stop",
		},
	}

	for _, e := range events {
		data, _ := json.Marshal(e)
		transformer.Transform(&sse.Event{Data: string(data)})
	}

	result := buf.String()

	// Should contain arguments field
	if !strings.Contains(result, `"arguments"`) {
		t.Error("Expected arguments in output")
	}
}

// TestResponsesTransformer_EmptyToolArgs tests empty tool arguments.
func TestResponsesTransformer_EmptyToolArgs(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesTransformer(&buf)

	events := []types.Event{
		{
			Type: "message_start",
			Message: &types.MessageInfo{
				ID:    "msg_abc",
				Model: "claude-3",
			},
		},
		{
			Type:         "content_block_start",
			Index:        intPtr(0),
			ContentBlock: mustMarshal(types.ContentBlock{Type: "tool_use", ID: "toolu_123", Name: "get_time"}),
		},
		{
			Type:  "content_block_delta",
			Index: intPtr(0),
			Delta: mustMarshal(types.InputJSONDelta{Type: "input_json_delta", PartialJSON: `{}`}),
		},
		{
			Type:  "content_block_stop",
			Index: intPtr(0),
		},
		{
			Type: "message_stop",
		},
	}

	for _, e := range events {
		data, _ := json.Marshal(e)
		transformer.Transform(&sse.Event{Data: string(data)})
	}

	result := buf.String()

	// Should contain empty arguments
	if !strings.Contains(result, `"arguments":"{}"`) {
		t.Error("Expected empty arguments in output")
	}
}

// ============================================================================
// PHASE 2 HIGH PRIORITY TESTS
// ============================================================================

// TestResponsesTransformer_ReasoningSummaryTextDelta tests response.reasoning_summary_text.delta.
// Category B2 (Responses → Chat transformation): HIGH priority
func TestResponsesTransformer_ReasoningSummaryTextDelta(t *testing.T) {
	tests := []struct {
		name     string
		events   []types.Event
		validate func(t *testing.T, output string)
	}{
		{
			name: "reasoning summary text delta streaming",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_abc123",
						Model: "claude-3-opus",
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
					Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "Let me"}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: " analyze this"}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: " step by step..."}),
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
				// Should contain response.reasoning_summary_text.delta events
				if !strings.Contains(output, `"type":"response.reasoning_summary_text.delta"`) {
					t.Error("Expected response.reasoning_summary_text.delta in output")
				}
				// Should have all deltas
				if !strings.Contains(output, `"delta":"Let me"`) {
					t.Error("Expected first reasoning delta")
				}
				if !strings.Contains(output, `"delta":" analyze this"`) {
					t.Error("Expected second reasoning delta")
				}
				if !strings.Contains(output, `"delta":" step by step..."`) {
					t.Error("Expected third reasoning delta")
				}
				// Should have correct output_index
				if !strings.Contains(output, `"output_index":0`) {
					t.Error("Expected output_index 0 for reasoning")
				}
				// Should have summary_index
				if !strings.Contains(output, `"summary_index":0`) {
					t.Error("Expected summary_index")
				}
			},
		},
		{
			name: "reasoning followed by text message",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_def456",
						Model: "claude-3-opus",
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
					Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "Thinking..."}),
				},
				{
					Type:  "content_block_stop",
					Index: intPtr(0),
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(1),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "text"}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(1),
					Delta: mustMarshal(types.TextDelta{Type: "text_delta", Text: "Answer: 42"}),
				},
				{
					Type:  "content_block_stop",
					Index: intPtr(1),
				},
				{
					Type: "message_stop",
				},
			},
			validate: func(t *testing.T, output string) {
				// Should have both reasoning and text events
				if !strings.Contains(output, `"type":"response.reasoning_summary_text.delta"`) {
					t.Error("Expected reasoning summary delta")
				}
				if !strings.Contains(output, `"type":"response.output_text.delta"`) {
					t.Error("Expected output text delta")
				}
				// Reasoning should come before text (output_index 0 vs 1)
				reasoningIdx := strings.Index(output, `"type":"response.reasoning_summary_text.delta"`)
				textIdx := strings.Index(output, `"type":"response.output_text.delta"`)
				if reasoningIdx == -1 || textIdx == -1 {
					t.Fatal("Missing expected events")
				}
				if reasoningIdx > textIdx {
					t.Error("Reasoning should come before text in output")
				}
				// Final output should have both items
				if !strings.Contains(output, `"type":"reasoning"`) {
					t.Error("Expected reasoning item in final output")
				}
				if !strings.Contains(output, `"type":"message"`) {
					t.Error("Expected message item in final output")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			transformer := NewResponsesTransformer(&buf)

			for _, e := range tt.events {
				data, _ := json.Marshal(e)
				err := transformer.Transform(&sse.Event{Data: string(data)})
				if err != nil {
					t.Errorf("Transform returned error: %v", err)
				}
			}

			output := buf.String()
			if tt.validate != nil {
				tt.validate(t, output)
			}
		})
	}
}

// TestResponsesTransformer_ResponseCompletedWithReasoning tests response.completed with reasoning item.
// Category B2 (Responses → Chat transformation): HIGH priority
func TestResponsesTransformer_ResponseCompletedWithReasoning(t *testing.T) {
	tests := []struct {
		name     string
		events   []types.Event
		validate func(t *testing.T, output string)
	}{
		{
			name: "response.completed with only reasoning",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_abc",
						Type:  "message",
						Role:  "assistant",
						Model: "claude-3-opus",
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
					Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "Internal reasoning"}),
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
				// Should have response.completed
				if !strings.Contains(output, `"type":"response.completed"`) {
					t.Error("Expected response.completed event")
				}
				// Should have reasoning output item
				if !strings.Contains(output, `"type":"reasoning"`) {
					t.Error("Expected reasoning item in output")
				}
				// Should have the reasoning ID
				if !strings.Contains(output, `"id":"rs_abc"`) {
					t.Error("Expected reasoning ID rs_abc")
				}
				// Should have summary text
				if !strings.Contains(output, `"text":"Internal reasoning"`) {
					t.Error("Expected reasoning summary text")
				}
				// Should have correct structure
				if !strings.Contains(output, `"type":"summary_text"`) {
					t.Error("Expected summary_text type in reasoning")
				}
			},
		},
		{
			name: "response.completed with reasoning and tool call",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_xyz",
						Model: "claude-3-opus",
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
					Delta: mustMarshal(types.ThinkingDelta{Type: "thinking_delta", Thinking: "Need to check weather"}),
				},
				{
					Type:  "content_block_stop",
					Index: intPtr(0),
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(1),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "tool_use", ID: "toolu_123", Name: "get_weather"}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(1),
					Delta: mustMarshal(types.InputJSONDelta{Type: "input_json_delta", PartialJSON: `{"city": "SF"}`}),
				},
				{
					Type:  "content_block_stop",
					Index: intPtr(1),
				},
				{
					Type: "message_stop",
				},
			},
			validate: func(t *testing.T, output string) {
				// Should have both reasoning and function_call in final output
				if !strings.Contains(output, `"type":"reasoning"`) {
					t.Error("Expected reasoning item")
				}
				if !strings.Contains(output, `"type":"function_call"`) {
					t.Error("Expected function_call item")
				}
				// Reasoning should come before function_call
				reasoningIdx := strings.Index(output, `"type":"reasoning"`)
				functionIdx := strings.Index(output, `"type":"function_call"`)
				if reasoningIdx == -1 || functionIdx == -1 {
					t.Fatal("Missing expected items")
				}
				if reasoningIdx > functionIdx {
					t.Error("Reasoning should come before function_call in output")
				}
				// Verify reasoning content
				if !strings.Contains(output, `"text":"Need to check weather"`) {
					t.Error("Expected reasoning text content")
				}
				// Verify function_call content
				if !strings.Contains(output, `"name":"get_weather"`) {
					t.Error("Expected function name")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			transformer := NewResponsesTransformer(&buf)

			for _, e := range tt.events {
				data, _ := json.Marshal(e)
				err := transformer.Transform(&sse.Event{Data: string(data)})
				if err != nil {
					t.Errorf("Transform returned error: %v", err)
				}
			}

			output := buf.String()
			if tt.validate != nil {
				tt.validate(t, output)
			}
		})
	}
}

// TestResponsesTransformer_FunctionCallArgumentsDelta tests response.function_call_arguments.delta.
// Category B2 (Responses → Chat transformation): HIGH priority
func TestResponsesTransformer_FunctionCallArgumentsDelta(t *testing.T) {
	tests := []struct {
		name     string
		events   []types.Event
		validate func(t *testing.T, output string)
	}{
		{
			name: "function_call_arguments.delta chunked streaming",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_tool123",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "tool_use", ID: "toolu_abc", Name: "search"}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.InputJSONDelta{Type: "input_json_delta", PartialJSON: `{"q`}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.InputJSONDelta{Type: "input_json_delta", PartialJSON: `uery": "`}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.InputJSONDelta{Type: "input_json_delta", PartialJSON: `hello`}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.InputJSONDelta{Type: "input_json_delta", PartialJSON: `"}`}),
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
				// Should have function_call_arguments.delta events
				if !strings.Contains(output, `"type":"response.function_call_arguments.delta"`) {
					t.Error("Expected response.function_call_arguments.delta")
				}
				// Should have all chunks
				count := strings.Count(output, `"type":"response.function_call_arguments.delta"`)
				if count != 4 {
					t.Errorf("Expected 4 argument delta events, got %d", count)
				}
				// Each chunk should have correct call_id
				if !strings.Contains(output, `"call_id":"toolu_abc"`) {
					t.Error("Expected call_id in argument deltas")
				}
				// Final arguments should be complete
				if !strings.Contains(output, `"arguments":"{\"query\": \"hello\"}"`) {
					t.Error("Expected complete arguments in final output")
				}
			},
		},
		{
			name: "function_call_arguments.delta with special characters",
			events: []types.Event{
				{
					Type: "message_start",
					Message: &types.MessageInfo{
						ID:    "msg_special",
						Model: "claude-3",
					},
				},
				{
					Type:         "content_block_start",
					Index:        intPtr(0),
					ContentBlock: mustMarshal(types.ContentBlock{Type: "tool_use", ID: "toolu_special", Name: "process_text"}),
				},
				{
					Type:  "content_block_delta",
					Index: intPtr(0),
					Delta: mustMarshal(types.InputJSONDelta{Type: "input_json_delta", PartialJSON: `{"text": "Hello
World"}`}),
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
				// Special characters should be handled correctly
				if !strings.Contains(output, `"type":"response.function_call_arguments.delta"`) {
					t.Error("Expected function_call_arguments.delta")
				}
				// Arguments should contain the text
				if !strings.Contains(output, `"arguments"`) {
					t.Error("Expected arguments field")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			transformer := NewResponsesTransformer(&buf)

			for _, e := range tt.events {
				data, _ := json.Marshal(e)
				err := transformer.Transform(&sse.Event{Data: string(data)})
				if err != nil {
					t.Errorf("Transform returned error: %v", err)
				}
			}

			output := buf.String()
			if tt.validate != nil {
				tt.validate(t, output)
			}
		})
	}
}
