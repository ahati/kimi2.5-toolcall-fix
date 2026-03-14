package convert

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"ai-proxy/types"

	"github.com/tmaxmax/go-sse"
)

func TestNewResponsesToChatConverter(t *testing.T) {
	converter := NewResponsesToChatConverter()
	if converter == nil {
		t.Fatal("expected converter to not be nil")
	}
}

func TestResponsesToChatConverter_Convert_StringInput(t *testing.T) {
	converter := NewResponsesToChatConverter()

	req := types.ResponsesRequest{
		Model:           "gpt-4",
		Input:           "Hello, world!",
		MaxOutputTokens: 100,
		Stream:          true,
		Temperature:     0.7,
		TopP:            0.9,
		Instructions:    "Be helpful",
	}

	body, _ := json.Marshal(req)
	result, err := converter.Convert(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var chatReq types.ChatCompletionRequest
	if err := json.Unmarshal(result, &chatReq); err != nil {
		t.Fatalf("failed to unmarshal result: %v", err)
	}

	if chatReq.Model != "gpt-4" {
		t.Errorf("expected model 'gpt-4', got '%s'", chatReq.Model)
	}
	if chatReq.MaxTokens != 100 {
		t.Errorf("expected max_tokens 100, got %d", chatReq.MaxTokens)
	}
	if !chatReq.Stream {
		t.Error("expected stream to be true")
	}
	if chatReq.Temperature != 0.7 {
		t.Errorf("expected temperature 0.7, got %f", chatReq.Temperature)
	}
	if chatReq.TopP != 0.9 {
		t.Errorf("expected top_p 0.9, got %f", chatReq.TopP)
	}
	if chatReq.System != "Be helpful" {
		t.Errorf("expected system 'Be helpful', got '%s'", chatReq.System)
	}
	if len(chatReq.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(chatReq.Messages))
	}
	if chatReq.Messages[0].Role != "user" {
		t.Errorf("expected role 'user', got '%s'", chatReq.Messages[0].Role)
	}
	if chatReq.Messages[0].Content != "Hello, world!" {
		t.Errorf("expected content 'Hello, world!', got '%v'", chatReq.Messages[0].Content)
	}
}

func TestResponsesToChatConverter_Convert_ArrayInput(t *testing.T) {
	converter := NewResponsesToChatConverter()

	input := []interface{}{
		map[string]interface{}{
			"type":    "message",
			"role":    "system",
			"content": "System message",
		},
		map[string]interface{}{
			"type":    "message",
			"role":    "user",
			"content": "User message",
		},
		map[string]interface{}{
			"type":    "message",
			"role":    "assistant",
			"content": "Assistant message",
		},
	}

	req := types.ResponsesRequest{
		Model: "gpt-4",
		Input: input,
	}

	body, _ := json.Marshal(req)
	result, err := converter.Convert(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var chatReq types.ChatCompletionRequest
	if err := json.Unmarshal(result, &chatReq); err != nil {
		t.Fatalf("failed to unmarshal result: %v", err)
	}

	if len(chatReq.Messages) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(chatReq.Messages))
	}
	if chatReq.Messages[0].Role != "system" {
		t.Errorf("expected role 'system', got '%s'", chatReq.Messages[0].Role)
	}
	if chatReq.Messages[1].Role != "user" {
		t.Errorf("expected role 'user', got '%s'", chatReq.Messages[1].Role)
	}
	if chatReq.Messages[2].Role != "assistant" {
		t.Errorf("expected role 'assistant', got '%s'", chatReq.Messages[2].Role)
	}
}

func TestResponsesToChatConverter_Convert_Tools(t *testing.T) {
	converter := NewResponsesToChatConverter()

	tools := []types.ResponsesTool{
		{
			Type:        "function",
			Name:        "get_weather",
			Description: "Get the weather",
			Parameters:  json.RawMessage(`{"type": "object"}`),
		},
	}

	req := types.ResponsesRequest{
		Model: "gpt-4",
		Input: "What's the weather?",
		Tools: tools,
	}

	body, _ := json.Marshal(req)
	result, err := converter.Convert(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var chatReq types.ChatCompletionRequest
	if err := json.Unmarshal(result, &chatReq); err != nil {
		t.Fatalf("failed to unmarshal result: %v", err)
	}

	if len(chatReq.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(chatReq.Tools))
	}
	if chatReq.Tools[0].Type != "function" {
		t.Errorf("expected tool type 'function', got '%s'", chatReq.Tools[0].Type)
	}
	if chatReq.Tools[0].Function.Name != "get_weather" {
		t.Errorf("expected function name 'get_weather', got '%s'", chatReq.Tools[0].Function.Name)
	}
}

func TestResponsesToChatConverter_Convert_ToolsWithFunction(t *testing.T) {
	converter := NewResponsesToChatConverter()

	tools := []types.ResponsesTool{
		{
			Type: "function",
			Function: &types.ResponsesToolFunction{
				Name:        "get_weather",
				Description: "Get the weather",
				Parameters:  json.RawMessage(`{"type": "object"}`),
			},
		},
	}

	req := types.ResponsesRequest{
		Model: "gpt-4",
		Input: "What's the weather?",
		Tools: tools,
	}

	body, _ := json.Marshal(req)
	result, err := converter.Convert(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var chatReq types.ChatCompletionRequest
	if err := json.Unmarshal(result, &chatReq); err != nil {
		t.Fatalf("failed to unmarshal result: %v", err)
	}

	if len(chatReq.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(chatReq.Tools))
	}
	if chatReq.Tools[0].Function.Name != "get_weather" {
		t.Errorf("expected function name 'get_weather', got '%s'", chatReq.Tools[0].Function.Name)
	}
}

func TestResponsesToChatConverter_Convert_InvalidJSON(t *testing.T) {
	converter := NewResponsesToChatConverter()

	_, err := converter.Convert([]byte("invalid json"))
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestResponsesToChatConverter_Convert_EmptyTools(t *testing.T) {
	converter := NewResponsesToChatConverter()

	req := types.ResponsesRequest{
		Model: "gpt-4",
		Input: "Hello",
		Tools: []types.ResponsesTool{},
	}

	body, _ := json.Marshal(req)
	result, err := converter.Convert(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var chatReq types.ChatCompletionRequest
	if err := json.Unmarshal(result, &chatReq); err != nil {
		t.Fatalf("failed to unmarshal result: %v", err)
	}

	if chatReq.Tools != nil {
		t.Errorf("expected nil tools, got %v", chatReq.Tools)
	}
}

func TestResponsesToChatConverter_Convert_NilInput(t *testing.T) {
	converter := NewResponsesToChatConverter()

	req := types.ResponsesRequest{
		Model: "gpt-4",
		Input: nil,
	}

	body, _ := json.Marshal(req)
	result, err := converter.Convert(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var chatReq types.ChatCompletionRequest
	if err := json.Unmarshal(result, &chatReq); err != nil {
		t.Fatalf("failed to unmarshal result: %v", err)
	}

	if len(chatReq.Messages) != 0 {
		t.Errorf("expected 0 messages, got %d", len(chatReq.Messages))
	}
}

func TestNewResponsesToChatTransformer(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesToChatTransformer(&buf)
	if transformer == nil {
		t.Fatal("expected transformer to not be nil")
	}
}

func TestResponsesToChatTransformer_Transform_ResponseCreated(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesToChatTransformer(&buf)

	event := types.ResponsesStreamEvent{
		Type: "response.created",
		Response: &types.ResponsesResponse{
			ID:    "resp_123",
			Model: "gpt-4",
		},
	}
	data, _ := json.Marshal(event)

	sseEvent := &sse.Event{Data: string(data)}
	err := transformer.Transform(sseEvent)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if buf.Len() != 0 {
		t.Error("expected no output for response.created event")
	}
}

func TestResponsesToChatTransformer_Transform_OutputTextDelta(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesToChatTransformer(&buf)

	transformer.messageID = "resp_123"
	transformer.model = "gpt-4"

	event := types.ResponsesStreamEvent{
		Type:  "response.output_text.delta",
		Delta: "Hello",
	}
	data, _ := json.Marshal(event)

	sseEvent := &sse.Event{Data: string(data)}
	err := transformer.Transform(sseEvent)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	output := buf.String()
	if !strings.Contains(output, "data: {") {
		t.Error("expected SSE formatted data")
	}
	if !strings.Contains(output, "Hello") {
		t.Error("expected output to contain 'Hello'")
	}
	if !strings.Contains(output, "chat.completion.chunk") {
		t.Error("expected chat.completion.chunk object type")
	}
}

func TestResponsesToChatTransformer_Transform_FunctionCallDelta(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesToChatTransformer(&buf)

	transformer.messageID = "resp_123"
	transformer.model = "gpt-4"

	event := types.ResponsesStreamEvent{
		Type:  "response.function_call_arguments.delta",
		Delta: `{"city": "Paris"}`,
		OutputItem: &types.OutputItem{
			ID:     "call_123",
			CallID: "call_123",
		},
	}
	data, _ := json.Marshal(event)

	sseEvent := &sse.Event{Data: string(data)}
	err := transformer.Transform(sseEvent)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	output := buf.String()
	if !strings.Contains(output, "tool_calls") {
		t.Error("expected output to contain tool_calls")
	}
}

func TestResponsesToChatTransformer_Transform_ResponseCompleted(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesToChatTransformer(&buf)

	transformer.messageID = "resp_123"
	transformer.model = "gpt-4"

	event := types.ResponsesStreamEvent{
		Type: "response.completed",
		Response: &types.ResponsesResponse{
			Usage: &types.ResponsesUsage{
				InputTokens:  10,
				OutputTokens: 20,
				TotalTokens:  30,
			},
		},
	}
	data, _ := json.Marshal(event)

	sseEvent := &sse.Event{Data: string(data)}
	err := transformer.Transform(sseEvent)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	output := buf.String()
	if !strings.Contains(output, "finish_reason") {
		t.Error("expected output to contain finish_reason")
	}
	if !strings.Contains(output, "stop") {
		t.Error("expected finish_reason to be 'stop'")
	}
}

func TestResponsesToChatTransformer_Transform_Done(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesToChatTransformer(&buf)

	sseEvent := &sse.Event{Data: "[DONE]"}
	err := transformer.Transform(sseEvent)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	output := buf.String()
	if !strings.Contains(output, "[DONE]") {
		t.Error("expected output to contain [DONE]")
	}
}

func TestResponsesToChatTransformer_Transform_EmptyData(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesToChatTransformer(&buf)

	sseEvent := &sse.Event{Data: ""}
	err := transformer.Transform(sseEvent)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if buf.Len() != 0 {
		t.Error("expected no output for empty data")
	}
}

func TestResponsesToChatTransformer_Transform_InvalidJSON(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesToChatTransformer(&buf)

	sseEvent := &sse.Event{Data: "invalid json"}
	err := transformer.Transform(sseEvent)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	output := buf.String()
	if !strings.Contains(output, "invalid json") {
		t.Error("expected passthrough of invalid JSON")
	}
}

func TestResponsesToChatTransformer_Flush(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesToChatTransformer(&buf)

	err := transformer.Flush()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestResponsesToChatTransformer_Close(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesToChatTransformer(&buf)

	err := transformer.Close()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestResponsesToChatConverter_Convert_ToolWithEmptyName(t *testing.T) {
	converter := NewResponsesToChatConverter()

	tools := []types.ResponsesTool{
		{
			Type:        "function",
			Description: "Get the weather",
		},
	}

	req := types.ResponsesRequest{
		Model: "gpt-4",
		Input: "Hello",
		Tools: tools,
	}

	body, _ := json.Marshal(req)
	result, err := converter.Convert(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var chatReq types.ChatCompletionRequest
	if err := json.Unmarshal(result, &chatReq); err != nil {
		t.Fatalf("failed to unmarshal result: %v", err)
	}

	if len(chatReq.Tools) != 0 {
		t.Errorf("expected 0 tools for empty name, got %d", len(chatReq.Tools))
	}
}

func TestResponsesToChatConverter_Convert_NonMessageInputItems(t *testing.T) {
	converter := NewResponsesToChatConverter()

	input := []interface{}{
		map[string]interface{}{
			"type": "file",
			"role": "user",
		},
	}

	req := types.ResponsesRequest{
		Model: "gpt-4",
		Input: input,
	}

	body, _ := json.Marshal(req)
	result, err := converter.Convert(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var chatReq types.ChatCompletionRequest
	if err := json.Unmarshal(result, &chatReq); err != nil {
		t.Fatalf("failed to unmarshal result: %v", err)
	}

	if len(chatReq.Messages) != 0 {
		t.Errorf("expected 0 messages for non-message type, got %d", len(chatReq.Messages))
	}
}

func TestResponsesToChatTransformer_Transform_UnknownEventType(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesToChatTransformer(&buf)

	event := types.ResponsesStreamEvent{
		Type: "response.unknown_event",
	}
	data, _ := json.Marshal(event)

	sseEvent := &sse.Event{Data: string(data)}
	err := transformer.Transform(sseEvent)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if buf.Len() != 0 {
		t.Error("expected no output for unknown event type")
	}
}

func TestResponsesToChatTransformer_Transform_ResponseInProgress(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesToChatTransformer(&buf)

	event := types.ResponsesStreamEvent{
		Type: "response.in_progress",
		Response: &types.ResponsesResponse{
			ID:    "resp_123",
			Model: "gpt-4",
		},
	}
	data, _ := json.Marshal(event)

	sseEvent := &sse.Event{Data: string(data)}
	err := transformer.Transform(sseEvent)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if transformer.messageID != "resp_123" {
		t.Errorf("expected messageID 'resp_123', got '%s'", transformer.messageID)
	}
	if transformer.model != "gpt-4" {
		t.Errorf("expected model 'gpt-4', got '%s'", transformer.model)
	}
}

func TestResponsesToChatTransformer_Transform_FunctionCallDeltaNilOutputItem(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewResponsesToChatTransformer(&buf)

	transformer.messageID = "resp_123"
	transformer.model = "gpt-4"

	event := types.ResponsesStreamEvent{
		Type:       "response.function_call_arguments.delta",
		Delta:      `{"city": "Paris"}`,
		OutputItem: nil,
	}
	data, _ := json.Marshal(event)

	sseEvent := &sse.Event{Data: string(data)}
	err := transformer.Transform(sseEvent)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if buf.Len() != 0 {
		t.Error("expected no output for nil OutputItem")
	}
}
