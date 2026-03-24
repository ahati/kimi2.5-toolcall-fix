package types

import (
	"encoding/json"
	"testing"
)

// boolPtr is a helper function to create a pointer to a bool value.
func boolPtr(b bool) *bool { return &b }

// TestResponsesRequest_MarshalUnmarshal tests marshaling and unmarshaling of ResponsesRequest.
func TestResponsesRequest_MarshalUnmarshal(t *testing.T) {
	tests := []struct {
		name     string
		request  ResponsesRequest
		wantJSON string
	}{
		{
			name: "basic request with string input",
			request: ResponsesRequest{
				Model:           "gpt-4o",
				Input:           "Hello, world!",
				Stream:          boolPtr(true),
				MaxOutputTokens: 100,
			},
			wantJSON: `{"model":"gpt-4o","input":"Hello, world!","max_output_tokens":100,"stream":true}`,
		},
		{
			name: "request with array input",
			request: ResponsesRequest{
				Model: "gpt-4o",
				Input: []InputItem{
					{Type: "message", Role: "user", Content: "Hello"},
					{Type: "message", Role: "assistant", Content: "Hi there"},
				},
				Stream: boolPtr(true),
			},
			wantJSON: `{"model":"gpt-4o","input":[{"type":"message","role":"user","content":"Hello"},{"type":"message","role":"assistant","content":"Hi there"}],"stream":true}`,
		},
		{
			name: "request with instructions",
			request: ResponsesRequest{
				Model:        "gpt-4o",
				Input:        "What is 2+2?",
				Instructions: "You are a helpful math tutor.",
				Stream:       boolPtr(true),
			},
			wantJSON: `{"model":"gpt-4o","input":"What is 2+2?","instructions":"You are a helpful math tutor.","stream":true}`,
		},
		{
			name: "request with reasoning config",
			request: ResponsesRequest{
				Model:  "o3",
				Input:  "Solve this complex problem",
				Stream: boolPtr(true),
				Reasoning: &ReasoningConfig{
					Summary: "detailed",
				},
			},
			wantJSON: `{"model":"o3","input":"Solve this complex problem","stream":true,"reasoning":{"summary":"detailed"}}`,
		},
		{
			name: "request with previous response ID",
			request: ResponsesRequest{
				Model:              "gpt-4o",
				Input:              "Tell me more",
				PreviousResponseID: "resp_123abc",
				Stream:             boolPtr(true),
			},
			wantJSON: `{"model":"gpt-4o","input":"Tell me more","stream":true,"previous_response_id":"resp_123abc"}`,
		},
		{
			name: "request with temperature and top_p",
			request: ResponsesRequest{
				Model:       "gpt-4o",
				Input:       "Creative writing",
				Temperature: 0.8,
				TopP:        0.9,
				Stream:      boolPtr(true),
			},
			wantJSON: `{"model":"gpt-4o","input":"Creative writing","stream":true,"temperature":0.8,"top_p":0.9}`,
		},
		{
			name: "empty request",
			request: ResponsesRequest{
				Model: "",
				Input: nil,
			},
			wantJSON: `{"model":"","input":null}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test marshaling
			gotJSON, err := json.Marshal(tt.request)
			if err != nil {
				t.Fatalf("json.Marshal failed: %v", err)
			}
			if string(gotJSON) != tt.wantJSON {
				t.Errorf("json.Marshal = %s, want %s", gotJSON, tt.wantJSON)
			}

			// Test unmarshaling
			var gotReq ResponsesRequest
			if err := json.Unmarshal([]byte(tt.wantJSON), &gotReq); err != nil {
				t.Fatalf("json.Unmarshal failed: %v", err)
			}

			// Verify model matches
			if gotReq.Model != tt.request.Model {
				t.Errorf("Model = %s, want %s", gotReq.Model, tt.request.Model)
			}
			// Compare stream values (handle nil pointers)
			var gotStream, wantStream bool
			if gotReq.Stream != nil {
				gotStream = *gotReq.Stream
			}
			if tt.request.Stream != nil {
				wantStream = *tt.request.Stream
			}
			if gotStream != wantStream {
				t.Errorf("Stream = %v, want %v", gotStream, wantStream)
			}
		})
	}
}

// TestResponsesRequest_WithTools tests request with tool definitions.
func TestResponsesRequest_WithTools(t *testing.T) {
	request := ResponsesRequest{
		Model: "gpt-4o",
		Input: "What's the weather?",
		Tools: []ResponsesTool{
			{
				Type: "function",
				Function: &ResponsesToolFunction{
					Name:        "get_weather",
					Description: "Get the current weather",
					Parameters:  json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}}}`),
					Strict:      true,
				},
			},
			{
				Type: "web_search",
			},
		},
		Stream: boolPtr(true),
	}

	data, err := json.Marshal(request)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}

	var got ResponsesRequest
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}

	if len(got.Tools) != 2 {
		t.Fatalf("len(Tools) = %d, want 2", len(got.Tools))
	}

	if got.Tools[0].Type != "function" {
		t.Errorf("Tools[0].Type = %s, want function", got.Tools[0].Type)
	}

	if got.Tools[0].Function == nil {
		t.Fatal("Tools[0].Function is nil")
	}

	if got.Tools[0].Function.Name != "get_weather" {
		t.Errorf("Tools[0].Function.Name = %s, want get_weather", got.Tools[0].Function.Name)
	}

	if got.Tools[1].Type != "web_search" {
		t.Errorf("Tools[1].Type = %s, want web_search", got.Tools[1].Type)
	}
}

// TestResponsesResponse_MarshalUnmarshal tests response marshaling.
func TestResponsesResponse_MarshalUnmarshal(t *testing.T) {
	response := ResponsesResponse{
		ID:        "resp_123abc",
		Object:    "response",
		CreatedAt: 1234567890,
		Status:    "completed",
		Model:     "gpt-4o",
		Output: []OutputItem{
			{
				Type:   "message",
				ID:     "msg_456",
				Status: "completed",
				Role:   "assistant",
				Content: []OutputContent{
					{
						Type: "output_text",
						Text: "Hello! How can I help you today?",
					},
				},
			},
		},
		Usage: &ResponsesUsage{
			InputTokens:  10,
			OutputTokens: 20,
			TotalTokens:  30,
		},
	}

	data, err := json.Marshal(response)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}

	var got ResponsesResponse
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}

	if got.ID != response.ID {
		t.Errorf("ID = %s, want %s", got.ID, response.ID)
	}

	if got.Status != response.Status {
		t.Errorf("Status = %s, want %s", got.Status, response.Status)
	}

	if len(got.Output) != 1 {
		t.Fatalf("len(Output) = %d, want 1", len(got.Output))
	}

	if got.Output[0].Type != "message" {
		t.Errorf("Output[0].Type = %s, want message", got.Output[0].Type)
	}
}

// TestResponsesResponse_WithError tests error response.
func TestResponsesResponse_WithError(t *testing.T) {
	response := ResponsesResponse{
		ID:     "resp_err",
		Object: "response",
		Status: "incomplete",
		Error: &ResponsesError{
			Code:    "rate_limit_exceeded",
			Message: "Rate limit exceeded",
		},
		IncompleteDetails: &IncompleteDetails{
			Reason: "max_output_tokens",
		},
	}

	data, err := json.Marshal(response)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}

	var got ResponsesResponse
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}

	if got.Error == nil {
		t.Fatal("Error is nil")
	}

	if got.Error.Code != "rate_limit_exceeded" {
		t.Errorf("Error.Code = %s, want rate_limit_exceeded", got.Error.Code)
	}

	if got.IncompleteDetails == nil {
		t.Fatal("IncompleteDetails is nil")
	}

	if got.IncompleteDetails.Reason != "max_output_tokens" {
		t.Errorf("IncompleteDetails.Reason = %s, want max_output_tokens", got.IncompleteDetails.Reason)
	}
}

// TestOutputContent_Types tests different output content types.
func TestOutputContent_Types(t *testing.T) {
	tests := []struct {
		name    string
		content OutputContent
	}{
		{
			name: "output_text",
			content: OutputContent{
				Type: "output_text",
				Text: "Hello world",
				Annotations: []Annotation{
					{Type: "url_citation", URL: "https://example.com", Title: "Example"},
				},
			},
		},
		{
			name: "function_call",
			content: OutputContent{
				Type:      "function_call",
				ID:        "call_123",
				CallID:    "call_123",
				Name:      "get_weather",
				Arguments: `{"location":"San Francisco"}`,
			},
		},
		{
			name: "reasoning",
			content: OutputContent{
				Type:    "reasoning",
				Summary: "I need to think about this carefully...",
			},
		},
		{
			name: "refusal",
			content: OutputContent{
				Type: "refusal",
				Text: "I cannot fulfill this request.",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.content)
			if err != nil {
				t.Fatalf("json.Marshal failed: %v", err)
			}

			var got OutputContent
			if err := json.Unmarshal(data, &got); err != nil {
				t.Fatalf("json.Unmarshal failed: %v", err)
			}

			if got.Type != tt.content.Type {
				t.Errorf("Type = %s, want %s", got.Type, tt.content.Type)
			}
		})
	}
}

// TestResponsesStreamEvent_Types tests different stream event types.
func TestResponsesStreamEvent_Types(t *testing.T) {
	tests := []struct {
		name  string
		event ResponsesStreamEvent
	}{
		{
			name: "response.created",
			event: ResponsesStreamEvent{
				Type: "response.created",
				Response: &ResponsesResponse{
					ID:     "resp_new",
					Status: "in_progress",
				},
			},
		},
		{
			name: "response.output_text.delta",
			event: ResponsesStreamEvent{
				Type:         "response.output_text.delta",
				ItemID:       "msg_1",
				ContentIndex: 0,
				Delta:        "Hello",
			},
		},
		{
			name: "response.function_call_arguments.delta",
			event: ResponsesStreamEvent{
				Type:         "response.function_call_arguments.delta",
				ItemID:       "msg_1",
				ContentIndex: 1,
				Delta:        `{"locat`,
			},
		},
		{
			name: "error",
			event: ResponsesStreamEvent{
				Type: "error",
				Error: &ResponsesError{
					Code:    "invalid_request",
					Message: "Invalid request",
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.event)
			if err != nil {
				t.Fatalf("json.Marshal failed: %v", err)
			}

			var got ResponsesStreamEvent
			if err := json.Unmarshal(data, &got); err != nil {
				t.Fatalf("json.Unmarshal failed: %v", err)
			}

			if got.Type != tt.event.Type {
				t.Errorf("Type = %s, want %s", got.Type, tt.event.Type)
			}
		})
	}
}

// TestInputItem_MarshalUnmarshal tests input item marshaling.
func TestInputItem_MarshalUnmarshal(t *testing.T) {
	tests := []struct {
		name string
		item InputItem
	}{
		{
			name: "simple message",
			item: InputItem{
				Type:    "message",
				Role:    "user",
				Content: "Hello",
			},
		},
		{
			name: "message with content parts",
			item: InputItem{
				Type: "message",
				Role: "user",
				Content: []ContentPart{
					{Type: "input_text", Text: "Hello"},
					{Type: "input_image", ImageURL: "https://example.com/image.png"},
				},
			},
		},
		{
			name: "file input",
			item: InputItem{
				Type: "file",
				Content: ContentPart{
					Type: "input_file",
					FileData: &FileData{
						Filename: "data.txt",
						FileData: "base64encoded...",
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.item)
			if err != nil {
				t.Fatalf("json.Marshal failed: %v", err)
			}

			var got InputItem
			if err := json.Unmarshal(data, &got); err != nil {
				t.Fatalf("json.Unmarshal failed: %v", err)
			}

			if got.Type != tt.item.Type {
				t.Errorf("Type = %s, want %s", got.Type, tt.item.Type)
			}
		})
	}
}

// TestResponsesUsage_WithDetails tests usage with token details.
func TestResponsesUsage_WithDetails(t *testing.T) {
	usage := ResponsesUsage{
		InputTokens:  100,
		OutputTokens: 50,
		TotalTokens:  150,
		InputTokensDetails: &InputTokensDetails{
			CachedTokens: 20,
		},
		OutputTokensDetails: &OutputTokensDetails{
			ReasoningTokens: 15,
		},
	}

	data, err := json.Marshal(usage)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}

	var got ResponsesUsage
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}

	if got.InputTokensDetails == nil {
		t.Fatal("InputTokensDetails is nil")
	}

	if got.InputTokensDetails.CachedTokens != 20 {
		t.Errorf("InputTokensDetails.CachedTokens = %d, want 20", got.InputTokensDetails.CachedTokens)
	}

	if got.OutputTokensDetails == nil {
		t.Fatal("OutputTokensDetails is nil")
	}

	if got.OutputTokensDetails.ReasoningTokens != 15 {
		t.Errorf("OutputTokensDetails.ReasoningTokens = %d, want 15", got.OutputTokensDetails.ReasoningTokens)
	}
}

// TestReasoningConfig_MarshalUnmarshal tests reasoning config.
func TestReasoningConfig_MarshalUnmarshal(t *testing.T) {
	config := ReasoningConfig{
		Summary: "detailed",
	}

	data, err := json.Marshal(config)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}

	var got ReasoningConfig
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}

	if got.Summary != "detailed" {
		t.Errorf("Summary = %s, want detailed", got.Summary)
	}
}

// TestResponsesTool_ComputerUse tests computer use tool.
func TestResponsesTool_ComputerUse(t *testing.T) {
	tool := ResponsesTool{
		Type:          "computer_use_preview",
		Name:          "browser",
		DisplayWidth:  1280,
		DisplayHeight: 800,
		Environment:   "browser",
	}

	data, err := json.Marshal(tool)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}

	var got ResponsesTool
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}

	if got.Type != "computer_use_preview" {
		t.Errorf("Type = %s, want computer_use_preview", got.Type)
	}

	if got.DisplayWidth != 1280 {
		t.Errorf("DisplayWidth = %d, want 1280", got.DisplayWidth)
	}
}

// TestAnnotation_Types tests different annotation types.
func TestAnnotation_Types(t *testing.T) {
	tests := []struct {
		name       string
		annotation Annotation
	}{
		{
			name: "url_citation",
			annotation: Annotation{
				Type:  "url_citation",
				URL:   "https://example.com",
				Title: "Example Site",
			},
		},
		{
			name: "file_citation",
			annotation: Annotation{
				Type:   "file_citation",
				FileID: "file_123",
			},
		},
		{
			name: "file_path",
			annotation: Annotation{
				Type:   "file_path",
				FileID: "file_456",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.annotation)
			if err != nil {
				t.Fatalf("json.Marshal failed: %v", err)
			}

			var got Annotation
			if err := json.Unmarshal(data, &got); err != nil {
				t.Fatalf("json.Unmarshal failed: %v", err)
			}

			if got.Type != tt.annotation.Type {
				t.Errorf("Type = %s, want %s", got.Type, tt.annotation.Type)
			}
		})
	}
}

// TestOutputItem_WithToolCall tests output item with tool call.
func TestOutputItem_WithToolCall(t *testing.T) {
	item := OutputItem{
		Type:   "message",
		ID:     "msg_789",
		Status: "completed",
		Role:   "assistant",
		Content: []OutputContent{
			{
				Type:      "function_call",
				ID:        "call_abc",
				CallID:    "call_abc",
				Name:      "get_weather",
				Arguments: `{"location":"San Francisco"}`,
			},
		},
	}

	data, err := json.Marshal(item)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}

	var got OutputItem
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}

	if len(got.Content) != 1 {
		t.Fatalf("len(Content) = %d, want 1", len(got.Content))
	}

	if got.Content[0].Type != "function_call" {
		t.Errorf("Content[0].Type = %s, want function_call", got.Content[0].Type)
	}

	if got.Content[0].Name != "get_weather" {
		t.Errorf("Content[0].Name = %s, want get_weather", got.Content[0].Name)
	}
}

// TestSafetyCheck_MarshalUnmarshal tests safety check.
func TestSafetyCheck_MarshalUnmarshal(t *testing.T) {
	check := SafetyCheck{
		Code:    "dangerous_action",
		Message: "This action may be dangerous",
	}

	data, err := json.Marshal(check)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}

	var got SafetyCheck
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}

	if got.Code != check.Code {
		t.Errorf("Code = %s, want %s", got.Code, check.Code)
	}
}

// TestParallelToolCalls tests parallel tool calls setting.
func TestParallelToolCalls(t *testing.T) {
	request := ResponsesRequest{
		Model:             "gpt-4o",
		Input:             "Call multiple tools",
		ParallelToolCalls: true,
		Stream:            boolPtr(true),
	}

	data, err := json.Marshal(request)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}

	var got ResponsesRequest
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}

	if !got.ParallelToolCalls {
		t.Error("ParallelToolCalls should be true")
	}
}

// TestMaxOutputTokens tests max output tokens.
func TestMaxOutputTokens(t *testing.T) {
	request := ResponsesRequest{
		Model:           "gpt-4o",
		Input:           "Hello",
		MaxOutputTokens: 500,
		Stream:          boolPtr(true),
	}

	data, err := json.Marshal(request)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}

	var got ResponsesRequest
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}

	if got.MaxOutputTokens != 500 {
		t.Errorf("MaxOutputTokens = %d, want 500", got.MaxOutputTokens)
	}
}

// BenchmarkResponsesRequest_Marshal benchmarks request marshaling.
func BenchmarkResponsesRequest_Marshal(b *testing.B) {
	request := ResponsesRequest{
		Model: "gpt-4o",
		Input: []InputItem{
			{Type: "message", Role: "user", Content: "Hello"},
		},
		Tools: []ResponsesTool{
			{
				Type: "function",
				Function: &ResponsesToolFunction{
					Name:       "test",
					Parameters: json.RawMessage(`{}`),
				},
			},
		},
		Stream: boolPtr(true),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := json.Marshal(request)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkResponsesResponse_Unmarshal benchmarks response unmarshaling.
func BenchmarkResponsesResponse_Unmarshal(b *testing.B) {
	data := []byte(`{
		"id": "resp_test",
		"object": "response",
		"created_at": 1234567890,
		"status": "completed",
		"model": "gpt-4o",
		"output": [{
			"type": "message",
			"id": "msg_1",
			"status": "completed",
			"role": "assistant",
			"content": [{"type": "output_text", "text": "Hello"}]
		}]
	}`)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var resp ResponsesResponse
		if err := json.Unmarshal(data, &resp); err != nil {
			b.Fatal(err)
		}
	}
}
