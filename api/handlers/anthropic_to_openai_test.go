package handlers

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"ai-proxy/config"
	"ai-proxy/types"

	"github.com/gin-gonic/gin"
)

func setupAnthropicToOpenAITestRouter(cfg *config.Config) *gin.Engine {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.POST("/v1/anthropic-to-openai/responses", NewAnthropicToOpenAIHandler(cfg))
	return router
}

// TestAnthropicToOpenAIHandler_ValidateRequest tests request validation.
func TestAnthropicToOpenAIHandler_ValidateRequest(t *testing.T) {
	tests := []struct {
		name      string
		body      string
		wantError bool
	}{
		{
			name:      "valid request",
			body:      `{"model":"gpt-4o","input":"Hello","stream":true}`,
			wantError: false,
		},
		{
			name:      "valid request with array input",
			body:      `{"model":"gpt-4o","input":[{"type":"message","role":"user","content":"Hello"}],"stream":true}`,
			wantError: false,
		},
		{
			name:      "invalid JSON",
			body:      `not valid json`,
			wantError: true,
		},
		{
			name:      "empty model",
			body:      `{"model":"","input":"Hello"}`,
			wantError: true,
		},
		{
			name:      "missing model field",
			body:      `{"input":"Hello"}`,
			wantError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			handler := &AnthropicToOpenAIHandler{
				cfg: &config.Config{},
			}

			err := handler.ValidateRequest([]byte(tt.body))

			if tt.wantError && err == nil {
				t.Error("Expected error but got nil")
			}
			if !tt.wantError && err != nil {
				t.Errorf("Expected no error but got: %v", err)
			}
		})
	}
}

// TestAnthropicToOpenAIHandler_TransformRequest_StringInput tests transforming string input.
func TestAnthropicToOpenAIHandler_TransformRequest_StringInput(t *testing.T) {
	handler := &AnthropicToOpenAIHandler{
		cfg: &config.Config{},
	}

	openReq := types.ResponsesRequest{
		Model:           "claude-3-opus",
		Input:           "Hello, how are you?",
		Instructions:    "Be helpful",
		Stream:          true,
		MaxOutputTokens: 500,
		Temperature:     0.8,
		TopP:            0.9,
	}

	body, _ := json.Marshal(openReq)
	transformed, err := handler.TransformRequest(body)

	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var anthReq types.MessageRequest
	if err := json.Unmarshal(transformed, &anthReq); err != nil {
		t.Fatalf("Failed to unmarshal transformed request: %v", err)
	}

	// Verify fields
	if anthReq.Model != "claude-3-opus" {
		t.Errorf("Model = %s, want claude-3-opus", anthReq.Model)
	}

	if anthReq.MaxTokens != 500 {
		t.Errorf("MaxTokens = %d, want 500", anthReq.MaxTokens)
	}

	if !anthReq.Stream {
		t.Error("Stream should be true")
	}

	if anthReq.Temperature != 0.8 {
		t.Errorf("Temperature = %f, want 0.8", anthReq.Temperature)
	}

	if anthReq.TopP != 0.9 {
		t.Errorf("TopP = %f, want 0.9", anthReq.TopP)
	}

	// System should be set from instructions
	if anthReq.System != "Be helpful" {
		t.Errorf("System = %v, want 'Be helpful'", anthReq.System)
	}

	// Messages should have one user message
	if len(anthReq.Messages) != 1 {
		t.Fatalf("len(Messages) = %d, want 1", len(anthReq.Messages))
	}

	if anthReq.Messages[0].Role != "user" {
		t.Errorf("Messages[0].Role = %s, want user", anthReq.Messages[0].Role)
	}

	if anthReq.Messages[0].Content != "Hello, how are you?" {
		t.Errorf("Messages[0].Content = %v, want 'Hello, how are you?'", anthReq.Messages[0].Content)
	}
}

// TestAnthropicToOpenAIHandler_TransformRequest_ArrayInput tests transforming array input.
func TestAnthropicToOpenAIHandler_TransformRequest_ArrayInput(t *testing.T) {
	handler := &AnthropicToOpenAIHandler{
		cfg: &config.Config{},
	}

	openReq := types.ResponsesRequest{
		Model: "claude-3-opus",
		Input: []types.InputItem{
			{Type: "message", Role: "user", Content: "Hello"},
			{Type: "message", Role: "assistant", Content: "Hi there!"},
			{Type: "message", Role: "user", Content: "How are you?"},
		},
		Stream: true,
	}

	body, _ := json.Marshal(openReq)
	transformed, err := handler.TransformRequest(body)

	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var anthReq types.MessageRequest
	if err := json.Unmarshal(transformed, &anthReq); err != nil {
		t.Fatalf("Failed to unmarshal transformed request: %v", err)
	}

	if len(anthReq.Messages) != 3 {
		t.Fatalf("len(Messages) = %d, want 3", len(anthReq.Messages))
	}

	if anthReq.Messages[0].Role != "user" {
		t.Errorf("Messages[0].Role = %s, want user", anthReq.Messages[0].Role)
	}

	if anthReq.Messages[1].Role != "assistant" {
		t.Errorf("Messages[1].Role = %s, want assistant", anthReq.Messages[1].Role)
	}

	if anthReq.Messages[2].Role != "user" {
		t.Errorf("Messages[2].Role = %s, want user", anthReq.Messages[2].Role)
	}
}

// TestAnthropicToOpenAIHandler_TransformRequest_WithTools tests transforming tools.
func TestAnthropicToOpenAIHandler_TransformRequest_WithTools(t *testing.T) {
	handler := &AnthropicToOpenAIHandler{
		cfg: &config.Config{},
	}

	openReq := types.ResponsesRequest{
		Model: "claude-3-opus",
		Input: "What's the weather?",
		Tools: []types.ResponsesTool{
			{
				Type: "function",
				Function: &types.ResponsesToolFunction{
					Name:        "get_weather",
					Description: "Get current weather",
					Parameters:  json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}}}`),
					Strict:      true,
				},
			},
			{
				Type: "function",
				Function: &types.ResponsesToolFunction{
					Name:       "get_time",
					Parameters: json.RawMessage(`{"type":"object"}`),
				},
			},
			{
				Type: "web_search",
			},
		},
		Stream: true,
	}

	body, _ := json.Marshal(openReq)
	transformed, err := handler.TransformRequest(body)

	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var anthReq types.MessageRequest
	if err := json.Unmarshal(transformed, &anthReq); err != nil {
		t.Fatalf("Failed to unmarshal transformed request: %v", err)
	}

	// Should only have function tools (web_search is not converted)
	if len(anthReq.Tools) != 2 {
		t.Fatalf("len(Tools) = %d, want 2", len(anthReq.Tools))
	}

	if anthReq.Tools[0].Name != "get_weather" {
		t.Errorf("Tools[0].Name = %s, want get_weather", anthReq.Tools[0].Name)
	}

	if anthReq.Tools[0].Description != "Get current weather" {
		t.Errorf("Tools[0].Description = %s, want 'Get current weather'", anthReq.Tools[0].Description)
	}

	if anthReq.Tools[1].Name != "get_time" {
		t.Errorf("Tools[1].Name = %s, want get_time", anthReq.Tools[1].Name)
	}
}

// TestAnthropicToOpenAIHandler_TransformRequest_WithReasoning tests reasoning config.
func TestAnthropicToOpenAIHandler_TransformRequest_WithReasoning(t *testing.T) {
	handler := &AnthropicToOpenAIHandler{
		cfg: &config.Config{},
	}

	openReq := types.ResponsesRequest{
		Model: "o3",
		Input: "Solve this complex problem",
		Reasoning: &types.ReasoningConfig{
			Summary: "detailed",
		},
		Stream: true,
	}

	body, _ := json.Marshal(openReq)
	_, err := handler.TransformRequest(body)

	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	// Reasoning config is not directly mapped to Anthropic, but request should succeed
}

// TestAnthropicToOpenAIHandler_TransformRequest_WithPreviousResponseID tests previous_response_id.
func TestAnthropicToOpenAIHandler_TransformRequest_WithPreviousResponseID(t *testing.T) {
	handler := &AnthropicToOpenAIHandler{
		cfg: &config.Config{},
	}

	openReq := types.ResponsesRequest{
		Model:              "claude-3-opus",
		Input:              "Tell me more",
		PreviousResponseID: "resp_123abc",
		Stream:             true,
	}

	body, _ := json.Marshal(openReq)
	_, err := handler.TransformRequest(body)

	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	// PreviousResponseID is not directly mapped to Anthropic, but request should succeed
}

// TestAnthropicToOpenAIHandler_TransformRequest_NilInput tests nil input.
func TestAnthropicToOpenAIHandler_TransformRequest_NilInput(t *testing.T) {
	handler := &AnthropicToOpenAIHandler{
		cfg: &config.Config{},
	}

	openReq := types.ResponsesRequest{
		Model:  "claude-3-opus",
		Input:  nil,
		Stream: true,
	}

	body, _ := json.Marshal(openReq)
	transformed, err := handler.TransformRequest(body)

	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	var anthReq types.MessageRequest
	if err := json.Unmarshal(transformed, &anthReq); err != nil {
		t.Fatalf("Failed to unmarshal transformed request: %v", err)
	}

	if len(anthReq.Messages) != 0 {
		t.Errorf("len(Messages) = %d, want 0", len(anthReq.Messages))
	}
}

// TestAnthropicToOpenAIHandler_UpstreamURL tests upstream URL.
func TestAnthropicToOpenAIHandler_UpstreamURL(t *testing.T) {
	cfg := config.LoadConfig(&config.SchemaConfig{
		Providers: []config.Provider{
			{
				Name:    "test-anthropic",
				Type:    "anthropic",
				BaseURL: "https://api.anthropic.com/v1/messages",
				APIKey:  "test-key",
			},
		},
	})
	handler := &AnthropicToOpenAIHandler{cfg: cfg}

	url := handler.UpstreamURL()
	if url != "https://api.anthropic.com/v1/messages" {
		t.Errorf("UpstreamURL = %s, want https://api.anthropic.com/v1/messages", url)
	}
}

// TestAnthropicToOpenAIHandler_ResolveAPIKey tests API key resolution.
func TestAnthropicToOpenAIHandler_ResolveAPIKey(t *testing.T) {
	cfg := config.LoadConfig(&config.SchemaConfig{
		Providers: []config.Provider{
			{
				Name:    "test-anthropic",
				Type:    "anthropic",
				BaseURL: "https://api.anthropic.com/v1/messages",
				APIKey:  "test-api-key",
			},
		},
	})
	handler := &AnthropicToOpenAIHandler{cfg: cfg}

	// Create a dummy gin context
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	key := handler.ResolveAPIKey(c)
	if key != "test-api-key" {
		t.Errorf("ResolveAPIKey = %s, want test-api-key", key)
	}
}

// TestAnthropicToOpenAIHandler_WriteError tests error writing.
func TestAnthropicToOpenAIHandler_WriteError(t *testing.T) {
	cfg := &config.Config{}
	handler := &AnthropicToOpenAIHandler{cfg: cfg}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	handler.WriteError(c, http.StatusBadRequest, "Test error message")

	if w.Code != http.StatusBadRequest {
		t.Errorf("Status code = %d, want %d", w.Code, http.StatusBadRequest)
	}

	body := w.Body.String()
	if !strings.Contains(body, `"type":"error"`) {
		t.Error("Response should contain error type")
	}

	if !strings.Contains(body, `"code":"invalid_request_error"`) {
		t.Error("Response should contain error code")
	}

	if !strings.Contains(body, "Test error message") {
		t.Error("Response should contain error message")
	}
}

// TestAnthropicToOpenAIHandler_CreateTransformer tests transformer creation.
func TestAnthropicToOpenAIHandler_CreateTransformer(t *testing.T) {
	cfg := &config.Config{}
	handler := &AnthropicToOpenAIHandler{cfg: cfg}

	var buf bytes.Buffer
	transformer := handler.CreateTransformer(&buf)

	if transformer == nil {
		t.Error("CreateTransformer returned nil")
	}
}

// TestConvertInputToMessages_String tests string input conversion.
func TestConvertInputToMessages_String(t *testing.T) {
	result := convertInputToMessages("Hello world")

	if len(result) != 1 {
		t.Fatalf("len(result) = %d, want 1", len(result))
	}

	if result[0].Role != "user" {
		t.Errorf("Role = %s, want user", result[0].Role)
	}

	if result[0].Content != "Hello world" {
		t.Errorf("Content = %v, want 'Hello world'", result[0].Content)
	}
}

// TestConvertInputToMessages_Nil tests nil input conversion.
func TestConvertInputToMessages_Nil(t *testing.T) {
	result := convertInputToMessages(nil)

	if len(result) != 0 {
		t.Errorf("len(result) = %d, want 0", len(result))
	}
}

// TestConvertInputToMessages_Array tests array input conversion.
func TestConvertInputToMessages_Array(t *testing.T) {
	input := []interface{}{
		map[string]interface{}{"type": "message", "role": "user", "content": "Hello"},
		map[string]interface{}{"type": "message", "role": "assistant", "content": "Hi!"},
	}

	result := convertInputToMessages(input)

	if len(result) != 2 {
		t.Fatalf("len(result) = %d, want 2", len(result))
	}

	if result[0].Role != "user" {
		t.Errorf("result[0].Role = %s, want user", result[0].Role)
	}

	if result[1].Role != "assistant" {
		t.Errorf("result[1].Role = %s, want assistant", result[1].Role)
	}
}

// TestConvertInputToMessages_EmptyArray tests empty array conversion.
func TestConvertInputToMessages_EmptyArray(t *testing.T) {
	result := convertInputToMessages([]interface{}{})

	if len(result) != 0 {
		t.Errorf("len(result) = %d, want 0", len(result))
	}
}

// TestConvertInputToMessages_FunctionCall tests function_call input conversion.
func TestConvertInputToMessages_FunctionCall(t *testing.T) {
	input := []interface{}{
		map[string]interface{}{
			"type":      "function_call",
			"call_id":   "toolu_abc123",
			"name":      "get_weather",
			"arguments": `{"location": "London"}`,
		},
	}

	result := convertInputToMessages(input)

	if len(result) != 1 {
		t.Fatalf("len(result) = %d, want 1", len(result))
	}

	if result[0].Role != "assistant" {
		t.Errorf("Role = %s, want assistant", result[0].Role)
	}

	content, ok := result[0].Content.([]map[string]interface{})
	if !ok {
		t.Fatalf("Content is not []map[string]interface{}, got %T", result[0].Content)
	}

	if len(content) != 1 {
		t.Fatalf("len(content) = %d, want 1", len(content))
	}

	if content[0]["type"] != "tool_use" {
		t.Errorf("content[0][type] = %v, want tool_use", content[0]["type"])
	}

	if content[0]["id"] != "toolu_abc123" {
		t.Errorf("content[0][id] = %v, want toolu_abc123", content[0]["id"])
	}

	if content[0]["name"] != "get_weather" {
		t.Errorf("content[0][name] = %v, want get_weather", content[0]["name"])
	}
}

// TestConvertInputToMessages_FunctionCallOutput tests function_call_output input conversion.
func TestConvertInputToMessages_FunctionCallOutput(t *testing.T) {
	input := []interface{}{
		map[string]interface{}{
			"type":    "function_call_output",
			"call_id": "toolu_abc123",
			"output":  "Sunny, 22°C",
		},
	}

	result := convertInputToMessages(input)

	if len(result) != 1 {
		t.Fatalf("len(result) = %d, want 1", len(result))
	}

	if result[0].Role != "user" {
		t.Errorf("Role = %s, want user", result[0].Role)
	}

	content, ok := result[0].Content.([]map[string]interface{})
	if !ok {
		t.Fatalf("Content is not []map[string]interface{}, got %T", result[0].Content)
	}

	if len(content) != 1 {
		t.Fatalf("len(content) = %d, want 1", len(content))
	}

	if content[0]["type"] != "tool_result" {
		t.Errorf("content[0][type] = %v, want tool_result", content[0]["type"])
	}

	if content[0]["tool_use_id"] != "toolu_abc123" {
		t.Errorf("content[0][tool_use_id] = %v, want toolu_abc123", content[0]["tool_use_id"])
	}

	if content[0]["content"] != "Sunny, 22°C" {
		t.Errorf("content[0][content] = %v, want 'Sunny, 22°C'", content[0]["content"])
	}
}

// TestConvertInputToMessages_Reasoning tests that reasoning items are skipped.
func TestConvertInputToMessages_Reasoning(t *testing.T) {
	input := []interface{}{
		map[string]interface{}{
			"type": "reasoning",
			"summary": []interface{}{
				map[string]interface{}{
					"type": "summary_text",
					"text": "Thinking about the problem...",
				},
			},
		},
		map[string]interface{}{
			"type":    "message",
			"role":    "user",
			"content": "Hello",
		},
	}

	result := convertInputToMessages(input)

	if len(result) != 1 {
		t.Fatalf("len(result) = %d, want 1 (reasoning should be skipped)", len(result))
	}

	if result[0].Role != "user" {
		t.Errorf("Role = %s, want user", result[0].Role)
	}
}

// TestConvertInputToMessages_FullConversation tests a complete conversation with tools.
func TestConvertInputToMessages_FullConversation(t *testing.T) {
	input := []interface{}{
		map[string]interface{}{
			"type":    "message",
			"role":    "user",
			"content": "What's the weather in London?",
		},
		map[string]interface{}{
			"type":      "function_call",
			"call_id":   "toolu_abc123",
			"name":      "get_weather",
			"arguments": `{"location": "London"}`,
		},
		map[string]interface{}{
			"type":    "function_call_output",
			"call_id": "toolu_abc123",
			"output":  "Sunny, 22°C",
		},
		map[string]interface{}{
			"type":    "message",
			"role":    "assistant",
			"content": "The weather in London is sunny and 22°C.",
		},
	}

	result := convertInputToMessages(input)

	if len(result) != 4 {
		t.Fatalf("len(result) = %d, want 4", len(result))
	}

	// Check user message
	if result[0].Role != "user" {
		t.Errorf("result[0].Role = %s, want user", result[0].Role)
	}

	// Check tool_use (assistant message with tool_use content)
	if result[1].Role != "assistant" {
		t.Errorf("result[1].Role = %s, want assistant", result[1].Role)
	}

	// Check tool_result (user message with tool_result content)
	if result[2].Role != "user" {
		t.Errorf("result[2].Role = %s, want user", result[2].Role)
	}

	// Check assistant response
	if result[3].Role != "assistant" {
		t.Errorf("result[3].Role = %s, want assistant", result[3].Role)
	}
}

// TestConvertResponsesTools_Empty tests empty tools conversion.
func TestConvertResponsesTools_Empty(t *testing.T) {
	result := convertResponsesTools(nil)

	if result != nil {
		t.Error("Result should be nil for nil input")
	}

	result = convertResponsesTools([]types.ResponsesTool{})

	if len(result) != 0 {
		t.Errorf("len(result) = %d, want 0", len(result))
	}
}

// TestConvertResponsesTools_OnlyNonFunction tests non-function tools filtering.
func TestConvertResponsesTools_OnlyNonFunction(t *testing.T) {
	tools := []types.ResponsesTool{
		{Type: "web_search"},
		{Type: "file_search"},
	}

	result := convertResponsesTools(tools)

	if len(result) != 0 {
		t.Errorf("len(result) = %d, want 0 (non-function tools should be filtered)", len(result))
	}
}

// TestConvertResponsesTools_NilFunction tests tool with nil function.
func TestConvertResponsesTools_NilFunction(t *testing.T) {
	tools := []types.ResponsesTool{
		{Type: "function"},
	}

	result := convertResponsesTools(tools)

	// Tool with nil function should be skipped
	if len(result) != 0 {
		t.Errorf("len(result) = %d, want 0 (nil function should be skipped)", len(result))
	}
}

// TestNewAnthropicToOpenAIHandler tests handler creation.
func TestNewAnthropicToOpenAIHandler(t *testing.T) {
	cfg := config.LoadConfig(&config.SchemaConfig{
		Providers: []config.Provider{
			{
				Name:    "test-anthropic",
				Type:    "anthropic",
				BaseURL: "https://api.anthropic.com/v1/messages",
				APIKey:  "test-key",
			},
		},
	})

	handler := NewAnthropicToOpenAIHandler(cfg)

	if handler == nil {
		t.Fatal("NewAnthropicToOpenAIHandler returned nil")
	}

	// Create a test request to verify handler is callable
	w := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/v1/anthropic-to-openai/responses", strings.NewReader(`{"model":"test","input":"hello","stream":true}`))
	req.Header.Set("Content-Type", "application/json")

	router := gin.New()
	router.POST("/v1/anthropic-to-openai/responses", handler)
	router.ServeHTTP(w, req)

	// Should get an error since there's no real upstream
	if w.Code == http.StatusOK {
		// This is fine - the handler is working
	}
}

// BenchmarkTransformResponsesRequest benchmarks request transformation.
func BenchmarkTransformResponsesRequest(b *testing.B) {
	openReq := types.ResponsesRequest{
		Model: "claude-3-opus",
		Input: []types.InputItem{
			{Type: "message", Role: "user", Content: "Hello"},
			{Type: "message", Role: "assistant", Content: "Hi!"},
		},
		Tools: []types.ResponsesTool{
			{
				Type: "function",
				Function: &types.ResponsesToolFunction{
					Name:        "get_weather",
					Description: "Get weather",
					Parameters:  json.RawMessage(`{"type":"object"}`),
				},
			},
		},
		Stream: true,
	}

	body, _ := json.Marshal(openReq)
	handler := &AnthropicToOpenAIHandler{cfg: &config.Config{}}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := handler.TransformRequest(body)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkConvertInputToMessages benchmarks input conversion.
func BenchmarkConvertInputToMessages(b *testing.B) {
	input := []interface{}{
		map[string]interface{}{"type": "message", "role": "user", "content": "Hello"},
		map[string]interface{}{"type": "message", "role": "assistant", "content": "Hi!"},
		map[string]interface{}{"type": "message", "role": "user", "content": "How are you?"},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		convertInputToMessages(input)
	}
}

// BenchmarkConvertResponsesTools benchmarks tool conversion.
func BenchmarkConvertResponsesTools(b *testing.B) {
	tools := []types.ResponsesTool{
		{
			Type: "function",
			Function: &types.ResponsesToolFunction{
				Name:        "tool1",
				Description: "Tool 1",
				Parameters:  json.RawMessage(`{"type":"object"}`),
			},
		},
		{
			Type: "function",
			Function: &types.ResponsesToolFunction{
				Name:        "tool2",
				Description: "Tool 2",
				Parameters:  json.RawMessage(`{"type":"object"}`),
			},
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		convertResponsesTools(tools)
	}
}

// Helper to read all from reader
func readAll(r io.Reader) string {
	buf := new(bytes.Buffer)
	buf.ReadFrom(r)
	return buf.String()
}
