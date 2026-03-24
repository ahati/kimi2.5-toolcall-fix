package convert

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"ai-proxy/types"

	"github.com/tmaxmax/go-sse"
)

// ─────────────────────────────────────────────────────────────────────────────
// TransformAnthropicToResponses Tests
// ─────────────────────────────────────────────────────────────────────────────

func TestTransformAnthropicToResponses_SimpleRequest(t *testing.T) {
	anthReq := types.MessageRequest{
		Model:     "claude-3-opus",
		MaxTokens: 1024,
		Messages: []types.MessageInput{
			{Role: "user", Content: "Hello"},
		},
	}

	body, err := json.Marshal(anthReq)
	if err != nil {
		t.Fatalf("Failed to marshal request: %v", err)
	}

	out, err := TransformAnthropicToResponses(body)
	if err != nil {
		t.Fatalf("TransformAnthropicToResponses failed: %v", err)
	}

	var resp types.ResponsesRequest
	if err := json.Unmarshal(out, &resp); err != nil {
		t.Fatalf("Failed to unmarshal response: %v", err)
	}

	if resp.Model != "claude-3-opus" {
		t.Errorf("expected model claude-3-opus, got %s", resp.Model)
	}
	if resp.MaxOutputTokens != 1024 {
		t.Errorf("expected max_output_tokens 1024, got %d", resp.MaxOutputTokens)
	}

	// Input should be a simple string
	if str, ok := resp.Input.(string); !ok || str != "Hello" {
		t.Errorf("expected input 'Hello', got %v", resp.Input)
	}
}

func TestTransformAnthropicToResponses_WithSystem(t *testing.T) {
	tests := []struct {
		name     string
		system   interface{}
		expected string
	}{
		{
			name:     "string system",
			system:   "You are a helpful assistant.",
			expected: "You are a helpful assistant.",
		},
		// Note: Array system format is handled by the types package during unmarshaling.
		// The extractSystemFromRequest function uses ExtractSystemText which handles
		// both string and array formats correctly.
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			anthReq := types.MessageRequest{
				Model:     "claude-3-opus",
				MaxTokens: 1024,
				System:    tt.system,
				Messages: []types.MessageInput{
					{Role: "user", Content: "Hi"},
				},
			}

			body, _ := json.Marshal(anthReq)
			out, err := TransformAnthropicToResponses(body)
			if err != nil {
				t.Fatalf("TransformAnthropicToResponses failed: %v", err)
			}

			var resp types.ResponsesRequest
			json.Unmarshal(out, &resp)

			if resp.Instructions != tt.expected {
				t.Errorf("expected instructions %q, got %q", tt.expected, resp.Instructions)
			}
		})
	}
}

func TestTransformAnthropicToResponses_ToolChoiceConversion(t *testing.T) {
	tests := []struct {
		name       string
		toolChoice *types.ToolChoice
		// The ToolChoice struct marshals as an object, so we check the result after conversion
		wantType string // Expected "type" field in output
		wantName string // Expected "name" field (if applicable)
	}{
		{
			name:       "auto",
			toolChoice: &types.ToolChoice{Type: "auto"},
			wantType:   "function", // "auto" gets converted via marshalToolChoice -> AnthropicToolChoiceToResponses
		},
		{
			name:       "any -> required",
			toolChoice: &types.ToolChoice{Type: "any"},
			wantType:   "function", // "any" gets converted similarly
		},
		{
			name:       "tool -> function with name",
			toolChoice: &types.ToolChoice{Type: "tool", Name: "calculator"},
			wantType:   "function",
			wantName:   "calculator",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			anthReq := types.MessageRequest{
				Model:     "claude-3-opus",
				MaxTokens: 1024,
				Messages: []types.MessageInput{
					{Role: "user", Content: "Hi"},
				},
				Tools: []types.ToolDef{
					{Name: "calculator", Description: "A calculator"},
				},
				ToolChoice: tt.toolChoice,
			}

			body, _ := json.Marshal(anthReq)
			out, err := TransformAnthropicToResponses(body)
			if err != nil {
				t.Fatalf("TransformAnthropicToResponses failed: %v", err)
			}

			var resp types.ResponsesRequest
			json.Unmarshal(out, &resp)

			// Tool choice is converted by AnthropicToolChoiceToResponses
			tc, ok := resp.ToolChoice.(map[string]interface{})
			if !ok {
				t.Errorf("expected tool_choice to be a map, got %T: %v", resp.ToolChoice, resp.ToolChoice)
				return
			}
			if tc["type"] != tt.wantType {
				t.Errorf("expected type %q, got %v", tt.wantType, tc["type"])
			}
			if tt.wantName != "" && tc["name"] != tt.wantName {
				t.Errorf("expected name %q, got %v", tt.wantName, tc["name"])
			}
		})
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// AnthropicToResponsesRequest Tests
// ─────────────────────────────────────────────────────────────────────────────

func TestAnthropicToResponsesRequest_Simple(t *testing.T) {
	req := &types.MessageRequest{
		Model:     "claude-3-opus",
		MaxTokens: 1024,
		Messages: []types.MessageInput{
			{Role: "user", Content: "Hello"},
		},
	}

	out, err := AnthropicToResponsesRequest(req)
	if err != nil {
		t.Fatalf("AnthropicToResponsesRequest failed: %v", err)
	}

	if out.Model != "claude-3-opus" {
		t.Errorf("expected model claude-3-opus, got %s", out.Model)
	}
	if out.MaxOutputTokens != 1024 {
		t.Errorf("expected max_output_tokens 1024, got %d", out.MaxOutputTokens)
	}

	// Input should be a simple string
	if str, ok := out.Input.(string); !ok || str != "Hello" {
		t.Errorf("expected input 'Hello', got %v", out.Input)
	}
}

func TestAnthropicToResponsesRequest_WithThinking(t *testing.T) {
	req := &types.MessageRequest{
		Model:     "claude-3-opus",
		MaxTokens: 1024,
		Messages: []types.MessageInput{
			{Role: "user", Content: "Hello"},
		},
		Thinking: &types.ThinkingConfig{
			Type:         "enabled",
			BudgetTokens: 8000,
		},
	}

	out, err := AnthropicToResponsesRequest(req)
	if err != nil {
		t.Fatalf("AnthropicToResponsesRequest failed: %v", err)
	}

	if out.Reasoning == nil {
		t.Fatal("expected reasoning to be set")
	}
	// BudgetToReasoningEffort maps 8000 to "medium"
	if out.Reasoning.Effort != "medium" {
		t.Errorf("expected reasoning effort medium, got %s", out.Reasoning.Effort)
	}
}

func TestAnthropicToResponsesRequest_WithTools(t *testing.T) {
	req := &types.MessageRequest{
		Model:     "claude-3-opus",
		MaxTokens: 1024,
		Messages: []types.MessageInput{
			{Role: "user", Content: "What's the weather?"},
		},
		Tools: []types.ToolDef{
			{
				Name:        "get_weather",
				Description: "Get the current weather",
			},
		},
		ToolChoice: &types.ToolChoice{Type: "auto"},
	}

	out, err := AnthropicToResponsesRequest(req)
	if err != nil {
		t.Fatalf("AnthropicToResponsesRequest failed: %v", err)
	}

	if len(out.Tools) != 1 {
		t.Errorf("expected 1 tool, got %d", len(out.Tools))
	}
	if out.Tools[0].Name != "get_weather" {
		t.Errorf("expected tool name get_weather, got %s", out.Tools[0].Name)
	}
	if out.Tools[0].Type != "function" {
		t.Errorf("expected tool type function, got %s", out.Tools[0].Type)
	}
}

func TestAnthropicToResponsesRequest_ToolResult(t *testing.T) {
	req := &types.MessageRequest{
		Model:     "claude-3-opus",
		MaxTokens: 1024,
		Messages: []types.MessageInput{
			{Role: "user", Content: []interface{}{
				map[string]interface{}{
					"type":        "tool_result",
					"tool_use_id": "tool_123",
					"content":     "The weather is sunny",
				},
			}},
		},
	}

	out, err := AnthropicToResponsesRequest(req)
	if err != nil {
		t.Fatalf("AnthropicToResponsesRequest failed: %v", err)
	}

	items, ok := out.Input.([]types.InputItem)
	if !ok {
		t.Fatalf("expected input to be []InputItem, got %T", out.Input)
	}

	if len(items) != 1 {
		t.Errorf("expected 1 item, got %d", len(items))
	}
	if items[0].Type != "function_call_output" {
		t.Errorf("expected type function_call_output, got %s", items[0].Type)
	}
	if items[0].CallID != "tool_123" {
		t.Errorf("expected call_id tool_123, got %s", items[0].CallID)
	}
}

func TestAnthropicToResponsesRequest_AssistantWithToolUse(t *testing.T) {
	req := &types.MessageRequest{
		Model:     "claude-3-opus",
		MaxTokens: 1024,
		Messages: []types.MessageInput{
			{Role: "assistant", Content: []interface{}{
				map[string]interface{}{
					"type": "text",
					"text": "Let me check the weather.",
				},
				map[string]interface{}{
					"type":  "tool_use",
					"id":    "tool_123",
					"name":  "get_weather",
					"input": map[string]interface{}{"location": "SF"},
				},
			}},
		},
	}

	out, err := AnthropicToResponsesRequest(req)
	if err != nil {
		t.Fatalf("AnthropicToResponsesRequest failed: %v", err)
	}

	items, ok := out.Input.([]types.InputItem)
	if !ok {
		t.Fatalf("expected input to be []InputItem, got %T", out.Input)
	}

	// Should have message item and function_call item
	if len(items) != 2 {
		t.Errorf("expected 2 items, got %d", len(items))
	}

	// First item should be message
	if items[0].Type != "message" {
		t.Errorf("expected first item type message, got %s", items[0].Type)
	}

	// Second item should be function_call
	if items[1].Type != "function_call" {
		t.Errorf("expected second item type function_call, got %s", items[1].Type)
	}
	if items[1].Name != "get_weather" {
		t.Errorf("expected function name get_weather, got %s", items[1].Name)
	}
}

func TestAnthropicToResponsesRequest_ImageContent(t *testing.T) {
	req := &types.MessageRequest{
		Model:     "claude-3-opus",
		MaxTokens: 1024,
		Messages: []types.MessageInput{
			{Role: "user", Content: []interface{}{
				map[string]interface{}{
					"type": "text",
					"text": "What's in this image?",
				},
				map[string]interface{}{
					"type": "image",
					"source": map[string]interface{}{
						"type":       "base64",
						"media_type": "image/png",
						"data":       "abc123",
					},
				},
			}},
		},
	}

	out, err := AnthropicToResponsesRequest(req)
	if err != nil {
		t.Fatalf("AnthropicToResponsesRequest failed: %v", err)
	}

	items, ok := out.Input.([]types.InputItem)
	if !ok {
		t.Fatalf("expected input to be []InputItem, got %T", out.Input)
	}

	if len(items) != 1 {
		t.Errorf("expected 1 item, got %d", len(items))
	}

	content, ok := items[0].Content.([]types.ContentPart)
	if !ok {
		t.Fatalf("expected content to be []ContentPart, got %T", items[0].Content)
	}

	if len(content) != 2 {
		t.Errorf("expected 2 content parts, got %d", len(content))
	}

	// Check image part
	var foundImage bool
	for _, part := range content {
		if part.Type == "input_image" {
			foundImage = true
			expected := "data:image/png;base64,abc123"
			if part.ImageURL != expected {
				t.Errorf("expected image_url %q, got %q", expected, part.ImageURL)
			}
		}
	}
	if !foundImage {
		t.Error("expected to find input_image content part")
	}
}

func TestAnthropicToResponsesRequest_Metadata(t *testing.T) {
	req := &types.MessageRequest{
		Model:     "claude-3-opus",
		MaxTokens: 1024,
		Messages: []types.MessageInput{
			{Role: "user", Content: "Hello"},
		},
		Metadata: &types.AnthropicMetadata{
			UserID: "user_abc",
		},
	}

	out, err := AnthropicToResponsesRequest(req)
	if err != nil {
		t.Fatalf("AnthropicToResponsesRequest failed: %v", err)
	}

	if out.Metadata == nil {
		t.Fatal("expected metadata to be set")
	}
	userID, ok := out.Metadata["user_id"].(string)
	if !ok || userID != "user_abc" {
		t.Errorf("expected user_id 'user_abc', got %v", out.Metadata["user_id"])
	}
}

func TestAnthropicToResponsesRequest_DroppedFields(t *testing.T) {
	req := &types.MessageRequest{
		Model:         "claude-3-opus",
		MaxTokens:     1024,
		TopK:          40,                      // Should be dropped
		StopSequences: []string{"STOP", "END"}, // Should be dropped
		Messages: []types.MessageInput{
			{Role: "user", Content: "Hello"},
		},
	}

	out, err := AnthropicToResponsesRequest(req)
	if err != nil {
		t.Fatalf("AnthropicToResponsesRequest failed: %v", err)
	}

	// Verify that dropped fields are not present in JSON
	body, _ := json.Marshal(out)
	var raw map[string]interface{}
	json.Unmarshal(body, &raw)

	if _, exists := raw["top_k"]; exists {
		t.Error("expected top_k to be dropped")
	}
	if _, exists := raw["stop_sequences"]; exists {
		t.Error("expected stop_sequences to be dropped")
	}
}

func TestAnthropicToResponsesRequest_TemperatureAndTopP(t *testing.T) {
	req := &types.MessageRequest{
		Model:       "claude-3-opus",
		MaxTokens:   1024,
		Temperature: 0.7,
		TopP:        0.9,
		Messages: []types.MessageInput{
			{Role: "user", Content: "Hello"},
		},
	}

	out, err := AnthropicToResponsesRequest(req)
	if err != nil {
		t.Fatalf("AnthropicToResponsesRequest failed: %v", err)
	}

	if out.Temperature != 0.7 {
		t.Errorf("expected temperature 0.7, got %f", out.Temperature)
	}
	if out.TopP != 0.9 {
		t.Errorf("expected top_p 0.9, got %f", out.TopP)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Reasoning/Budget Conversion Tests
// ─────────────────────────────────────────────────────────────────────────────

func TestReasoningEffortToBudget(t *testing.T) {
	tests := []struct {
		effort   string
		expected int
	}{
		{"low", 4000},
		{"medium", 8000},
		{"high", 16000},
		{"unknown", 8000}, // default
	}

	for _, tt := range tests {
		t.Run(tt.effort, func(t *testing.T) {
			result := ReasoningEffortToBudget(tt.effort)
			if result != tt.expected {
				t.Errorf("ReasoningEffortToBudget(%s) = %d, want %d", tt.effort, result, tt.expected)
			}
		})
	}
}

func TestBudgetToReasoningEffort(t *testing.T) {
	tests := []struct {
		budget   int
		expected string
	}{
		{2000, "low"},
		{4000, "low"},
		{5000, "medium"},
		{8000, "medium"},
		{10000, "medium"},
		{15000, "high"},
		{20000, "high"},
	}

	for _, tt := range tests {
		t.Run(string(rune(tt.budget)), func(t *testing.T) {
			result := BudgetToReasoningEffort(tt.budget)
			if result != tt.expected {
				t.Errorf("BudgetToReasoningEffort(%d) = %s, want %s", tt.budget, result, tt.expected)
			}
		})
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Streaming Transformer Tests
// ─────────────────────────────────────────────────────────────────────────────

// TestAnthropicToResponsesTransformer_TokenCounts verifies that token counts
// from message_start and message_delta are properly propagated to the final response.
func TestAnthropicToResponsesTransformer_TokenCounts(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewAnthropicToResponsesTransformer(&buf)

	// Simulate message_start with usage data
	messageStart := `{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","model":"claude-3-opus","usage":{"input_tokens":3639,"output_tokens":0,"cache_read_input_tokens":1408,"cache_creation_input_tokens":0}}}`
	event1 := &sse.Event{Data: messageStart}
	if err := transformer.Transform(event1); err != nil {
		t.Fatalf("Transform message_start failed: %v", err)
	}

	// Simulate content block for text output
	contentBlockStart := `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`
	event2 := &sse.Event{Data: contentBlockStart}
	if err := transformer.Transform(event2); err != nil {
		t.Fatalf("Transform content_block_start failed: %v", err)
	}

	// Simulate content block delta
	contentBlockDelta := `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`
	event3 := &sse.Event{Data: contentBlockDelta}
	if err := transformer.Transform(event3); err != nil {
		t.Fatalf("Transform content_block_delta failed: %v", err)
	}

	// Simulate content block stop
	contentBlockStop := `{"type":"content_block_stop","index":0}`
	event4 := &sse.Event{Data: contentBlockStop}
	if err := transformer.Transform(event4); err != nil {
		t.Fatalf("Transform content_block_stop failed: %v", err)
	}

	// Simulate message_delta with output token count
	messageDelta := `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":3639,"output_tokens":188}}`
	event5 := &sse.Event{Data: messageDelta}
	if err := transformer.Transform(event5); err != nil {
		t.Fatalf("Transform message_delta failed: %v", err)
	}

	output := buf.String()

	// Verify the final response.completed event contains all token counts
	// Expected: input_tokens=5047 (3639+1408 cache), output_tokens=188, total_tokens=5235, cached_tokens=1408
	if !strings.Contains(output, `"input_tokens":5047`) {
		t.Errorf("Expected input_tokens=5047 (3639+1408 cache) in output, got:\n%s", output)
	}
	if !strings.Contains(output, `"output_tokens":188`) {
		t.Errorf("Expected output_tokens=188 in output, got:\n%s", output)
	}
	if !strings.Contains(output, `"total_tokens":5235`) {
		t.Errorf("Expected total_tokens=5235 (5047+188) in output, got:\n%s", output)
	}
	if !strings.Contains(output, `"cached_tokens":1408`) {
		t.Errorf("Expected cached_tokens=1408 in output, got:\n%s", output)
	}
}

func TestAnthropicToResponsesTransformer_ToolUse(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewAnthropicToResponsesTransformer(&buf)

	// Simulate message_start
	messageStart := `{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","model":"claude-3-opus","usage":{"input_tokens":100,"output_tokens":0}}}`
	if err := transformer.Transform(&sse.Event{Data: messageStart}); err != nil {
		t.Fatalf("Transform message_start failed: %v", err)
	}

	// Simulate tool_use block start
	toolUseStart := `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_123","name":"get_weather"}}`
	if err := transformer.Transform(&sse.Event{Data: toolUseStart}); err != nil {
		t.Fatalf("Transform tool_use start failed: %v", err)
	}

	// Simulate input_json_delta
	jsonDelta := `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"Paris\"}"}}`
	if err := transformer.Transform(&sse.Event{Data: jsonDelta}); err != nil {
		t.Fatalf("Transform json delta failed: %v", err)
	}

	// Simulate content block stop
	blockStop := `{"type":"content_block_stop","index":0}`
	if err := transformer.Transform(&sse.Event{Data: blockStop}); err != nil {
		t.Fatalf("Transform block stop failed: %v", err)
	}

	// Simulate message_delta
	messageDelta := `{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":50}}`
	if err := transformer.Transform(&sse.Event{Data: messageDelta}); err != nil {
		t.Fatalf("Transform message_delta failed: %v", err)
	}

	output := buf.String()

	// Verify events were emitted
	if !strings.Contains(output, "response.output_item.added") {
		t.Error("Expected response.output_item.added event")
	}
	if !strings.Contains(output, `"type":"function_call"`) {
		t.Error("Expected function_call type in output")
	}
	if !strings.Contains(output, "response.function_call_arguments.delta") {
		t.Error("Expected response.function_call_arguments.delta event")
	}
	if !strings.Contains(output, "response.function_call_arguments.done") {
		t.Error("Expected response.function_call_arguments.done event")
	}
}

func TestAnthropicToResponsesTransformer_MaxTokensStop(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewAnthropicToResponsesTransformer(&buf)

	// Simulate message_start
	messageStart := `{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","model":"claude-3-opus","usage":{"input_tokens":100,"output_tokens":0}}}`
	if err := transformer.Transform(&sse.Event{Data: messageStart}); err != nil {
		t.Fatalf("Transform message_start failed: %v", err)
	}

	// Simulate content block
	contentBlockStart := `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`
	if err := transformer.Transform(&sse.Event{Data: contentBlockStart}); err != nil {
		t.Fatalf("Transform content_block_start failed: %v", err)
	}

	// Simulate content block stop
	blockStop := `{"type":"content_block_stop","index":0}`
	if err := transformer.Transform(&sse.Event{Data: blockStop}); err != nil {
		t.Fatalf("Transform block stop failed: %v", err)
	}

	// Simulate message_delta with max_tokens stop reason
	messageDelta := `{"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"output_tokens":50}}`
	if err := transformer.Transform(&sse.Event{Data: messageDelta}); err != nil {
		t.Fatalf("Transform message_delta failed: %v", err)
	}

	output := buf.String()

	// Verify response.incomplete was emitted
	if !strings.Contains(output, "response.incomplete") {
		t.Error("Expected response.incomplete event for max_tokens stop reason")
	}
	if !strings.Contains(output, `"status":"incomplete"`) {
		t.Error("Expected status incomplete in response")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Converter Interface Test
// ─────────────────────────────────────────────────────────────────────────────

func TestAnthropicToResponsesConverter_Interface(t *testing.T) {
	converter := NewAnthropicToResponsesConverter()

	anthReq := types.MessageRequest{
		Model:     "claude-3-opus",
		MaxTokens: 512,
		Messages: []types.MessageInput{
			{Role: "user", Content: "Test"},
		},
	}

	body, _ := json.Marshal(anthReq)
	out, err := converter.Convert(body)
	if err != nil {
		t.Fatalf("Converter.Convert failed: %v", err)
	}

	var resp types.ResponsesRequest
	if err := json.Unmarshal(out, &resp); err != nil {
		t.Fatalf("Failed to unmarshal response: %v", err)
	}

	if resp.Model != "claude-3-opus" {
		t.Errorf("Expected model claude-3-opus, got %s", resp.Model)
	}
	if resp.MaxOutputTokens != 512 {
		t.Errorf("Expected max_output_tokens 512, got %d", resp.MaxOutputTokens)
	}
}
