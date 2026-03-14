package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"ai-proxy/config"
	"ai-proxy/router"
	"ai-proxy/types"

	"github.com/gin-gonic/gin"
)

func newTestRoute(providerType, baseURL, apiKey, model string) *router.ResolvedRoute {
	return &router.ResolvedRoute{
		Provider: config.Provider{
			Type:    providerType,
			BaseURL: baseURL,
			APIKey:  apiKey,
		},
		Model: model,
	}
}

func TestBridgeHandler_ValidateRequest(t *testing.T) {
	h := &BridgeHandler{}

	tests := []struct {
		name    string
		body    []byte
		wantErr bool
	}{
		{
			name:    "empty body",
			body:    []byte{},
			wantErr: false,
		},
		{
			name:    "valid json",
			body:    []byte(`{"model": "claude-3", "stream": true}`),
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := h.ValidateRequest(tt.body)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateRequest() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestBridgeHandler_TransformRequest(t *testing.T) {
	h := &BridgeHandler{}

	tests := []struct {
		name    string
		body    []byte
		wantErr bool
	}{
		{
			name:    "empty body - passes through",
			body:    []byte{},
			wantErr: false,
		},
		{
			name:    "valid anthropic request",
			body:    []byte(`{"model": "claude-3", "messages": [{"role": "user", "content": "hello"}], "stream": true}`),
			wantErr: false,
		},
		{
			name:    "request with tools",
			body:    []byte(`{"model": "claude-3", "messages": [{"role": "user", "content": "hello"}], "tools": [{"name": "test", "input_schema": {}}], "stream": true}`),
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := h.TransformRequest(tt.body)
			if (err != nil) != tt.wantErr {
				t.Errorf("TransformRequest() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestBridgeHandler_UpstreamURLWithRoute(t *testing.T) {
	h := &BridgeHandler{}

	tests := []struct {
		name     string
		route    *router.ResolvedRoute
		expected string
	}{
		{
			name:     "openai provider",
			route:    newTestRoute("openai", "https://api.example.com/v1", "test-key", "gpt-4"),
			expected: "https://api.example.com/v1/chat/completions",
		},
		{
			name:     "openai provider with chat/completions suffix",
			route:    newTestRoute("openai", "https://api.example.com/v1/chat/completions", "test-key", "gpt-4"),
			expected: "https://api.example.com/v1/chat/completions",
		},
		{
			name:     "anthropic provider",
			route:    newTestRoute("anthropic", "https://api.anthropic.com/v1/messages", "test-key", "claude-3"),
			expected: "https://api.anthropic.com/v1/messages",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := h.UpstreamURLWithRoute(tt.route)
			if got != tt.expected {
				t.Errorf("UpstreamURLWithRoute() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestBridgeHandler_ResolveAPIKeyWithRoute(t *testing.T) {
	h := &BridgeHandler{}

	route := newTestRoute("openai", "https://api.example.com/v1", "test-api-key", "gpt-4")

	got := h.ResolveAPIKeyWithRoute(route)
	if got != "test-api-key" {
		t.Errorf("ResolveAPIKeyWithRoute() = %v, want %v", got, "test-api-key")
	}
}

func TestBridgeHandler_ForwardHeadersWithRoute(t *testing.T) {
	h := &BridgeHandler{}

	tests := []struct {
		name            string
		route           *router.ResolvedRoute
		requestHeaders  map[string]string
		expectedHeaders map[string]string
	}{
		{
			name:            "openai - no custom headers",
			route:           newTestRoute("openai", "https://api.example.com/v1", "test-key", "gpt-4"),
			requestHeaders:  map[string]string{},
			expectedHeaders: map[string]string{},
		},
		{
			name:            "openai - X- header forwarded",
			route:           newTestRoute("openai", "https://api.example.com/v1", "test-key", "gpt-4"),
			requestHeaders:  map[string]string{"X-Custom": "value1"},
			expectedHeaders: map[string]string{"X-Custom": "value1"},
		},
		{
			name:            "anthropic - X- and Anthropic headers forwarded",
			route:           newTestRoute("anthropic", "https://api.anthropic.com/v1/messages", "test-key", "claude-3"),
			requestHeaders:  map[string]string{"X-Custom": "value1", "Anthropic-Version": "2023-06-01"},
			expectedHeaders: map[string]string{"X-Custom": "value1", "Anthropic-Version": "2023-06-01"},
		},
		{
			name:            "openai - non-forwarded headers ignored",
			route:           newTestRoute("openai", "https://api.example.com/v1", "test-key", "gpt-4"),
			requestHeaders:  map[string]string{"Authorization": "Bearer token", "Content-Type": "application/json"},
			expectedHeaders: map[string]string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			c.Request = httptest.NewRequest(http.MethodPost, "/", nil)
			for k, v := range tt.requestHeaders {
				c.Request.Header.Set(k, v)
			}

			upstreamReq := httptest.NewRequest(http.MethodPost, "https://upstream.example.com", nil)
			h.ForwardHeadersWithRoute(c, upstreamReq, tt.route)

			for k, v := range tt.expectedHeaders {
				if upstreamReq.Header.Get(k) != v {
					t.Errorf("expected header %s = %s, got %s", k, v, upstreamReq.Header.Get(k))
				}
			}
		})
	}
}

func TestBridgeHandler_WriteError(t *testing.T) {
	h := &BridgeHandler{}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	h.WriteError(c, http.StatusBadRequest, "test error")

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected status %d, got %d", http.StatusBadRequest, w.Code)
	}
}

func TestBridgeHandler_CreateTransformerWithRoute(t *testing.T) {
	h := &BridgeHandler{}

	w := httptest.NewRecorder()
	route := newTestRoute("openai", "https://api.example.com/v1", "test-key", "gpt-4")

	transformer := h.CreateTransformerWithRoute(w, route)

	if transformer == nil {
		t.Error("expected non-nil transformer")
	}
}

func TestTransformAnthropicToOpenAI(t *testing.T) {
	tests := []struct {
		name        string
		input       []byte
		wantErr     bool
		checkModel  string
		checkStream bool
	}{
		{
			name:        "empty body",
			input:       []byte(`{}`),
			wantErr:     false,
			checkModel:  "",
			checkStream: false,
		},
		{
			name:        "simple message",
			input:       []byte(`{"model": "claude-3", "messages": [{"role": "user", "content": "hello"}], "stream": true}`),
			wantErr:     false,
			checkModel:  "claude-3",
			checkStream: true,
		},
		{
			name:    "invalid json",
			input:   []byte(`{invalid}`),
			wantErr: true,
		},
		{
			name:        "with system message string",
			input:       []byte(`{"model": "claude-3", "system": "You are helpful", "messages": [{"role": "user", "content": "hi"}], "stream": true}`),
			wantErr:     false,
			checkModel:  "claude-3",
			checkStream: true,
		},
		{
			name:        "with system message array",
			input:       []byte(`{"model": "claude-3", "system": [{"type": "text", "text": "You are helpful"}], "messages": [{"role": "user", "content": "hi"}], "stream": true}`),
			wantErr:     false,
			checkModel:  "claude-3",
			checkStream: true,
		},
		{
			name:        "with tools",
			input:       []byte(`{"model": "claude-3", "messages": [{"role": "user", "content": "hi"}], "tools": [{"name": "bash", "input_schema": {}}], "stream": true}`),
			wantErr:     false,
			checkModel:  "claude-3",
			checkStream: true,
		},
		{
			name:        "with temperature and top_p",
			input:       []byte(`{"model": "claude-3", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.7, "top_p": 0.9, "stream": true}`),
			wantErr:     false,
			checkModel:  "claude-3",
			checkStream: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := transformAnthropicToOpenAI(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("transformAnthropicToOpenAI() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr {
				return
			}

			var req types.ChatCompletionRequest
			if err := json.Unmarshal(result, &req); err != nil {
				t.Errorf("failed to unmarshal result: %v", err)
				return
			}

			if req.Model != tt.checkModel {
				t.Errorf("expected model %q, got %q", tt.checkModel, req.Model)
			}
			if req.Stream != tt.checkStream {
				t.Errorf("expected stream %v, got %v", tt.checkStream, req.Stream)
			}
		})
	}
}

func TestExtractSystemMessage(t *testing.T) {
	tests := []struct {
		name     string
		system   interface{}
		expected string
	}{
		{
			name:     "nil system",
			system:   nil,
			expected: "",
		},
		{
			name:     "string system",
			system:   "You are helpful",
			expected: "You are helpful",
		},
		{
			name:     "empty string system",
			system:   "",
			expected: "",
		},
		{
			name:     "array system single text",
			system:   []interface{}{map[string]interface{}{"type": "text", "text": "You are helpful"}},
			expected: "You are helpful",
		},
		{
			name:     "array system multiple texts",
			system:   []interface{}{map[string]interface{}{"type": "text", "text": "You are helpful"}, map[string]interface{}{"type": "text", "text": "Be concise"}},
			expected: "You are helpfulBe concise",
		},
		{
			name:     "array system with non-text items",
			system:   []interface{}{map[string]interface{}{"type": "image"}, map[string]interface{}{"type": "text", "text": "Hello"}},
			expected: "Hello",
		},
		{
			name:     "array system empty",
			system:   []interface{}{},
			expected: "",
		},
		{
			name:     "invalid type",
			system:   123,
			expected: "",
		},
		{
			name:     "array with non-map items",
			system:   []interface{}{"string", 123},
			expected: "",
		},
		{
			name:     "array with missing text field",
			system:   []interface{}{map[string]interface{}{"type": "text"}},
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractSystemMessage(tt.system)
			if result != tt.expected {
				t.Errorf("extractSystemMessage() = %q, want %q", result, tt.expected)
			}
		})
	}
}

func TestConvertMessages(t *testing.T) {
	tests := []struct {
		name          string
		anthMsgs      []types.MessageInput
		expectedCount int
	}{
		{
			name:          "empty messages",
			anthMsgs:      []types.MessageInput{},
			expectedCount: 0,
		},
		{
			name: "single user message",
			anthMsgs: []types.MessageInput{
				{Role: "user", Content: "hello"},
			},
			expectedCount: 1,
		},
		{
			name: "multiple messages",
			anthMsgs: []types.MessageInput{
				{Role: "user", Content: "hello"},
				{Role: "assistant", Content: "hi there"},
			},
			expectedCount: 2,
		},
		{
			name: "message with content blocks",
			anthMsgs: []types.MessageInput{
				{Role: "user", Content: []interface{}{map[string]interface{}{"type": "text", "text": "hello"}}},
			},
			expectedCount: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := convertMessages(tt.anthMsgs)
			if len(result) != tt.expectedCount {
				t.Errorf("convertMessages() returned %d messages, want %d", len(result), tt.expectedCount)
			}
		})
	}
}

func TestConvertMessage(t *testing.T) {
	tests := []struct {
		name            string
		anthMsg         types.MessageInput
		expectedRole    string
		expectedContent string
	}{
		{
			name:            "string content",
			anthMsg:         types.MessageInput{Role: "user", Content: "hello"},
			expectedRole:    "user",
			expectedContent: "hello",
		},
		{
			name:            "assistant message",
			anthMsg:         types.MessageInput{Role: "assistant", Content: "hi there"},
			expectedRole:    "assistant",
			expectedContent: "hi there",
		},
		{
			name: "content blocks with text",
			anthMsg: types.MessageInput{
				Role: "user",
				Content: []interface{}{
					map[string]interface{}{"type": "text", "text": "hello"},
				},
			},
			expectedRole:    "user",
			expectedContent: "hello",
		},
		{
			name: "content blocks with tool_use",
			anthMsg: types.MessageInput{
				Role: "assistant",
				Content: []interface{}{
					map[string]interface{}{"type": "text", "text": "using tool"},
					map[string]interface{}{
						"type":  "tool_use",
						"id":    "tool-123",
						"name":  "bash",
						"input": map[string]interface{}{"command": "ls"},
					},
				},
			},
			expectedRole:    "assistant",
			expectedContent: "using tool",
		},
		{
			name: "content blocks with tool_result",
			anthMsg: types.MessageInput{
				Role: "user",
				Content: []interface{}{
					map[string]interface{}{
						"type":        "tool_result",
						"tool_use_id": "tool-123",
					},
				},
			},
			expectedRole: "user",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := convertMessage(tt.anthMsg)
			if result.Role != tt.expectedRole {
				t.Errorf("convertMessage() role = %q, want %q", result.Role, tt.expectedRole)
			}
			if tt.expectedContent != "" {
				if content, ok := result.Content.(string); !ok || content != tt.expectedContent {
					t.Errorf("convertMessage() content = %v, want %q", result.Content, tt.expectedContent)
				}
			}
		})
	}
}

func TestConvertContentBlocks(t *testing.T) {
	tests := []struct {
		name              string
		blocks            []interface{}
		expectedText      string
		expectedToolCalls int
		expectedToolID    string
	}{
		{
			name:              "empty blocks",
			blocks:            []interface{}{},
			expectedText:      "",
			expectedToolCalls: 0,
			expectedToolID:    "",
		},
		{
			name: "single text block",
			blocks: []interface{}{
				map[string]interface{}{"type": "text", "text": "hello"},
			},
			expectedText:      "hello",
			expectedToolCalls: 0,
			expectedToolID:    "",
		},
		{
			name: "multiple text blocks",
			blocks: []interface{}{
				map[string]interface{}{"type": "text", "text": "hello"},
				map[string]interface{}{"type": "text", "text": "world"},
			},
			expectedText:      "hello\nworld",
			expectedToolCalls: 0,
			expectedToolID:    "",
		},
		{
			name: "tool_use block",
			blocks: []interface{}{
				map[string]interface{}{
					"type":  "tool_use",
					"id":    "tool-123",
					"name":  "bash",
					"input": map[string]interface{}{"command": "ls"},
				},
			},
			expectedText:      "",
			expectedToolCalls: 1,
			expectedToolID:    "",
		},
		{
			name: "tool_result block",
			blocks: []interface{}{
				map[string]interface{}{
					"type":        "tool_result",
					"tool_use_id": "tool-456",
				},
			},
			expectedText:      "",
			expectedToolCalls: 0,
			expectedToolID:    "tool-456",
		},
		{
			name: "mixed blocks",
			blocks: []interface{}{
				map[string]interface{}{"type": "text", "text": "using tool"},
				map[string]interface{}{
					"type":  "tool_use",
					"id":    "tool-123",
					"name":  "bash",
					"input": map[string]interface{}{"command": "ls"},
				},
				map[string]interface{}{
					"type":        "tool_result",
					"tool_use_id": "tool-456",
				},
			},
			expectedText:      "using tool",
			expectedToolCalls: 1,
			expectedToolID:    "tool-456",
		},
		{
			name: "non-map block",
			blocks: []interface{}{
				"string value",
				map[string]interface{}{"type": "text", "text": "hello"},
			},
			expectedText:      "hello",
			expectedToolCalls: 0,
			expectedToolID:    "",
		},
		{
			name: "tool_use missing fields",
			blocks: []interface{}{
				map[string]interface{}{"type": "tool_use"},
			},
			expectedText:      "",
			expectedToolCalls: 0,
			expectedToolID:    "",
		},
		{
			name: "unknown block type",
			blocks: []interface{}{
				map[string]interface{}{"type": "unknown", "data": "something"},
			},
			expectedText:      "",
			expectedToolCalls: 0,
			expectedToolID:    "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			text, toolCalls, toolID := convertContentBlocks(tt.blocks)
			if text != tt.expectedText {
				t.Errorf("convertContentBlocks() text = %q, want %q", text, tt.expectedText)
			}
			if len(toolCalls) != tt.expectedToolCalls {
				t.Errorf("convertContentBlocks() toolCalls count = %d, want %d", len(toolCalls), tt.expectedToolCalls)
			}
			if toolID != tt.expectedToolID {
				t.Errorf("convertContentBlocks() toolID = %q, want %q", toolID, tt.expectedToolID)
			}
		})
	}
}

func TestConvertTools(t *testing.T) {
	tests := []struct {
		name          string
		anthTools     []types.ToolDef
		expectedCount int
	}{
		{
			name:          "empty tools",
			anthTools:     []types.ToolDef{},
			expectedCount: 0,
		},
		{
			name: "single tool",
			anthTools: []types.ToolDef{
				{Name: "bash", Description: "Run bash commands", InputSchema: json.RawMessage(`{"type": "object"}`)},
			},
			expectedCount: 1,
		},
		{
			name: "multiple tools",
			anthTools: []types.ToolDef{
				{Name: "bash", Description: "Run bash commands", InputSchema: json.RawMessage(`{"type": "object"}`)},
				{Name: "read", Description: "Read files", InputSchema: json.RawMessage(`{"type": "object"}`)},
			},
			expectedCount: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := convertTools(tt.anthTools)
			if len(result) != tt.expectedCount {
				t.Errorf("convertTools() returned %d tools, want %d", len(result), tt.expectedCount)
			}
			for i, tool := range result {
				if tool.Type != "function" {
					t.Errorf("convertTools() tool %d type = %q, want %q", i, tool.Type, "function")
				}
				if tool.Function.Name != tt.anthTools[i].Name {
					t.Errorf("convertTools() tool %d name = %q, want %q", i, tool.Function.Name, tt.anthTools[i].Name)
				}
			}
		})
	}
}
