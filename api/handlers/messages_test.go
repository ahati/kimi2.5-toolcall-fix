package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"ai-proxy/config"
	"ai-proxy/router"
	"ai-proxy/types"

	"github.com/gin-gonic/gin"
)

func TestMessagesHandler_ValidateRequest(t *testing.T) {
	cfg := &config.Config{}
	h := &MessagesHandler{cfg: cfg}

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
		{
			name:    "invalid json passes validation",
			body:    []byte(`{invalid}`),
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

func TestMessagesHandler_TransformRequest(t *testing.T) {
	cfg := &config.Config{}
	h := &MessagesHandler{cfg: cfg}

	tests := []struct {
		name    string
		body    []byte
		want    []byte
		wantErr bool
	}{
		{
			name:    "empty body",
			body:    []byte{},
			want:    []byte{},
			wantErr: false,
		},
		{
			name:    "valid request",
			body:    []byte(`{"model": "claude-3", "stream": true}`),
			want:    []byte(`{"model": "claude-3", "stream": true}`),
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := h.TransformRequest(context.TODO(), tt.body)
			if (err != nil) != tt.wantErr {
				t.Errorf("TransformRequest(context.TODO(), ) error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !bytes.Equal(got, tt.want) {
				t.Errorf("TransformRequest(context.TODO(), ) = %s, want %s", got, tt.want)
			}
		})
	}
}

func TestMessagesHandler_UpstreamURL(t *testing.T) {
	provider := config.Provider{
		Name:      "anthropic",
		Endpoints: map[string]string{"anthropic": "https://api.anthropic.com/v1/messages"},
	}
	h := &MessagesHandler{
		route: &router.ResolvedRoute{
			Provider:       provider,
			OutputProtocol: "anthropic",
		},
	}

	expectedURL := "https://api.anthropic.com/v1/messages"
	if got := h.UpstreamURL(); got != expectedURL {
		t.Errorf("UpstreamURL() = %v, want %v", got, expectedURL)
	}
}

func TestMessagesHandler_ResolveAPIKey(t *testing.T) {
	provider := config.Provider{
		Name:      "anthropic",
		Endpoints: map[string]string{"anthropic": "https://api.anthropic.com"},
		APIKey:    "anthropic-api-key",
	}
	h := &MessagesHandler{
		route: &router.ResolvedRoute{
			Provider:       provider,
			OutputProtocol: "anthropic",
		},
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/", nil)

	got := h.ResolveAPIKey(c)
	if got != "anthropic-api-key" {
		t.Errorf("ResolveAPIKey() = %v, want %v", got, "anthropic-api-key")
	}
}

func TestMessagesHandler_ForwardHeaders(t *testing.T) {
	cfg := &config.Config{}
	h := &MessagesHandler{cfg: cfg}

	tests := []struct {
		name            string
		requestHeaders  map[string]string
		expectedHeaders map[string]string
	}{
		{
			name:            "no custom headers",
			requestHeaders:  map[string]string{},
			expectedHeaders: map[string]string{},
		},
		{
			name: "X- header forwarded",
			requestHeaders: map[string]string{
				"X-Custom": "value1",
			},
			expectedHeaders: map[string]string{
				"X-Custom": "value1",
			},
		},
		{
			name: "Anthropic-Version header forwarded",
			requestHeaders: map[string]string{
				"Anthropic-Version": "2023-06-01",
			},
			expectedHeaders: map[string]string{
				"Anthropic-Version": "2023-06-01",
			},
		},
		{
			name: "Anthropic-Beta header forwarded",
			requestHeaders: map[string]string{
				"Anthropic-Beta": "some-beta",
			},
			expectedHeaders: map[string]string{
				"Anthropic-Beta": "some-beta",
			},
		},
		{
			name: "non-forwarded headers ignored",
			requestHeaders: map[string]string{
				"Authorization": "Bearer token",
				"Content-Type":  "application/json",
			},
			expectedHeaders: map[string]string{},
		},
		{
			name: "mixed headers",
			requestHeaders: map[string]string{
				"X-Request-Id":      "123",
				"Anthropic-Version": "2023-06-01",
				"Authorization":     "Bearer token",
			},
			expectedHeaders: map[string]string{
				"X-Request-Id":      "123",
				"Anthropic-Version": "2023-06-01",
			},
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
			h.ForwardHeaders(c, upstreamReq)

			for k, v := range tt.expectedHeaders {
				if upstreamReq.Header.Get(k) != v {
					t.Errorf("expected header %s = %s, got %s", k, v, upstreamReq.Header.Get(k))
				}
			}

			for k := range tt.requestHeaders {
				if _, expected := tt.expectedHeaders[k]; !expected {
					if upstreamReq.Header.Get(k) != "" {
						t.Errorf("unexpected header %s forwarded", k)
					}
				}
			}
		})
	}
}

func TestConvertAnthropicMessage_MixedUserToolResultStaysUser(t *testing.T) {
	msg := types.MessageInput{
		Role: "user",
		Content: []interface{}{
			map[string]interface{}{
				"type": "text",
				"text": "Here is my note.",
			},
			map[string]interface{}{
				"type":        "tool_result",
				"tool_use_id": "tool_123",
				"content":     "42",
			},
		},
	}

	got := convertAnthropicMessage(msg)

	if got.Role != "user" {
		t.Fatalf("expected mixed message to remain user role, got %s", got.Role)
	}
	if got.ToolCallID != "tool_123" {
		t.Fatalf("expected tool_call_id tool_123, got %s", got.ToolCallID)
	}
	if got.Content != "Here is my note.\n42" {
		t.Fatalf("expected merged text content, got %#v", got.Content)
	}
}

func TestConvertAnthropicMessage_PureToolResultBecomesTool(t *testing.T) {
	msg := types.MessageInput{
		Role: "user",
		Content: []interface{}{
			map[string]interface{}{
				"type":        "tool_result",
				"tool_use_id": "tool_123",
				"content":     "42",
			},
		},
	}

	got := convertAnthropicMessage(msg)

	if got.Role != "tool" {
		t.Fatalf("expected pure tool_result turn to become tool role, got %s", got.Role)
	}
	if got.ToolCallID != "tool_123" {
		t.Fatalf("expected tool_call_id tool_123, got %s", got.ToolCallID)
	}
	if got.Content != "42" {
		t.Fatalf("expected tool content 42, got %#v", got.Content)
	}
}

func TestMessagesHandler_WriteError(t *testing.T) {
	cfg := &config.Config{}
	h := &MessagesHandler{cfg: cfg}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	h.WriteError(c, http.StatusBadRequest, "test error")

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected status %d, got %d", http.StatusBadRequest, w.Code)
	}
}

func TestMessagesHandler_CreateTransformer(t *testing.T) {
	cfg := &config.Config{}
	h := &MessagesHandler{cfg: cfg}

	w := httptest.NewRecorder()
	transformer := h.CreateTransformer(w)

	if transformer == nil {
		t.Error("expected non-nil transformer")
	}
}

func TestMessagesHandler_TransformRequest_PassthroughNormalizesWebSearchToolResult(t *testing.T) {
	// This test verifies that web_search_tool_result blocks are normalized
	// even when IsPassthrough is true (which happens when provider supports
	// the incoming protocol). This is needed because the proxy internally
	// injects web_search_tool_result blocks that upstream providers don't understand.
	provider := config.Provider{
		Name:      "test-provider",
		Endpoints: map[string]string{"anthropic": "https://api.example.com/v1/messages"},
	}
	h := &MessagesHandler{
		cfg: &config.Config{},
		route: &router.ResolvedRoute{
			Provider:       provider,
			OutputProtocol: "anthropic",
			Model:          "test-model",
			IsPassthrough:  true,
		},
	}

	// Request with web_search_tool_result that should be normalized
	body := []byte(`{
		"model": "test-model",
		"messages": [{
			"role": "user",
			"content": [{
				"type": "web_search_tool_result",
				"tool_use_id": "toolu_123",
				"content": [
					{"type": "web_search_result", "title": "Test", "url": "https://example.com", "content": "Test content"}
				]
			}]
		}]
	}`)

	result, err := h.TransformRequest(context.TODO(), body)
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	// Verify the result is valid JSON
	var req map[string]interface{}
	if err := json.Unmarshal(result, &req); err != nil {
		t.Fatalf("failed to unmarshal result: %v", err)
	}

	// Verify web_search_tool_result was converted to tool_result
	messages := req["messages"].([]interface{})
	msg := messages[0].(map[string]interface{})
	content := msg["content"].([]interface{})
	block := content[0].(map[string]interface{})

	if block["type"] != "tool_result" {
		t.Errorf("expected type 'tool_result', got '%v'", block["type"])
	}
	if block["tool_use_id"] != "toolu_123" {
		t.Errorf("expected tool_use_id 'toolu_123', got '%v'", block["tool_use_id"])
	}
	// Content should be converted to a string, not an array
	if _, ok := block["content"].(string); !ok {
		t.Errorf("expected content to be string, got %T", block["content"])
	}
}
