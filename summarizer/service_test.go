package summarizer

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"ai-proxy/config"
)

func TestNewService_Disabled(t *testing.T) {
	cfg := config.SummarizerConfig{
		Enabled: false,
	}
	providers := map[string]config.Provider{}

	svc := NewService(cfg, providers)
	if svc != nil {
		t.Error("expected nil service when disabled")
	}
}

func TestNewService_MissingProvider(t *testing.T) {
	cfg := config.SummarizerConfig{
		Enabled:  true,
		Provider: "nonexistent",
		Model:    "test-model",
	}
	providers := map[string]config.Provider{}

	svc := NewService(cfg, providers)
	if svc != nil {
		t.Error("expected nil service when provider not found")
	}
}

func TestNewService_MissingModel(t *testing.T) {
	cfg := config.SummarizerConfig{
		Enabled:  true,
		Provider: "test-provider",
		Model:    "",
	}
	providers := map[string]config.Provider{
		"test-provider": {Name: "test-provider"},
	}

	svc := NewService(cfg, providers)
	if svc != nil {
		t.Error("expected nil service when model not configured")
	}
}

func TestNewService_Success(t *testing.T) {
	cfg := config.SummarizerConfig{
		Enabled:  true,
		Provider: "test-provider",
		Model:    "test-model",
		Prompt:   "Custom prompt:",
	}
	providers := map[string]config.Provider{
		"test-provider": {
			Name: "test-provider",
			Endpoints: map[string]string{
				"openai": "https://api.example.com/v1/chat/completions",
			},
		},
	}

	svc := NewService(cfg, providers)
	if svc == nil {
		t.Fatal("expected non-nil service")
	}
	if !svc.IsEnabled() {
		t.Error("expected IsEnabled to return true")
	}
}

func TestNewService_LocalModeMissingModel(t *testing.T) {
	// Local mode should fail gracefully if model file doesn't exist
	cfg := config.SummarizerConfig{
		Enabled: true,
		Mode:    "local",
		Local: config.LocalSummarizerConfig{
			ModelPath: "/nonexistent/model.gguf",
		},
	}
	providers := map[string]config.Provider{}

	svc := NewService(cfg, providers)
	// Should be nil because model file doesn't exist
	if svc != nil {
		t.Error("expected nil service when model file doesn't exist")
	}
}

func TestNewService_UnknownMode(t *testing.T) {
	cfg := config.SummarizerConfig{
		Enabled:  true,
		Mode:     "unknown",
		Provider: "test-provider",
		Model:    "test-model",
	}
	providers := map[string]config.Provider{
		"test-provider": {Name: "test-provider"},
	}

	svc := NewService(cfg, providers)
	if svc != nil {
		t.Error("expected nil service for unknown mode")
	}
}

func TestSummarize_Success(t *testing.T) {
	// Create test server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify request
		if r.Method != "POST" {
			t.Errorf("expected POST, got %s", r.Method)
		}

		// Parse request
		var req SummarizeRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("failed to decode request: %v", err)
		}
		if req.Model != "test-model" {
			t.Errorf("expected model test-model, got %s", req.Model)
		}
		if req.Stream {
			t.Error("expected stream=false")
		}

		// Send response
		resp := SummarizeResponse{
			ID:      "test-id",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   "test-model",
			Choices: []struct {
				Index   int `json:"index"`
				Message struct {
					Role    string `json:"role"`
					Content string `json:"content"`
				} `json:"message"`
				FinishReason string `json:"finish_reason"`
			}{
				{
					Index:        0,
					FinishReason: "stop",
				},
			},
			Usage: struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
				TotalTokens      int `json:"total_tokens"`
			}{
				PromptTokens:     100,
				CompletionTokens: 50,
				TotalTokens:      150,
			},
		}
		resp.Choices[0].Message.Role = "assistant"
		resp.Choices[0].Message.Content = "This is a summary."

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	cfg := config.SummarizerConfig{
		Enabled:  true,
		Provider: "test-provider",
		Model:    "test-model",
	}
	providers := map[string]config.Provider{
		"test-provider": {
			Name: "test-provider",
			Endpoints: map[string]string{
				"openai": server.URL,
			},
		},
	}

	svc := NewService(cfg, providers)
	if svc == nil {
		t.Fatal("expected non-nil service")
	}

	summary, err := svc.Summarize(context.Background(), "Long reasoning content here...")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if summary != "This is a summary." {
		t.Errorf("expected summary 'This is a summary.', got '%s'", summary)
	}
}

func TestSummarize_Error(t *testing.T) {
	// Create test server that returns error
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte("internal error"))
	}))
	defer server.Close()

	cfg := config.SummarizerConfig{
		Enabled:  true,
		Provider: "test-provider",
		Model:    "test-model",
	}
	providers := map[string]config.Provider{
		"test-provider": {
			Name: "test-provider",
			Endpoints: map[string]string{
				"openai": server.URL,
			},
		},
	}

	svc := NewService(cfg, providers)
	if svc == nil {
		t.Fatal("expected non-nil service")
	}

	_, err := svc.Summarize(context.Background(), "content")
	if err == nil {
		t.Error("expected error from summarizer")
	}
}

func TestSummarize_NilService(t *testing.T) {
	var svc *Service
	_, err := svc.Summarize(context.Background(), "content")
	if err == nil {
		t.Error("expected error from nil service")
	}
}

func TestSummarize_UsesDefaultPrompt(t *testing.T) {
	receivedPrompt := ""

	// Create test server that captures the prompt
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req SummarizeRequest
		json.NewDecoder(r.Body).Decode(&req)
		if len(req.Messages) > 0 {
			receivedPrompt = req.Messages[0].Content
		}

		resp := SummarizeResponse{
			ID:      "test-id",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   "test-model",
			Choices: []struct {
				Index   int `json:"index"`
				Message struct {
					Role    string `json:"role"`
					Content string `json:"content"`
				} `json:"message"`
				FinishReason string `json:"finish_reason"`
			}{
				{Index: 0, FinishReason: "stop"},
			},
		}
		resp.Choices[0].Message.Role = "assistant"
		resp.Choices[0].Message.Content = "summary"

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	// Test with empty prompt (should use default)
	cfg := config.SummarizerConfig{
		Enabled:  true,
		Provider: "test-provider",
		Model:    "test-model",
		Prompt:   "",
	}
	providers := map[string]config.Provider{
		"test-provider": {
			Name: "test-provider",
			Endpoints: map[string]string{
				"openai": server.URL,
			},
		},
	}

	svc := NewService(cfg, providers)
	svc.Summarize(context.Background(), "content")

	if receivedPrompt != DefaultPrompt {
		t.Errorf("expected default prompt, got '%s'", receivedPrompt)
	}
}

func TestSummarize_FallbackWhenSummaryLonger(t *testing.T) {
	// Create test server that returns a summary longer than input
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := SummarizeResponse{
			ID:      "test-id",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   "test-model",
			Choices: []struct {
				Index   int `json:"index"`
				Message struct {
					Role    string `json:"role"`
					Content string `json:"content"`
				} `json:"message"`
				FinishReason string `json:"finish_reason"`
			}{
				{Index: 0, FinishReason: "stop"},
			},
		}
		resp.Choices[0].Message.Role = "assistant"
		// Return a very long summary that exceeds input length
		resp.Choices[0].Message.Content = "This is a very long summary that is definitely much longer than the original input content and should trigger the fallback mechanism."

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	cfg := config.SummarizerConfig{
		Enabled:  true,
		Provider: "test-provider",
		Model:    "test-model",
	}
	providers := map[string]config.Provider{
		"test-provider": {
			Name: "test-provider",
			Endpoints: map[string]string{
				"openai": server.URL,
			},
		},
	}

	svc := NewService(cfg, providers)
	if svc == nil {
		t.Fatal("expected non-nil service")
	}

	// Short input - summary will be longer
	input := "Short input"
	summary, err := svc.Summarize(context.Background(), input)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// Should return original input since summary is longer
	if summary != input {
		t.Errorf("expected original input '%s' when summary longer, got '%s'", input, summary)
	}
}