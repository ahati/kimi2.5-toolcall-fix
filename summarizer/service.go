// Package summarizer provides a service for summarizing reasoning content using a small fast model.
// The summarizer is triggered when a request includes reasoning.summary parameter.
// This file contains the HTTP API-based summarizer implementation.
package summarizer

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"

	"ai-proxy/config"
	"ai-proxy/logging"
)

// DefaultPrompt is the default prompt used for summarization when none is configured.
// It uses JSON format for reliable extraction and requests short summaries.
const DefaultPrompt = `Extract the main point in under 10 words.
Output as JSON: {"summary": "your summary here"}`

// DefaultService is the global summarizer service instance.
// It is initialized by InitDefaultService and accessed via GetDefaultService.
var DefaultService *Service

// localSummarizer is an interface for local summarization (implemented in service_local.go).
type localSummarizer interface {
	Summarize(ctx context.Context, reasoning string) (localSummary, error)
	Close()
}

// localSummary is the result from local summarization.
type localSummary struct {
	Text      string
	Latency   interface{}
	Truncated bool
}

// Service provides reasoning summarization using either HTTP API calls or local inference.
// It supports both modes and can be configured via config.json.
type Service struct {
	// cfg is the summarizer configuration.
	cfg config.SummarizerConfig
	// provider is the resolved provider configuration (for HTTP mode).
	provider config.Provider
	// client is the HTTP client for making API requests (for HTTP mode).
	client *http.Client
	// local is the local summarizer instance (for local mode, nil if not available).
	local localSummarizer
	// mu protects access to local summarizer during Close.
	mu sync.Mutex
}

// InitDefaultService initializes the global summarizer service from the schema.
// This should be called once at startup, similar to conversation.InitDefaultStore.
//
// @param schema - the loaded configuration schema containing providers and summarizer config
// @post DefaultService is initialized (or nil if disabled/misconfigured)
func InitDefaultService(schema *config.Schema) {
	if schema == nil {
		logging.DebugMsg("Summarizer not initialized: schema is nil")
		return
	}

	// Build providers map
	providersMap := make(map[string]config.Provider)
	for _, p := range schema.Providers {
		providersMap[p.Name] = p
	}

	DefaultService = NewService(schema.Summarizer, providersMap)
	if DefaultService != nil {
		logging.InfoMsg("Summarizer service initialized and ready")
	}
}

// GetDefaultService returns the global summarizer service instance.
// Returns nil if InitDefaultService hasn't been called or if summarizer is disabled.
func GetDefaultService() *Service {
	return DefaultService
}

// NewService creates a new summarizer service.
// Returns nil if summarizer is disabled or configuration is invalid.
//
// @param cfg - the summarizer configuration from config.json
// @param providers - map of provider name to Provider configuration
// @return *Service - initialized service, or nil if disabled/invalid
func NewService(cfg config.SummarizerConfig, providers map[string]config.Provider) *Service {
	if !cfg.Enabled {
		logging.DebugMsg("Summarizer is disabled in configuration")
		return nil
	}

	mode := cfg.Mode
	if mode == "" {
		mode = "http" // Default to HTTP mode
	}

	s := &Service{cfg: cfg}

	switch mode {
	case "local":
		// Initialize local llama.cpp summarizer (if available)
		return newLocalService(cfg, s)

	case "http":
		// Initialize HTTP API summarizer
		provider, ok := providers[cfg.Provider]
		if !ok {
			logging.ErrorMsg("Summarizer provider '%s' not found, summarizer disabled", cfg.Provider)
			return nil
		}

		if cfg.Model == "" {
			logging.ErrorMsg("Summarizer model not configured, summarizer disabled")
			return nil
		}

		s.provider = provider
		s.client = &http.Client{
			Timeout: 0, // No timeout - let context handle cancellation
		}
		logging.InfoMsg("Summarizer initialized: mode=http, provider=%s, model=%s", cfg.Provider, cfg.Model)

	default:
		logging.ErrorMsg("Unknown summarizer mode '%s', summarizer disabled", mode)
		return nil
	}

	return s
}

// newLocalService is implemented in service_local.go (or stub_local.go if not built with localsummarizer tag).
// It returns nil if local summarizer is not available.

// SummarizeRequest represents the request body for the summarization API call.
type SummarizeRequest struct {
	Model    string         `json:"model"`
	Messages []SummarizeMsg `json:"messages"`
	Stream   bool           `json:"stream"`
}

// SummarizeMsg represents a message in the summarization request.
type SummarizeMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// SummarizeResponse represents the response from the summarization API.
type SummarizeResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index   int `json:"index"`
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// Summarize generates a summary of the given reasoning content.
// It uses either local inference or HTTP API calls based on configuration.
//
// @param ctx - context for cancellation
// @param reasoningContent - the reasoning text to summarize
// @return summary text, or error if summarization fails
func (s *Service) Summarize(ctx context.Context, reasoningContent string) (string, error) {
	if s == nil {
		return "", fmt.Errorf("summarizer service not initialized")
	}

	// Use local or HTTP based on mode
	if s.local != nil {
		return s.summarizeLocal(ctx, reasoningContent)
	}
	return s.summarizeHTTP(ctx, reasoningContent)
}

// summarizeLocal uses local llama.cpp inference for summarization.
func (s *Service) summarizeLocal(ctx context.Context, reasoningContent string) (string, error) {
	logging.InfoMsg("📝 Summarizer (local): summarizing reasoning content (%d bytes)", len(reasoningContent))

	summary, err := s.local.Summarize(ctx, reasoningContent)
	if err != nil {
		logging.ErrorMsg("Local summarizer failed: %v", err)
		return "", fmt.Errorf("local summarizer failed: %w", err)
	}

	// Safety check: if summary is longer than input, return original
	if len(summary.Text) > len(reasoningContent) {
		logging.DebugMsg("Summarizer output (%d bytes) longer than input (%d bytes), using original",
			len(summary.Text), len(reasoningContent))
		return reasoningContent, nil
	}

	logging.InfoMsg("✅ Summarizer (local): generated summary (%d bytes) from reasoning (%d bytes)",
		len(summary.Text), len(reasoningContent))
	if summary.Truncated {
		logging.DebugMsg("Input was truncated before summarization")
	}

	return summary.Text, nil
}

// summarizeHTTP uses HTTP API calls for summarization.
func (s *Service) summarizeHTTP(ctx context.Context, reasoningContent string) (string, error) {
	// Use configured prompt or default
	prompt := s.cfg.Prompt
	if prompt == "" {
		prompt = DefaultPrompt
	}

	// Build the summarization request with JSON format
	reqBody := SummarizeRequest{
		Model: s.cfg.Model,
		Messages: []SummarizeMsg{
			{
				Role:    "system",
				Content: prompt,
			},
			{
				Role:    "user",
				Content: reasoningContent,
			},
		},
		Stream: false, // Non-streaming for simplicity
	}

	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	// Get the OpenAI-compatible endpoint
	endpoint := s.provider.GetEndpoint("openai")
	if endpoint == "" {
		return "", fmt.Errorf("no openai endpoint configured for provider '%s'", s.cfg.Provider)
	}

	// Create HTTP request
	req, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewReader(bodyBytes))
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+s.provider.GetAPIKey())

	// Log that we're about to summarize
	logging.InfoMsg("📝 Summarizer (http): summarizing reasoning content (%d bytes) using model %s",
		len(reasoningContent), s.cfg.Model)

	// Execute request
	resp, err := s.client.Do(req)
	if err != nil {
		logging.ErrorMsg("Summarizer request failed: %v", err)
		return "", fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	// Read response
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read response: %w", err)
	}

	// Check status code
	if resp.StatusCode != http.StatusOK {
		logging.ErrorMsg("Summarizer returned status %d: %s", resp.StatusCode, string(respBody))
		return "", fmt.Errorf("summarizer returned status %d: %s", resp.StatusCode, string(respBody))
	}

	// Parse response
	var summarizeResp SummarizeResponse
	if err := json.Unmarshal(respBody, &summarizeResp); err != nil {
		return "", fmt.Errorf("parse response: %w", err)
	}

	// Extract summary from response
	if len(summarizeResp.Choices) == 0 {
		return "", fmt.Errorf("no choices in summarizer response")
	}

	summary := summarizeResp.Choices[0].Message.Content

	// Extract from JSON if present
	if strings.Contains(summary, `"summary"`) {
		if idx := strings.Index(summary, `"summary": "`); idx >= 0 {
			start := idx + len(`"summary": "`)
			if endIdx := strings.Index(summary[start:], `"`); endIdx > 0 {
				summary = summary[start : start+endIdx]
			}
		}
	}

	// Safety check: if summary is longer than input, return original instead
	// This ensures we never expand, only compress
	if len(summary) > len(reasoningContent) {
		logging.DebugMsg("Summarizer output (%d bytes) longer than input (%d bytes), using original",
			len(summary), len(reasoningContent))
		return reasoningContent, nil
	}

	// Log successful summarization
	logging.InfoMsg("✅ Summarizer (http): generated summary (%d bytes) from reasoning (%d bytes), tokens: prompt=%d completion=%d",
		len(summary), len(reasoningContent),
		summarizeResp.Usage.PromptTokens, summarizeResp.Usage.CompletionTokens)

	return summary, nil
}

// Close releases resources held by the service.
// This is important for local mode to free GPU memory.
func (s *Service) Close() {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.local != nil {
		s.local.Close()
		s.local = nil
	}
}

// IsEnabled returns true if the summarizer service is enabled and ready.
func (s *Service) IsEnabled() bool {
	return s != nil
}