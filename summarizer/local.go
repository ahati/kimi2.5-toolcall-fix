//go:build llama

// Package summarizer provides local summarization using llama.cpp.
// This file is only compiled with -tags llama. Use 'make build' to include it.
package summarizer

import (
	"context"
	"errors"
	"fmt"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"ai-proxy/config"
	"ai-proxy/llama"
	"ai-proxy/logging"
)

// llamaSummarizer performs reasoning summarization using a local LLM via llama.cpp.
type llamaSummarizer struct {
	cfg   llamaConfig
	model *llama.Model
	ctx   *llama.Context
	mu    sync.Mutex
}

// llamaConfig holds configuration for the llama.cpp summarizer.
type llamaConfig struct {
	ModelPath         string
	ContextSize       int
	Threads           int
	MaxSummaryTokens  int
	MaxReasoningChars int
	GPULayers         int
}

// defaultLlamaConfig returns a Config with sensible defaults for the given model path.
func defaultLlamaConfig(modelPath string) llamaConfig {
	return llamaConfig{
		ModelPath:         modelPath,
		ContextSize:       2048,
		Threads:           0,
		MaxSummaryTokens:  256,
		MaxReasoningChars: 6000,
		GPULayers:         0,
	}
}

// newLlamaSummarizer creates a new llama.cpp summarizer with the given configuration.
// The llama.cpp backend is initialized and the model is loaded into memory.
func newLlamaSummarizer(cfg llamaConfig) (*llamaSummarizer, error) {
	if cfg.ModelPath == "" {
		return nil, errors.New("model path required")
	}
	if _, err := os.Stat(cfg.ModelPath); err != nil {
		return nil, fmt.Errorf("model not found: %w", err)
	}
	if cfg.Threads == 0 {
		cfg.Threads = runtime.NumCPU()
	}

	logging.InfoMsg("Initializing llama backend")
	llama.BackendInit()
	llama.LoadBackends()

	start := time.Now()
	logging.InfoMsg("Loading model: path=%s, ctx=%d", cfg.ModelPath, cfg.ContextSize)

	model, err := llama.LoadModel(cfg.ModelPath, cfg.GPULayers)
	if err != nil {
		llama.BackendFree()
		return nil, err
	}

	ctx, err := llama.NewContext(model, cfg.ContextSize, cfg.Threads)
	if err != nil {
		model.Close()
		llama.BackendFree()
		return nil, err
	}

	logging.InfoMsg("Model loaded in %v", time.Since(start).Round(time.Millisecond))
	return &llamaSummarizer{cfg: cfg, model: model, ctx: ctx}, nil
}

// Close releases all resources held by the summarizer.
func (s *llamaSummarizer) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ctx != nil {
		s.ctx.Close()
		s.ctx = nil
	}
	if s.model != nil {
		s.model.Close()
		s.model = nil
	}
	llama.BackendFree()
}

// Summarize generates a concise summary of the given reasoning text.
func (s *llamaSummarizer) Summarize(ctx context.Context, reasoning string) (localSummary, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.model == nil {
		return localSummary{}, errors.New("summarizer closed")
	}

	// Clear KV cache for new sequence
	s.ctx.ClearMemory()

	start := time.Now()
	truncated := false
	if s.cfg.MaxReasoningChars > 0 && len(reasoning) > s.cfg.MaxReasoningChars {
		reasoning = truncateMiddle(reasoning, s.cfg.MaxReasoningChars)
		truncated = true
	}

	// Prompt uses aggressive compression.
	// Qwen 3.5 is a reasoning model that outputs thinking tokens.
	// We request JSON format to extract the summary reliably.
	var prompt string
	if strings.Contains(strings.ToLower(s.cfg.ModelPath), "tinyllama") || strings.Contains(strings.ToLower(s.cfg.ModelPath), "llama") {
		// Llama-2 format for TinyLlama
		prompt = fmt.Sprintf(
			`<>
Key words from this text (max 5 words):
%s
Key words:</s>
`, reasoning)
	} else {
		// ChatML format for Qwen - request JSON output
		prompt = fmt.Sprintf(
			`<|im_start|>system
Extract the main point in under 10 words.
Output as JSON: {"summary": "your summary here"}<|im_end|>
<|im_start|>user
%s<|im_end|>
<|im_start|>assistant
{"summary": "`, reasoning)
	}

	tokens, err := s.ctx.Tokenize(prompt)
	if err != nil {
		return localSummary{}, err
	}

	logging.DebugMsg("Inference started: tokens=%d", len(tokens))
	if err := s.ctx.Decode(tokens); err != nil {
		return localSummary{}, err
	}

	sampler := llama.NewGreedySampler()
	defer sampler.Close()

	var output strings.Builder
	eos := s.model.EOSToken()

	for i := 0; i < s.cfg.MaxSummaryTokens; i++ {
		token := sampler.Sample(s.ctx)
		if token == eos {
			break
		}
		piece := s.model.TokenToPiece(token)
		if piece == "" || piece == "</s>" || piece == "<|im_end|>" {
			break
		}
		output.WriteString(piece)
		if err := s.ctx.Decode([]int32{token}); err != nil {
			break
		}
	}

	latency := time.Since(start)
	rawOutput := strings.TrimSpace(output.String())

	// Clean up thinking tokens and artifacts that some models emit
	cleanedOutput := cleanSummaryOutput(rawOutput)

	logging.DebugMsg("Inference complete: latency=%v, chars=%d, raw_chars=%d", latency, len(cleanedOutput), len(rawOutput))
	return localSummary{Text: cleanedOutput, Latency: latency, Truncated: truncated}, nil
}

// cleanSummaryOutput removes thinking tokens and artifacts from model output.
func cleanSummaryOutput(s string) string {
	// Qwen 3.5 reasoning models output thinking in this format:
	// ང thinking_content ང actual_response
	// The thinking token is Unicode U+0F04 (Tibetan mark initial form)

	// Count occurrences of thinking token
	thinkingToken := "<tool_call>"
	count := strings.Count(s, thinkingToken)

	if count >= 2 {
		// Normal case: thinking_content ང actual_content
		// Find the position after the second thinking token
		idx := strings.Index(s, thinkingToken)
		rest := s[idx+len(thinkingToken):]
		idx2 := strings.Index(rest, thinkingToken)
		if idx2 >= 0 {
			s = strings.TrimSpace(rest[idx2+len(thinkingToken):])
		}
	} else if count == 1 {
		// Only one thinking token - check what comes after
		idx := strings.Index(s, thinkingToken)
		after := strings.TrimSpace(s[idx+len(thinkingToken):])

		// If it starts with thinking process markers, the model failed
		if strings.HasPrefix(after, "Thinking Process") ||
			strings.HasPrefix(after, "Analyze") ||
			strings.HasPrefix(after, "**Task") ||
			strings.HasPrefix(after, "* Task") {
			// Return empty to trigger fallback
			return ""
		}
		s = after
	}

	// Extract summary from JSON if present
	// Look for {"summary": "..."} or just the value after {"summary": "
	if strings.Contains(s, `"summary"`) {
		// Find the start of the summary value
		if idx := strings.Index(s, `"summary": "`); idx >= 0 {
			start := idx + len(`"summary": "`)
			// Find the end quote
			if endIdx := strings.Index(s[start:], `"`); endIdx > 0 {
				s = s[start : start+endIdx]
			} else if endIdx := strings.Index(s[start:], `"}`); endIdx > 0 {
				s = s[start : start+endIdx]
			}
		}
	}

	// Remove stop tokens
	s = strings.Split(s, "<|im_end|")[0]
	s = strings.Split(s, "<|im_start|")[0]
	s = strings.Split(s, "</s>")[0]
	s = strings.Split(s, `"}`)[0]
	s = strings.Split(s, `"}`)[0]

	// Remove leading labels
	s = strings.TrimPrefix(s, "Summary:")
	s = strings.TrimPrefix(s, "Summary: ")

	// Remove markdown bold
	s = strings.ReplaceAll(s, "**", "")

	// Remove newlines
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.ReplaceAll(s, "\r", "")

	// Collapse multiple spaces
	for strings.Contains(s, "  ") {
		s = strings.ReplaceAll(s, "  ", " ")
	}

	return strings.TrimSpace(s)
}

func truncateMiddle(s string, maxLen int) string {
	if utf8.RuneCountInString(s) <= maxLen {
		return s
	}
	runes := []rune(s)
	h, t := maxLen/2, maxLen-maxLen/2
	result := make([]rune, maxLen)
	copy(result, runes[:h])
	copy(result[h:], runes[len(runes)-t:])
	return string(result)
}

// localImpl wraps llamaSummarizer to implement the localSummarizer interface.
type localImpl struct {
	inner *llamaSummarizer
}

func (l *localImpl) Summarize(ctx context.Context, reasoning string) (localSummary, error) {
	return l.inner.Summarize(ctx, reasoning)
}

func (l *localImpl) Close() {
	l.inner.Close()
}

// newLocalService creates a local summarizer service when built without the nosummarizer tag.
func newLocalService(cfg config.SummarizerConfig, s *Service) *Service {
	if cfg.Local.ModelPath == "" {
		logging.ErrorMsg("Local summarizer requires model_path, summarizer disabled")
		return nil
	}

	llamaCfg := defaultLlamaConfig(cfg.Local.ModelPath)
	if cfg.Local.ContextSize > 0 {
		llamaCfg.ContextSize = cfg.Local.ContextSize
	}
	if cfg.Local.Threads > 0 {
		llamaCfg.Threads = cfg.Local.Threads
	}
	if cfg.Local.GPULayers > 0 {
		llamaCfg.GPULayers = cfg.Local.GPULayers
	}
	if cfg.Local.MaxSummaryTokens > 0 {
		llamaCfg.MaxSummaryTokens = cfg.Local.MaxSummaryTokens
	}
	if cfg.Local.MaxReasoningChars > 0 {
		llamaCfg.MaxReasoningChars = cfg.Local.MaxReasoningChars
	}

	llama, err := newLlamaSummarizer(llamaCfg)
	if err != nil {
		logging.ErrorMsg("Failed to initialize local summarizer: %v", err)
		return nil
	}

	s.local = &localImpl{inner: llama}
	logging.InfoMsg("Summarizer initialized: mode=local, model=%s", cfg.Local.ModelPath)
	return s
}