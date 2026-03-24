//go:build !nosummarizer

// Package summarizer provides local summarization using llama.cpp.
// This file is compiled by default. Use -tags nosummarizer to exclude it.
package summarizer

import (
	"context"

	"ai-proxy/config"
	"ai-proxy/logging"

	localsum "github.com/ahati/reasoning-summarizer/summarizer"
)

// localImpl wraps the actual local summarizer.
type localImpl struct {
	inner *localsum.Summarizer
}

func (l *localImpl) Summarize(ctx context.Context, reasoning string) (localSummary, error) {
	summary, err := l.inner.Summarize(ctx, reasoning)
	if err != nil {
		return localSummary{}, err
	}
	return localSummary{
		Text:      summary.Text,
		Latency:   summary.Latency,
		Truncated: summary.Truncated,
	}, nil
}

func (l *localImpl) Close() {
	l.inner.Close()
}

// newLocalService creates a local summarizer service when built with the localsummarizer tag.
func newLocalService(cfg config.SummarizerConfig, s *Service) *Service {
	if cfg.Local.ModelPath == "" {
		logging.ErrorMsg("Local summarizer requires model_path, summarizer disabled")
		return nil
	}

	localCfg := localsum.DefaultConfig(cfg.Local.ModelPath)
	if cfg.Local.ContextSize > 0 {
		localCfg.ContextSize = cfg.Local.ContextSize
	}
	if cfg.Local.Threads > 0 {
		localCfg.Threads = cfg.Local.Threads
	}
	if cfg.Local.GPULayers > 0 {
		localCfg.GPULayers = cfg.Local.GPULayers
	}
	if cfg.Local.MaxSummaryTokens > 0 {
		localCfg.MaxSummaryTokens = cfg.Local.MaxSummaryTokens
	}
	if cfg.Local.MaxReasoningChars > 0 {
		localCfg.MaxReasoningChars = cfg.Local.MaxReasoningChars
	}

	local, err := localsum.New(localCfg)
	if err != nil {
		logging.ErrorMsg("Failed to initialize local summarizer: %v", err)
		return nil
	}

	s.local = &localImpl{inner: local}
	logging.InfoMsg("Summarizer initialized: mode=local, model=%s", cfg.Local.ModelPath)
	return s
}