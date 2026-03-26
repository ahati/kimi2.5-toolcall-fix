//go:build !llama

// Package summarizer provides a stub for local summarization.
// This file is compiled by default. Use -tags llama to include llama.cpp support.
package summarizer

import (
	"ai-proxy/config"
	"ai-proxy/logging"
)

// newLocalService returns nil when local summarizer is excluded via build tag.
func newLocalService(_ config.SummarizerConfig, _ *Service) *Service {
	logging.ErrorMsg("Local summarizer mode requested, but binary was built without llama.cpp support")
	logging.ErrorMsg("Rebuild with -tags llama (or use 'make build') to enable local summarization")
	return nil
}