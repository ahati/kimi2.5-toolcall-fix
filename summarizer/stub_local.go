//go:build nosummarizer

// Package summarizer provides a stub for local summarization.
// This file is only compiled when the "nosummarizer" build tag is set.
// Use this to build without llama.cpp CGo dependencies.
package summarizer

import (
	"ai-proxy/config"
	"ai-proxy/logging"
)

// newLocalService returns nil when local summarizer is excluded via build tag.
func newLocalService(_ config.SummarizerConfig, _ *Service) *Service {
	logging.ErrorMsg("Local summarizer mode requested, but binary was built with 'nosummarizer' tag")
	logging.ErrorMsg("Rebuild without -tags nosummarizer to enable local summarization")
	return nil
}