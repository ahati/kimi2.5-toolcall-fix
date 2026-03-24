// Package config provides configuration structs for the multi-provider multi-protocol proxy.
// This file defines the JSON schema for provider and model configuration.
package config

import (
	"os"
)

// Provider defines an upstream API provider configuration.
// A provider represents an external API service that can handle requests.
type Provider struct {
	// Name is the unique identifier for this provider.
	Name string `json:"name"`
	// Endpoints maps protocol names to their specific endpoint URLs.
	// Required: at least one endpoint must be specified.
	// Protocols: "openai", "anthropic", "responses"
	// Example: {"openai": "https://api.provider.com/v1/chat/completions"}
	Endpoints map[string]string `json:"endpoints"`
	// Default specifies the default protocol for multi-protocol providers.
	// Required when Endpoints has more than one entry.
	// Must be one of: "openai", "anthropic", "responses".
	Default string `json:"default,omitempty"`
	// APIKey is the direct API key for authentication (optional).
	// If not set, EnvAPIKey is used to fetch from environment.
	APIKey string `json:"apiKey,omitempty"`
	// EnvAPIKey is the environment variable name containing the API key.
	// Used when APIKey is not directly set.
	EnvAPIKey string `json:"envApiKey,omitempty"`
}

// GetAPIKey returns the API key for this provider.
// If APIKey is set directly, it is returned.
// Otherwise, the value is fetched from the environment variable specified by EnvAPIKey.
//
// @return string - the resolved API key, or empty string if not configured
func (p *Provider) GetAPIKey() string {
	if p.APIKey != "" {
		return p.APIKey
	}
	return os.Getenv(p.EnvAPIKey)
}

// GetEndpoint returns the endpoint URL for the specified protocol.
//
// @param protocol - the protocol name ("openai", "anthropic", "responses")
// @return the endpoint URL, or empty string if not found
func (p *Provider) GetEndpoint(protocol string) string {
	return p.Endpoints[protocol]
}

// SupportedProtocols returns the list of protocols this provider supports.
//
// @return slice of supported protocol names
func (p *Provider) SupportedProtocols() []string {
	protocols := make([]string, 0, len(p.Endpoints))
	for protocol := range p.Endpoints {
		protocols = append(protocols, protocol)
	}
	return protocols
}

// HasProtocol checks if the provider supports the given protocol.
//
// @param protocol - the protocol to check
// @return true if the protocol is supported
func (p *Provider) HasProtocol(protocol string) bool {
	_, exists := p.Endpoints[protocol]
	return exists
}

// GetDefaultProtocol returns the default protocol for this provider.
// Returns the configured Default field, or the only protocol if single-endpoint.
//
// @return the default protocol name, or empty string if none configured
func (p *Provider) GetDefaultProtocol() string {
	if p.Default != "" {
		return p.Default
	}
	// Single protocol: return the only one
	if len(p.Endpoints) == 1 {
		for protocol := range p.Endpoints {
			return protocol
		}
	}
	return ""
}

// ModelConfig defines how a model alias maps to a specific provider and model.
// This allows routing requests to the appropriate upstream provider.
type ModelConfig struct {
	// Provider is the name of the provider to use for this model.
	Provider string `json:"provider"`
	// Model is the actual model identifier to use on the upstream provider.
	Model string `json:"model"`
	// Type specifies the output protocol: "openai", "anthropic", or "auto".
	// "auto" means use the incoming request's protocol for passthrough.
	// Empty defaults to provider's default protocol.
	Type string `json:"type,omitempty"`
	// KimiToolCallTransform enables tool call transformation for this model.
	// When true, tool calls are transformed between OpenAI and Anthropic formats.
	KimiToolCallTransform bool `json:"kimi_tool_call_transform"`
	// GLM5ToolCallTransform enables GLM-5 style XML tool call extraction.
	// When true, extracts tool calls from <tool_call> tags in reasoning_content.
	GLM5ToolCallTransform bool `json:"glm5_tool_call_transform"`
	// ReasoningSplit enables separate reasoning output for providers that support it.
	// When true, adds "reasoning_split": true to the ChatCompletionRequest.
	// Supported by MiniMax M2.7 to return reasoning in reasoning_details field
	// instead of embedded aisaI tags in content.
	ReasoningSplit bool `json:"reasoning_split,omitempty"`
}

// FallbackConfig defines the fallback behavior when a request fails.
// Fallback allows routing failed requests to an alternative provider.
type FallbackConfig struct {
	// Enabled determines whether fallback is active.
	Enabled bool `json:"enabled"`
	// Provider is the name of the fallback provider.
	Provider string `json:"provider"`
	// Model is the model to use for fallback requests.
	Model string `json:"model"`
	// Type specifies the output protocol for fallback: "openai", "anthropic", or "auto".
	Type string `json:"type,omitempty"`
	// KimiToolCallTransform enables tool call transformation for fallback requests.
	KimiToolCallTransform bool `json:"kimi_tool_call_transform"`
	// GLM5ToolCallTransform enables GLM-5 style XML tool call extraction for fallback.
	GLM5ToolCallTransform bool `json:"glm5_tool_call_transform"`
	// ReasoningSplit enables separate reasoning output for fallback requests.
	ReasoningSplit bool `json:"reasoning_split,omitempty"`
}

// SummarizerConfig defines the configuration for the reasoning summarizer.
// The summarizer uses a small fast model to generate concise summaries of
// the model's internal reasoning process.
type SummarizerConfig struct {
	// Enabled determines whether summarization is active.
	Enabled bool `json:"enabled"`
	// Mode specifies the summarizer mode: "http" (API calls) or "local" (llama.cpp).
	// Default is "http".
	Mode string `json:"mode,omitempty"`
	// Provider is the name of the provider to use for HTTP summarization.
	Provider string `json:"provider"`
	// Model is the model to use for summarization (e.g., "gpt-4o-mini", "claude-3-haiku").
	Model string `json:"model"`
	// Prompt is an optional custom prompt for summarization.
	// If empty, a default prompt is used.
	Prompt string `json:"prompt,omitempty"`
	// Local contains configuration for local llama.cpp summarization.
	Local LocalSummarizerConfig `json:"local,omitempty"`
}

// LocalSummarizerConfig defines configuration for local llama.cpp summarization.
type LocalSummarizerConfig struct {
	// ModelPath is the path to the GGUF model file.
	ModelPath string `json:"model_path"`
	// ContextSize is the context window size for the model.
	ContextSize int `json:"context_size,omitempty"`
	// Threads is the number of CPU threads to use (0 = auto).
	Threads int `json:"threads,omitempty"`
	// GPULayers is the number of layers to offload to GPU (0 = CPU-only).
	GPULayers int `json:"gpu_layers,omitempty"`
	// MaxSummaryTokens is the maximum number of tokens to generate.
	MaxSummaryTokens int `json:"max_summary_tokens,omitempty"`
	// MaxReasoningChars limits input reasoning text length (0 = unlimited).
	MaxReasoningChars int `json:"max_reasoning_chars,omitempty"`
}

// Schema is the root configuration structure for the multi-provider proxy.
// It contains all provider definitions, model mappings, and fallback settings.
type Schema struct {
	// Providers is a list of available upstream providers.
	Providers []Provider `json:"providers"`
	// Models maps model aliases to their provider and model configuration.
	Models map[string]ModelConfig `json:"models"`
	// Fallback defines the fallback behavior for failed requests.
	Fallback FallbackConfig `json:"fallback"`
	// Summarizer defines the summarizer configuration.
	Summarizer SummarizerConfig `json:"summarizer"`
}
