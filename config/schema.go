// Package config provides configuration structs for the multi-provider multi-protocol proxy.
// This file defines the JSON schema for provider and model configuration.
package config

import "os"

// Provider defines an upstream API provider configuration.
// A provider represents an external API service that can handle requests.
type Provider struct {
	// Name is the unique identifier for this provider.
	Name string `json:"name"`
	// Type specifies the provider API format: "openai" or "anthropic".
	Type string `json:"type"`
	// BaseURL is the base URL for the provider's API endpoint.
	BaseURL string `json:"base_url"`
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

// ModelConfig defines how a model alias maps to a specific provider and model.
// This allows routing requests to the appropriate upstream provider.
type ModelConfig struct {
	// Provider is the name of the provider to use for this model.
	Provider string `json:"provider"`
	// Model is the actual model identifier to use on the upstream provider.
	Model string `json:"model"`
	// ToolCallTransform enables tool call transformation for this model.
	// When true, tool calls are transformed between OpenAI and Anthropic formats.
	ToolCallTransform bool `json:"tool_call_transform"`
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
	// ToolCallTransform enables tool call transformation for fallback requests.
	ToolCallTransform bool `json:"tool_call_transform"`
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
}
