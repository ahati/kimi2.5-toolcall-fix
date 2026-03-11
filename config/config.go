// Package config provides configuration loading from environment variables and command-line flags.
// Configuration values are loaded with precedence: flags > environment variables > defaults.
package config

import (
	"flag"
	"os"
)

// Config holds all configuration settings for the proxy server.
// All settings are loaded at startup and remain immutable during runtime.
type Config struct {
	// OpenAIUpstreamURL is the base URL for OpenAI-compatible API requests.
	// Default: "https://llm.chutes.ai/v1/chat/completions"
	// Must be a valid HTTPS URL for secure communication.
	OpenAIUpstreamURL string
	// OpenAIUpstreamAPIKey is the API key for authenticating with the OpenAI-compatible API.
	// Must be set via UPSTREAM_API_KEY environment variable or --upstream-api-key flag.
	// Empty string results in authentication errors from the upstream API.
	OpenAIUpstreamAPIKey string
	// AnthropicUpstreamURL is the base URL for Anthropic-compatible API requests.
	// Default: "https://coding-intl.dashscope.aliyuncs.com/apps/anthropic/v1/messages"
	// Must be a valid HTTPS URL for secure communication.
	AnthropicUpstreamURL string
	// AnthropicAPIKey is the API key for authenticating with the Anthropic-compatible API.
	// Must be set via ALIBABA_ANTHROPIC_API_KEY environment variable or --anthropic-api-key flag.
	// Empty string results in authentication errors from the upstream API.
	AnthropicAPIKey string
	// Port is the TCP port on which the proxy server listens.
	// Default: "8080". Must be a valid port number (1-65535).
	// Ports below 1024 may require elevated privileges.
	Port string
	// SSELogDir is the directory path for logging SSE request/response data.
	// Default: "" (disabled). When set, all requests are logged to JSON files.
	// Directory must exist and be writable; subdirectories are created per date.
	SSELogDir string
}

// Load reads configuration from command-line flags and environment variables.
// Flags take precedence over environment variables, which take precedence over defaults.
//
// @return *Config - a fully initialized Config instance
// @post All configuration values are populated with resolved values
// @post Flag.Parse() has been called, consuming command-line arguments
// @note Environment variables: UPSTREAM_URL, UPSTREAM_API_KEY, ANTHROPIC_UPSTREAM_URL,
//
//	ALIBABA_ANTHROPIC_API_KEY, PORT, SSELOG_DIR
func Load() *Config {
	// Define command-line flags with empty defaults to allow env var fallback
	upstreamURL := flag.String("upstream-url", "", "Upstream URL (default: https://llm.chutes.ai/v1/chat/completions)")
	upstreamAPIKey := flag.String("upstream-api-key", "", "Upstream API Key")
	anthropicUpstreamURL := flag.String("anthropic-upstream-url", "", "Anthropic Upstream URL (default: https://api.anthropic.com/v1/messages)")
	anthropicAPIKey := flag.String("anthropic-api-key", "", "Anthropic Upstream API Key")
	port := flag.String("port", "", "Server port (default: 8080)")
	sseLogDir := flag.String("sse-log-dir", "", "Directory to log SSE responses")

	flag.Parse()

	// Build config with precedence: flag > env var > default
	return &Config{
		OpenAIUpstreamURL:    getEnvOrFlag("UPSTREAM_URL", *upstreamURL, "https://llm.chutes.ai/v1/chat/completions"),
		OpenAIUpstreamAPIKey: getEnvOrFlag("UPSTREAM_API_KEY", *upstreamAPIKey, ""),
		AnthropicUpstreamURL: getEnvOrFlag("ANTHROPIC_UPSTREAM_URL", *anthropicUpstreamURL, "https://coding-intl.dashscope.aliyuncs.com/apps/anthropic/v1/messages"),
		AnthropicAPIKey:      getEnvOrFlag("ALIBABA_ANTHROPIC_API_KEY", *anthropicAPIKey, ""),
		Port:                 getEnvOrFlag("PORT", *port, "8080"),
		SSELogDir:            getEnvOrFlag("SSELOG_DIR", *sseLogDir, ""),
	}
}

// getEnvOrFlag returns the flag value if non-empty, otherwise the environment variable if set, otherwise the default value.
// This implements the precedence chain: flag > environment > default.
//
// @param envKey - the environment variable name to check
// @param flagValue - the flag value (may be empty if not provided)
// @param defaultValue - the fallback default value
// @return string - the resolved configuration value
// @post Returns non-empty string (defaultValue may be empty for optional settings)
// @note Flag values take highest precedence for explicit command-line control
func getEnvOrFlag(envKey, flagValue, defaultValue string) string {
	// Flag value takes highest precedence when explicitly provided
	if flagValue != "" {
		return flagValue
	}
	// Environment variable is second precedence
	if value, ok := os.LookupEnv(envKey); ok {
		return value
	}
	// Default is the fallback when neither flag nor env is set
	return defaultValue
}
