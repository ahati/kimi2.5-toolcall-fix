// Package config provides configuration loading from environment variables and command-line flags.
// It centralizes all application configuration with sensible defaults.
package config

import (
	"flag"
	"os"
)

// Config holds all application configuration settings.
//
// @brief Application configuration container.
//
// Fields:
//   - OpenAIUpstreamURL: URL for OpenAI-compatible upstream API
//   - OpenAIUpstreamAPIKey: API key for OpenAI-compatible upstream
//   - AnthropicUpstreamURL: URL for Anthropic-compatible upstream API
//   - AnthropicAPIKey: API key for Anthropic-compatible upstream
//   - Port: Server listen port
//   - SSELogDir: Directory for SSE capture logs (empty to disable)
type Config struct {
	OpenAIUpstreamURL    string
	OpenAIUpstreamAPIKey string
	AnthropicUpstreamURL string
	AnthropicAPIKey      string
	Port                 string
	SSELogDir            string
}

// Load reads configuration from environment variables and command-line flags.
//
// @brief    Loads application configuration with defaults.
// @return   Pointer to populated Config struct.
//
// @note     Environment variables take precedence over defaults.
// @note     Command-line flags take precedence over environment variables.
// @note     Flags are parsed on first call; subsequent calls return same result.
//
// @pre      None.
// @post     Flag package state is modified (flags parsed).
func Load() *Config {
	upstreamURL := flag.String("upstream-url", "", "Upstream URL (default: https://llm.chutes.ai/v1/chat/completions)")
	upstreamAPIKey := flag.String("upstream-api-key", "", "Upstream API Key")
	anthropicUpstreamURL := flag.String("anthropic-upstream-url", "", "Anthropic Upstream URL (default: https://api.anthropic.com/v1/messages)")
	anthropicAPIKey := flag.String("anthropic-api-key", "", "Anthropic Upstream API Key")
	port := flag.String("port", "", "Server port (default: 8080)")
	sseLogDir := flag.String("sse-log-dir", "", "Directory to log SSE responses")

	flag.Parse()

	return &Config{
		OpenAIUpstreamURL:    getEnvOrFlag("UPSTREAM_URL", *upstreamURL, "https://llm.chutes.ai/v1/chat/completions"),
		OpenAIUpstreamAPIKey: getEnvOrFlag("UPSTREAM_API_KEY", *upstreamAPIKey, ""),
		AnthropicUpstreamURL: getEnvOrFlag("ANTHROPIC_UPSTREAM_URL", *anthropicUpstreamURL, "https://coding-intl.dashscope.aliyuncs.com/apps/anthropic/v1/messages"),
		AnthropicAPIKey:      getEnvOrFlag("ALIBABA_ANTHROPIC_API_KEY", *anthropicAPIKey, ""),
		Port:                 getEnvOrFlag("PORT", *port, "8080"),
		SSELogDir:            getEnvOrFlag("SSELOG_DIR", *sseLogDir, ""),
	}
}

// getEnvOrFlag resolves a configuration value from flag, environment variable, or default.
//
// @brief    Returns configuration value with fallback precedence.
// @param    envKey       Environment variable name to check.
// @param    flagValue    Value from command-line flag (empty if not set).
// @param    defaultValue Default value if neither flag nor env is set.
// @return   Resolved configuration value as string.
//
// @note     Precedence order: flag > environment variable > default.
//
// @pre      None.
// @post     None.
func getEnvOrFlag(envKey, flagValue, defaultValue string) string {
	if flagValue != "" {
		return flagValue
	}
	if value, ok := os.LookupEnv(envKey); ok {
		return value
	}
	return defaultValue
}
