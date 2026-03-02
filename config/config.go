package config

import (
	"flag"
	"os"
)

type Config struct {
	OpenAIUpstreamURL    string
	OpenAIUpstreamAPIKey string
	AnthropicUpstreamURL string
	AnthropicAPIKey      string
	Port                 string
	SSELogDir            string
}

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

func getEnvOrFlag(envKey, flagValue, defaultValue string) string {
	if flagValue != "" {
		return flagValue
	}
	if value, ok := os.LookupEnv(envKey); ok {
		return value
	}
	return defaultValue
}
