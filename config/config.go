package config

import "os"

type Config struct {
	UpstreamURL    string
	UpstreamAPIKey string
	Port           string
	SSELogDir      string
}

func Load() *Config {
	return &Config{
		UpstreamURL:    getEnv("UPSTREAM_URL", "https://llm.chutes.ai/v1/chat/completions"),
		UpstreamAPIKey: getEnv("UPSTREAM_API_KEY", ""),
		Port:           getEnv("PORT", "8080"),
		SSELogDir:      getEnvWithEmptyDefault("SSELOG_DIR", ""),
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvWithEmptyDefault(key, defaultValue string) string {
	if value, ok := os.LookupEnv(key); ok {
		return value
	}
	return defaultValue
}
