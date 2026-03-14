package config

import "os"

type Provider struct {
	Name      string `json:"name"`
	Type      string `json:"type"`
	BaseURL   string `json:"base_url"`
	APIKey    string `json:"apiKey,omitempty"`
	EnvAPIKey string `json:"envApiKey,omitempty"`
}

func (p *Provider) GetAPIKey() string {
	if p.EnvAPIKey != "" {
		if key := os.Getenv(p.EnvAPIKey); key != "" {
			return key
		}
	}
	return p.APIKey
}

type ModelConfig struct {
	Provider          string `json:"provider"`
	Model             string `json:"model"`
	ToolCallTransform bool   `json:"tool_call_transform"`
}

type FallbackConfig struct {
	Enabled           bool   `json:"enabled"`
	Provider          string `json:"provider"`
	Model             string `json:"model"`
	ToolCallTransform bool   `json:"tool_call_transform"`
}

type AppConfig struct {
	Providers []Provider             `json:"providers"`
	Models    map[string]ModelConfig `json:"models"`
	Fallback  FallbackConfig         `json:"fallback"`
}
