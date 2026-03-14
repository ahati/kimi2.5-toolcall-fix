package config

import (
	"fmt"
	"os"
)

type Config struct {
	AppConfig *AppConfig
	Port      string
	SSELogDir string
}

func Load(path string) (*Config, error) {
	appCfg, err := LoadFromPath(path)
	if err != nil {
		return nil, err
	}

	cfg := &Config{
		AppConfig: appCfg,
		Port:      getEnvWithDefault("PORT", "8080"),
		SSELogDir: os.Getenv("SSELOG_DIR"),
	}

	return cfg, nil
}

func LoadFromEnv() (*Config, error) {
	path := os.Getenv("CONFIG_FILE")
	if path == "" {
		return nil, fmt.Errorf("CONFIG_FILE environment variable not set")
	}
	return Load(path)
}

func (c *Config) GetProvider(name string) (Provider, bool) {
	for _, p := range c.AppConfig.Providers {
		if p.Name == name {
			return p, true
		}
	}
	return Provider{}, false
}

func (c *Config) GetModel(name string) (ModelConfig, bool) {
	mc, ok := c.AppConfig.Models[name]
	return mc, ok
}

func (c *Config) GetOpenAIProvider() (Provider, bool) {
	for _, p := range c.AppConfig.Providers {
		if p.Type == "openai" {
			return p, true
		}
	}
	return Provider{}, false
}

func (c *Config) GetAnthropicProvider() (Provider, bool) {
	for _, p := range c.AppConfig.Providers {
		if p.Type == "anthropic" {
			return p, true
		}
	}
	return Provider{}, false
}

func getEnvWithDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func NewTestConfig(providers []Provider, models map[string]ModelConfig) *Config {
	if providers == nil {
		providers = []Provider{}
	}
	if models == nil {
		models = map[string]ModelConfig{}
	}
	return &Config{
		AppConfig: &AppConfig{
			Providers: providers,
			Models:    models,
			Fallback:  FallbackConfig{Enabled: false},
		},
		Port:      "8080",
		SSELogDir: "",
	}
}

func NewTestConfigWithOpenAI(baseURL, apiKey string) *Config {
	return NewTestConfig([]Provider{
		{Name: "openai", Type: "openai", BaseURL: baseURL, APIKey: apiKey},
	}, nil)
}

func NewTestConfigWithAnthropic(baseURL, apiKey string) *Config {
	return NewTestConfig([]Provider{
		{Name: "anthropic", Type: "anthropic", BaseURL: baseURL, APIKey: apiKey},
	}, nil)
}

func NewTestConfigWithBoth(openaiURL, openaiKey, anthropicURL, anthropicKey string) *Config {
	return NewTestConfig([]Provider{
		{Name: "openai", Type: "openai", BaseURL: openaiURL, APIKey: openaiKey},
		{Name: "anthropic", Type: "anthropic", BaseURL: anthropicURL, APIKey: anthropicKey},
	}, nil)
}
