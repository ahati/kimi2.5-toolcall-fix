package config

import (
	"flag"
	"fmt"
	"os"
)

type Config struct {
	Port       string
	SSELogDir  string
	ConfigFile string
	AppConfig  *SchemaConfig
}

// GetOpenAIProvider returns the first OpenAI-type provider
func (c *Config) GetOpenAIProvider() *Provider {
	if c.AppConfig == nil {
		return nil
	}
	for _, p := range c.AppConfig.Providers {
		if p.Type == "openai" {
			return &p
		}
	}
	return nil
}

// GetAnthropicProvider returns the first Anthropic-type provider
func (c *Config) GetAnthropicProvider() *Provider {
	if c.AppConfig == nil {
		return nil
	}
	for _, p := range c.AppConfig.Providers {
		if p.Type == "anthropic" {
			return &p
		}
	}
	return nil
}

// OpenAIUpstreamURL returns the base URL for OpenAI-compatible API
func (c *Config) OpenAIUpstreamURL() string {
	p := c.GetOpenAIProvider()
	if p != nil {
		return p.BaseURL
	}
	return ""
}

// OpenAIUpstreamAPIKey returns the API key for OpenAI-compatible API
func (c *Config) OpenAIUpstreamAPIKey() string {
	p := c.GetOpenAIProvider()
	if p != nil {
		return p.GetAPIKey()
	}
	return ""
}

// AnthropicUpstreamURL returns the base URL for Anthropic-compatible API
func (c *Config) AnthropicUpstreamURL() string {
	p := c.GetAnthropicProvider()
	if p != nil {
		return p.BaseURL
	}
	return ""
}

// AnthropicAPIKey returns the API key for Anthropic-compatible API
func (c *Config) AnthropicAPIKey() string {
	p := c.GetAnthropicProvider()
	if p != nil {
		return p.GetAPIKey()
	}
	return ""
}

// LoadConfig creates a Config for testing without loading from file
func LoadConfig(appCfg *SchemaConfig) *Config {
	return &Config{
		Port:      "8080",
		AppConfig: appCfg,
	}
}

func Load() (*Config, error) {
	configFile := flag.String("config-file", "", "Path to configuration file")
	port := flag.String("port", "", "Server port (default: 8080)")
	sseLogDir := flag.String("sse-log-dir", "", "Directory to log SSE responses")

	flag.Parse()

	cfg := &Config{
		Port:       getEnvOrFlag("PORT", *port, "8080"),
		SSELogDir:  getEnvOrFlag("SSELOG_DIR", *sseLogDir, ""),
		ConfigFile: getEnvOrFlag("CONFIG_FILE", *configFile, ""),
	}

	if cfg.ConfigFile == "" {
		return nil, fmt.Errorf("--config-file or CONFIG_FILE env var required")
	}

	loader := NewLoader()
	appCfg, err := loader.Load(cfg.ConfigFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}
	cfg.AppConfig = appCfg

	return cfg, nil
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
