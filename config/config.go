// Package config provides configuration loading from environment variables and command-line flags.
// Configuration values are loaded with precedence: flags > environment variables > defaults.
package config

import (
	"os"
)

// Config holds all configuration settings for the proxy server.
// All settings are loaded at startup and remain immutable during runtime.
type Config struct {
	// Port is the TCP port on which the proxy server listens.
	// Default: "8080". Must be a valid port number (1-65535).
	// Ports below 1024 may require elevated privileges.
	Port string
	// SSELogDir is the directory path for logging SSE request/response data.
	// Default: "" (disabled). When set, all requests are logged to JSON files.
	// Directory must exist and be writable; subdirectories are created per date.
	SSELogDir string
	// ConfigFile is the path to the JSON configuration file.
	// Set via --config-file flag or CONFIG_FILE environment variable.
	ConfigFile string
	// AppConfig holds the loaded JSON configuration schema.
	// Contains provider definitions, model mappings, and fallback settings.
	AppConfig *Schema
}

// Load reads configuration from command-line flags, environment variables, and JSON config file.
// Flags take precedence over environment variables for Port and SSELogDir.
// JSON config file is required and loaded via --config-file flag or CONFIG_FILE env var.
//
// @return *Config - a fully initialized Config instance
// @post All configuration values are populated with resolved values
// @post Flag.Parse() has been called, consuming command-line arguments
// @note Environment variables: PORT, SSELOG_DIR, CONFIG_FILE
func Load() *Config {
	// Parse CLI flags to get config file path
	configFile, err := ParseFlags()
	if err != nil {
		// If config file is required but not provided, return config with error info
		// The caller should check AppConfig == nil
		return &Config{
			Port:      getEnvOrFlag("PORT", "", "8080"),
			SSELogDir: getEnvOrFlag("SSELOG_DIR", "", ""),
		}
	}

	// Load JSON config file
	loader := NewLoader()
	appConfig, err := loader.Load(configFile)
	if err != nil {
		// Return config with nil AppConfig, caller should handle error
		return &Config{
			Port:       getEnvOrFlag("PORT", "", "8080"),
			SSELogDir:  getEnvOrFlag("SSELOG_DIR", "", ""),
			ConfigFile: configFile,
		}
	}

	// Build config with precedence: flag > env var > default
	return &Config{
		Port:       getEnvOrFlag("PORT", "", "8080"),
		SSELogDir:  getEnvOrFlag("SSELOG_DIR", "", ""),
		ConfigFile: configFile,
		AppConfig:  appConfig,
	}
}

// GetSchema returns the loaded configuration schema.
// Returns nil if the config file was not loaded successfully.
//
// @return *Schema - the loaded schema, or nil if not loaded
func (c *Config) GetSchema() *Schema {
	return c.AppConfig
}

// GetAnthropicUpstreamURL returns the base URL for the first Anthropic-type provider.
// Returns empty string if no Anthropic provider is configured.
//
// @return string - the base URL for Anthropic API, or empty if not found
func (c *Config) GetAnthropicUpstreamURL() string {
	provider := c.getProviderByType("anthropic")
	if provider != nil {
		return provider.BaseURL
	}
	return ""
}

// GetAnthropicAPIKey returns the API key for the first Anthropic-type provider.
// Returns empty string if no Anthropic provider is configured.
//
// @return string - the API key for Anthropic API, or empty if not found
func (c *Config) GetAnthropicAPIKey() string {
	provider := c.getProviderByType("anthropic")
	if provider != nil {
		return provider.GetAPIKey()
	}
	return ""
}

// GetOpenAIUpstreamURL returns the base URL for the first OpenAI-type provider.
// Returns empty string if no OpenAI provider is configured.
//
// @return string - the base URL for OpenAI API, or empty if not found
func (c *Config) GetOpenAIUpstreamURL() string {
	provider := c.getProviderByType("openai")
	if provider != nil {
		return provider.BaseURL
	}
	return ""
}

// GetOpenAIUpstreamAPIKey returns the API key for the first OpenAI-type provider.
// Returns empty string if no OpenAI provider is configured.
//
// @return string - the API key for OpenAI API, or empty if not found
func (c *Config) GetOpenAIUpstreamAPIKey() string {
	provider := c.getProviderByType("openai")
	if provider != nil {
		return provider.GetAPIKey()
	}
	return ""
}

// getProviderByType returns the first provider matching the given type.
// Returns nil if no matching provider is found or if AppConfig is nil.
//
// @param providerType - the type to match (e.g., "anthropic", "openai")
// @return *Provider - the matching provider, or nil if not found
func (c *Config) getProviderByType(providerType string) *Provider {
	if c.AppConfig == nil {
		return nil
	}
	for i := range c.AppConfig.Providers {
		if c.AppConfig.Providers[i].Type == providerType {
			return &c.AppConfig.Providers[i]
		}
	}
	return nil
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
