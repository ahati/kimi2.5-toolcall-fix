package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
)

type Loader struct{}

func NewLoader() *Loader {
	return &Loader{}
}

func (l *Loader) Load(path string) (*SchemaConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}
	var cfg SchemaConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config JSON: %w", err)
	}
	if err := l.validate(&cfg); err != nil {
		return nil, err
	}
	if err := l.resolveEnvVars(&cfg); err != nil {
		return nil, err
	}
	return &cfg, nil
}

func (l *Loader) validate(cfg *SchemaConfig) error {
	if len(cfg.Providers) == 0 {
		return errors.New("at least one provider is required")
	}
	providerNames := make(map[string]bool)
	for i, p := range cfg.Providers {
		if err := l.validateProvider(p, i); err != nil {
			return err
		}
		providerNames[p.Name] = true
	}
	for modelName, mc := range cfg.Models {
		if !providerNames[mc.Provider] {
			return fmt.Errorf("model %q references unknown provider %q", modelName, mc.Provider)
		}
	}
	if cfg.Fallback.Enabled {
		if cfg.Fallback.Provider == "" {
			return errors.New("fallback provider is required when fallback is enabled")
		}
		if !providerNames[cfg.Fallback.Provider] {
			return fmt.Errorf("fallback provider %q does not exist", cfg.Fallback.Provider)
		}
	}
	return nil
}

func (l *Loader) validateProvider(p Provider, index int) error {
	if p.Name == "" {
		return fmt.Errorf("provider at index %d: name is required", index)
	}
	if p.Type == "" {
		return fmt.Errorf("provider %q: type is required", p.Name)
	}
	if p.Type != "openai" && p.Type != "anthropic" {
		return fmt.Errorf("provider %q: type must be \"openai\" or \"anthropic\", got %q", p.Name, p.Type)
	}
	if p.BaseURL == "" {
		return fmt.Errorf("provider %q: base_url is required", p.Name)
	}
	if p.APIKey == "" && p.EnvAPIKey == "" {
		return fmt.Errorf("provider %q: at least one apiKey or envApiKey is required", p.Name)
	}
	return nil
}

func (l *Loader) resolveEnvVars(cfg *SchemaConfig) error {
	for i := range cfg.Providers {
		p := &cfg.Providers[i]
		if p.EnvAPIKey != "" {
			if value := os.Getenv(p.EnvAPIKey); value != "" {
				p.APIKey = value
			}
		}
		if p.BaseURL != "" && strings.HasPrefix(p.BaseURL, "$") {
			envVar := strings.TrimPrefix(p.BaseURL, "$")
			if value := os.Getenv(envVar); value != "" {
				p.BaseURL = value
			}
		}
	}
	if cfg.Fallback.Model != "" && strings.HasPrefix(cfg.Fallback.Model, "$") {
		envVar := strings.TrimPrefix(cfg.Fallback.Model, "$")
		if value := os.Getenv(envVar); value != "" {
			cfg.Fallback.Model = value
		}
	}
	return nil
}
