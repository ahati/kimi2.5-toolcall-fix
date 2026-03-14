package config

import (
	"encoding/json"
	"fmt"
	"os"
)

type Loader struct{}

func NewLoader() *Loader {
	return &Loader{}
}

func (l *Loader) Load(path string) (*AppConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config file: %w", err)
	}

	var cfg AppConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config file: %w", err)
	}

	if err := l.validate(&cfg); err != nil {
		return nil, fmt.Errorf("validate config: %w", err)
	}

	return &cfg, nil
}

func (l *Loader) validate(cfg *AppConfig) error {
	if len(cfg.Providers) == 0 {
		return fmt.Errorf("at least one provider is required")
	}

	providerNames := make(map[string]bool)
	for i, p := range cfg.Providers {
		if p.Name == "" {
			return fmt.Errorf("provider[%d]: name is required", i)
		}
		if providerNames[p.Name] {
			return fmt.Errorf("provider[%d]: duplicate name %q", i, p.Name)
		}
		providerNames[p.Name] = true

		if p.Type == "" {
			return fmt.Errorf("provider[%q]: type is required", p.Name)
		}
		if p.Type != "openai" && p.Type != "anthropic" {
			return fmt.Errorf("provider[%q]: type must be 'openai' or 'anthropic', got %q", p.Name, p.Type)
		}
		if p.BaseURL == "" {
			return fmt.Errorf("provider[%q]: base_url is required", p.Name)
		}
		if p.APIKey == "" && p.EnvAPIKey == "" {
			return fmt.Errorf("provider[%q]: apiKey or envApiKey is required", p.Name)
		}
	}

	for modelName, mc := range cfg.Models {
		if mc.Provider == "" {
			return fmt.Errorf("model[%q]: provider is required", modelName)
		}
		if !providerNames[mc.Provider] {
			return fmt.Errorf("model[%q]: unknown provider %q", modelName, mc.Provider)
		}
		if mc.Model == "" {
			return fmt.Errorf("model[%q]: model is required", modelName)
		}
	}

	if cfg.Fallback.Enabled {
		if cfg.Fallback.Provider == "" {
			return fmt.Errorf("fallback: provider is required when enabled")
		}
		if !providerNames[cfg.Fallback.Provider] {
			return fmt.Errorf("fallback: unknown provider %q", cfg.Fallback.Provider)
		}
		if cfg.Fallback.Model == "" {
			return fmt.Errorf("fallback: model is required when enabled")
		}
	}

	return nil
}

func LoadFromPath(path string) (*AppConfig, error) {
	return NewLoader().Load(path)
}
