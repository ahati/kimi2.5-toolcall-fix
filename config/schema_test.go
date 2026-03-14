package config

import (
	"encoding/json"
	"os"
	"testing"
)

func TestSchemaUnmarshal(t *testing.T) {
	jsonData := `{
		"providers": [
			{
				"name": "openai-primary",
				"type": "openai",
				"base_url": "https://api.openai.com/v1",
				"apiKey": "sk-direct-key"
			},
			{
				"name": "anthropic-backup",
				"type": "anthropic",
				"base_url": "https://api.anthropic.com/v1",
				"envApiKey": "ANTHROPIC_API_KEY"
			}
		],
		"models": {
			"gpt-4": {
				"provider": "openai-primary",
				"model": "gpt-4",
				"tool_call_transform": true
			},
			"claude-3": {
				"provider": "anthropic-backup",
				"model": "claude-3-opus",
				"tool_call_transform": false
			}
		},
		"fallback": {
			"enabled": true,
			"provider": "openai-primary",
			"model": "gpt-3.5-turbo",
			"tool_call_transform": false
		}
	}`

	var cfg SchemaConfig
	if err := json.Unmarshal([]byte(jsonData), &cfg); err != nil {
		t.Fatalf("Failed to unmarshal: %v", err)
	}

	if len(cfg.Providers) != 2 {
		t.Errorf("Expected 2 providers, got %d", len(cfg.Providers))
	}

	if cfg.Providers[0].Name != "openai-primary" {
		t.Errorf("Expected provider name 'openai-primary', got %q", cfg.Providers[0].Name)
	}

	if cfg.Providers[0].Type != "openai" {
		t.Errorf("Expected provider type 'openai', got %q", cfg.Providers[0].Type)
	}

	if cfg.Providers[0].BaseURL != "https://api.openai.com/v1" {
		t.Errorf("Expected base_url 'https://api.openai.com/v1', got %q", cfg.Providers[0].BaseURL)
	}

	if cfg.Providers[0].APIKey != "sk-direct-key" {
		t.Errorf("Expected apiKey 'sk-direct-key', got %q", cfg.Providers[0].APIKey)
	}

	if cfg.Providers[1].EnvAPIKey != "ANTHROPIC_API_KEY" {
		t.Errorf("Expected envApiKey 'ANTHROPIC_API_KEY', got %q", cfg.Providers[1].EnvAPIKey)
	}

	if len(cfg.Models) != 2 {
		t.Errorf("Expected 2 models, got %d", len(cfg.Models))
	}

	gpt4, ok := cfg.Models["gpt-4"]
	if !ok {
		t.Fatal("Expected 'gpt-4' model in models map")
	}

	if gpt4.Provider != "openai-primary" {
		t.Errorf("Expected model provider 'openai-primary', got %q", gpt4.Provider)
	}

	if gpt4.Model != "gpt-4" {
		t.Errorf("Expected model name 'gpt-4', got %q", gpt4.Model)
	}

	if !gpt4.ToolCallTransform {
		t.Error("Expected ToolCallTransform to be true for gpt-4")
	}

	claude := cfg.Models["claude-3"]
	if claude.ToolCallTransform {
		t.Error("Expected ToolCallTransform to be false for claude-3")
	}

	if !cfg.Fallback.Enabled {
		t.Error("Expected fallback enabled to be true")
	}

	if cfg.Fallback.Provider != "openai-primary" {
		t.Errorf("Expected fallback provider 'openai-primary', got %q", cfg.Fallback.Provider)
	}

	if cfg.Fallback.Model != "gpt-3.5-turbo" {
		t.Errorf("Expected fallback model 'gpt-3.5-turbo', got %q", cfg.Fallback.Model)
	}
}

func TestSchemaUnmarshalEmptyModels(t *testing.T) {
	jsonData := `{
		"providers": [],
		"models": {},
		"fallback": {
			"enabled": false,
			"provider": "",
			"model": "",
			"tool_call_transform": false
		}
	}`

	var cfg SchemaConfig
	if err := json.Unmarshal([]byte(jsonData), &cfg); err != nil {
		t.Fatalf("Failed to unmarshal: %v", err)
	}

	if len(cfg.Providers) != 0 {
		t.Errorf("Expected 0 providers, got %d", len(cfg.Providers))
	}

	if len(cfg.Models) != 0 {
		t.Errorf("Expected 0 models, got %d", len(cfg.Models))
	}

	if cfg.Fallback.Enabled {
		t.Error("Expected fallback enabled to be false")
	}
}

func TestSchemaUnmarshalFieldPresence(t *testing.T) {
	jsonData := `{
		"providers": [
			{
				"name": "test-provider",
				"type": "openai",
				"base_url": "https://example.com"
			}
		],
		"models": {
			"test-model": {
				"provider": "test-provider",
				"model": "test-model",
				"tool_call_transform": true
			}
		},
		"fallback": {
			"enabled": true,
			"provider": "test-provider",
			"model": "fallback-model",
			"tool_call_transform": false
		}
	}`

	var cfg SchemaConfig
	if err := json.Unmarshal([]byte(jsonData), &cfg); err != nil {
		t.Fatalf("Failed to unmarshal: %v", err)
	}

	provider := cfg.Providers[0]
	if provider.APIKey != "" {
		t.Error("Expected APIKey to be empty when not provided")
	}

	if provider.EnvAPIKey != "" {
		t.Error("Expected EnvAPIKey to be empty when not provided")
	}

	if provider.Name == "" {
		t.Error("Expected Name to be present")
	}

	if provider.Type == "" {
		t.Error("Expected Type to be present")
	}

	if provider.BaseURL == "" {
		t.Error("Expected BaseURL to be present")
	}
}

func TestProviderGetAPIKeyWithExplicitKey(t *testing.T) {
	provider := Provider{
		Name:    "test",
		Type:    "openai",
		BaseURL: "https://example.com",
		APIKey:  "explicit-key",
	}

	got := provider.GetAPIKey()
	if got != "explicit-key" {
		t.Errorf("GetAPIKey() = %q, want 'explicit-key'", got)
	}
}

func TestProviderGetAPIKeyWithEnvVar(t *testing.T) {
	envKey := "TEST_API_KEY_SCHEMA_12345"
	os.Setenv(envKey, "env-key-value")
	defer os.Unsetenv(envKey)

	provider := Provider{
		Name:      "test",
		Type:      "openai",
		BaseURL:   "https://example.com",
		APIKey:    "explicit-key",
		EnvAPIKey: envKey,
	}

	got := provider.GetAPIKey()
	if got != "env-key-value" {
		t.Errorf("GetAPIKey() = %q, want 'env-key-value'", got)
	}
}

func TestProviderGetAPIKeyWithEmptyEnvVar(t *testing.T) {
	envKey := "TEST_API_KEY_EMPTY_12345"
	os.Setenv(envKey, "")
	defer os.Unsetenv(envKey)

	provider := Provider{
		Name:      "test",
		Type:      "openai",
		BaseURL:   "https://example.com",
		APIKey:    "explicit-key",
		EnvAPIKey: envKey,
	}

	got := provider.GetAPIKey()
	if got != "explicit-key" {
		t.Errorf("GetAPIKey() = %q, want 'explicit-key' (should fall back when env is empty)", got)
	}
}

func TestProviderGetAPIKeyWithUnsetEnvVar(t *testing.T) {
	envKey := "TEST_API_KEY_UNSET_12345"
	os.Unsetenv(envKey)

	provider := Provider{
		Name:      "test",
		Type:      "openai",
		BaseURL:   "https://example.com",
		APIKey:    "explicit-key",
		EnvAPIKey: envKey,
	}

	got := provider.GetAPIKey()
	if got != "explicit-key" {
		t.Errorf("GetAPIKey() = %q, want 'explicit-key' (should fall back when env is unset)", got)
	}
}

func TestProviderGetAPIKeyNoEnvKeyConfigured(t *testing.T) {
	provider := Provider{
		Name:    "test",
		Type:    "openai",
		BaseURL: "https://example.com",
		APIKey:  "explicit-key",
	}

	got := provider.GetAPIKey()
	if got != "explicit-key" {
		t.Errorf("GetAPIKey() = %q, want 'explicit-key'", got)
	}
}

func TestProviderGetAPIKeyNoKeyAtAll(t *testing.T) {
	provider := Provider{
		Name:    "test",
		Type:    "openai",
		BaseURL: "https://example.com",
	}

	got := provider.GetAPIKey()
	if got != "" {
		t.Errorf("GetAPIKey() = %q, want empty string", got)
	}
}

func TestProviderGetAPIKeyEnvVarOnly(t *testing.T) {
	envKey := "TEST_API_KEY_ONLY_12345"
	os.Setenv(envKey, "only-env-key")
	defer os.Unsetenv(envKey)

	provider := Provider{
		Name:      "test",
		Type:      "openai",
		BaseURL:   "https://example.com",
		EnvAPIKey: envKey,
	}

	got := provider.GetAPIKey()
	if got != "only-env-key" {
		t.Errorf("GetAPIKey() = %q, want 'only-env-key'", got)
	}
}
