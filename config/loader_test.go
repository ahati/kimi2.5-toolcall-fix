package config

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLoader_Load_ValidConfig(t *testing.T) {
	jsonContent := `{
		"providers": [
			{"name": "openai", "type": "openai", "base_url": "https://api.openai.com/v1", "apiKey": "sk-test"}
		],
		"models": {
			"gpt-4": {"provider": "openai", "model": "gpt-4", "tool_call_transform": false}
		},
		"fallback": {"enabled": false}
	}`

	tmpFile := createTempConfig(t, jsonContent)
	defer os.Remove(tmpFile)

	cfg, err := LoadFromPath(tmpFile)
	require.NoError(t, err)
	require.NotNil(t, cfg)

	assert.Len(t, cfg.Providers, 1)
	assert.Equal(t, "openai", cfg.Providers[0].Name)
	assert.Equal(t, "openai", cfg.Providers[0].Type)
	assert.Equal(t, "https://api.openai.com/v1", cfg.Providers[0].BaseURL)

	assert.Len(t, cfg.Models, 1)
	assert.Equal(t, "openai", cfg.Models["gpt-4"].Provider)
	assert.Equal(t, "gpt-4", cfg.Models["gpt-4"].Model)
}

func TestLoader_Load_MissingFile(t *testing.T) {
	l := NewLoader()
	_, err := l.Load("/nonexistent/path/config.json")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "read config file")
}

func TestLoader_Load_InvalidJSON(t *testing.T) {
	jsonContent := `{invalid json`

	tmpFile := createTempConfig(t, jsonContent)
	defer os.Remove(tmpFile)

	l := NewLoader()
	_, err := l.Load(tmpFile)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "parse config file")
}

func TestLoader_Validate_NoProviders(t *testing.T) {
	cfg := &AppConfig{
		Providers: []Provider{},
		Models:    map[string]ModelConfig{},
	}

	l := NewLoader()
	err := l.validate(cfg)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "at least one provider is required")
}

func TestLoader_Validate_ProviderMissingName(t *testing.T) {
	cfg := &AppConfig{
		Providers: []Provider{
			{Name: "", Type: "openai", BaseURL: "https://api.openai.com", APIKey: "key"},
		},
		Models: map[string]ModelConfig{},
	}

	l := NewLoader()
	err := l.validate(cfg)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "provider[0]: name is required")
}

func TestLoader_Validate_ProviderMissingType(t *testing.T) {
	cfg := &AppConfig{
		Providers: []Provider{
			{Name: "test", Type: "", BaseURL: "https://api.test.com", APIKey: "key"},
		},
		Models: map[string]ModelConfig{},
	}

	l := NewLoader()
	err := l.validate(cfg)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "provider[\"test\"]: type is required")
}

func TestLoader_Validate_ProviderInvalidType(t *testing.T) {
	cfg := &AppConfig{
		Providers: []Provider{
			{Name: "test", Type: "invalid", BaseURL: "https://api.test.com", APIKey: "key"},
		},
		Models: map[string]ModelConfig{},
	}

	l := NewLoader()
	err := l.validate(cfg)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "type must be 'openai' or 'anthropic'")
}

func TestLoader_Validate_ProviderMissingBaseURL(t *testing.T) {
	cfg := &AppConfig{
		Providers: []Provider{
			{Name: "test", Type: "openai", BaseURL: "", APIKey: "key"},
		},
		Models: map[string]ModelConfig{},
	}

	l := NewLoader()
	err := l.validate(cfg)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "base_url is required")
}

func TestLoader_Validate_ProviderMissingAPIKey(t *testing.T) {
	cfg := &AppConfig{
		Providers: []Provider{
			{Name: "test", Type: "openai", BaseURL: "https://api.test.com"},
		},
		Models: map[string]ModelConfig{},
	}

	l := NewLoader()
	err := l.validate(cfg)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "apiKey or envApiKey is required")
}

func TestLoader_Validate_ProviderDuplicateName(t *testing.T) {
	cfg := &AppConfig{
		Providers: []Provider{
			{Name: "test", Type: "openai", BaseURL: "https://api1.test.com", APIKey: "key1"},
			{Name: "test", Type: "anthropic", BaseURL: "https://api2.test.com", APIKey: "key2"},
		},
		Models: map[string]ModelConfig{},
	}

	l := NewLoader()
	err := l.validate(cfg)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "duplicate name")
}

func TestLoader_Validate_ModelMissingProvider(t *testing.T) {
	cfg := &AppConfig{
		Providers: []Provider{
			{Name: "test", Type: "openai", BaseURL: "https://api.test.com", APIKey: "key"},
		},
		Models: map[string]ModelConfig{
			"gpt-4": {Provider: "", Model: "gpt-4"},
		},
	}

	l := NewLoader()
	err := l.validate(cfg)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "model[\"gpt-4\"]: provider is required")
}

func TestLoader_Validate_ModelUnknownProvider(t *testing.T) {
	cfg := &AppConfig{
		Providers: []Provider{
			{Name: "openai", Type: "openai", BaseURL: "https://api.openai.com", APIKey: "key"},
		},
		Models: map[string]ModelConfig{
			"gpt-4": {Provider: "anthropic", Model: "gpt-4"},
		},
	}

	l := NewLoader()
	err := l.validate(cfg)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "unknown provider")
}

func TestLoader_Validate_ModelMissingModel(t *testing.T) {
	cfg := &AppConfig{
		Providers: []Provider{
			{Name: "openai", Type: "openai", BaseURL: "https://api.openai.com", APIKey: "key"},
		},
		Models: map[string]ModelConfig{
			"gpt-4": {Provider: "openai", Model: ""},
		},
	}

	l := NewLoader()
	err := l.validate(cfg)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "model[\"gpt-4\"]: model is required")
}

func TestLoader_Validate_FallbackMissingProvider(t *testing.T) {
	cfg := &AppConfig{
		Providers: []Provider{
			{Name: "openai", Type: "openai", BaseURL: "https://api.openai.com", APIKey: "key"},
		},
		Models: map[string]ModelConfig{},
		Fallback: FallbackConfig{
			Enabled:  true,
			Provider: "",
			Model:    "fallback-model",
		},
	}

	l := NewLoader()
	err := l.validate(cfg)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "fallback: provider is required when enabled")
}

func TestLoader_Validate_FallbackUnknownProvider(t *testing.T) {
	cfg := &AppConfig{
		Providers: []Provider{
			{Name: "openai", Type: "openai", BaseURL: "https://api.openai.com", APIKey: "key"},
		},
		Models: map[string]ModelConfig{},
		Fallback: FallbackConfig{
			Enabled:  true,
			Provider: "unknown",
			Model:    "fallback-model",
		},
	}

	l := NewLoader()
	err := l.validate(cfg)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "fallback: unknown provider")
}

func TestLoader_Validate_FallbackMissingModel(t *testing.T) {
	cfg := &AppConfig{
		Providers: []Provider{
			{Name: "openai", Type: "openai", BaseURL: "https://api.openai.com", APIKey: "key"},
		},
		Models: map[string]ModelConfig{},
		Fallback: FallbackConfig{
			Enabled:  true,
			Provider: "openai",
			Model:    "",
		},
	}

	l := NewLoader()
	err := l.validate(cfg)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "fallback: model is required when enabled")
}

func TestLoader_Validate_FallbackDisabled(t *testing.T) {
	cfg := &AppConfig{
		Providers: []Provider{
			{Name: "openai", Type: "openai", BaseURL: "https://api.openai.com", APIKey: "key"},
		},
		Models: map[string]ModelConfig{},
		Fallback: FallbackConfig{
			Enabled: false,
		},
	}

	l := NewLoader()
	err := l.validate(cfg)
	assert.NoError(t, err)
}

func TestLoader_Validate_ValidProviderWithEnvAPIKey(t *testing.T) {
	cfg := &AppConfig{
		Providers: []Provider{
			{Name: "openai", Type: "openai", BaseURL: "https://api.openai.com", EnvAPIKey: "OPENAI_KEY"},
		},
		Models: map[string]ModelConfig{},
	}

	l := NewLoader()
	err := l.validate(cfg)
	assert.NoError(t, err)
}

func TestLoader_Validate_ValidProviderWithAPIKey(t *testing.T) {
	cfg := &AppConfig{
		Providers: []Provider{
			{Name: "openai", Type: "openai", BaseURL: "https://api.openai.com", APIKey: "sk-test"},
		},
		Models: map[string]ModelConfig{},
	}

	l := NewLoader()
	err := l.validate(cfg)
	assert.NoError(t, err)
}

func TestLoader_Validate_ValidFallback(t *testing.T) {
	cfg := &AppConfig{
		Providers: []Provider{
			{Name: "openai", Type: "openai", BaseURL: "https://api.openai.com", APIKey: "key"},
		},
		Models: map[string]ModelConfig{},
		Fallback: FallbackConfig{
			Enabled:  true,
			Provider: "openai",
			Model:    "gpt-3.5-turbo",
		},
	}

	l := NewLoader()
	err := l.validate(cfg)
	assert.NoError(t, err)
}

func TestLoader_Validate_AnthropicProvider(t *testing.T) {
	cfg := &AppConfig{
		Providers: []Provider{
			{Name: "anthropic", Type: "anthropic", BaseURL: "https://api.anthropic.com", APIKey: "key"},
		},
		Models: map[string]ModelConfig{},
	}

	l := NewLoader()
	err := l.validate(cfg)
	assert.NoError(t, err)
}

func TestLoader_Load_ComplexConfig(t *testing.T) {
	jsonContent := `{
		"providers": [
			{"name": "openai", "type": "openai", "base_url": "https://api.openai.com/v1", "envApiKey": "OPENAI_API_KEY"},
			{"name": "anthropic", "type": "anthropic", "base_url": "https://api.anthropic.com", "apiKey": "sk-ant-test"}
		],
		"models": {
			"gpt-4": {"provider": "openai", "model": "gpt-4-turbo", "tool_call_transform": false},
			"claude": {"provider": "anthropic", "model": "claude-3-opus", "tool_call_transform": true}
		},
		"fallback": {"enabled": true, "provider": "openai", "model": "{model}", "tool_call_transform": false}
	}`

	tmpFile := createTempConfig(t, jsonContent)
	defer os.Remove(tmpFile)

	cfg, err := LoadFromPath(tmpFile)
	require.NoError(t, err)

	assert.Len(t, cfg.Providers, 2)
	assert.Len(t, cfg.Models, 2)
	assert.True(t, cfg.Fallback.Enabled)
	assert.Equal(t, "{model}", cfg.Fallback.Model)
}

func createTempConfig(t *testing.T, content string) string {
	t.Helper()
	tmpDir := t.TempDir()
	tmpFile := filepath.Join(tmpDir, "config.json")
	err := os.WriteFile(tmpFile, []byte(content), 0644)
	require.NoError(t, err)
	return tmpFile
}
