package config

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewLoader(t *testing.T) {
	loader := NewLoader()
	assert.NotNil(t, loader)
}

func TestLoader_Load_MissingFile(t *testing.T) {
	loader := NewLoader()
	cfg, err := loader.Load("/nonexistent/path/config.json")
	assert.Error(t, err)
	assert.Nil(t, cfg)
	assert.Contains(t, err.Error(), "failed to read config file")
}

func TestLoader_Load_InvalidJSON(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "invalid.json")
	err := os.WriteFile(configPath, []byte("not valid json"), 0644)
	require.NoError(t, err)

	loader := NewLoader()
	cfg, err := loader.Load(configPath)
	assert.Error(t, err)
	assert.Nil(t, cfg)
	assert.Contains(t, err.Error(), "failed to parse config JSON")
}

func TestLoader_Load_ValidConfig(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.json")
	configJSON := `{
		"providers": [
			{
				"name": "openai-provider",
				"type": "openai",
				"base_url": "https://api.openai.com/v1",
				"apiKey": "sk-test"
			}
		],
		"models": {
			"gpt-4": {
				"provider": "openai-provider",
				"model": "gpt-4",
				"tool_call_transform": true
			}
		},
		"fallback": {
			"enabled": false
		}
	}`
	err := os.WriteFile(configPath, []byte(configJSON), 0644)
	require.NoError(t, err)

	loader := NewLoader()
	cfg, err := loader.Load(configPath)
	assert.NoError(t, err)
	assert.NotNil(t, cfg)
	assert.Len(t, cfg.Providers, 1)
	assert.Equal(t, "openai-provider", cfg.Providers[0].Name)
	assert.Equal(t, "openai", cfg.Providers[0].Type)
	assert.Equal(t, "gpt-4", cfg.Models["gpt-4"].Model)
}

func TestLoader_Validate_NoProviders(t *testing.T) {
	loader := NewLoader()
	cfg := &SchemaConfig{
		Providers: []Provider{},
		Models:    map[string]ModelConfig{},
	}
	err := loader.validate(cfg)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "at least one provider is required")
}

func TestLoader_Validate_MissingProviderName(t *testing.T) {
	loader := NewLoader()
	cfg := &SchemaConfig{
		Providers: []Provider{
			{Type: "openai", BaseURL: "https://api.openai.com", APIKey: "key"},
		},
	}
	err := loader.validate(cfg)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "name is required")
}

func TestLoader_Validate_MissingProviderType(t *testing.T) {
	loader := NewLoader()
	cfg := &SchemaConfig{
		Providers: []Provider{
			{Name: "test", BaseURL: "https://api.openai.com", APIKey: "key"},
		},
	}
	err := loader.validate(cfg)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "type is required")
}

func TestLoader_Validate_InvalidProviderType(t *testing.T) {
	loader := NewLoader()
	cfg := &SchemaConfig{
		Providers: []Provider{
			{Name: "test", Type: "invalid", BaseURL: "https://api.openai.com", APIKey: "key"},
		},
	}
	err := loader.validate(cfg)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "type must be")
}

func TestLoader_Validate_MissingBaseURL(t *testing.T) {
	loader := NewLoader()
	cfg := &SchemaConfig{
		Providers: []Provider{
			{Name: "test", Type: "openai", APIKey: "key"},
		},
	}
	err := loader.validate(cfg)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "base_url is required")
}

func TestLoader_Validate_MissingAPIKey(t *testing.T) {
	loader := NewLoader()
	cfg := &SchemaConfig{
		Providers: []Provider{
			{Name: "test", Type: "openai", BaseURL: "https://api.openai.com"},
		},
	}
	err := loader.validate(cfg)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "at least one apiKey or envApiKey is required")
}

func TestLoader_Validate_EnvAPIKeyOnly(t *testing.T) {
	loader := NewLoader()
	cfg := &SchemaConfig{
		Providers: []Provider{
			{Name: "test", Type: "openai", BaseURL: "https://api.openai.com", EnvAPIKey: "API_KEY"},
		},
	}
	err := loader.validate(cfg)
	assert.NoError(t, err)
}

func TestLoader_Validate_InvalidModelProviderReference(t *testing.T) {
	loader := NewLoader()
	cfg := &SchemaConfig{
		Providers: []Provider{
			{Name: "provider1", Type: "openai", BaseURL: "https://api.openai.com", APIKey: "key"},
		},
		Models: map[string]ModelConfig{
			"model1": {Provider: "nonexistent", Model: "gpt-4"},
		},
	}
	err := loader.validate(cfg)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "references unknown provider")
}

func TestLoader_Validate_FallbackEnabledNoProvider(t *testing.T) {
	loader := NewLoader()
	cfg := &SchemaConfig{
		Providers: []Provider{
			{Name: "provider1", Type: "openai", BaseURL: "https://api.openai.com", APIKey: "key"},
		},
		Fallback: FallbackConfig{
			Enabled: true,
		},
	}
	err := loader.validate(cfg)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "fallback provider is required")
}

func TestLoader_Validate_FallbackProviderNotExist(t *testing.T) {
	loader := NewLoader()
	cfg := &SchemaConfig{
		Providers: []Provider{
			{Name: "provider1", Type: "openai", BaseURL: "https://api.openai.com", APIKey: "key"},
		},
		Fallback: FallbackConfig{
			Enabled:  true,
			Provider: "nonexistent",
		},
	}
	err := loader.validate(cfg)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "fallback provider")
	assert.Contains(t, err.Error(), "does not exist")
}

func TestLoader_Validate_ValidFallback(t *testing.T) {
	loader := NewLoader()
	cfg := &SchemaConfig{
		Providers: []Provider{
			{Name: "provider1", Type: "openai", BaseURL: "https://api.openai.com", APIKey: "key"},
		},
		Fallback: FallbackConfig{
			Enabled:  true,
			Provider: "provider1",
			Model:    "gpt-4",
		},
	}
	err := loader.validate(cfg)
	assert.NoError(t, err)
}

func TestLoader_Validate_AnthropicProvider(t *testing.T) {
	loader := NewLoader()
	cfg := &SchemaConfig{
		Providers: []Provider{
			{Name: "anthropic", Type: "anthropic", BaseURL: "https://api.anthropic.com", APIKey: "key"},
		},
	}
	err := loader.validate(cfg)
	assert.NoError(t, err)
}

func TestLoader_ResolveEnvVars_EnvAPIKey(t *testing.T) {
	os.Setenv("TEST_API_KEY", "resolved-key")
	defer os.Unsetenv("TEST_API_KEY")

	loader := NewLoader()
	cfg := &SchemaConfig{
		Providers: []Provider{
			{Name: "test", Type: "openai", BaseURL: "https://api.openai.com", EnvAPIKey: "TEST_API_KEY"},
		},
	}
	err := loader.resolveEnvVars(cfg)
	require.NoError(t, err)
	assert.Equal(t, "resolved-key", cfg.Providers[0].APIKey)
}

func TestLoader_ResolveEnvVars_EnvAPIKeyNotSet(t *testing.T) {
	os.Unsetenv("MISSING_KEY")

	loader := NewLoader()
	cfg := &SchemaConfig{
		Providers: []Provider{
			{Name: "test", Type: "openai", BaseURL: "https://api.openai.com", APIKey: "fallback-key", EnvAPIKey: "MISSING_KEY"},
		},
	}
	err := loader.resolveEnvVars(cfg)
	require.NoError(t, err)
	assert.Equal(t, "fallback-key", cfg.Providers[0].APIKey)
}

func TestLoader_ResolveEnvVars_BaseURLEnvVar(t *testing.T) {
	os.Setenv("BASE_URL_VAR", "https://resolved.url.com")
	defer os.Unsetenv("BASE_URL_VAR")

	loader := NewLoader()
	cfg := &SchemaConfig{
		Providers: []Provider{
			{Name: "test", Type: "openai", BaseURL: "$BASE_URL_VAR", APIKey: "key"},
		},
	}
	err := loader.resolveEnvVars(cfg)
	require.NoError(t, err)
	assert.Equal(t, "https://resolved.url.com", cfg.Providers[0].BaseURL)
}

func TestLoader_ResolveEnvVars_FallbackModelEnvVar(t *testing.T) {
	os.Setenv("FALLBACK_MODEL", "gpt-4o")
	defer os.Unsetenv("FALLBACK_MODEL")

	loader := NewLoader()
	cfg := &SchemaConfig{
		Providers: []Provider{
			{Name: "test", Type: "openai", BaseURL: "https://api.openai.com", APIKey: "key"},
		},
		Fallback: FallbackConfig{
			Model: "$FALLBACK_MODEL",
		},
	}
	err := loader.resolveEnvVars(cfg)
	require.NoError(t, err)
	assert.Equal(t, "gpt-4o", cfg.Fallback.Model)
}

func TestLoader_ResolveEnvVars_BaseURLEnvNotSet(t *testing.T) {
	os.Unsetenv("MISSING_URL")

	loader := NewLoader()
	cfg := &SchemaConfig{
		Providers: []Provider{
			{Name: "test", Type: "openai", BaseURL: "$MISSING_URL", APIKey: "key"},
		},
	}
	err := loader.resolveEnvVars(cfg)
	require.NoError(t, err)
	assert.Equal(t, "$MISSING_URL", cfg.Providers[0].BaseURL)
}

func TestLoader_Load_EndToEnd(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.json")
	configJSON := `{
		"providers": [
			{
				"name": "openai",
				"type": "openai",
				"base_url": "https://api.openai.com/v1",
				"envApiKey": "OPENAI_KEY"
			},
			{
				"name": "anthropic",
				"type": "anthropic",
				"base_url": "https://api.anthropic.com",
				"apiKey": "sk-ant"
			}
		],
		"models": {
			"gpt-4": {
				"provider": "openai",
				"model": "gpt-4-turbo",
				"tool_call_transform": true
			},
			"claude": {
				"provider": "anthropic",
				"model": "claude-3-opus",
				"tool_call_transform": false
			}
		},
		"fallback": {
			"enabled": true,
			"provider": "openai",
			"model": "gpt-3.5-turbo",
			"tool_call_transform": true
		}
	}`
	err := os.WriteFile(configPath, []byte(configJSON), 0644)
	require.NoError(t, err)

	os.Setenv("OPENAI_KEY", "sk-openai-env")
	defer os.Unsetenv("OPENAI_KEY")

	loader := NewLoader()
	cfg, err := loader.Load(configPath)
	require.NoError(t, err)

	assert.Len(t, cfg.Providers, 2)
	assert.Equal(t, "sk-openai-env", cfg.Providers[0].APIKey)
	assert.Equal(t, "sk-ant", cfg.Providers[1].APIKey)
	assert.Len(t, cfg.Models, 2)
	assert.True(t, cfg.Fallback.Enabled)
	assert.Equal(t, "openai", cfg.Fallback.Provider)
}

func TestLoader_Validate_TableDriven(t *testing.T) {
	tests := []struct {
		name    string
		cfg     *SchemaConfig
		wantErr bool
		errMsg  string
	}{
		{
			name: "valid single provider",
			cfg: &SchemaConfig{
				Providers: []Provider{
					{Name: "p1", Type: "openai", BaseURL: "https://api.openai.com", APIKey: "key"},
				},
			},
			wantErr: false,
		},
		{
			name: "valid multiple providers",
			cfg: &SchemaConfig{
				Providers: []Provider{
					{Name: "p1", Type: "openai", BaseURL: "https://api.openai.com", APIKey: "key1"},
					{Name: "p2", Type: "anthropic", BaseURL: "https://api.anthropic.com", APIKey: "key2"},
				},
			},
			wantErr: false,
		},
		{
			name: "no providers",
			cfg: &SchemaConfig{
				Providers: []Provider{},
			},
			wantErr: true,
			errMsg:  "at least one provider is required",
		},
		{
			name: "provider missing name",
			cfg: &SchemaConfig{
				Providers: []Provider{
					{Type: "openai", BaseURL: "https://api.openai.com", APIKey: "key"},
				},
			},
			wantErr: true,
			errMsg:  "name is required",
		},
		{
			name: "provider missing type",
			cfg: &SchemaConfig{
				Providers: []Provider{
					{Name: "test", BaseURL: "https://api.openai.com", APIKey: "key"},
				},
			},
			wantErr: true,
			errMsg:  "type is required",
		},
		{
			name: "provider invalid type",
			cfg: &SchemaConfig{
				Providers: []Provider{
					{Name: "test", Type: "gemini", BaseURL: "https://api.gemini.com", APIKey: "key"},
				},
			},
			wantErr: true,
			errMsg:  "type must be",
		},
		{
			name: "provider missing base_url",
			cfg: &SchemaConfig{
				Providers: []Provider{
					{Name: "test", Type: "openai", APIKey: "key"},
				},
			},
			wantErr: true,
			errMsg:  "base_url is required",
		},
		{
			name: "provider missing both keys",
			cfg: &SchemaConfig{
				Providers: []Provider{
					{Name: "test", Type: "openai", BaseURL: "https://api.openai.com"},
				},
			},
			wantErr: true,
			errMsg:  "apiKey or envApiKey is required",
		},
		{
			name: "valid with envApiKey only",
			cfg: &SchemaConfig{
				Providers: []Provider{
					{Name: "test", Type: "openai", BaseURL: "https://api.openai.com", EnvAPIKey: "KEY"},
				},
			},
			wantErr: false,
		},
		{
			name: "valid with apiKey only",
			cfg: &SchemaConfig{
				Providers: []Provider{
					{Name: "test", Type: "anthropic", BaseURL: "https://api.anthropic.com", APIKey: "key"},
				},
			},
			wantErr: false,
		},
		{
			name: "valid with model mapping",
			cfg: &SchemaConfig{
				Providers: []Provider{
					{Name: "p1", Type: "openai", BaseURL: "https://api.openai.com", APIKey: "key"},
				},
				Models: map[string]ModelConfig{
					"model1": {Provider: "p1", Model: "gpt-4"},
				},
			},
			wantErr: false,
		},
		{
			name: "model references nonexistent provider",
			cfg: &SchemaConfig{
				Providers: []Provider{
					{Name: "p1", Type: "openai", BaseURL: "https://api.openai.com", APIKey: "key"},
				},
				Models: map[string]ModelConfig{
					"model1": {Provider: "p2", Model: "gpt-4"},
				},
			},
			wantErr: true,
			errMsg:  "references unknown provider",
		},
		{
			name: "fallback enabled without provider",
			cfg: &SchemaConfig{
				Providers: []Provider{
					{Name: "p1", Type: "openai", BaseURL: "https://api.openai.com", APIKey: "key"},
				},
				Fallback: FallbackConfig{Enabled: true},
			},
			wantErr: true,
			errMsg:  "fallback provider is required",
		},
		{
			name: "fallback enabled with invalid provider",
			cfg: &SchemaConfig{
				Providers: []Provider{
					{Name: "p1", Type: "openai", BaseURL: "https://api.openai.com", APIKey: "key"},
				},
				Fallback: FallbackConfig{Enabled: true, Provider: "p2"},
			},
			wantErr: true,
			errMsg:  "does not exist",
		},
		{
			name: "valid fallback enabled",
			cfg: &SchemaConfig{
				Providers: []Provider{
					{Name: "p1", Type: "openai", BaseURL: "https://api.openai.com", APIKey: "key"},
				},
				Fallback: FallbackConfig{Enabled: true, Provider: "p1"},
			},
			wantErr: false,
		},
		{
			name: "valid fallback disabled",
			cfg: &SchemaConfig{
				Providers: []Provider{
					{Name: "p1", Type: "openai", BaseURL: "https://api.openai.com", APIKey: "key"},
				},
				Fallback: FallbackConfig{Enabled: false, Provider: "nonexistent"},
			},
			wantErr: false,
		},
	}

	loader := NewLoader()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := loader.validate(tt.cfg)
			if tt.wantErr {
				assert.Error(t, err)
				if tt.errMsg != "" {
					assert.Contains(t, err.Error(), tt.errMsg)
				}
			} else {
				assert.NoError(t, err)
			}
		})
	}
}
