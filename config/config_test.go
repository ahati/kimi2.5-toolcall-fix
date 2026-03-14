package config

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func createTestConfig(t *testing.T, content string) string {
	t.Helper()
	tmpDir := t.TempDir()
	tmpFile := filepath.Join(tmpDir, "config.json")
	err := os.WriteFile(tmpFile, []byte(content), 0644)
	require.NoError(t, err)
	return tmpFile
}

func TestLoad_ValidConfig(t *testing.T) {
	jsonContent := `{
		"providers": [
			{"name": "openai", "type": "openai", "base_url": "https://api.openai.com/v1", "apiKey": "sk-test"}
		],
		"models": {
			"gpt-4": {"provider": "openai", "model": "gpt-4", "tool_call_transform": false}
		},
		"fallback": {"enabled": false}
	}`

	tmpFile := createTestConfig(t, jsonContent)
	defer os.Remove(tmpFile)

	cfg, err := Load(tmpFile)
	require.NoError(t, err)
	require.NotNil(t, cfg)

	assert.NotNil(t, cfg.AppConfig)
	assert.Len(t, cfg.AppConfig.Providers, 1)
	assert.Equal(t, "8080", cfg.Port)
	assert.Equal(t, "", cfg.SSELogDir)
}

func TestLoad_WithEnvVars(t *testing.T) {
	jsonContent := `{
		"providers": [
			{"name": "openai", "type": "openai", "base_url": "https://api.openai.com/v1", "apiKey": "sk-test"}
		],
		"models": {},
		"fallback": {"enabled": false}
	}`

	tmpFile := createTestConfig(t, jsonContent)
	defer os.Remove(tmpFile)

	os.Setenv("PORT", "9090")
	os.Setenv("SSELOG_DIR", "/var/log/sse")
	defer os.Unsetenv("PORT")
	defer os.Unsetenv("SSELOG_DIR")

	cfg, err := Load(tmpFile)
	require.NoError(t, err)

	assert.Equal(t, "9090", cfg.Port)
	assert.Equal(t, "/var/log/sse", cfg.SSELogDir)
}

func TestLoad_MissingFile(t *testing.T) {
	_, err := Load("/nonexistent/path/config.json")
	assert.Error(t, err)
}

func TestLoadFromEnv_Valid(t *testing.T) {
	jsonContent := `{
		"providers": [
			{"name": "openai", "type": "openai", "base_url": "https://api.openai.com/v1", "apiKey": "sk-test"}
		],
		"models": {},
		"fallback": {"enabled": false}
	}`

	tmpFile := createTestConfig(t, jsonContent)
	defer os.Remove(tmpFile)

	os.Setenv("CONFIG_FILE", tmpFile)
	defer os.Unsetenv("CONFIG_FILE")

	cfg, err := LoadFromEnv()
	require.NoError(t, err)
	assert.NotNil(t, cfg)
}

func TestLoadFromEnv_NotSet(t *testing.T) {
	os.Unsetenv("CONFIG_FILE")

	_, err := LoadFromEnv()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "CONFIG_FILE")
}

func TestGetProvider_Found(t *testing.T) {
	jsonContent := `{
		"providers": [
			{"name": "openai", "type": "openai", "base_url": "https://api.openai.com/v1", "apiKey": "sk-test"},
			{"name": "anthropic", "type": "anthropic", "base_url": "https://api.anthropic.com", "apiKey": "ant-test"}
		],
		"models": {},
		"fallback": {"enabled": false}
	}`

	tmpFile := createTestConfig(t, jsonContent)
	defer os.Remove(tmpFile)

	cfg, err := Load(tmpFile)
	require.NoError(t, err)

	provider, ok := cfg.GetProvider("openai")
	assert.True(t, ok)
	assert.Equal(t, "openai", provider.Name)
	assert.Equal(t, "https://api.openai.com/v1", provider.BaseURL)

	provider, ok = cfg.GetProvider("anthropic")
	assert.True(t, ok)
	assert.Equal(t, "anthropic", provider.Name)
}

func TestGetProvider_NotFound(t *testing.T) {
	jsonContent := `{
		"providers": [
			{"name": "openai", "type": "openai", "base_url": "https://api.openai.com/v1", "apiKey": "sk-test"}
		],
		"models": {},
		"fallback": {"enabled": false}
	}`

	tmpFile := createTestConfig(t, jsonContent)
	defer os.Remove(tmpFile)

	cfg, err := Load(tmpFile)
	require.NoError(t, err)

	_, ok := cfg.GetProvider("nonexistent")
	assert.False(t, ok)
}

func TestGetModel_Found(t *testing.T) {
	jsonContent := `{
		"providers": [
			{"name": "openai", "type": "openai", "base_url": "https://api.openai.com/v1", "apiKey": "sk-test"}
		],
		"models": {
			"gpt-4": {"provider": "openai", "model": "gpt-4-turbo", "tool_call_transform": true}
		},
		"fallback": {"enabled": false}
	}`

	tmpFile := createTestConfig(t, jsonContent)
	defer os.Remove(tmpFile)

	cfg, err := Load(tmpFile)
	require.NoError(t, err)

	model, ok := cfg.GetModel("gpt-4")
	assert.True(t, ok)
	assert.Equal(t, "openai", model.Provider)
	assert.Equal(t, "gpt-4-turbo", model.Model)
	assert.True(t, model.ToolCallTransform)
}

func TestGetModel_NotFound(t *testing.T) {
	jsonContent := `{
		"providers": [
			{"name": "openai", "type": "openai", "base_url": "https://api.openai.com/v1", "apiKey": "sk-test"}
		],
		"models": {},
		"fallback": {"enabled": false}
	}`

	tmpFile := createTestConfig(t, jsonContent)
	defer os.Remove(tmpFile)

	cfg, err := Load(tmpFile)
	require.NoError(t, err)

	_, ok := cfg.GetModel("nonexistent")
	assert.False(t, ok)
}

func TestGetOpenAIProvider(t *testing.T) {
	jsonContent := `{
		"providers": [
			{"name": "openai", "type": "openai", "base_url": "https://api.openai.com/v1", "apiKey": "sk-test"},
			{"name": "anthropic", "type": "anthropic", "base_url": "https://api.anthropic.com", "apiKey": "ant-test"}
		],
		"models": {},
		"fallback": {"enabled": false}
	}`

	tmpFile := createTestConfig(t, jsonContent)
	defer os.Remove(tmpFile)

	cfg, err := Load(tmpFile)
	require.NoError(t, err)

	provider, ok := cfg.GetOpenAIProvider()
	assert.True(t, ok)
	assert.Equal(t, "openai", provider.Type)
	assert.Equal(t, "https://api.openai.com/v1", provider.BaseURL)
}

func TestGetOpenAIProvider_NotFound(t *testing.T) {
	jsonContent := `{
		"providers": [
			{"name": "anthropic", "type": "anthropic", "base_url": "https://api.anthropic.com", "apiKey": "ant-test"}
		],
		"models": {},
		"fallback": {"enabled": false}
	}`

	tmpFile := createTestConfig(t, jsonContent)
	defer os.Remove(tmpFile)

	cfg, err := Load(tmpFile)
	require.NoError(t, err)

	_, ok := cfg.GetOpenAIProvider()
	assert.False(t, ok)
}

func TestGetAnthropicProvider(t *testing.T) {
	jsonContent := `{
		"providers": [
			{"name": "openai", "type": "openai", "base_url": "https://api.openai.com/v1", "apiKey": "sk-test"},
			{"name": "anthropic", "type": "anthropic", "base_url": "https://api.anthropic.com", "apiKey": "ant-test"}
		],
		"models": {},
		"fallback": {"enabled": false}
	}`

	tmpFile := createTestConfig(t, jsonContent)
	defer os.Remove(tmpFile)

	cfg, err := Load(tmpFile)
	require.NoError(t, err)

	provider, ok := cfg.GetAnthropicProvider()
	assert.True(t, ok)
	assert.Equal(t, "anthropic", provider.Type)
	assert.Equal(t, "https://api.anthropic.com", provider.BaseURL)
}

func TestGetAnthropicProvider_NotFound(t *testing.T) {
	jsonContent := `{
		"providers": [
			{"name": "openai", "type": "openai", "base_url": "https://api.openai.com/v1", "apiKey": "sk-test"}
		],
		"models": {},
		"fallback": {"enabled": false}
	}`

	tmpFile := createTestConfig(t, jsonContent)
	defer os.Remove(tmpFile)

	cfg, err := Load(tmpFile)
	require.NoError(t, err)

	_, ok := cfg.GetAnthropicProvider()
	assert.False(t, ok)
}

func TestGetEnvWithDefault(t *testing.T) {
	tests := []struct {
		name         string
		key          string
		envValue     string
		setEnv       bool
		defaultValue string
		want         string
	}{
		{
			name:         "env set",
			key:          "TEST_KEY_1",
			setEnv:       true,
			envValue:     "value",
			defaultValue: "default",
			want:         "value",
		},
		{
			name:         "env not set",
			key:          "TEST_KEY_2",
			setEnv:       false,
			defaultValue: "default",
			want:         "default",
		},
		{
			name:         "env empty",
			key:          "TEST_KEY_3",
			setEnv:       true,
			envValue:     "",
			defaultValue: "default",
			want:         "default",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.setEnv {
				os.Setenv(tt.key, tt.envValue)
				defer os.Unsetenv(tt.key)
			} else {
				os.Unsetenv(tt.key)
			}

			got := getEnvWithDefault(tt.key, tt.defaultValue)
			assert.Equal(t, tt.want, got)
		})
	}
}
