package router

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"ai-proxy/config"
)

func newTestConfig() *config.AppConfig {
	return &config.AppConfig{
		Providers: []config.Provider{
			{
				Name:      "test-openai",
				Type:      "openai",
				BaseURL:   "https://api.example.com/v1",
				EnvAPIKey: "TEST_API_KEY",
			},
			{
				Name:      "test-anthropic",
				Type:      "anthropic",
				BaseURL:   "https://api.anthropic.com/v1",
				EnvAPIKey: "ANTHROPIC_API_KEY",
			},
		},
		Models: map[string]config.ModelConfig{
			"gpt-4": {
				Provider:          "test-openai",
				Model:             "gpt-4-turbo",
				ToolCallTransform: false,
			},
		},
		Fallback: config.FallbackConfig{
			Enabled:           true,
			Provider:          "test-openai",
			Model:             "{model}",
			ToolCallTransform: true,
		},
	}
}

func TestNewRouter_ValidConfig(t *testing.T) {
	cfg := newTestConfig()
	router, err := NewRouter(cfg)
	require.NoError(t, err)
	assert.NotNil(t, router)
}

func TestNewRouter_NilConfig(t *testing.T) {
	router, err := NewRouter(nil)
	require.Error(t, err)
	assert.Nil(t, router)
	assert.Equal(t, "config is required", err.Error())
}

func TestResolve_ExactMatch(t *testing.T) {
	router, err := NewRouter(newTestConfig())
	require.NoError(t, err)

	route, err := router.Resolve("gpt-4")
	require.NoError(t, err)
	assert.Equal(t, "test-openai", route.Provider.Name)
	assert.Equal(t, "gpt-4-turbo", route.Model)
	assert.Equal(t, "openai", route.OutputProtocol)
	assert.False(t, route.ToolCallTransform)
}

func TestResolve_UnknownModel(t *testing.T) {
	cfg := newTestConfig()
	cfg.Fallback.Enabled = false
	router, err := NewRouter(cfg)
	require.NoError(t, err)

	route, err := router.Resolve("unknown-model")
	require.Error(t, err)
	assert.Nil(t, route)
	assert.Contains(t, err.Error(), "unknown model")
}

func TestResolve_FallbackEnabled(t *testing.T) {
	router, err := NewRouter(newTestConfig())
	require.NoError(t, err)

	route, err := router.Resolve("claude-3-opus")
	require.NoError(t, err)
	assert.Equal(t, "test-openai", route.Provider.Name)
	assert.Equal(t, "claude-3-opus", route.Model)
	assert.True(t, route.ToolCallTransform)
}

func TestResolve_FallbackWithPlaceholder(t *testing.T) {
	cfg := newTestConfig()
	cfg.Fallback.Model = "prefix/{model}/suffix"
	router, err := NewRouter(cfg)
	require.NoError(t, err)

	route, err := router.Resolve("my-model")
	require.NoError(t, err)
	assert.Equal(t, "prefix/my-model/suffix", route.Model)
}

func TestResolve_EmptyModelName(t *testing.T) {
	router, err := NewRouter(newTestConfig())
	require.NoError(t, err)

	route, err := router.Resolve("")
	require.Error(t, err)
	assert.Nil(t, route)
	assert.Equal(t, "model name is required", err.Error())
}

func TestResolve_InvalidProviderInMapping(t *testing.T) {
	cfg := newTestConfig()
	cfg.Models["bad-model"] = config.ModelConfig{
		Provider: "nonexistent-provider",
		Model:    "some-model",
	}
	router, err := NewRouter(cfg)
	require.NoError(t, err)

	route, err := router.Resolve("bad-model")
	require.Error(t, err)
	assert.Nil(t, route)
	assert.Contains(t, err.Error(), "unknown provider")
}

func TestResolve_FallbackDisabled(t *testing.T) {
	cfg := newTestConfig()
	cfg.Fallback.Enabled = false
	router, err := NewRouter(cfg)
	require.NoError(t, err)

	route, err := router.Resolve("unknown-model")
	require.Error(t, err)
	assert.Nil(t, route)
	assert.Contains(t, err.Error(), "unknown model")
}

func TestResolve_FallbackInvalidProvider(t *testing.T) {
	cfg := newTestConfig()
	cfg.Fallback.Provider = "nonexistent-provider"
	router, err := NewRouter(cfg)
	require.NoError(t, err)

	route, err := router.Resolve("unknown-model")
	require.Error(t, err)
	assert.Nil(t, route)
	assert.Contains(t, err.Error(), "unknown provider")
}

func TestGetProvider_Exists(t *testing.T) {
	router, err := NewRouter(newTestConfig())
	require.NoError(t, err)

	provider, ok := router.GetProvider("test-openai")
	assert.True(t, ok)
	assert.Equal(t, "test-openai", provider.Name)
	assert.Equal(t, "openai", provider.Type)
}

func TestGetProvider_NotExists(t *testing.T) {
	router, err := NewRouter(newTestConfig())
	require.NoError(t, err)

	provider, ok := router.GetProvider("nonexistent")
	assert.False(t, ok)
	assert.Empty(t, provider.Name)
}

func TestListModels(t *testing.T) {
	router, err := NewRouter(newTestConfig())
	require.NoError(t, err)

	models := router.ListModels()
	assert.Len(t, models, 1)
	assert.Contains(t, models, "gpt-4")
}

func TestListProviders(t *testing.T) {
	router, err := NewRouter(newTestConfig())
	require.NoError(t, err)

	providers := router.ListProviders()
	assert.Len(t, providers, 2)
	assert.Contains(t, providers, "test-openai")
	assert.Contains(t, providers, "test-anthropic")
}

func TestResolve_AnthropicProvider(t *testing.T) {
	cfg := newTestConfig()
	cfg.Models["claude-model"] = config.ModelConfig{
		Provider:          "test-anthropic",
		Model:             "claude-3-opus",
		ToolCallTransform: true,
	}
	router, err := NewRouter(cfg)
	require.NoError(t, err)

	route, err := router.Resolve("claude-model")
	require.NoError(t, err)
	assert.Equal(t, "test-anthropic", route.Provider.Name)
	assert.Equal(t, "anthropic", route.OutputProtocol)
	assert.True(t, route.ToolCallTransform)
}

func TestListModels_Empty(t *testing.T) {
	cfg := newTestConfig()
	cfg.Models = nil
	router, err := NewRouter(cfg)
	require.NoError(t, err)

	models := router.ListModels()
	assert.Empty(t, models)
}

func TestListProviders_Empty(t *testing.T) {
	cfg := &config.AppConfig{
		Providers: []config.Provider{},
		Models:    nil,
		Fallback:  config.FallbackConfig{},
	}
	router, err := NewRouter(cfg)
	require.NoError(t, err)

	providers := router.ListProviders()
	assert.Empty(t, providers)
}
