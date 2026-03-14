package router

import (
	"testing"

	"ai-proxy/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewRouter(t *testing.T) {
	t.Run("nil config returns error", func(t *testing.T) {
		_, err := NewRouter(nil)
		assert.Error(t, err)
	})

	t.Run("valid config returns router", func(t *testing.T) {
		cfg := &config.SchemaConfig{
			Providers: []config.Provider{
				{Name: "openai", Type: "openai", BaseURL: "https://api.openai.com"},
			},
		}
		router, err := NewRouter(cfg)
		require.NoError(t, err)
		assert.NotNil(t, router)
	})
}

func TestResolve_ExactMatch(t *testing.T) {
	cfg := &config.SchemaConfig{
		Providers: []config.Provider{
			{Name: "openai", Type: "openai", BaseURL: "https://api.openai.com"},
		},
		Models: map[string]config.ModelConfig{
			"gpt-4": {
				Provider:          "openai",
				Model:             "gpt-4-turbo",
				ToolCallTransform: true,
			},
		},
		Fallback: config.FallbackConfig{Enabled: false},
	}

	router, err := NewRouter(cfg)
	require.NoError(t, err)

	route, err := router.Resolve("gpt-4")
	require.NoError(t, err)
	assert.Equal(t, "gpt-4-turbo", route.Model)
	assert.Equal(t, "openai", route.OutputProtocol)
	assert.True(t, route.ToolCallTransform)
	assert.NotNil(t, route.Provider)
	assert.Equal(t, "openai", route.Provider.Name)
}

func TestResolve_FallbackWithPlaceholder(t *testing.T) {
	cfg := &config.SchemaConfig{
		Providers: []config.Provider{
			{Name: "openai", Type: "openai", BaseURL: "https://api.openai.com"},
		},
		Models: map[string]config.ModelConfig{
			"gpt-4": {Provider: "openai", Model: "gpt-4"},
		},
		Fallback: config.FallbackConfig{
			Enabled:           true,
			Provider:          "openai",
			Model:             "custom/{model}/latest",
			ToolCallTransform: false,
		},
	}

	router, err := NewRouter(cfg)
	require.NoError(t, err)

	route, err := router.Resolve("unknown-model")
	require.NoError(t, err)
	assert.Equal(t, "custom/unknown-model/latest", route.Model)
	assert.Equal(t, "openai", route.OutputProtocol)
	assert.False(t, route.ToolCallTransform)
}

func TestResolve_FallbackDisabled(t *testing.T) {
	cfg := &config.SchemaConfig{
		Providers: []config.Provider{
			{Name: "openai", Type: "openai", BaseURL: "https://api.openai.com"},
		},
		Models: map[string]config.ModelConfig{
			"gpt-4": {Provider: "openai", Model: "gpt-4"},
		},
		Fallback: config.FallbackConfig{
			Enabled:  false,
			Provider: "openai",
			Model:    "gpt-4",
		},
	}

	router, err := NewRouter(cfg)
	require.NoError(t, err)

	_, err = router.Resolve("unknown-model")
	assert.ErrorIs(t, err, ErrModelNotFound)
}

func TestResolve_UnknownModel(t *testing.T) {
	cfg := &config.SchemaConfig{
		Providers: []config.Provider{
			{Name: "openai", Type: "openai", BaseURL: "https://api.openai.com"},
		},
		Models: map[string]config.ModelConfig{
			"gpt-4": {Provider: "openai", Model: "gpt-4"},
		},
		Fallback: config.FallbackConfig{Enabled: false},
	}

	router, err := NewRouter(cfg)
	require.NoError(t, err)

	_, err = router.Resolve("nonexistent-model")
	assert.ErrorIs(t, err, ErrModelNotFound)
}

func TestResolve_MissingProvider(t *testing.T) {
	cfg := &config.SchemaConfig{
		Providers: []config.Provider{
			{Name: "openai", Type: "openai", BaseURL: "https://api.openai.com"},
		},
		Models: map[string]config.ModelConfig{
			"gpt-4": {Provider: "nonexistent-provider", Model: "gpt-4"},
		},
		Fallback: config.FallbackConfig{Enabled: false},
	}

	router, err := NewRouter(cfg)
	require.NoError(t, err)

	_, err = router.Resolve("gpt-4")
	assert.ErrorIs(t, err, ErrProviderNotFound)
}

func TestResolve_FallbackMissingProvider(t *testing.T) {
	cfg := &config.SchemaConfig{
		Providers: []config.Provider{
			{Name: "openai", Type: "openai", BaseURL: "https://api.openai.com"},
		},
		Models: map[string]config.ModelConfig{},
		Fallback: config.FallbackConfig{
			Enabled:  true,
			Provider: "nonexistent-provider",
			Model:    "{model}",
		},
	}

	router, err := NewRouter(cfg)
	require.NoError(t, err)

	_, err = router.Resolve("any-model")
	assert.ErrorIs(t, err, ErrProviderNotFound)
}

func TestGetProvider(t *testing.T) {
	cfg := &config.SchemaConfig{
		Providers: []config.Provider{
			{Name: "openai", Type: "openai", BaseURL: "https://api.openai.com"},
			{Name: "anthropic", Type: "anthropic", BaseURL: "https://api.anthropic.com"},
		},
	}

	router, err := NewRouter(cfg)
	require.NoError(t, err)

	t.Run("existing provider", func(t *testing.T) {
		provider, found := router.GetProvider("openai")
		assert.True(t, found)
		assert.Equal(t, "openai", provider.Name)
		assert.Equal(t, "openai", provider.Type)
	})

	t.Run("nonexistent provider", func(t *testing.T) {
		provider, found := router.GetProvider("nonexistent")
		assert.False(t, found)
		assert.Nil(t, provider)
	})
}

func TestListModels(t *testing.T) {
	cfg := &config.SchemaConfig{
		Providers: []config.Provider{
			{Name: "openai", Type: "openai", BaseURL: "https://api.openai.com"},
		},
		Models: map[string]config.ModelConfig{
			"gpt-4":      {Provider: "openai", Model: "gpt-4"},
			"gpt-3.5":    {Provider: "openai", Model: "gpt-3.5-turbo"},
			"gpt-4-turbo": {Provider: "openai", Model: "gpt-4-turbo"},
		},
		Fallback: config.FallbackConfig{Enabled: false},
	}

	router, err := NewRouter(cfg)
	require.NoError(t, err)

	models := router.ListModels()
	assert.Len(t, models, 3)
	assert.Contains(t, models, "gpt-4")
	assert.Contains(t, models, "gpt-3.5")
	assert.Contains(t, models, "gpt-4-turbo")
}

func TestListModels_Empty(t *testing.T) {
	cfg := &config.SchemaConfig{
		Providers: []config.Provider{},
		Models:    map[string]config.ModelConfig{},
		Fallback:  config.FallbackConfig{Enabled: false},
	}

	router, err := NewRouter(cfg)
	require.NoError(t, err)

	models := router.ListModels()
	assert.Empty(t, models)
	assert.NotNil(t, models)
}

func TestResolve_AnthropicProtocol(t *testing.T) {
	cfg := &config.SchemaConfig{
		Providers: []config.Provider{
			{Name: "anthropic", Type: "anthropic", BaseURL: "https://api.anthropic.com"},
		},
		Models: map[string]config.ModelConfig{
			"claude": {
				Provider:          "anthropic",
				Model:             "claude-3-opus",
				ToolCallTransform: false,
			},
		},
		Fallback: config.FallbackConfig{Enabled: false},
	}

	router, err := NewRouter(cfg)
	require.NoError(t, err)

	route, err := router.Resolve("claude")
	require.NoError(t, err)
	assert.Equal(t, "anthropic", route.OutputProtocol)
}

func TestResolve_MultiplePlaceholders(t *testing.T) {
	cfg := &config.SchemaConfig{
		Providers: []config.Provider{
			{Name: "openai", Type: "openai", BaseURL: "https://api.openai.com"},
		},
		Models: map[string]config.ModelConfig{},
		Fallback: config.FallbackConfig{
			Enabled:  true,
			Provider: "openai",
			Model:    "{model}/{model}-custom",
		},
	}

	router, err := NewRouter(cfg)
	require.NoError(t, err)

	route, err := router.Resolve("gpt-4")
	require.NoError(t, err)
	assert.Equal(t, "gpt-4/gpt-4-custom", route.Model)
}
