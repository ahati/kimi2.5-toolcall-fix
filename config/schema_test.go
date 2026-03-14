package config

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestProvider_JSONUnmarshal(t *testing.T) {
	tests := []struct {
		name     string
		json     string
		expected Provider
	}{
		{
			name: "full provider config",
			json: `{"name":"openai","type":"openai","base_url":"https://api.openai.com/v1","apiKey":"sk-test","envApiKey":"OPENAI_KEY"}`,
			expected: Provider{
				Name:      "openai",
				Type:      "openai",
				BaseURL:   "https://api.openai.com/v1",
				APIKey:    "sk-test",
				EnvAPIKey: "OPENAI_KEY",
			},
		},
		{
			name: "minimal provider config",
			json: `{"name":"anthropic","type":"anthropic","base_url":"https://api.anthropic.com"}`,
			expected: Provider{
				Name:    "anthropic",
				Type:    "anthropic",
				BaseURL: "https://api.anthropic.com",
			},
		},
		{
			name: "provider with only env key",
			json: `{"name":"kimi","type":"openai","base_url":"https://api.kimi.ai","envApiKey":"KIMI_API_KEY"}`,
			expected: Provider{
				Name:      "kimi",
				Type:      "openai",
				BaseURL:   "https://api.kimi.ai",
				EnvAPIKey: "KIMI_API_KEY",
			},
		},
		{
			name:     "empty provider config",
			json:     `{}`,
			expected: Provider{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var p Provider
			err := json.Unmarshal([]byte(tt.json), &p)
			require.NoError(t, err)
			assert.Equal(t, tt.expected, p)
		})
	}
}

func TestProvider_GetAPIKey(t *testing.T) {
	tests := []struct {
		name     string
		provider Provider
		envKey   string
		envValue string
		expected string
	}{
		{
			name: "returns APIKey when EnvAPIKey is empty",
			provider: Provider{
				APIKey: "direct-api-key",
			},
			expected: "direct-api-key",
		},
		{
			name: "returns env value when EnvAPIKey is set and env exists",
			provider: Provider{
				EnvAPIKey: "TEST_API_KEY_1",
				APIKey:    "fallback-key",
			},
			envKey:   "TEST_API_KEY_1",
			envValue: "env-api-key",
			expected: "env-api-key",
		},
		{
			name: "returns APIKey when EnvAPIKey is set but env is empty",
			provider: Provider{
				EnvAPIKey: "TEST_API_KEY_2",
				APIKey:    "fallback-key",
			},
			envKey:   "TEST_API_KEY_2",
			envValue: "",
			expected: "fallback-key",
		},
		{
			name: "returns APIKey when EnvAPIKey is set but env not set",
			provider: Provider{
				EnvAPIKey: "TEST_API_KEY_NOT_SET",
				APIKey:    "fallback-key",
			},
			expected: "fallback-key",
		},
		{
			name:     "returns empty string when both are empty",
			provider: Provider{},
			expected: "",
		},
		{
			name: "env value takes precedence over APIKey",
			provider: Provider{
				EnvAPIKey: "TEST_API_KEY_3",
				APIKey:    "direct-key",
			},
			envKey:   "TEST_API_KEY_3",
			envValue: "env-key",
			expected: "env-key",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.envKey != "" {
				if tt.envValue != "" {
					os.Setenv(tt.envKey, tt.envValue)
					defer os.Unsetenv(tt.envKey)
				} else {
					os.Unsetenv(tt.envKey)
				}
			}

			result := tt.provider.GetAPIKey()
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestModelConfig_JSONUnmarshal(t *testing.T) {
	tests := []struct {
		name     string
		json     string
		expected ModelConfig
	}{
		{
			name: "full model config",
			json: `{"provider":"openai","model":"gpt-4","tool_call_transform":true}`,
			expected: ModelConfig{
				Provider:          "openai",
				Model:             "gpt-4",
				ToolCallTransform: true,
			},
		},
		{
			name: "minimal model config",
			json: `{"provider":"anthropic","model":"claude-3"}`,
			expected: ModelConfig{
				Provider: "anthropic",
				Model:    "claude-3",
			},
		},
		{
			name:     "empty model config",
			json:     `{}`,
			expected: ModelConfig{},
		},
		{
			name: "tool_call_transform false explicitly",
			json: `{"provider":"openai","model":"gpt-3.5","tool_call_transform":false}`,
			expected: ModelConfig{
				Provider:          "openai",
				Model:             "gpt-3.5",
				ToolCallTransform: false,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var m ModelConfig
			err := json.Unmarshal([]byte(tt.json), &m)
			require.NoError(t, err)
			assert.Equal(t, tt.expected, m)
		})
	}
}

func TestFallbackConfig_JSONUnmarshal(t *testing.T) {
	tests := []struct {
		name     string
		json     string
		expected FallbackConfig
	}{
		{
			name: "full fallback config",
			json: `{"enabled":true,"provider":"openai","model":"gpt-4o-mini","tool_call_transform":true}`,
			expected: FallbackConfig{
				Enabled:           true,
				Provider:          "openai",
				Model:             "gpt-4o-mini",
				ToolCallTransform: true,
			},
		},
		{
			name: "fallback with model placeholder",
			json: `{"enabled":true,"provider":"anthropic","model":"{model}","tool_call_transform":false}`,
			expected: FallbackConfig{
				Enabled:           true,
				Provider:          "anthropic",
				Model:             "{model}",
				ToolCallTransform: false,
			},
		},
		{
			name:     "empty fallback config",
			json:     `{}`,
			expected: FallbackConfig{},
		},
		{
			name: "disabled fallback",
			json: `{"enabled":false}`,
			expected: FallbackConfig{
				Enabled: false,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var f FallbackConfig
			err := json.Unmarshal([]byte(tt.json), &f)
			require.NoError(t, err)
			assert.Equal(t, tt.expected, f)
		})
	}
}

func TestAppConfig_JSONUnmarshal(t *testing.T) {
	tests := []struct {
		name     string
		json     string
		expected AppConfig
	}{
		{
			name: "full app config",
			json: `{
				"providers": [
					{"name":"openai","type":"openai","base_url":"https://api.openai.com/v1","envApiKey":"OPENAI_KEY"},
					{"name":"anthropic","type":"anthropic","base_url":"https://api.anthropic.com","envApiKey":"ANTHROPIC_KEY"}
				],
				"models": {
					"gpt-4": {"provider":"openai","model":"gpt-4","tool_call_transform":false},
					"claude": {"provider":"anthropic","model":"claude-3-opus","tool_call_transform":false}
				},
				"fallback": {"enabled":true,"provider":"openai","model":"gpt-3.5-turbo","tool_call_transform":false}
			}`,
			expected: AppConfig{
				Providers: []Provider{
					{Name: "openai", Type: "openai", BaseURL: "https://api.openai.com/v1", EnvAPIKey: "OPENAI_KEY"},
					{Name: "anthropic", Type: "anthropic", BaseURL: "https://api.anthropic.com", EnvAPIKey: "ANTHROPIC_KEY"},
				},
				Models: map[string]ModelConfig{
					"gpt-4":  {Provider: "openai", Model: "gpt-4", ToolCallTransform: false},
					"claude": {Provider: "anthropic", Model: "claude-3-opus", ToolCallTransform: false},
				},
				Fallback: FallbackConfig{Enabled: true, Provider: "openai", Model: "gpt-3.5-turbo", ToolCallTransform: false},
			},
		},
		{
			name: "minimal app config",
			json: `{"providers":[],"models":{},"fallback":{"enabled":false}}`,
			expected: AppConfig{
				Providers: []Provider{},
				Models:    map[string]ModelConfig{},
				Fallback:  FallbackConfig{Enabled: false},
			},
		},
		{
			name: "empty app config",
			json: `{}`,
			expected: AppConfig{
				Models: map[string]ModelConfig(nil),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var a AppConfig
			err := json.Unmarshal([]byte(tt.json), &a)
			require.NoError(t, err)
			assert.Equal(t, tt.expected, a)
		})
	}
}

func TestProvider_Validation(t *testing.T) {
	tests := []struct {
		name        string
		json        string
		shouldError bool
	}{
		{
			name:        "valid provider with all required fields",
			json:        `{"name":"test","type":"openai","base_url":"https://api.test.com"}`,
			shouldError: false,
		},
		{
			name:        "empty provider is valid JSON",
			json:        `{}`,
			shouldError: false,
		},
		{
			name:        "provider with extra fields ignores them",
			json:        `{"name":"test","type":"openai","base_url":"https://api.test.com","extra":"ignored"}`,
			shouldError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var p Provider
			err := json.Unmarshal([]byte(tt.json), &p)
			if tt.shouldError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestModelConfig_Validation(t *testing.T) {
	tests := []struct {
		name        string
		json        string
		shouldError bool
	}{
		{
			name:        "valid model config",
			json:        `{"provider":"openai","model":"gpt-4"}`,
			shouldError: false,
		},
		{
			name:        "empty model config is valid JSON",
			json:        `{}`,
			shouldError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var m ModelConfig
			err := json.Unmarshal([]byte(tt.json), &m)
			if tt.shouldError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestAppConfig_WithMultipleProviders(t *testing.T) {
	jsonStr := `{
		"providers": [
			{"name":"p1","type":"openai","base_url":"https://p1.com"},
			{"name":"p2","type":"anthropic","base_url":"https://p2.com"},
			{"name":"p3","type":"openai","base_url":"https://p3.com"}
		],
		"models": {
			"m1": {"provider":"p1","model":"model1"},
			"m2": {"provider":"p2","model":"model2"}
		},
		"fallback": {"enabled":true,"provider":"p3","model":"fallback-model"}
	}`

	var a AppConfig
	err := json.Unmarshal([]byte(jsonStr), &a)
	require.NoError(t, err)

	assert.Len(t, a.Providers, 3)
	assert.Len(t, a.Models, 2)
	assert.True(t, a.Fallback.Enabled)
	assert.Equal(t, "p3", a.Fallback.Provider)
}

func TestProvider_GetAPIKey_EdgeCases(t *testing.T) {
	t.Run("whitespace env value", func(t *testing.T) {
		os.Setenv("TEST_WHITESPACE_KEY", "   ")
		defer os.Unsetenv("TEST_WHITESPACE_KEY")

		p := Provider{
			EnvAPIKey: "TEST_WHITESPACE_KEY",
			APIKey:    "fallback",
		}
		result := p.GetAPIKey()
		assert.Equal(t, "   ", result)
	})

	t.Run("special characters in env value", func(t *testing.T) {
		os.Setenv("TEST_SPECIAL_KEY", "sk-test_123!@#$%^&*()")
		defer os.Unsetenv("TEST_SPECIAL_KEY")

		p := Provider{
			EnvAPIKey: "TEST_SPECIAL_KEY",
		}
		result := p.GetAPIKey()
		assert.Equal(t, "sk-test_123!@#$%^&*()", result)
	})
}

func TestAppConfig_EmptyProviders(t *testing.T) {
	jsonStr := `{"providers":[],"models":{},"fallback":{"enabled":false}}`

	var a AppConfig
	err := json.Unmarshal([]byte(jsonStr), &a)
	require.NoError(t, err)

	assert.Empty(t, a.Providers)
	assert.Empty(t, a.Models)
	assert.False(t, a.Fallback.Enabled)
}

func TestModelConfig_ProviderRequired(t *testing.T) {
	jsonStr := `{"model":"some-model"}`
	var m ModelConfig
	err := json.Unmarshal([]byte(jsonStr), &m)
	require.NoError(t, err)
	assert.Empty(t, m.Provider)
	assert.Equal(t, "some-model", m.Model)
}

func TestFallbackConfig_DisabledByDefault(t *testing.T) {
	jsonStr := `{}`
	var f FallbackConfig
	err := json.Unmarshal([]byte(jsonStr), &f)
	require.NoError(t, err)
	assert.False(t, f.Enabled)
}
