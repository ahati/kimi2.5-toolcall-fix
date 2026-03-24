package config

import (
	"encoding/json"
	"os"
	"testing"
)

func TestProviderGetAPIKey(t *testing.T) {
	tests := []struct {
		name     string
		provider Provider
		envKey   string
		envValue string
		wantKey  string
	}{
		{
			name: "direct APIKey takes precedence",
			provider: Provider{
				Name:      "test-provider",
				Endpoints: map[string]string{"openai": "https://api.example.com/v1"},
				APIKey:    "direct-api-key",
			},
			wantKey: "direct-api-key",
		},
		{
			name: "EnvAPIKey used when APIKey is empty",
			provider: Provider{
				Name:      "test-provider",
				Endpoints: map[string]string{"anthropic": "https://api.anthropic.com"},
				EnvAPIKey: "TEST_API_KEY_ENV",
			},
			envKey:   "TEST_API_KEY_ENV",
			envValue: "env-api-key-value",
			wantKey:  "env-api-key-value",
		},
		{
			name: "empty string when neither set",
			provider: Provider{
				Name:      "test-provider",
				Endpoints: map[string]string{"openai": "https://api.example.com/v1"},
			},
			wantKey: "",
		},
		{
			name: "APIKey takes precedence over EnvAPIKey",
			provider: Provider{
				Name:      "test-provider",
				Endpoints: map[string]string{"openai": "https://api.example.com/v1"},
				APIKey:    "direct-key",
				EnvAPIKey: "TEST_API_KEY_OVERRIDE",
			},
			envKey:   "TEST_API_KEY_OVERRIDE",
			envValue: "env-key-should-not-be-used",
			wantKey:  "direct-key",
		},
		{
			name: "EnvAPIKey with empty environment value",
			provider: Provider{
				Name:      "test-provider",
				Endpoints: map[string]string{"openai": "https://api.example.com/v1"},
				EnvAPIKey: "EMPTY_ENV_VAR",
			},
			envKey:   "EMPTY_ENV_VAR",
			envValue: "",
			wantKey:  "",
		},
		{
			name: "EnvAPIKey referencing non-existent env var",
			provider: Provider{
				Name:      "test-provider",
				Endpoints: map[string]string{"openai": "https://api.example.com/v1"},
				EnvAPIKey: "NON_EXISTENT_VAR_12345",
			},
			wantKey: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.envKey != "" {
				os.Setenv(tt.envKey, tt.envValue)
				defer os.Unsetenv(tt.envKey)
			}

			got := tt.provider.GetAPIKey()
			if got != tt.wantKey {
				t.Errorf("Provider.GetAPIKey() = %q, want %q", got, tt.wantKey)
			}
		})
	}
}

func TestSchemaJSONUnmarshal(t *testing.T) {
	jsonData := `{
		"providers": [
			{
				"name": "openai-main",
				"endpoints": {"openai": "https://api.openai.com/v1"},
				"apiKey": "sk-test-key-123"
			},
			{
				"name": "anthropic-main",
				"endpoints": {"anthropic": "https://api.anthropic.com"},
				"envApiKey": "ANTHROPIC_API_KEY"
			}
		],
		"models": {
			"gpt-4": {
				"provider": "openai-main",
				"model": "gpt-4-turbo",
				"kimi_tool_call_transform": true
			},
			"claude-3": {
				"provider": "anthropic-main",
				"model": "claude-3-opus-20240229",
				"kimi_tool_call_transform": false
			}
		},
		"fallback": {
			"enabled": true,
			"provider": "openai-main",
			"model": "gpt-3.5-turbo",
			"kimi_tool_call_transform": true
		}
	}`

	var schema Schema
	err := json.Unmarshal([]byte(jsonData), &schema)
	if err != nil {
		t.Fatalf("Failed to unmarshal JSON: %v", err)
	}

	if len(schema.Providers) != 2 {
		t.Errorf("Expected 2 providers, got %d", len(schema.Providers))
	}

	openaiProvider := schema.Providers[0]
	if openaiProvider.Name != "openai-main" {
		t.Errorf("Expected provider name 'openai-main', got %q", openaiProvider.Name)
	}
	if openaiProvider.GetEndpoint("openai") != "https://api.openai.com/v1" {
		t.Errorf("Expected openai endpoint 'https://api.openai.com/v1', got %q", openaiProvider.GetEndpoint("openai"))
	}
	if openaiProvider.APIKey != "sk-test-key-123" {
		t.Errorf("Expected API key 'sk-test-key-123', got %q", openaiProvider.APIKey)
	}

	anthropicProvider := schema.Providers[1]
	if anthropicProvider.Name != "anthropic-main" {
		t.Errorf("Expected provider name 'anthropic-main', got %q", anthropicProvider.Name)
	}
	if anthropicProvider.EnvAPIKey != "ANTHROPIC_API_KEY" {
		t.Errorf("Expected env API key 'ANTHROPIC_API_KEY', got %q", anthropicProvider.EnvAPIKey)
	}

	gpt4Config, ok := schema.Models["gpt-4"]
	if !ok {
		t.Error("Expected 'gpt-4' model config to exist")
	} else {
		if gpt4Config.Provider != "openai-main" {
			t.Errorf("Expected model provider 'openai-main', got %q", gpt4Config.Provider)
		}
		if gpt4Config.Model != "gpt-4-turbo" {
			t.Errorf("Expected model 'gpt-4-turbo', got %q", gpt4Config.Model)
		}
		if !gpt4Config.KimiToolCallTransform {
			t.Error("Expected KimiToolCallTransform to be true")
		}
	}

	claudeConfig, ok := schema.Models["claude-3"]
	if !ok {
		t.Error("Expected 'claude-3' model config to exist")
	} else {
		if claudeConfig.Provider != "anthropic-main" {
			t.Errorf("Expected model provider 'anthropic-main', got %q", claudeConfig.Provider)
		}
		if claudeConfig.KimiToolCallTransform {
			t.Error("Expected KimiToolCallTransform to be false")
		}
	}

	if !schema.Fallback.Enabled {
		t.Error("Expected fallback to be enabled")
	}
	if schema.Fallback.Provider != "openai-main" {
		t.Errorf("Expected fallback provider 'openai-main', got %q", schema.Fallback.Provider)
	}
	if schema.Fallback.Model != "gpt-3.5-turbo" {
		t.Errorf("Expected fallback model 'gpt-3.5-turbo', got %q", schema.Fallback.Model)
	}
}

func TestSchemaJSONUnmarshalMinimal(t *testing.T) {
	jsonData := `{
		"providers": [],
		"models": {},
		"fallback": {
			"enabled": false,
			"provider": "",
			"model": "",
			"kimi_tool_call_transform": false
		}
	}`

	var schema Schema
	err := json.Unmarshal([]byte(jsonData), &schema)
	if err != nil {
		t.Fatalf("Failed to unmarshal minimal JSON: %v", err)
	}

	if len(schema.Providers) != 0 {
		t.Errorf("Expected 0 providers, got %d", len(schema.Providers))
	}
	if len(schema.Models) != 0 {
		t.Errorf("Expected 0 models, got %d", len(schema.Models))
	}
	if schema.Fallback.Enabled {
		t.Error("Expected fallback to be disabled")
	}
}

func TestSchemaJSONUnmarshalPartial(t *testing.T) {
	jsonData := `{
		"providers": [
			{
				"name": "minimal-provider",
				"endpoints": {"openai": "https://api.example.com"}
			}
		],
		"models": {
			"test-model": {
				"provider": "minimal-provider",
				"model": "test-model-v1"
			}
		},
		"fallback": {
			"enabled": false
		}
	}`

	var schema Schema
	err := json.Unmarshal([]byte(jsonData), &schema)
	if err != nil {
		t.Fatalf("Failed to unmarshal partial JSON: %v", err)
	}

	if len(schema.Providers) != 1 {
		t.Errorf("Expected 1 provider, got %d", len(schema.Providers))
	}
	if schema.Providers[0].APIKey != "" {
		t.Errorf("Expected empty APIKey for partial unmarshal, got %q", schema.Providers[0].APIKey)
	}
	if schema.Providers[0].EnvAPIKey != "" {
		t.Errorf("Expected empty EnvAPIKey for partial unmarshal, got %q", schema.Providers[0].EnvAPIKey)
	}

	testModel, ok := schema.Models["test-model"]
	if !ok {
		t.Error("Expected 'test-model' to exist in models")
	} else {
		if testModel.KimiToolCallTransform {
			t.Error("Expected KimiToolCallTransform to default to false")
		}
	}
}

func TestSchemaJSONMarshal(t *testing.T) {
	schema := Schema{
		Providers: []Provider{
			{
				Name:      "test-provider",
				Endpoints: map[string]string{"openai": "https://api.test.com/v1"},
				APIKey:    "test-key",
			},
		},
		Models: map[string]ModelConfig{
			"test-model": {
				Provider:              "test-provider",
				Model:                 "test-model-v1",
				KimiToolCallTransform: true,
			},
		},
		Fallback: FallbackConfig{
			Enabled:               true,
			Provider:              "test-provider",
			Model:                 "fallback-model",
			KimiToolCallTransform: false,
		},
	}

	data, err := json.Marshal(schema)
	if err != nil {
		t.Fatalf("Failed to marshal schema: %v", err)
	}

	var unmarshaled Schema
	err = json.Unmarshal(data, &unmarshaled)
	if err != nil {
		t.Fatalf("Failed to unmarshal marshaled data: %v", err)
	}

	if unmarshaled.Providers[0].Name != schema.Providers[0].Name {
		t.Errorf("Provider name mismatch: got %q, want %q", unmarshaled.Providers[0].Name, schema.Providers[0].Name)
	}
	if unmarshaled.Models["test-model"].Model != schema.Models["test-model"].Model {
		t.Errorf("Model config mismatch")
	}
	if unmarshaled.Fallback.Enabled != schema.Fallback.Enabled {
		t.Errorf("Fallback enabled mismatch")
	}
}

func TestProviderJSONUnmarshal(t *testing.T) {
	tests := []struct {
		name     string
		jsonData string
		want     Provider
	}{
		{
			name: "full provider with all fields",
			jsonData: `{
				"name": "full-provider",
				"endpoints": {"openai": "https://api.example.com/v1"},
				"apiKey": "secret-key",
				"envApiKey": "API_KEY_ENV"
			}`,
			want: Provider{
				Name:      "full-provider",
				Endpoints: map[string]string{"openai": "https://api.example.com/v1"},
				APIKey:    "secret-key",
				EnvAPIKey: "API_KEY_ENV",
			},
		},
		{
			name: "minimal provider",
			jsonData: `{
				"name": "minimal",
				"endpoints": {"anthropic": "https://api.anthropic.com"}
			}`,
			want: Provider{
				Name:      "minimal",
				Endpoints: map[string]string{"anthropic": "https://api.anthropic.com"},
			},
		},
		{
			name: "provider with only env key",
			jsonData: `{
				"name": "env-only",
				"endpoints": {"openai": "https://api.example.com"},
				"envApiKey": "MY_API_KEY"
			}`,
			want: Provider{
				Name:      "env-only",
				Endpoints: map[string]string{"openai": "https://api.example.com"},
				EnvAPIKey: "MY_API_KEY",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got Provider
			err := json.Unmarshal([]byte(tt.jsonData), &got)
			if err != nil {
				t.Fatalf("Failed to unmarshal: %v", err)
			}
			if got.Name != tt.want.Name {
				t.Errorf("Name = %q, want %q", got.Name, tt.want.Name)
			}
			if got.GetEndpoint("openai") != tt.want.GetEndpoint("openai") {
				t.Errorf("OpenAI Endpoint = %q, want %q", got.GetEndpoint("openai"), tt.want.GetEndpoint("openai"))
			}
			if got.GetEndpoint("anthropic") != tt.want.GetEndpoint("anthropic") {
				t.Errorf("Anthropic Endpoint = %q, want %q", got.GetEndpoint("anthropic"), tt.want.GetEndpoint("anthropic"))
			}
			if got.APIKey != tt.want.APIKey {
				t.Errorf("APIKey = %q, want %q", got.APIKey, tt.want.APIKey)
			}
			if got.EnvAPIKey != tt.want.EnvAPIKey {
				t.Errorf("EnvAPIKey = %q, want %q", got.EnvAPIKey, tt.want.EnvAPIKey)
			}
		})
	}
}

func TestModelConfigJSONUnmarshal(t *testing.T) {
	jsonData := `{
		"provider": "test-provider",
		"model": "gpt-4",
		"kimi_tool_call_transform": true
	}`

	var config ModelConfig
	err := json.Unmarshal([]byte(jsonData), &config)
	if err != nil {
		t.Fatalf("Failed to unmarshal: %v", err)
	}
	if config.Provider != "test-provider" {
		t.Errorf("Provider = %q, want 'test-provider'", config.Provider)
	}
	if config.Model != "gpt-4" {
		t.Errorf("Model = %q, want 'gpt-4'", config.Model)
	}
	if !config.KimiToolCallTransform {
		t.Error("KimiToolCallTransform should be true")
	}
}

func TestFallbackConfigJSONUnmarshal(t *testing.T) {
	jsonData := `{
		"enabled": true,
		"provider": "fallback-provider",
		"model": "fallback-model",
		"kimi_tool_call_transform": false
	}`

	var config FallbackConfig
	err := json.Unmarshal([]byte(jsonData), &config)
	if err != nil {
		t.Fatalf("Failed to unmarshal: %v", err)
	}
	if !config.Enabled {
		t.Error("Enabled should be true")
	}
	if config.Provider != "fallback-provider" {
		t.Errorf("Provider = %q, want 'fallback-provider'", config.Provider)
	}
	if config.Model != "fallback-model" {
		t.Errorf("Model = %q, want 'fallback-model'", config.Model)
	}
	if config.KimiToolCallTransform {
		t.Error("KimiToolCallTransform should be false")
	}
}

func TestProviderGetEndpoint(t *testing.T) {
	provider := Provider{
		Name: "test-provider",
		Endpoints: map[string]string{
			"openai":    "https://api.example.com/v1/chat/completions",
			"anthropic": "https://api.example.com/v1/messages",
		},
	}

	if got := provider.GetEndpoint("openai"); got != "https://api.example.com/v1/chat/completions" {
		t.Errorf("GetEndpoint(openai) = %q, want %q", got, "https://api.example.com/v1/chat/completions")
	}

	if got := provider.GetEndpoint("anthropic"); got != "https://api.example.com/v1/messages" {
		t.Errorf("GetEndpoint(anthropic) = %q, want %q", got, "https://api.example.com/v1/messages")
	}

	if got := provider.GetEndpoint("unknown"); got != "" {
		t.Errorf("GetEndpoint(unknown) = %q, want empty string", got)
	}
}
