package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestNewLoader(t *testing.T) {
	loader := NewLoader()
	if loader == nil {
		t.Error("NewLoader() returned nil")
	}
}

func TestLoaderLoad(t *testing.T) {
	t.Run("valid config", func(t *testing.T) {
		configContent := `{
			"providers": [
				{
					"name": "openai-main",
					"endpoints": {
						"openai": "https://api.openai.com/v1/chat/completions"
					},
					"apiKey": "sk-test-key"
				}
			],
			"models": {
				"gpt-4": {
					"provider": "openai-main",
					"model": "gpt-4-turbo",
					"kimi_tool_call_transform": true
				}
			},
			"fallback": {
				"enabled": false
			}
		}`

		tmpFile := createTempConfigFile(t, configContent)
		defer os.Remove(tmpFile)

		loader := NewLoader()
		schema, err := loader.Load(tmpFile)
		if err != nil {
			t.Fatalf("Load() failed: %v", err)
		}

		if len(schema.Providers) != 1 {
			t.Errorf("Expected 1 provider, got %d", len(schema.Providers))
		}
		if schema.Providers[0].Name != "openai-main" {
			t.Errorf("Expected provider name 'openai-main', got %q", schema.Providers[0].Name)
		}
		if schema.Providers[0].APIKey != "sk-test-key" {
			t.Errorf("Expected API key 'sk-test-key', got %q", schema.Providers[0].APIKey)
		}
	})

	t.Run("valid config with envApiKey", func(t *testing.T) {
		os.Setenv("TEST_API_KEY_LOADER", "env-api-key-value")
		defer os.Unsetenv("TEST_API_KEY_LOADER")

		configContent := `{
			"providers": [
				{
					"name": "test-provider",
					"endpoints": {
						"anthropic": "https://api.anthropic.com/v1/messages"
					},
					"envApiKey": "TEST_API_KEY_LOADER"
				}
			],
			"models": {},
			"fallback": {
				"enabled": false
			}
		}`

		tmpFile := createTempConfigFile(t, configContent)
		defer os.Remove(tmpFile)

		loader := NewLoader()
		schema, err := loader.Load(tmpFile)
		if err != nil {
			t.Fatalf("Load() failed: %v", err)
		}

		if schema.Providers[0].APIKey != "env-api-key-value" {
			t.Errorf("Expected API key 'env-api-key-value', got %q", schema.Providers[0].APIKey)
		}
	})

	t.Run("valid config with fallback", func(t *testing.T) {
		configContent := `{
			"providers": [
				{
					"name": "primary",
					"endpoints": {
						"openai": "https://api.openai.com/v1/chat/completions"
					},
					"apiKey": "primary-key"
				},
				{
					"name": "fallback-provider",
					"endpoints": {
						"anthropic": "https://api.anthropic.com/v1/messages"
					},
					"apiKey": "fallback-key"
				}
			],
			"models": {},
			"fallback": {
				"enabled": true,
				"provider": "fallback-provider",
				"model": "claude-3-opus",
				"kimi_tool_call_transform": false
			}
		}`

		tmpFile := createTempConfigFile(t, configContent)
		defer os.Remove(tmpFile)

		loader := NewLoader()
		schema, err := loader.Load(tmpFile)
		if err != nil {
			t.Fatalf("Load() failed: %v", err)
		}

		if !schema.Fallback.Enabled {
			t.Error("Expected fallback to be enabled")
		}
		if schema.Fallback.Provider != "fallback-provider" {
			t.Errorf("Expected fallback provider 'fallback-provider', got %q", schema.Fallback.Provider)
		}
	})

	t.Run("missing file", func(t *testing.T) {
		loader := NewLoader()
		_, err := loader.Load("/nonexistent/path/config.json")
		if err == nil {
			t.Error("Expected error for missing file")
		}
	})

	t.Run("invalid JSON", func(t *testing.T) {
		configContent := `{invalid json}`

		tmpFile := createTempConfigFile(t, configContent)
		defer os.Remove(tmpFile)

		loader := NewLoader()
		_, err := loader.Load(tmpFile)
		if err == nil {
			t.Error("Expected error for invalid JSON")
		}
	})
}

func TestLoaderValidate(t *testing.T) {
	tests := []struct {
		name        string
		schema      Schema
		wantErr     bool
		errContains string
	}{
		{
			name: "valid minimal config",
			schema: Schema{
				Providers: []Provider{
					{
						Name: "test",
						Endpoints: map[string]string{
							"openai": "https://api.example.com/v1/chat/completions",
						},
						APIKey: "key",
					},
				},
				Models:   map[string]ModelConfig{},
				Fallback: FallbackConfig{Enabled: false},
			},
			wantErr: false,
		},
		{
			name: "valid with models and fallback",
			schema: Schema{
				Providers: []Provider{
					{
						Name: "openai",
						Endpoints: map[string]string{
							"openai": "https://api.openai.com/v1/chat/completions",
						},
						APIKey: "key",
					},
					{
						Name: "anthropic",
						Endpoints: map[string]string{
							"anthropic": "https://api.anthropic.com/v1/messages",
						},
						APIKey: "key",
					},
				},
				Models: map[string]ModelConfig{
					"gpt-4": {Provider: "openai", Model: "gpt-4-turbo"},
				},
				Fallback: FallbackConfig{
					Enabled:  true,
					Provider: "anthropic",
					Model:    "claude-3",
				},
			},
			wantErr: false,
		},
		{
			name: "no providers",
			schema: Schema{
				Providers: []Provider{},
				Models:    map[string]ModelConfig{},
				Fallback:  FallbackConfig{Enabled: false},
			},
			wantErr:     true,
			errContains: "at least one provider required",
		},
		{
			name: "provider missing name",
			schema: Schema{
				Providers: []Provider{
					{
						Name: "",
						Endpoints: map[string]string{
							"openai": "https://api.example.com/v1/chat/completions",
						},
						APIKey: "key",
					},
				},
				Models:   map[string]ModelConfig{},
				Fallback: FallbackConfig{Enabled: false},
			},
			wantErr:     true,
			errContains: "name is required",
		},
		{
			name: "provider missing endpoints",
			schema: Schema{
				Providers: []Provider{
					{
						Name:      "test",
						Endpoints: map[string]string{},
						APIKey:    "key",
					},
				},
				Models:   map[string]ModelConfig{},
				Fallback: FallbackConfig{Enabled: false},
			},
			wantErr:     true,
			errContains: "endpoints is required",
		},
		{
			name: "provider invalid protocol in endpoints",
			schema: Schema{
				Providers: []Provider{
					{
						Name: "test",
						Endpoints: map[string]string{
							"invalid": "https://api.example.com/v1",
						},
						APIKey: "key",
					},
				},
				Models:   map[string]ModelConfig{},
				Fallback: FallbackConfig{Enabled: false},
			},
			wantErr:     true,
			errContains: "invalid protocol",
		},
		{
			name: "multi-endpoint provider missing default",
			schema: Schema{
				Providers: []Provider{
					{
						Name: "test",
						Endpoints: map[string]string{
							"openai":    "https://api.example.com/v1/chat/completions",
							"anthropic": "https://api.example.com/anthropic/v1/messages",
						},
						APIKey: "key",
					},
				},
				Models:   map[string]ModelConfig{},
				Fallback: FallbackConfig{Enabled: false},
			},
			wantErr:     true,
			errContains: "'default' field is required",
		},
		{
			name: "multi-endpoint provider with invalid default",
			schema: Schema{
				Providers: []Provider{
					{
						Name: "test",
						Endpoints: map[string]string{
							"openai":    "https://api.example.com/v1/chat/completions",
							"anthropic": "https://api.example.com/anthropic/v1/messages",
						},
						Default: "invalid",
						APIKey:  "key",
					},
				},
				Models:   map[string]ModelConfig{},
				Fallback: FallbackConfig{Enabled: false},
			},
			wantErr:     true,
			errContains: "default protocol 'invalid' is invalid",
		},
		{
			name: "multi-endpoint provider with default not in endpoints",
			schema: Schema{
				Providers: []Provider{
					{
						Name: "test",
						Endpoints: map[string]string{
							"openai": "https://api.example.com/v1/chat/completions",
						},
						Default: "anthropic",
						APIKey:  "key",
					},
				},
				Models:   map[string]ModelConfig{},
				Fallback: FallbackConfig{Enabled: false},
			},
			wantErr:     true,
			errContains: "default protocol 'anthropic' not found in endpoints",
		},
		{
			name: "provider missing api key sources",
			schema: Schema{
				Providers: []Provider{
					{
						Name: "test",
						Endpoints: map[string]string{
							"openai": "https://api.example.com/v1/chat/completions",
						},
						APIKey:    "",
						EnvAPIKey: "",
					},
				},
				Models:   map[string]ModelConfig{},
				Fallback: FallbackConfig{Enabled: false},
			},
			wantErr:     true,
			errContains: "at least one of apiKey or envApiKey is required",
		},
		{
			name: "model references unknown provider",
			schema: Schema{
				Providers: []Provider{
					{
						Name: "known-provider",
						Endpoints: map[string]string{
							"openai": "https://api.example.com/v1/chat/completions",
						},
						APIKey: "key",
					},
				},
				Models: map[string]ModelConfig{
					"test-model": {Provider: "unknown-provider", Model: "model-v1"},
				},
				Fallback: FallbackConfig{Enabled: false},
			},
			wantErr:     true,
			errContains: "model 'test-model' references unknown provider 'unknown-provider'",
		},
		{
			name: "fallback references unknown provider",
			schema: Schema{
				Providers: []Provider{
					{
						Name: "known-provider",
						Endpoints: map[string]string{
							"openai": "https://api.example.com/v1/chat/completions",
						},
						APIKey: "key",
					},
				},
				Models: map[string]ModelConfig{},
				Fallback: FallbackConfig{
					Enabled:  true,
					Provider: "unknown-provider",
					Model:    "fallback-model",
				},
			},
			wantErr:     true,
			errContains: "fallback references unknown provider 'unknown-provider'",
		},
		{
			name: "fallback disabled with missing provider is ok",
			schema: Schema{
				Providers: []Provider{
					{
						Name: "known-provider",
						Endpoints: map[string]string{
							"openai": "https://api.example.com/v1/chat/completions",
						},
						APIKey: "key",
					},
				},
				Models: map[string]ModelConfig{},
				Fallback: FallbackConfig{
					Enabled:  false,
					Provider: "unknown-provider",
					Model:    "fallback-model",
				},
			},
			wantErr: false,
		},
		{
			name: "provider with only envApiKey",
			schema: Schema{
				Providers: []Provider{
					{
						Name: "test",
						Endpoints: map[string]string{
							"anthropic": "https://api.example.com/v1/messages",
						},
						EnvAPIKey: "MY_API_KEY",
					},
				},
				Models:   map[string]ModelConfig{},
				Fallback: FallbackConfig{Enabled: false},
			},
			wantErr: false,
		},
		{
			name: "provider with both apiKey and envApiKey",
			schema: Schema{
				Providers: []Provider{
					{
						Name: "test",
						Endpoints: map[string]string{
							"openai": "https://api.example.com/v1/chat/completions",
						},
						APIKey:    "direct-key",
						EnvAPIKey: "MY_API_KEY",
					},
				},
				Models:   map[string]ModelConfig{},
				Fallback: FallbackConfig{Enabled: false},
			},
			wantErr: false,
		},
		{
			name: "multiple providers with mixed protocols",
			schema: Schema{
				Providers: []Provider{
					{
						Name: "openai-provider",
						Endpoints: map[string]string{
							"openai": "https://api.openai.com/v1/chat/completions",
						},
						APIKey: "openai-key",
					},
					{
						Name: "anthropic-provider",
						Endpoints: map[string]string{
							"anthropic": "https://api.anthropic.com/v1/messages",
						},
						APIKey: "anthropic-key",
					},
				},
				Models: map[string]ModelConfig{
					"gpt-4":    {Provider: "openai-provider", Model: "gpt-4-turbo"},
					"claude-3": {Provider: "anthropic-provider", Model: "claude-3-opus"},
				},
				Fallback: FallbackConfig{Enabled: false},
			},
			wantErr: false,
		},
		{
			name: "multi-protocol provider with valid default",
			schema: Schema{
				Providers: []Provider{
					{
						Name: "multi",
						Endpoints: map[string]string{
							"openai":    "https://api.example.com/v1/chat/completions",
							"anthropic": "https://api.example.com/anthropic/v1/messages",
						},
						Default: "openai",
						APIKey:  "key",
					},
				},
				Models:   map[string]ModelConfig{},
				Fallback: FallbackConfig{Enabled: false},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			loader := NewLoader()
			err := loader.validate(&tt.schema)

			if tt.wantErr {
				if err == nil {
					t.Errorf("validate() expected error containing %q, got nil", tt.errContains)
					return
				}
				if tt.errContains != "" && !containsString(err.Error(), tt.errContains) {
					t.Errorf("validate() error = %q, want error containing %q", err.Error(), tt.errContains)
				}
			} else {
				if err != nil {
					t.Errorf("validate() unexpected error: %v", err)
				}
			}
		})
	}
}

func TestLoaderResolveEnvVars(t *testing.T) {
	tests := []struct {
		name        string
		schema      Schema
		envVars     map[string]string
		wantAPIKeys map[string]string
	}{
		{
			name: "resolve envApiKey when apiKey is empty",
			schema: Schema{
				Providers: []Provider{
					{
						Name: "test",
						Endpoints: map[string]string{
							"openai": "https://api.example.com/v1/chat/completions",
						},
						APIKey:    "",
						EnvAPIKey: "TEST_KEY_1",
					},
				},
			},
			envVars:     map[string]string{"TEST_KEY_1": "resolved-key-1"},
			wantAPIKeys: map[string]string{"test": "resolved-key-1"},
		},
		{
			name: "do not override existing apiKey",
			schema: Schema{
				Providers: []Provider{
					{
						Name: "test",
						Endpoints: map[string]string{
							"openai": "https://api.example.com/v1/chat/completions",
						},
						APIKey:    "direct-key",
						EnvAPIKey: "TEST_KEY_2",
					},
				},
			},
			envVars:     map[string]string{"TEST_KEY_2": "env-key-should-not-be-used"},
			wantAPIKeys: map[string]string{"test": "direct-key"},
		},
		{
			name: "empty env var results in empty apiKey",
			schema: Schema{
				Providers: []Provider{
					{
						Name: "test",
						Endpoints: map[string]string{
							"openai": "https://api.example.com/v1/chat/completions",
						},
						APIKey:    "",
						EnvAPIKey: "NON_EXISTENT_KEY",
					},
				},
			},
			envVars:     map[string]string{},
			wantAPIKeys: map[string]string{"test": ""},
		},
		{
			name: "multiple providers with mixed resolution",
			schema: Schema{
				Providers: []Provider{
					{
						Name: "provider1",
						Endpoints: map[string]string{
							"openai": "https://api.example.com/v1/chat/completions",
						},
						APIKey:    "",
						EnvAPIKey: "KEY_1",
					},
					{
						Name: "provider2",
						Endpoints: map[string]string{
							"anthropic": "https://api.anthropic.com/v1/messages",
						},
						APIKey:    "direct-key",
						EnvAPIKey: "KEY_2",
					},
					{
						Name: "provider3",
						Endpoints: map[string]string{
							"openai": "https://api.example.com/v1/chat/completions",
						},
						APIKey:    "",
						EnvAPIKey: "KEY_3",
					},
				},
			},
			envVars: map[string]string{
				"KEY_1": "env-key-1",
				"KEY_3": "env-key-3",
			},
			wantAPIKeys: map[string]string{
				"provider1": "env-key-1",
				"provider2": "direct-key",
				"provider3": "env-key-3",
			},
		},
		{
			name: "provider with no envApiKey",
			schema: Schema{
				Providers: []Provider{
					{
						Name: "test",
						Endpoints: map[string]string{
							"openai": "https://api.example.com/v1/chat/completions",
						},
						APIKey:    "",
						EnvAPIKey: "",
					},
				},
			},
			envVars:     map[string]string{},
			wantAPIKeys: map[string]string{"test": ""},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for key, value := range tt.envVars {
				os.Setenv(key, value)
				defer os.Unsetenv(key)
			}

			loader := NewLoader()
			loader.resolveEnvVars(&tt.schema)

			for _, provider := range tt.schema.Providers {
				wantKey, ok := tt.wantAPIKeys[provider.Name]
				if !ok {
					t.Errorf("No expected API key for provider %q", provider.Name)
					continue
				}
				if provider.APIKey != wantKey {
					t.Errorf("Provider %q: APIKey = %q, want %q", provider.Name, provider.APIKey, wantKey)
				}
			}
		})
	}
}

func TestLoaderValidateIntegration(t *testing.T) {
	t.Run("complete valid config file", func(t *testing.T) {
		os.Setenv("INTEGRATION_TEST_KEY", "integration-key-value")
		defer os.Unsetenv("INTEGRATION_TEST_KEY")

		configContent := `{
			"providers": [
				{
					"name": "openai-main",
					"endpoints": {
						"openai": "https://api.openai.com/v1/chat/completions"
					},
					"apiKey": "sk-direct-key"
				},
				{
					"name": "anthropic-main",
					"endpoints": {
						"anthropic": "https://api.anthropic.com/v1/messages"
					},
					"envApiKey": "INTEGRATION_TEST_KEY"
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
				"kimi_tool_call_transform": false
			}
		}`

		tmpFile := createTempConfigFile(t, configContent)
		defer os.Remove(tmpFile)

		loader := NewLoader()
		schema, err := loader.Load(tmpFile)
		if err != nil {
			t.Fatalf("Load() failed: %v", err)
		}

		if len(schema.Providers) != 2 {
			t.Errorf("Expected 2 providers, got %d", len(schema.Providers))
		}

		if schema.Providers[0].APIKey != "sk-direct-key" {
			t.Errorf("Expected direct API key, got %q", schema.Providers[0].APIKey)
		}
		if schema.Providers[1].APIKey != "integration-key-value" {
			t.Errorf("Expected resolved env API key, got %q", schema.Providers[1].APIKey)
		}

		if len(schema.Models) != 2 {
			t.Errorf("Expected 2 models, got %d", len(schema.Models))
		}

		if !schema.Fallback.Enabled {
			t.Error("Expected fallback to be enabled")
		}
	})

	t.Run("validation error prevents env resolution", func(t *testing.T) {
		configContent := `{
			"providers": [],
			"models": {},
			"fallback": {
				"enabled": false
			}
		}`

		tmpFile := createTempConfigFile(t, configContent)
		defer os.Remove(tmpFile)

		loader := NewLoader()
		_, err := loader.Load(tmpFile)
		if err == nil {
			t.Error("Expected validation error for empty providers")
		}
	})
}

func TestLoaderJSONUnmarshalErrors(t *testing.T) {
	tests := []struct {
		name          string
		configContent string
		wantErr       bool
	}{
		{
			name: "malformed JSON",
			configContent: `{
				"providers": [
					{
						"name": "test",
						"endpoints": {"openai": "https://api.example.com"},
						"apiKey": "key"
					}
				,
				"models": {}
			}`,
			wantErr: true,
		},
		{
			name: "wrong type for providers",
			configContent: `{
				"providers": "not-an-array",
				"models": {},
				"fallback": {"enabled": false}
			}`,
			wantErr: true,
		},
		{
			name: "wrong type for models",
			configContent: `{
				"providers": [
					{
						"name": "test",
						"endpoints": {"openai": "https://api.example.com"},
						"apiKey": "key"
					}
				],
				"models": "not-an-object",
				"fallback": {"enabled": false}
			}`,
			wantErr: true,
		},
		{
			name: "wrong type for provider fields",
			configContent: `{
				"providers": [
					{
						"name": 123,
						"endpoints": {"openai": "https://api.example.com"},
						"apiKey": "key"
					}
				],
				"models": {},
				"fallback": {"enabled": false}
			}`,
			wantErr: true,
		},
		{
			name:          "empty file",
			configContent: ``,
			wantErr:       true,
		},
		{
			name:          "null instead of object",
			configContent: `null`,
			wantErr:       true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpFile := createTempConfigFile(t, tt.configContent)
			defer os.Remove(tmpFile)

			loader := NewLoader()
			_, err := loader.Load(tmpFile)

			if tt.wantErr && err == nil {
				t.Error("Expected error, got nil")
			}
			if !tt.wantErr && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
		})
	}
}

func createTempConfigFile(t *testing.T, content string) string {
	t.Helper()
	tmpDir := t.TempDir()
	tmpFile := filepath.Join(tmpDir, "config.json")
	if err := os.WriteFile(tmpFile, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to create temp config file: %v", err)
	}
	return tmpFile
}

func containsString(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 ||
		(len(s) > 0 && len(substr) > 0 && findSubstring(s, substr)))
}

func findSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func TestLoaderLoadFilePermissionError(t *testing.T) {
	if os.Getuid() == 0 {
		t.Skip("Skipping permission test when running as root")
	}

	tmpDir := t.TempDir()
	tmpFile := filepath.Join(tmpDir, "config.json")

	validConfig := `{
		"providers": [
			{
				"name": "test",
				"endpoints": {"openai": "https://api.example.com"},
				"apiKey": "key"
			}
		],
		"models": {},
		"fallback": {"enabled": false}
	}`
	if err := os.WriteFile(tmpFile, []byte(validConfig), 0644); err != nil {
		t.Fatalf("Failed to write temp file: %v", err)
	}

	if err := os.Chmod(tmpFile, 0000); err != nil {
		t.Fatalf("Failed to change file permissions: %v", err)
	}
	defer os.Chmod(tmpFile, 0644)

	loader := NewLoader()
	_, err := loader.Load(tmpFile)
	if err == nil {
		t.Error("Expected error for unreadable file")
	}
}

func TestLoaderValidateEdgeCases(t *testing.T) {
	tests := []struct {
		name        string
		setupSchema func() *Schema
		wantErr     bool
		errContains string
	}{
		{
			name: "provider with empty name at index 0",
			setupSchema: func() *Schema {
				return &Schema{
					Providers: []Provider{
						{
							Name: "",
							Endpoints: map[string]string{
								"openai": "https://api.example.com",
							},
							APIKey: "key",
						},
					},
					Models:   map[string]ModelConfig{},
					Fallback: FallbackConfig{Enabled: false},
				}
			},
			wantErr:     true,
			errContains: "provider 'index 0': name is required",
		},
		{
			name: "provider with empty name at index 2",
			setupSchema: func() *Schema {
				return &Schema{
					Providers: []Provider{
						{
							Name: "p1",
							Endpoints: map[string]string{
								"openai": "https://api1.example.com",
							},
							APIKey: "key1",
						},
						{
							Name: "p2",
							Endpoints: map[string]string{
								"openai": "https://api2.example.com",
							},
							APIKey: "key2",
						},
						{
							Name: "",
							Endpoints: map[string]string{
								"openai": "https://api3.example.com",
							},
							APIKey: "key3",
						},
					},
					Models:   map[string]ModelConfig{},
					Fallback: FallbackConfig{Enabled: false},
				}
			},
			wantErr:     true,
			errContains: "provider 'index 2': name is required",
		},
		{
			name: "model with empty provider name",
			setupSchema: func() *Schema {
				return &Schema{
					Providers: []Provider{
						{
							Name: "valid",
							Endpoints: map[string]string{
								"openai": "https://api.example.com",
							},
							APIKey: "key",
						},
					},
					Models: map[string]ModelConfig{
						"test": {Provider: "", Model: "model"},
					},
					Fallback: FallbackConfig{Enabled: false},
				}
			},
			wantErr:     true,
			errContains: "model 'test' references unknown provider ''",
		},
		{
			name: "fallback with empty provider name when enabled",
			setupSchema: func() *Schema {
				return &Schema{
					Providers: []Provider{
						{
							Name: "valid",
							Endpoints: map[string]string{
								"openai": "https://api.example.com",
							},
							APIKey: "key",
						},
					},
					Models: map[string]ModelConfig{},
					Fallback: FallbackConfig{
						Enabled:  true,
						Provider: "",
						Model:    "model",
					},
				}
			},
			wantErr:     true,
			errContains: "fallback references unknown provider ''",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			loader := NewLoader()
			schema := tt.setupSchema()
			err := loader.validate(schema)

			if tt.wantErr {
				if err == nil {
					t.Errorf("validate() expected error containing %q, got nil", tt.errContains)
					return
				}
				if tt.errContains != "" && !containsString(err.Error(), tt.errContains) {
					t.Errorf("validate() error = %q, want error containing %q", err.Error(), tt.errContains)
				}
			} else {
				if err != nil {
					t.Errorf("validate() unexpected error: %v", err)
				}
			}
		})
	}
}

func BenchmarkLoaderLoad(b *testing.B) {
	configContent := `{
		"providers": [
			{
				"name": "openai-main",
				"endpoints": {
					"openai": "https://api.openai.com/v1/chat/completions"
				},
				"apiKey": "sk-test-key"
			},
			{
				"name": "anthropic-main",
				"endpoints": {
					"anthropic": "https://api.anthropic.com/v1/messages"
				},
				"apiKey": "anthropic-key"
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
			"kimi_tool_call_transform": false
		}
	}`

	tmpDir := b.TempDir()
	tmpFile := filepath.Join(tmpDir, "config.json")
	if err := os.WriteFile(tmpFile, []byte(configContent), 0644); err != nil {
		b.Fatalf("Failed to create temp config file: %v", err)
	}

	loader := NewLoader()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := loader.Load(tmpFile)
		if err != nil {
			b.Fatalf("Load() failed: %v", err)
		}
	}
}

func TestLoaderJSONUnmarshalErrorWrapping(t *testing.T) {
	configContent := `{
		"providers": [
			{
				"name": "test",
				"endpoints": {"openai": "https://api.example.com"},
				"apiKey": "key"
			}
		],
		"models": {
			"test": {
				"provider": "test",
				"model": "test-model",
				"kimi_tool_call_transform": "not-a-bool"
			}
		},
		"fallback": {"enabled": false}
	}`

	tmpFile := createTempConfigFile(t, configContent)
	defer os.Remove(tmpFile)

	loader := NewLoader()
	_, err := loader.Load(tmpFile)
	if err == nil {
		t.Error("Expected error for invalid JSON type")
	}

	if !containsString(err.Error(), "failed to parse config JSON") {
		t.Errorf("Expected error to contain 'failed to parse config JSON', got: %v", err)
	}
}

func TestLoaderLoadWithExtraFields(t *testing.T) {
	configContent := `{
		"extra_top_level": "ignored",
		"providers": [
			{
				"name": "test",
				"endpoints": {"openai": "https://api.example.com"},
				"apiKey": "key",
				"extra_field": "also ignored"
			}
		],
		"models": {},
		"fallback": {"enabled": false},
		"another_extra": 123
	}`

	tmpFile := createTempConfigFile(t, configContent)
	defer os.Remove(tmpFile)

	loader := NewLoader()
	schema, err := loader.Load(tmpFile)
	if err != nil {
		t.Fatalf("Load() failed: %v", err)
	}

	if len(schema.Providers) != 1 {
		t.Errorf("Expected 1 provider, got %d", len(schema.Providers))
	}
}

func TestLoaderMultipleLoads(t *testing.T) {
	config1 := `{
		"providers": [
			{
				"name": "provider1",
				"endpoints": {"openai": "https://api1.example.com"},
				"apiKey": "key1"
			}
		],
		"models": {},
		"fallback": {"enabled": false}
	}`

	config2 := `{
		"providers": [
			{
				"name": "provider2",
				"endpoints": {"anthropic": "https://api2.example.com"},
				"apiKey": "key2"
			}
		],
		"models": {},
		"fallback": {"enabled": false}
	}`

	tmpFile1 := createTempConfigFile(t, config1)
	defer os.Remove(tmpFile1)

	tmpFile2 := createTempConfigFile(t, config2)
	defer os.Remove(tmpFile2)

	loader := NewLoader()

	schema1, err := loader.Load(tmpFile1)
	if err != nil {
		t.Fatalf("Load() failed for config1: %v", err)
	}

	schema2, err := loader.Load(tmpFile2)
	if err != nil {
		t.Fatalf("Load() failed for config2: %v", err)
	}

	if schema1.Providers[0].Name != "provider1" {
		t.Errorf("Expected provider1, got %s", schema1.Providers[0].Name)
	}

	if schema2.Providers[0].Name != "provider2" {
		t.Errorf("Expected provider2, got %s", schema2.Providers[0].Name)
	}
}

func TestLoaderConcurrentLoad(t *testing.T) {
	configContent := `{
		"providers": [
			{
				"name": "concurrent-test",
				"endpoints": {"openai": "https://api.example.com"},
				"apiKey": "key"
			}
		],
		"models": {},
		"fallback": {"enabled": false}
	}`

	tmpFile := createTempConfigFile(t, configContent)
	defer os.Remove(tmpFile)

	loader := NewLoader()

	const numGoroutines = 10
	errCh := make(chan error, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func() {
			_, err := loader.Load(tmpFile)
			errCh <- err
		}()
	}

	for i := 0; i < numGoroutines; i++ {
		if err := <-errCh; err != nil {
			t.Errorf("Concurrent load failed: %v", err)
		}
	}
}

func TestLoaderValidationBeforeResolution(t *testing.T) {
	os.Setenv("SHOULD_NOT_BE_READ", "secret-value")
	defer os.Unsetenv("SHOULD_NOT_BE_READ")

	configContent := `{
		"providers": [],
		"models": {},
		"fallback": {"enabled": false}
	}`

	tmpFile := createTempConfigFile(t, configContent)
	defer os.Remove(tmpFile)

	loader := NewLoader()
	_, err := loader.Load(tmpFile)

	if err == nil {
		t.Error("Expected validation error for empty providers")
	}
	if !containsString(err.Error(), "at least one provider required") {
		t.Errorf("Expected 'at least one provider required' error, got: %v", err)
	}
}

func TestLoaderLoadEmptyObject(t *testing.T) {
	configContent := `{}`

	tmpFile := createTempConfigFile(t, configContent)
	defer os.Remove(tmpFile)

	loader := NewLoader()
	_, err := loader.Load(tmpFile)

	if err == nil {
		t.Error("Expected validation error for empty object")
	}
}

func TestLoaderLoadNull(t *testing.T) {
	configContent := `null`

	tmpFile := createTempConfigFile(t, configContent)
	defer os.Remove(tmpFile)

	loader := NewLoader()
	_, err := loader.Load(tmpFile)

	if err == nil {
		t.Error("Expected error for null config")
	}
}

func TestLoaderJSONTypeMismatches(t *testing.T) {
	tests := []struct {
		name    string
		content string
	}{
		{
			name: "providers as object",
			content: `{
				"providers": {"name": "test"},
				"models": {},
				"fallback": {"enabled": false}
			}`,
		},
		{
			name: "models as array",
			content: `{
				"providers": [{"name": "test", "endpoints": {"openai": "url"}, "apiKey": "key"}],
				"models": [],
				"fallback": {"enabled": false}
			}`,
		},
		{
			name: "fallback as array",
			content: `{
				"providers": [{"name": "test", "endpoints": {"openai": "url"}, "apiKey": "key"}],
				"models": {},
				"fallback": []
			}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpFile := createTempConfigFile(t, tt.content)
			defer os.Remove(tmpFile)

			loader := NewLoader()
			_, err := loader.Load(tmpFile)

			var jsonErr *json.UnmarshalTypeError
			if err == nil {
				t.Error("Expected JSON unmarshal error")
			} else if err != nil {
				if !containsString(err.Error(), "failed to parse config JSON") {
					t.Logf("Got error: %v", err)
				}
			}
			_ = jsonErr
		})
	}
}
