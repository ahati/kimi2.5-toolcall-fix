package config

import (
	"flag"
	"os"
	"testing"
)

func cleanupEnv() {
	os.Unsetenv("PORT")
	os.Unsetenv("SSELOG_DIR")
	os.Unsetenv("CONFIG_FILE")
}

func TestGetEnvOrFlag(t *testing.T) {
	tests := []struct {
		name         string
		envKey       string
		envValue     string
		setEnv       bool
		flagValue    string
		defaultValue string
		want         string
	}{
		{
			name:         "flag value takes precedence over env and default",
			envKey:       "TEST_KEY",
			setEnv:       true,
			envValue:     "env-value",
			flagValue:    "flag-value",
			defaultValue: "default-value",
			want:         "flag-value",
		},
		{
			name:         "flag value takes precedence when env is empty",
			envKey:       "TEST_KEY",
			setEnv:       false,
			flagValue:    "flag-value",
			defaultValue: "default-value",
			want:         "flag-value",
		},
		{
			name:         "env value used when flag is empty",
			envKey:       "TEST_KEY",
			setEnv:       true,
			envValue:     "env-value",
			flagValue:    "",
			defaultValue: "default-value",
			want:         "env-value",
		},
		{
			name:         "default value used when flag and env are empty",
			envKey:       "TEST_KEY",
			setEnv:       false,
			flagValue:    "",
			defaultValue: "default-value",
			want:         "default-value",
		},
		{
			name:         "empty default when nothing set",
			envKey:       "TEST_KEY",
			setEnv:       false,
			flagValue:    "",
			defaultValue: "",
			want:         "",
		},
		{
			name:         "env value with special characters",
			envKey:       "TEST_KEY_SPECIAL",
			setEnv:       true,
			envValue:     "value-with-special_chars.123",
			flagValue:    "",
			defaultValue: "default",
			want:         "value-with-special_chars.123",
		},
		{
			name:         "env value with spaces",
			envKey:       "TEST_KEY_SPACES",
			setEnv:       true,
			envValue:     "value with spaces",
			flagValue:    "",
			defaultValue: "default",
			want:         "value with spaces",
		},
		{
			name:         "env value with url",
			envKey:       "TEST_KEY_URL",
			setEnv:       true,
			envValue:     "https://example.com/path?query=value",
			flagValue:    "",
			defaultValue: "default",
			want:         "https://example.com/path?query=value",
		},
		{
			name:         "flag with empty env",
			envKey:       "TEST_EMPTY_ENV",
			setEnv:       true,
			envValue:     "",
			flagValue:    "flag-val",
			defaultValue: "default",
			want:         "flag-val",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.setEnv {
				os.Setenv(tt.envKey, tt.envValue)
				defer os.Unsetenv(tt.envKey)
			}

			got := getEnvOrFlag(tt.envKey, tt.flagValue, tt.defaultValue)
			if got != tt.want {
				t.Errorf("getEnvOrFlag() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestGetEnvOrFlag_EmptyEnvValue(t *testing.T) {
	key := "TEST_EMPTY_ENV_VALUE"
	os.Setenv(key, "")
	defer os.Unsetenv(key)

	got := getEnvOrFlag(key, "", "default")
	if got != "" {
		t.Errorf("getEnvOrFlag() = %q, want empty string (env was set to empty)", got)
	}
}

func TestGetEnvOrFlag_EnvNotSet(t *testing.T) {
	key := "TEST_KEY_NOT_SET_12345"
	os.Unsetenv(key)

	got := getEnvOrFlag(key, "", "default")
	if got != "default" {
		t.Errorf("getEnvOrFlag() = %q, want %q", got, "default")
	}
}

func TestGetEnvOrFlag_FlagPrecedenceOverEnv(t *testing.T) {
	key := "TEST_PRECEDENCE"
	os.Setenv(key, "env-value")
	defer os.Unsetenv(key)

	got := getEnvOrFlag(key, "flag-value", "default")
	if got != "flag-value" {
		t.Errorf("getEnvOrFlag() = %q, want %q (flag should win)", got, "flag-value")
	}
}

func TestGetEnvOrFlag_AllCombinations(t *testing.T) {
	tests := []struct {
		name         string
		flagValue    string
		setEnv       bool
		envValue     string
		defaultValue string
		expected     string
	}{
		{"flag set, env set, default set", "flag", true, "env", "default", "flag"},
		{"flag set, env set, default empty", "flag", true, "env", "", "flag"},
		{"flag set, env empty, default set", "flag", true, "", "default", "flag"},
		{"flag set, env not set, default set", "flag", false, "", "default", "flag"},
		{"flag empty, env set, default set", "", true, "env", "default", "env"},
		{"flag empty, env set, default empty", "", true, "env", "", "env"},
		{"flag empty, env empty, default set", "", true, "", "default", ""},
		{"flag empty, env not set, default set", "", false, "", "default", "default"},
		{"flag empty, env not set, default empty", "", false, "", "", ""},
		{"flag set, env not set, default empty", "flag", false, "", "", "flag"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			envKey := "TEST_COMBO_" + tt.name
			if tt.setEnv {
				os.Setenv(envKey, tt.envValue)
				defer os.Unsetenv(envKey)
			} else {
				os.Unsetenv(envKey)
			}

			result := getEnvOrFlag(envKey, tt.flagValue, tt.defaultValue)
			if result != tt.expected {
				t.Errorf("getEnvOrFlag() = %q, want %q", result, tt.expected)
			}
		})
	}
}

func TestLoad_DefaultValues(t *testing.T) {
	flag.CommandLine = flag.NewFlagSet("test", flag.ContinueOnError)
	os.Args = []string{"test"}
	defer func() { os.Args = []string{"test"} }()
	cleanupEnv()

	cfg := Load()

	if cfg.Port != "8080" {
		t.Errorf("Port = %q, want 8080", cfg.Port)
	}
	if cfg.SSELogDir != "" {
		t.Errorf("SSELogDir = %q, want empty", cfg.SSELogDir)
	}
	// Without a config file, AppConfig should be nil
	if cfg.AppConfig != nil {
		t.Errorf("AppConfig should be nil when no config file is provided")
	}
}

func TestLoad_EnvironmentVariables(t *testing.T) {
	flag.CommandLine = flag.NewFlagSet("test", flag.ContinueOnError)
	os.Args = []string{"test"}
	defer func() { os.Args = []string{"test"} }()
	cleanupEnv()

	os.Setenv("PORT", "9090")
	os.Setenv("SSELOG_DIR", "/var/log/sse")
	defer cleanupEnv()

	cfg := Load()

	if cfg.Port != "9090" {
		t.Errorf("Port = %q, want 9090", cfg.Port)
	}
	if cfg.SSELogDir != "/var/log/sse" {
		t.Errorf("SSELogDir = %q, want /var/log/sse", cfg.SSELogDir)
	}
}

func TestLoad_PartialEnvironmentVariables(t *testing.T) {
	flag.CommandLine = flag.NewFlagSet("test", flag.ContinueOnError)
	os.Args = []string{"test"}
	defer func() { os.Args = []string{"test"} }()
	cleanupEnv()

	os.Setenv("PORT", "3000")
	defer cleanupEnv()

	cfg := Load()

	if cfg.Port != "3000" {
		t.Errorf("Port = %q, want 3000", cfg.Port)
	}
	if cfg.SSELogDir != "" {
		t.Errorf("SSELogDir = %q, want empty", cfg.SSELogDir)
	}
}

func TestLoad_ConfigFile(t *testing.T) {
	flag.CommandLine = flag.NewFlagSet("test", flag.ContinueOnError)

	// Create a temporary config file
	tmpFile, err := os.CreateTemp("", "config-*.json")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(tmpFile.Name())

	configContent := `{
		"providers": [
			{
				"name": "test-provider",
				"type": "openai",
				"base_url": "https://api.example.com/v1",
				"apiKey": "test-key"
			}
		],
		"models": {
			"test-model": {
				"provider": "test-provider",
				"model": "gpt-4"
			}
		},
		"fallback": {
			"enabled": false
		}
	}`

	if _, err := tmpFile.WriteString(configContent); err != nil {
		t.Fatalf("Failed to write config: %v", err)
	}
	tmpFile.Close()

	os.Args = []string{"test", "-config-file=" + tmpFile.Name()}
	defer func() { os.Args = []string{"test"} }()
	cleanupEnv()

	cfg := Load()

	if cfg.ConfigFile != tmpFile.Name() {
		t.Errorf("ConfigFile = %q, want %q", cfg.ConfigFile, tmpFile.Name())
	}
	if cfg.AppConfig == nil {
		t.Fatal("AppConfig should not be nil")
	}
	if len(cfg.AppConfig.Providers) != 1 {
		t.Errorf("Providers count = %d, want 1", len(cfg.AppConfig.Providers))
	}
	if cfg.AppConfig.Providers[0].Name != "test-provider" {
		t.Errorf("Provider name = %q, want test-provider", cfg.AppConfig.Providers[0].Name)
	}
	if _, ok := cfg.AppConfig.Models["test-model"]; !ok {
		t.Error("Expected test-model in models map")
	}
}

func TestLoad_ConfigFileFromEnv(t *testing.T) {
	// Create a temporary config file
	tmpFile, err := os.CreateTemp("", "config-*.json")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(tmpFile.Name())

	configContent := `{
		"providers": [
			{
				"name": "env-provider",
				"type": "anthropic",
				"base_url": "https://api.anthropic.com",
				"apiKey": "test-api-key"
			}
		],
		"models": {},
		"fallback": {}
	}`

	if _, err := tmpFile.WriteString(configContent); err != nil {
		t.Fatalf("Failed to write config: %v", err)
	}
	tmpFile.Close()

	flag.CommandLine = flag.NewFlagSet("test", flag.ContinueOnError)
	originalArgs := os.Args
	os.Args = []string{"test"}
	defer func() { os.Args = originalArgs }()

	os.Unsetenv("PORT")
	os.Unsetenv("SSELOG_DIR")
	os.Setenv("CONFIG_FILE", tmpFile.Name())
	defer os.Unsetenv("CONFIG_FILE")

	cfg := Load()

	if cfg.ConfigFile != tmpFile.Name() {
		t.Errorf("ConfigFile = %q, want %q", cfg.ConfigFile, tmpFile.Name())
	}
	if cfg.AppConfig == nil {
		t.Fatal("AppConfig should not be nil")
	}
	if cfg.AppConfig.Providers[0].Name != "env-provider" {
		t.Errorf("Provider name = %q, want env-provider", cfg.AppConfig.Providers[0].Name)
	}
}

func TestLoad_NonexistentConfigFile(t *testing.T) {
	flag.CommandLine = flag.NewFlagSet("test", flag.ContinueOnError)
	os.Args = []string{"test", "-config-file=/nonexistent/path/config.json"}
	defer func() { os.Args = []string{"test"} }()
	cleanupEnv()

	cfg := Load()

	// Should have nil AppConfig when file doesn't exist
	if cfg.AppConfig != nil {
		t.Error("AppConfig should be nil for nonexistent file")
	}
}

func TestGetSchema(t *testing.T) {
	// Test with nil AppConfig
	cfg := &Config{AppConfig: nil}
	if cfg.GetSchema() != nil {
		t.Error("GetSchema() should return nil when AppConfig is nil")
	}

	// Test with valid AppConfig
	schema := &Schema{
		Providers: []Provider{{Name: "test", Type: "openai", BaseURL: "https://example.com"}},
	}
	cfg = &Config{AppConfig: schema}
	if cfg.GetSchema() != schema {
		t.Error("GetSchema() should return the AppConfig")
	}
}

func TestLoad_PortAndSSELogDirWithConfigFile(t *testing.T) {
	flag.CommandLine = flag.NewFlagSet("test", flag.ContinueOnError)

	// Create a temporary config file
	tmpFile, err := os.CreateTemp("", "config-*.json")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(tmpFile.Name())

	configContent := `{
		"providers": [
			{
				"name": "test-provider",
				"type": "openai",
				"base_url": "https://api.example.com",
				"apiKey": "test-key"
			}
		],
		"models": {},
		"fallback": {}
	}`
	if _, err := tmpFile.WriteString(configContent); err != nil {
		t.Fatalf("Failed to write config: %v", err)
	}
	tmpFile.Close()

	os.Args = []string{"test", "-config-file=" + tmpFile.Name()}
	defer func() { os.Args = []string{"test"} }()
	cleanupEnv()
	os.Setenv("PORT", "7070")
	os.Setenv("SSELOG_DIR", "/var/log/test")
	defer cleanupEnv()

	cfg := Load()

	if cfg.Port != "7070" {
		t.Errorf("Port = %q, want 7070", cfg.Port)
	}
	if cfg.SSELogDir != "/var/log/test" {
		t.Errorf("SSELogDir = %q, want /var/log/test", cfg.SSELogDir)
	}
}
