package config

import (
	"flag"
	"os"
	"testing"
)

func cleanupEnv() {
	os.Unsetenv("UPSTREAM_URL")
	os.Unsetenv("UPSTREAM_API_KEY")
	os.Unsetenv("ANTHROPIC_UPSTREAM_URL")
	os.Unsetenv("ALIBABA_ANTHROPIC_API_KEY")
	os.Unsetenv("PORT")
	os.Unsetenv("SSELOG_DIR")
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
	cleanupEnv()

	cfg := Load()

	if cfg.OpenAIUpstreamURL != "https://llm.chutes.ai/v1/chat/completions" {
		t.Errorf("OpenAIUpstreamURL = %q, want default", cfg.OpenAIUpstreamURL)
	}
	if cfg.OpenAIUpstreamAPIKey != "" {
		t.Errorf("OpenAIUpstreamAPIKey = %q, want empty", cfg.OpenAIUpstreamAPIKey)
	}
	if cfg.AnthropicUpstreamURL != "https://coding-intl.dashscope.aliyuncs.com/apps/anthropic/v1/messages" {
		t.Errorf("AnthropicUpstreamURL = %q, want default", cfg.AnthropicUpstreamURL)
	}
	if cfg.AnthropicAPIKey != "" {
		t.Errorf("AnthropicAPIKey = %q, want empty", cfg.AnthropicAPIKey)
	}
	if cfg.Port != "8080" {
		t.Errorf("Port = %q, want 8080", cfg.Port)
	}
	if cfg.SSELogDir != "" {
		t.Errorf("SSELogDir = %q, want empty", cfg.SSELogDir)
	}
}

func TestLoad_EnvironmentVariables(t *testing.T) {
	flag.CommandLine = flag.NewFlagSet("test", flag.ContinueOnError)
	cleanupEnv()

	os.Setenv("UPSTREAM_URL", "https://custom.upstream.com/v1")
	os.Setenv("UPSTREAM_API_KEY", "custom-api-key-123")
	os.Setenv("ANTHROPIC_UPSTREAM_URL", "https://custom.anthropic.com/v1")
	os.Setenv("ALIBABA_ANTHROPIC_API_KEY", "anthropic-key-456")
	os.Setenv("PORT", "9090")
	os.Setenv("SSELOG_DIR", "/var/log/sse")

	defer cleanupEnv()

	cfg := Load()

	if cfg.OpenAIUpstreamURL != "https://custom.upstream.com/v1" {
		t.Errorf("OpenAIUpstreamURL = %q, want custom URL", cfg.OpenAIUpstreamURL)
	}
	if cfg.OpenAIUpstreamAPIKey != "custom-api-key-123" {
		t.Errorf("OpenAIUpstreamAPIKey = %q, want custom key", cfg.OpenAIUpstreamAPIKey)
	}
	if cfg.AnthropicUpstreamURL != "https://custom.anthropic.com/v1" {
		t.Errorf("AnthropicUpstreamURL = %q, want custom URL", cfg.AnthropicUpstreamURL)
	}
	if cfg.AnthropicAPIKey != "anthropic-key-456" {
		t.Errorf("AnthropicAPIKey = %q, want custom key", cfg.AnthropicAPIKey)
	}
	if cfg.Port != "9090" {
		t.Errorf("Port = %q, want 9090", cfg.Port)
	}
	if cfg.SSELogDir != "/var/log/sse" {
		t.Errorf("SSELogDir = %q, want /var/log/sse", cfg.SSELogDir)
	}
}

func TestLoad_PartialEnvironmentVariables(t *testing.T) {
	flag.CommandLine = flag.NewFlagSet("test", flag.ContinueOnError)
	cleanupEnv()

	os.Setenv("PORT", "3000")
	os.Setenv("SSELOG_DIR", "./logs")

	defer cleanupEnv()

	cfg := Load()

	if cfg.OpenAIUpstreamURL != "https://llm.chutes.ai/v1/chat/completions" {
		t.Errorf("OpenAIUpstreamURL = %q, want default", cfg.OpenAIUpstreamURL)
	}
	if cfg.OpenAIUpstreamAPIKey != "" {
		t.Errorf("OpenAIUpstreamAPIKey = %q, want empty", cfg.OpenAIUpstreamAPIKey)
	}
	if cfg.Port != "3000" {
		t.Errorf("Port = %q, want 3000", cfg.Port)
	}
	if cfg.SSELogDir != "./logs" {
		t.Errorf("SSELogDir = %q, want ./logs", cfg.SSELogDir)
	}
}

func TestLoad_WithFlags(t *testing.T) {
	flag.CommandLine = flag.NewFlagSet("test", flag.ContinueOnError)
	os.Args = []string{
		"test",
		"-upstream-url=https://flag.upstream.com/v1",
		"-upstream-api-key=flag-api-key",
		"-anthropic-upstream-url=https://flag.anthropic.com/v1",
		"-anthropic-api-key=flag-anthropic-key",
		"-port=7070",
		"-sse-log-dir=/flag/log/dir",
	}
	defer func() { os.Args = []string{"test"} }()

	cleanupEnv()

	cfg := Load()

	if cfg.OpenAIUpstreamURL != "https://flag.upstream.com/v1" {
		t.Errorf("OpenAIUpstreamURL = %q, want flag value", cfg.OpenAIUpstreamURL)
	}
	if cfg.OpenAIUpstreamAPIKey != "flag-api-key" {
		t.Errorf("OpenAIUpstreamAPIKey = %q, want flag value", cfg.OpenAIUpstreamAPIKey)
	}
	if cfg.AnthropicUpstreamURL != "https://flag.anthropic.com/v1" {
		t.Errorf("AnthropicUpstreamURL = %q, want flag value", cfg.AnthropicUpstreamURL)
	}
	if cfg.AnthropicAPIKey != "flag-anthropic-key" {
		t.Errorf("AnthropicAPIKey = %q, want flag value", cfg.AnthropicAPIKey)
	}
	if cfg.Port != "7070" {
		t.Errorf("Port = %q, want 7070", cfg.Port)
	}
	if cfg.SSELogDir != "/flag/log/dir" {
		t.Errorf("SSELogDir = %q, want /flag/log/dir", cfg.SSELogDir)
	}
}

func TestLoad_FlagOverridesEnv(t *testing.T) {
	flag.CommandLine = flag.NewFlagSet("test", flag.ContinueOnError)
	os.Args = []string{"test", "-port=8888"}
	defer func() { os.Args = []string{"test"} }()

	os.Setenv("PORT", "9999")
	defer cleanupEnv()

	cfg := Load()

	if cfg.Port != "8888" {
		t.Errorf("Port = %q, want 8888 (flag should override env)", cfg.Port)
	}
}

func TestLoad_FlagOverridesAllEnv(t *testing.T) {
	flag.CommandLine = flag.NewFlagSet("test", flag.ContinueOnError)
	os.Args = []string{
		"test",
		"-upstream-url=https://flag.override/v1",
		"-upstream-api-key=flag-key-override",
		"-port=5555",
	}
	defer func() { os.Args = []string{"test"} }()

	os.Setenv("UPSTREAM_URL", "https://env.override/v1")
	os.Setenv("UPSTREAM_API_KEY", "env-key-override")
	os.Setenv("PORT", "6666")
	defer cleanupEnv()

	cfg := Load()

	if cfg.OpenAIUpstreamURL != "https://flag.override/v1" {
		t.Errorf("OpenAIUpstreamURL = %q, want flag value", cfg.OpenAIUpstreamURL)
	}
	if cfg.OpenAIUpstreamAPIKey != "flag-key-override" {
		t.Errorf("OpenAIUpstreamAPIKey = %q, want flag value", cfg.OpenAIUpstreamAPIKey)
	}
	if cfg.Port != "5555" {
		t.Errorf("Port = %q, want 5555 (flag should override env)", cfg.Port)
	}
}

func TestConfig_AllFieldsSetFromEnv(t *testing.T) {
	flag.CommandLine = flag.NewFlagSet("test", flag.ContinueOnError)
	cleanupEnv()

	os.Setenv("UPSTREAM_URL", "https://test1.com/v1")
	os.Setenv("UPSTREAM_API_KEY", "key1")
	os.Setenv("ANTHROPIC_UPSTREAM_URL", "https://test2.com/v1")
	os.Setenv("ALIBABA_ANTHROPIC_API_KEY", "key2")
	os.Setenv("PORT", "1234")
	os.Setenv("SSELOG_DIR", "/test/logs")
	defer cleanupEnv()

	cfg := Load()

	if cfg.OpenAIUpstreamURL != "https://test1.com/v1" {
		t.Errorf("OpenAIUpstreamURL = %q, want https://test1.com/v1", cfg.OpenAIUpstreamURL)
	}
	if cfg.OpenAIUpstreamAPIKey != "key1" {
		t.Errorf("OpenAIUpstreamAPIKey = %q, want key1", cfg.OpenAIUpstreamAPIKey)
	}
	if cfg.AnthropicUpstreamURL != "https://test2.com/v1" {
		t.Errorf("AnthropicUpstreamURL = %q, want https://test2.com/v1", cfg.AnthropicUpstreamURL)
	}
	if cfg.AnthropicAPIKey != "key2" {
		t.Errorf("AnthropicAPIKey = %q, want key2", cfg.AnthropicAPIKey)
	}
	if cfg.Port != "1234" {
		t.Errorf("Port = %q, want 1234", cfg.Port)
	}
	if cfg.SSELogDir != "/test/logs" {
		t.Errorf("SSELogDir = %q, want /test/logs", cfg.SSELogDir)
	}
}
