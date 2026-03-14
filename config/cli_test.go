package config

import (
	"flag"
	"os"
	"testing"
)

func TestParseFlags(t *testing.T) {
	tests := []struct {
		name           string
		args           []string
		envValue       string
		setEnv         bool
		expectedPath   string
		expectError    bool
		expectedErrMsg string
	}{
		{
			name:           "flag provided",
			args:           []string{"test", "--config-file=/path/to/config.yaml"},
			envValue:       "",
			setEnv:         false,
			expectedPath:   "/path/to/config.yaml",
			expectError:    false,
			expectedErrMsg: "",
		},
		{
			name:           "flag with equals sign",
			args:           []string{"test", "--config-file=/etc/app/config.json"},
			envValue:       "",
			setEnv:         false,
			expectedPath:   "/etc/app/config.json",
			expectError:    false,
			expectedErrMsg: "",
		},
		{
			name:           "flag with separate value",
			args:           []string{"test", "--config-file", "/opt/config.toml"},
			envValue:       "",
			setEnv:         false,
			expectedPath:   "/opt/config.toml",
			expectError:    false,
			expectedErrMsg: "",
		},
		{
			name:           "env var fallback when no flag",
			args:           []string{"test"},
			envValue:       "/env/config.yaml",
			setEnv:         true,
			expectedPath:   "/env/config.yaml",
			expectError:    false,
			expectedErrMsg: "",
		},
		{
			name:           "flag overrides env var",
			args:           []string{"test", "--config-file=/flag/config.yaml"},
			envValue:       "/env/config.yaml",
			setEnv:         true,
			expectedPath:   "/flag/config.yaml",
			expectError:    false,
			expectedErrMsg: "",
		},
		{
			name:           "neither flag nor env var provided",
			args:           []string{"test"},
			envValue:       "",
			setEnv:         false,
			expectedPath:   "",
			expectError:    true,
			expectedErrMsg: "config file required: use --config-file or CONFIG_FILE environment variable",
		},
		{
			name:           "env var set to empty string",
			args:           []string{"test"},
			envValue:       "",
			setEnv:         true,
			expectedPath:   "",
			expectError:    true,
			expectedErrMsg: "config file required: use --config-file or CONFIG_FILE environment variable",
		},
		{
			name:           "flag with relative path",
			args:           []string{"test", "--config-file=./local-config.yaml"},
			envValue:       "",
			setEnv:         false,
			expectedPath:   "./local-config.yaml",
			expectError:    false,
			expectedErrMsg: "",
		},
		{
			name:           "flag with spaces in path",
			args:           []string{"test", "--config-file=/path with spaces/config.yaml"},
			envValue:       "",
			setEnv:         false,
			expectedPath:   "/path with spaces/config.yaml",
			expectError:    false,
			expectedErrMsg: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			flag.CommandLine = flag.NewFlagSet("test", flag.ContinueOnError)

			originalArgs := os.Args
			os.Args = tt.args
			defer func() { os.Args = originalArgs }()

			if tt.setEnv {
				os.Setenv("CONFIG_FILE", tt.envValue)
			} else {
				os.Unsetenv("CONFIG_FILE")
			}
			defer os.Unsetenv("CONFIG_FILE")

			path, err := ParseFlags()

			if tt.expectError {
				if err == nil {
					t.Errorf("ParseFlags() expected error, got nil")
				}
				if err != nil && err.Error() != tt.expectedErrMsg {
					t.Errorf("ParseFlags() error = %q, want %q", err.Error(), tt.expectedErrMsg)
				}
				if path != "" {
					t.Errorf("ParseFlags() path = %q, want empty string on error", path)
				}
			} else {
				if err != nil {
					t.Errorf("ParseFlags() unexpected error: %v", err)
				}
				if path != tt.expectedPath {
					t.Errorf("ParseFlags() path = %q, want %q", path, tt.expectedPath)
				}
			}
		})
	}
}

func TestParseFlags_ErrConfigFileRequired(t *testing.T) {
	flag.CommandLine = flag.NewFlagSet("test", flag.ContinueOnError)
	os.Args = []string{"test"}
	os.Unsetenv("CONFIG_FILE")

	err := ErrConfigFileRequired

	if err == nil {
		t.Error("ErrConfigFileRequired should not be nil")
	}

	expectedMsg := "config file required: use --config-file or CONFIG_FILE environment variable"
	if err.Error() != expectedMsg {
		t.Errorf("ErrConfigFileRequired.Error() = %q, want %q", err.Error(), expectedMsg)
	}
}

func TestParseFlags_MultipleFlags(t *testing.T) {
	flag.CommandLine = flag.NewFlagSet("test", flag.ContinueOnError)
	os.Args = []string{"test", "--config-file=/config.yaml", "--port=8080"}
	os.Unsetenv("CONFIG_FILE")

	path, err := ParseFlags()

	if err != nil {
		t.Errorf("ParseFlags() unexpected error: %v", err)
	}
	if path != "/config.yaml" {
		t.Errorf("ParseFlags() path = %q, want /config.yaml", path)
	}
}

func TestParseFlags_FlagPrecedenceOverEnv(t *testing.T) {
	flag.CommandLine = flag.NewFlagSet("test", flag.ContinueOnError)
	os.Args = []string{"test", "--config-file=/flag-config.yaml"}
	os.Setenv("CONFIG_FILE", "/env-config.yaml")
	defer os.Unsetenv("CONFIG_FILE")

	path, err := ParseFlags()

	if err != nil {
		t.Errorf("ParseFlags() unexpected error: %v", err)
	}
	if path != "/flag-config.yaml" {
		t.Errorf("ParseFlags() path = %q, want /flag-config.yaml (flag should override env)", path)
	}
}

func TestParseFlags_EnvWithSpecialChars(t *testing.T) {
	flag.CommandLine = flag.NewFlagSet("test", flag.ContinueOnError)
	os.Args = []string{"test"}
	os.Setenv("CONFIG_FILE", "/path/with-special_chars.123/config.yaml")
	defer os.Unsetenv("CONFIG_FILE")

	path, err := ParseFlags()

	if err != nil {
		t.Errorf("ParseFlags() unexpected error: %v", err)
	}
	if path != "/path/with-special_chars.123/config.yaml" {
		t.Errorf("ParseFlags() path = %q, want /path/with-special_chars.123/config.yaml", path)
	}
}