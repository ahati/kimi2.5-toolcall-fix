package main

import (
	"os"
	"testing"
)

func TestParseConfigPath_Flag(t *testing.T) {
	result := parseConfigPath([]string{"--config-file", "/path/to/config.yaml"})
	if result != "/path/to/config.yaml" {
		t.Errorf("expected /path/to/config.yaml, got %s", result)
	}
}

func TestParseConfigPath_EnvVar(t *testing.T) {
	os.Setenv("CONFIG_FILE", "/env/config.yaml")
	defer os.Unsetenv("CONFIG_FILE")

	result := parseConfigPath([]string{})
	if result != "/env/config.yaml" {
		t.Errorf("expected /env/config.yaml, got %s", result)
	}
}

func TestParseConfigPath_FlagPrecedence(t *testing.T) {
	os.Setenv("CONFIG_FILE", "/env/config.yaml")
	defer os.Unsetenv("CONFIG_FILE")

	result := parseConfigPath([]string{"--config-file", "/flag/config.yaml"})
	if result != "/flag/config.yaml" {
		t.Errorf("expected /flag/config.yaml (flag takes precedence), got %s", result)
	}
}

func TestParseConfigPath_Empty(t *testing.T) {
	os.Unsetenv("CONFIG_FILE")

	result := parseConfigPath([]string{})
	if result != "" {
		t.Errorf("expected empty string, got %s", result)
	}
}
