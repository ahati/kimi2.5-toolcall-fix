package config

import (
	"os"
	"testing"
)

func TestGetEnvOrFlag_FlagValue(t *testing.T) {
	result := getEnvOrFlag("NONEXISTENT_ENV", "flag-value", "default")
	if result != "flag-value" {
		t.Errorf("Expected flag-value, got %s", result)
	}
}

func TestGetEnvOrFlag_EnvValue(t *testing.T) {
	os.Setenv("TEST_ENV_VAR", "env-value")
	defer os.Unsetenv("TEST_ENV_VAR")

	result := getEnvOrFlag("TEST_ENV_VAR", "", "default")
	if result != "env-value" {
		t.Errorf("Expected env-value, got %s", result)
	}
}

func TestGetEnvOrFlag_DefaultValue(t *testing.T) {
	result := getEnvOrFlag("NONEXISTENT_ENV", "", "default")
	if result != "default" {
		t.Errorf("Expected default, got %s", result)
	}
}

func TestGetEnvOrFlag_FlagOverridesEnv(t *testing.T) {
	os.Setenv("TEST_OVERRIDE", "env-value")
	defer os.Unsetenv("TEST_OVERRIDE")

	result := getEnvOrFlag("TEST_OVERRIDE", "flag-value", "default")
	if result != "flag-value" {
		t.Errorf("Expected flag-value, got %s", result)
	}
}
