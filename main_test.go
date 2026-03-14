package main

import (
	"flag"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

func resetFlags() {
	flag.CommandLine = flag.NewFlagSet("test", flag.ContinueOnError)
	configFile = ""
}

func TestResolveConfigPath_Flag(t *testing.T) {
	resetFlags()
	os.Unsetenv("CONFIG_FILE")

	configFile = "/path/to/config.json"
	path, err := resolveConfigPath()

	assert.NoError(t, err)
	assert.Equal(t, "/path/to/config.json", path)
}

func TestResolveConfigPath_EnvVar(t *testing.T) {
	resetFlags()
	os.Setenv("CONFIG_FILE", "/env/config.json")
	defer os.Unsetenv("CONFIG_FILE")

	path, err := resolveConfigPath()

	assert.NoError(t, err)
	assert.Equal(t, "/env/config.json", path)
}

func TestResolveConfigPath_FlagPrecedence(t *testing.T) {
	resetFlags()
	os.Setenv("CONFIG_FILE", "/env/config.json")
	defer os.Unsetenv("CONFIG_FILE")

	configFile = "/flag/config.json"
	path, err := resolveConfigPath()

	assert.NoError(t, err)
	assert.Equal(t, "/flag/config.json", path)
}

func TestResolveConfigPath_Error(t *testing.T) {
	resetFlags()
	os.Unsetenv("CONFIG_FILE")

	path, err := resolveConfigPath()

	assert.Error(t, err)
	assert.Equal(t, "", path)
	assert.Contains(t, err.Error(), "config file required")
}

func TestResolveConfigPath_EnvEmptyString(t *testing.T) {
	resetFlags()
	os.Setenv("CONFIG_FILE", "")
	defer os.Unsetenv("CONFIG_FILE")

	path, err := resolveConfigPath()

	assert.Error(t, err)
	assert.Equal(t, "", path)
}
