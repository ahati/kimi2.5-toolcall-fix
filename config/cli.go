// Package config provides configuration loading from environment variables and command-line flags.
// Configuration values are loaded with precedence: flags > environment variables > defaults.
package config

import (
	"errors"
	"flag"
	"os"
)

// ErrConfigFileRequired is returned when no config file path is provided.
var ErrConfigFileRequired = errors.New("config file required: use --config-file or CONFIG_FILE environment variable")

// ParseFlags parses CLI flags and returns the config file path.
// Priority: --config-file flag > CONFIG_FILE env var > error
//
// @return string - the config file path
// @return error - ErrConfigFileRequired if neither flag nor env var is provided
// @post flag.Parse() has been called, consuming command-line arguments
// @note The returned path may be empty only if an error is returned
func ParseFlags() (configFile string, err error) {
	configFilePath := flag.String("config-file", "", "Path to configuration file")

	flag.Parse()

	if *configFilePath != "" {
		return *configFilePath, nil
	}

	if envPath, ok := os.LookupEnv("CONFIG_FILE"); ok && envPath != "" {
		return envPath, nil
	}

	return "", ErrConfigFileRequired
}