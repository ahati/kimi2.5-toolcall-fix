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

// CLIFlags holds the parsed command-line flags.
type CLIFlags struct {
	ConfigFile            string
	SSELogDir             string
	Port                  string
	ConversationStoreSize int
	ConversationStoreTTL  string
}

// ParseFlags parses CLI flags and returns the parsed flags.
// Priority for config file: --config-file flag > CONFIG_FILE env var > error
// Priority for other flags: flag > environment variable > default
//
// @return CLIFlags - the parsed flags
// @return error - ErrConfigFileRequired if neither flag nor env var is provided for config file
// @post flag.Parse() has been called, consuming command-line arguments
func ParseFlags() (CLIFlags, error) {
	configFilePath := flag.String("config-file", "", "Path to configuration file")
	sseLogDir := flag.String("sse-log-dir", "", "Directory for SSE request/response logging")
	port := flag.String("port", "", "Server port (default: 8080)")
	conversationStoreSize := flag.Int("conversation-store-size", 0, "Max conversations in memory (default: 1000)")
	conversationStoreTTL := flag.String("conversation-store-ttl", "", "Conversation TTL duration (default: 24h)")

	flag.Parse()

	flags := CLIFlags{
		SSELogDir:             *sseLogDir,
		Port:                  *port,
		ConversationStoreSize: *conversationStoreSize,
		ConversationStoreTTL:  *conversationStoreTTL,
	}

	if *configFilePath != "" {
		flags.ConfigFile = *configFilePath
		return flags, nil
	}

	if envPath, ok := os.LookupEnv("CONFIG_FILE"); ok && envPath != "" {
		flags.ConfigFile = envPath
		return flags, nil
	}

	return flags, ErrConfigFileRequired
}
