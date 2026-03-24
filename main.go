// Package main is the entry point for the ai-proxy HTTP server.
// It initializes configuration, logging, and starts the proxy server.
package main

import (
	"fmt"
	"os"

	"ai-proxy/api"
	"ai-proxy/config"
	"ai-proxy/conversation"
	"ai-proxy/logging"
	"ai-proxy/summarizer"
)

// main loads configuration, initializes logging and storage, then starts the HTTP server.
// This is the application entry point and orchestrates all startup tasks.
//
// @pre Environment variables and command-line flags are available for configuration
// @post Server is running and listening on the configured port
// @post All capture middleware is initialized if SSELogDir is configured
// @note Exits with code 1 if config file is missing or server fails to start
// @note Blocks until server is stopped (SIGINT, SIGTERM, or fatal error)
func main() {
	// Load configuration from config file, flags, environment variables, and defaults.
	// config.Load() internally parses CLI flags and loads the JSON config file.
	cfg := config.Load()

	// Verify config was loaded successfully (AppConfig is nil if loading failed)
	if cfg.AppConfig == nil {
		fmt.Fprintln(os.Stderr, "Error: Failed to load configuration")
		fmt.Fprintln(os.Stderr, "A config file is required. Use --config-file flag or CONFIG_FILE environment variable.")
		os.Exit(1)
	}

	// Initialize logging early so subsequent messages are captured
	logging.Init()

	// Initialize conversation store for previous_response_id support
	conversation.InitDefaultStore(conversation.Config{
		MaxSize: cfg.ConversationStoreSize,
		TTL:     cfg.ConversationStoreTTL,
	})
	logging.InfoMsg("Conversation store initialized: maxSize=%d, ttl=%v", cfg.ConversationStoreSize, cfg.ConversationStoreTTL)

	// Initialize summarizer service for reasoning summarization
	summarizer.InitDefaultService(cfg.AppConfig)

	// Initialize storage for request capture if logging is enabled
	storage := api.InitStorage(cfg.SSELogDir)
	if storage != nil {
		logging.InfoMsg("SSE capture enabled, logging to: %s", cfg.SSELogDir)
	} else {
		logging.InfoMsg("SSE capture disabled (use --sse-log-dir to enable)")
	}

	// Create server with loaded configuration
	// Middleware is added first so it applies to all routes
	server := api.NewServer(cfg, api.NewCaptureMiddleware(storage).Handler())

	// Build listen address from configured port
	addr := ":" + cfg.Port
	logging.InfoMsg("ai-proxy server starting on %s", addr)

	// Start server; this blocks until server stops
	if err := server.Run(addr); err != nil {
		logging.ErrorMsg("Failed to start server: %v", err)
		os.Exit(1) // Exit with error code if server fails to start
	}
}
