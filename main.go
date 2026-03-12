// Package main is the entry point for the ai-proxy HTTP server.
// It initializes configuration, logging, and starts the proxy server.
package main

import (
	"os"

	"ai-proxy/api"
	"ai-proxy/config"
	"ai-proxy/logging"
)

// main loads configuration, initializes logging and storage, then starts the HTTP server.
// This is the application entry point and orchestrates all startup tasks.
//
// @pre Environment variables and command-line flags are available for configuration
// @post Server is running and listening on the configured port
// @post All capture middleware is initialized if SSELogDir is configured
// @note Exits with code 1 if server fails to start
// @note Blocks until server is stopped (SIGINT, SIGTERM, or fatal error)
func main() {
	// Load configuration from flags, environment variables, and defaults
	cfg := config.Load()

	// Initialize logging early so subsequent messages are captured
	logging.Init()

	// Initialize storage for request capture if logging is enabled
	storage := api.InitStorage(cfg.SSELogDir)

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
