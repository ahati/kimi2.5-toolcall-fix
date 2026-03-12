// Package main provides the entry point for the AI proxy server.
// It initializes configuration, logging, and starts the HTTP server.
package main

import (
	"os"

	"ai-proxy/config"
	"ai-proxy/downstream"
	"ai-proxy/downstream/protocols"
	"ai-proxy/logging"

	"github.com/gin-gonic/gin"
)

// main is the application entry point.
//
// @brief    Initializes and starts the AI proxy HTTP server.
//
// @note     Loads configuration from environment variables and command-line flags.
// @note     Initializes logging with optional SSE capture storage.
// @note     Registers HTTP endpoints for health, models, and streaming handlers.
// @note     Server listens on the configured port (default: 8080).
//
// @pre      Configuration environment variables or flags are optional.
// @post     HTTP server runs until interrupted or fatal error.
func main() {
	cfg := config.Load()
	logging.Init()

	var storage *logging.Storage
	if cfg.SSELogDir != "" {
		storage = logging.NewStorage(cfg.SSELogDir)
	}

	r := gin.Default()
	r.Use(logging.CaptureMiddleware(storage))

	r.GET("/health", downstream.HealthCheck)

	r.GET("/v1/models", downstream.ListModels(cfg))

	r.POST("/v1/chat/completions", downstream.StreamHandler(cfg, &protocols.OpenAIAdapter{}))

	r.POST("/v1/messages", downstream.StreamHandler(cfg, &protocols.AnthropicAdapter{}))

	r.POST("/v1/openai-to-anthropic/messages", downstream.StreamHandler(cfg, &protocols.BridgeAdapter{}))

	addr := ":" + cfg.Port
	logging.InfoMsg("ai-proxy server starting on %s", addr)
	if err := r.Run(addr); err != nil {
		logging.ErrorMsg("Failed to start server: %v", err)
		os.Exit(1)
	}
}
