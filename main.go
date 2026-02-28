package main

import (
	"os"

	"ai-proxy/config"
	"ai-proxy/downstream"
	"ai-proxy/logging"
	"github.com/gin-gonic/gin"
)

func main() {
	cfg := config.Load()
	logging.Init()

	r := gin.Default()

	r.GET("/health", downstream.HealthCheck)

	r.GET("/v1/models", downstream.ListModels(cfg))

	r.POST("/v1/chat/completions", downstream.Completions(cfg))

	addr := ":" + cfg.Port
	logging.InfoMsg("ai-proxy server starting on %s", addr)
	if err := r.Run(addr); err != nil {
		logging.ErrorMsg("Failed to start server: %v", err)
		os.Exit(1)
	}
}
