package main

import (
	"os"

	"github.com/gin-gonic/gin"
	"proxy/config"
	"proxy/downstream"
	"proxy/logging"
)

func main() {
	cfg := config.Load()
	logging.Init()

	r := gin.Default()

	r.GET("/health", downstream.HealthCheck)

	r.GET("/v1/models", downstream.ListModels(cfg))

	r.POST("/v1/chat/completions", downstream.Completions(cfg))

	addr := ":" + cfg.Port
	logging.InfoMsg("Proxy server starting on %s", addr)
	if err := r.Run(addr); err != nil {
		logging.ErrorMsg("Failed to start server: %v", err)
		os.Exit(1)
	}
}
