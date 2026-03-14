package main

import (
	"log"
	"os"
	"os/signal"
	"syscall"

	"ai-proxy/api"
	"ai-proxy/config"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("Failed to load configuration: %v", err)
	}

	log.Printf("Starting AI Proxy server on port %s", cfg.Port)
	log.Printf("Loaded configuration from: %s", cfg.ConfigFile)

	// Log provider information
	for _, provider := range cfg.AppConfig.Providers {
		log.Printf("Configured provider: %s (type: %s, baseURL: %s)", provider.Name, provider.Type, provider.BaseURL)
	}

	// Create and configure server
	server := api.NewServer(cfg)

	// Setup graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigChan
		log.Println("Shutting down server...")
		os.Exit(0)
	}()

	// Start server
	addr := ":" + cfg.Port
	log.Printf("Server ready to accept connections on %s", addr)
	if err := server.Run(addr); err != nil {
		log.Fatalf("Server error: %v", err)
	}
}
