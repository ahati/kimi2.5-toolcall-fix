package main

import (
	"flag"
	"fmt"
	"os"

	"ai-proxy/api"
	"ai-proxy/config"
	"ai-proxy/logging"
	"ai-proxy/router"
)

var configFile string

func init() {
	flag.StringVar(&configFile, "config-file", "", "Path to JSON configuration file")
}

func resolveConfigPath() (string, error) {
	if configFile != "" {
		return configFile, nil
	}
	if envPath := os.Getenv("CONFIG_FILE"); envPath != "" {
		return envPath, nil
	}
	return "", fmt.Errorf("config file required: use --config-file flag or CONFIG_FILE environment variable")
}

func main() {
	flag.Parse()

	path, err := resolveConfigPath()
	if err != nil {
		logging.ErrorMsg("Config error: %v", err)
		os.Exit(1)
	}

	cfg, err := config.Load(path)
	if err != nil {
		logging.ErrorMsg("Failed to load config: %v", err)
		os.Exit(1)
	}

	logging.InfoMsg("Config file: %s", path)

	logging.Init()

	storage := api.InitStorage(cfg.SSELogDir)
	if storage != nil {
		logging.InfoMsg("SSE capture enabled, logging to: %s", cfg.SSELogDir)
	} else {
		logging.InfoMsg("SSE capture disabled (use --sse-log-dir to enable)")
	}

	appRouter, err := router.NewRouter(cfg.AppConfig)
	if err != nil {
		logging.ErrorMsg("Failed to create router: %v", err)
		os.Exit(1)
	}

	server := api.NewServer(cfg, appRouter, api.NewCaptureMiddleware(storage).Handler())

	addr := ":" + cfg.Port
	logging.InfoMsg("ai-proxy server starting on %s", addr)

	if err := server.Run(addr); err != nil {
		logging.ErrorMsg("Failed to start server: %v", err)
		os.Exit(1)
	}
}
