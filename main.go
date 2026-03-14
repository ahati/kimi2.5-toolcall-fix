package main

import (
	"flag"
	"fmt"
	"os"
)

func main() {
	configPath := parseConfigPath(os.Args[1:])
	if configPath == "" {
		fmt.Fprintln(os.Stderr, "Error: --config-file or CONFIG_FILE env var required")
		os.Exit(1)
	}
	fmt.Println("Config path:", configPath)
}

func parseConfigPath(args []string) string {
	var configFile string
	fs := flag.NewFlagSet("main", flag.ContinueOnError)
	fs.StringVar(&configFile, "config-file", "", "Path to configuration file")
	fs.Parse(args)
	if configFile != "" {
		return configFile
	}
	return os.Getenv("CONFIG_FILE")
}
