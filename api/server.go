// Package api provides the HTTP server and routing configuration for the AI proxy.
// This package implements the core server infrastructure responsible for handling
// incoming HTTP requests and routing them to appropriate handlers.
package api

import (
	"ai-proxy/api/handlers"
	"ai-proxy/config"
	"ai-proxy/router"

	"github.com/gin-gonic/gin"
)

// Server represents the HTTP server that handles all API routes.
type Server struct {
	router *gin.Engine
	config *config.Config
	rtr    router.Router
}

// NewServer creates and initializes a new Server instance with the given configuration.
// It sets up all routes and middleware during initialization.
//
// @param cfg - Application configuration containing upstream URLs and API keys.
//
//	Must not be nil. Caller retains ownership.
//
// @param middleware - Optional middleware functions to apply before route handlers.
//
// @return Pointer to newly allocated Server instance. Never returns nil.
//
// @pre cfg != nil
// @post Returned Server has all routes registered and is ready to accept connections.
// @post Gin is set to ReleaseMode if cfg.Port is non-empty (production mode).
func NewServer(cfg *config.Config, middleware ...gin.HandlerFunc) *Server {
	if cfg.Port != "" {
		gin.SetMode(gin.ReleaseMode)
	}

	rtr, err := router.NewRouter(cfg.AppConfig)
	if err != nil {
		panic(err)
	}

	s := &Server{
		router: gin.Default(),
		config: cfg,
		rtr:    rtr,
	}

	for _, m := range middleware {
		s.router.Use(m)
	}

	s.setupRoutes()

	return s
}

// setupRoutes registers all API endpoints with their corresponding handlers.
// This method is called once during server initialization.
//
// @pre s.router != nil
// @pre s.config != nil
// @post All endpoints are registered and accessible through the router.
func (s *Server) setupRoutes() {
	s.router.GET("/health", handlers.HealthCheck)
	s.router.GET("/v1/models", handlers.NewModelsHandler(s.config))
	s.router.POST("/v1/chat/completions", handlers.NewCompletionsHandler(s.rtr))
	s.router.POST("/v1/messages", handlers.NewMessagesHandler(s.rtr))
	s.router.POST("/v1/messages/count_tokens", handlers.NewCountTokensHandler(s.config))
	s.router.POST("/v1/openai-to-anthropic/messages", handlers.NewBridgeHandler(s.rtr))
	s.router.POST("/v1/anthropic-to-openai/responses", handlers.NewAnthropicToOpenAIHandler(s.config))
	s.router.POST("/v1/responses", handlers.NewResponsesHandler(s.rtr))
}

// Use adds middleware to the server's router chain.
// Middleware is executed in the order it is added, before route handlers.
//
// @param middleware - One or more Gin middleware functions to add to the chain.
//
//	May be empty, in which case no action is taken.
//
// @pre s.router != nil
// @post Middleware functions are added to the processing chain.
func (s *Server) Use(middleware ...gin.HandlerFunc) {
	s.router.Use(middleware...)
}

// Run starts the HTTP server on the specified address.
// This method blocks until the server encounters an error or is shut down.
//
// @param addr - Network address to listen on in "host:port" format.
//
//	Empty host defaults to all interfaces. Empty port uses default.
//
// @return Error if server fails to start or encounters a fatal error.
//
//	Returns nil only if server shuts down cleanly (rare).
//
// @pre s.router != nil
// @post HTTP server is running and accepting connections (until error occurs).
// @note This is a blocking call. Use in goroutine if non-blocking start needed.
func (s *Server) Run(addr string) error {
	return s.router.Run(addr)
}
