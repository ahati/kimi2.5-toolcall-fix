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
// It encapsulates the Gin router and configuration for the proxy service.
// The Server is responsible for:
//   - Managing the HTTP listener lifecycle
//   - Routing requests to appropriate handlers
//   - Applying middleware to request processing pipeline
//
// @invariant s.router != nil (router is always initialized)
// @invariant s.config != nil (config is always initialized)
type Server struct {
	// router is the underlying Gin engine that handles HTTP routing.
	// Must not be nil after initialization via NewServer.
	router *gin.Engine

	// config holds the application configuration including upstream URLs,
	// API keys, and other runtime settings. Must not be nil after initialization.
	config *config.Config

	// modelRouter resolves model names to providers.
	// May be nil if no config file was loaded.
	modelRouter router.Router
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
	// Set Gin to release mode for production when port is configured
	// to reduce console noise and improve performance
	if cfg.Port != "" {
		gin.SetMode(gin.ReleaseMode)
	}

	s := &Server{
		router: gin.Default(),
		config: cfg,
	}

	// Create model router if config is loaded
	if cfg.AppConfig != nil {
		if r, err := router.NewRouter(cfg.AppConfig); err == nil {
			s.modelRouter = r
		}
	}

	// Apply middleware first so it runs before routes
	for _, m := range middleware {
		s.router.Use(m)
	}

	// Register all API routes after middleware is set up
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
	// Health check endpoint - used by load balancers and monitoring systems
	// to verify service availability. Does not require authentication.
	s.router.GET("/health", handlers.HealthCheck)

	// Models endpoint - returns list of available models from upstream API
	// Supports OpenAI-compatible response format.
	s.router.GET("/v1/models", handlers.NewModelsHandler(s.config))

	// Chat completions endpoint - primary OpenAI-compatible endpoint
	// for streaming chat completions with tool call support.
	s.router.POST("/v1/chat/completions", handlers.NewCompletionsHandler(s.config, s.modelRouter))

	// Messages endpoint - native Anthropic API format endpoint
	// for streaming messages with tool call support.
	s.router.POST("/v1/messages", handlers.NewMessagesHandler(s.config))

	// Messages count tokens endpoint - Anthropic API format endpoint
	// for counting tokens in messages before sending to upstream.
	s.router.POST("/v1/messages/count_tokens", handlers.NewCountTokensHandler(s.config))

	// Bridge endpoint - converts Anthropic format requests to OpenAI format
	// before forwarding to upstream, then converts responses back to Anthropic format.
	s.router.POST("/v1/openai-to-anthropic/messages", handlers.NewBridgeHandler(s.config, s.modelRouter))

	// Anthropic-to-OpenAI Responses endpoint - converts OpenAI Responses API format requests
	// to Anthropic format before forwarding to upstream, then converts responses back to
	// OpenAI Responses API format.
	s.router.POST("/v1/anthropic-to-openai/responses", handlers.NewAnthropicToOpenAIHandler(s.config))

	// Responses endpoint - unified OpenAI Responses API endpoint that routes to the
	// appropriate provider based on model configuration.
	if s.modelRouter != nil {
		s.router.POST("/v1/responses", handlers.NewResponsesHandler(s.config, s.modelRouter))
	}
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
