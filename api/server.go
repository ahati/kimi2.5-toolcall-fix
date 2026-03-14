package api

import (
	"ai-proxy/api/handlers"
	"ai-proxy/config"
	"ai-proxy/router"

	"github.com/gin-gonic/gin"
)

type Server struct {
	router    *gin.Engine
	config    *config.Config
	appRouter router.Router
}

func NewServer(cfg *config.Config, r router.Router, middleware ...gin.HandlerFunc) *Server {
	if cfg.Port != "" {
		gin.SetMode(gin.ReleaseMode)
	}

	s := &Server{
		router:    gin.Default(),
		config:    cfg,
		appRouter: r,
	}

	for _, m := range middleware {
		s.router.Use(m)
	}

	s.setupRoutes()

	return s
}

func (s *Server) setupRoutes() {
	s.router.GET("/health", handlers.HealthCheck)

	s.router.GET("/v1/models", handlers.NewModelsHandler(s.config))

	s.router.POST("/v1/chat/completions", handlers.NewCompletionsHandler(s.appRouter))

	s.router.POST("/v1/messages", handlers.NewMessagesHandler(s.appRouter))

	s.router.POST("/v1/messages/count_tokens", handlers.NewCountTokensHandler(s.config))

	s.router.POST("/v1/openai-to-anthropic/messages", handlers.NewBridgeHandler(s.appRouter))

	s.router.POST("/v1/anthropic-to-openai/responses", handlers.NewAnthropicToOpenAIHandler(s.config))

	s.router.POST("/v1/responses", handlers.NewResponsesHandler(s.appRouter))
}

func (s *Server) Use(middleware ...gin.HandlerFunc) {
	s.router.Use(middleware...)
}

func (s *Server) Run(addr string) error {
	return s.router.Run(addr)
}
