package router

import (
	"fmt"
	"strings"

	"ai-proxy/config"
)

type Router interface {
	Resolve(modelName string) (*ResolvedRoute, error)
	GetProvider(name string) (config.Provider, bool)
	ListModels() []string
	ListProviders() []string
}

type ResolvedRoute struct {
	Provider          config.Provider
	Model             string
	OutputProtocol    string
	ToolCallTransform bool
}

type router struct {
	providers map[string]config.Provider
	models    map[string]config.ModelConfig
	fallback  config.FallbackConfig
}

func NewRouter(cfg *config.AppConfig) (Router, error) {
	if cfg == nil {
		return nil, fmt.Errorf("config is required")
	}

	providers := make(map[string]config.Provider)
	for _, p := range cfg.Providers {
		providers[p.Name] = p
	}

	return &router{
		providers: providers,
		models:    cfg.Models,
		fallback:  cfg.Fallback,
	}, nil
}

func (r *router) Resolve(modelName string) (*ResolvedRoute, error) {
	if modelName == "" {
		return nil, fmt.Errorf("model name is required")
	}

	if mc, ok := r.models[modelName]; ok {
		return r.buildRoute(mc.Provider, mc.Model, mc.ToolCallTransform)
	}

	if r.fallback.Enabled {
		targetModel := strings.ReplaceAll(r.fallback.Model, "{model}", modelName)
		return r.buildRoute(r.fallback.Provider, targetModel, r.fallback.ToolCallTransform)
	}

	return nil, fmt.Errorf("unknown model: %s", modelName)
}

func (r *router) buildRoute(providerName, model string, toolCallTransform bool) (*ResolvedRoute, error) {
	provider, ok := r.providers[providerName]
	if !ok {
		return nil, fmt.Errorf("unknown provider: %s", providerName)
	}

	return &ResolvedRoute{
		Provider:          provider,
		Model:             model,
		OutputProtocol:    provider.Type,
		ToolCallTransform: toolCallTransform,
	}, nil
}

func (r *router) GetProvider(name string) (config.Provider, bool) {
	p, ok := r.providers[name]
	return p, ok
}

func (r *router) ListModels() []string {
	models := make([]string, 0, len(r.models))
	for name := range r.models {
		models = append(models, name)
	}
	return models
}

func (r *router) ListProviders() []string {
	providers := make([]string, 0, len(r.providers))
	for name := range r.providers {
		providers = append(providers, name)
	}
	return providers
}
