package router

import (
	"errors"
	"strings"

	"ai-proxy/config"
)

var (
	ErrModelNotFound   = errors.New("model not found")
	ErrProviderNotFound = errors.New("provider not found")
)

type Router interface {
	Resolve(modelName string) (*ResolvedRoute, error)
	GetProvider(name string) (*config.Provider, bool)
	ListModels() []string
}

type ResolvedRoute struct {
	Provider          *config.Provider
	Model             string
	OutputProtocol    string
	ToolCallTransform bool
}

type router struct {
	config *config.SchemaConfig
}

func NewRouter(cfg *config.SchemaConfig) (Router, error) {
	if cfg == nil {
		return nil, errors.New("config is nil")
	}
	return &router{config: cfg}, nil
}

func (r *router) Resolve(modelName string) (*ResolvedRoute, error) {
	if modelConfig, ok := r.config.Models[modelName]; ok {
		provider, found := r.findProvider(modelConfig.Provider)
		if !found {
			return nil, ErrProviderNotFound
		}
		return &ResolvedRoute{
			Provider:          provider,
			Model:             modelConfig.Model,
			OutputProtocol:    r.determineProtocol(provider.Type),
			ToolCallTransform: modelConfig.ToolCallTransform,
		}, nil
	}

	if r.config.Fallback.Enabled {
		provider, found := r.findProvider(r.config.Fallback.Provider)
		if !found {
			return nil, ErrProviderNotFound
		}
		model := strings.ReplaceAll(r.config.Fallback.Model, "{model}", modelName)
		return &ResolvedRoute{
			Provider:          provider,
			Model:             model,
			OutputProtocol:    r.determineProtocol(provider.Type),
			ToolCallTransform: r.config.Fallback.ToolCallTransform,
		}, nil
	}

	return nil, ErrModelNotFound
}

func (r *router) GetProvider(name string) (*config.Provider, bool) {
	return r.findProvider(name)
}

func (r *router) ListModels() []string {
	models := make([]string, 0, len(r.config.Models))
	for name := range r.config.Models {
		models = append(models, name)
	}
	return models
}

func (r *router) findProvider(name string) (*config.Provider, bool) {
	for i := range r.config.Providers {
		if r.config.Providers[i].Name == name {
			return &r.config.Providers[i], true
		}
	}
	return nil, false
}

func (r *router) determineProtocol(providerType string) string {
	if providerType == "anthropic" {
		return "anthropic"
	}
	return "openai"
}
