// Package router provides model resolution with fallback support.
// It implements routing logic to map model names to providers.
package router

import (
	"fmt"
	"strings"

	"ai-proxy/config"
)

// Router defines the interface for model resolution.
type Router interface {
	// Resolve resolves a model name to a route with provider information.
	Resolve(modelName string) (*ResolvedRoute, error)
	// GetProvider retrieves a provider by name.
	GetProvider(name string) (config.Provider, bool)
	// ListModels returns all configured model names.
	ListModels() []string
}

// ResolvedRoute contains the resolved routing information for a model.
type ResolvedRoute struct {
	// Provider is the upstream provider configuration.
	Provider config.Provider
	// Model is the actual model identifier on the upstream provider.
	Model string
	// OutputProtocol specifies the protocol to use: "openai" or "anthropic".
	OutputProtocol string
	// ToolCallTransform enables tool call transformation for this route.
	ToolCallTransform bool
}

// router implements the Router interface.
type router struct {
	schema *config.Schema
	// providersMap is a lookup map from provider name to Provider.
	providersMap map[string]config.Provider
}

// NewRouter creates a new Router from the given schema.
// Returns an error if the schema is nil.
func NewRouter(s *config.Schema) (Router, error) {
	if s == nil {
		return nil, fmt.Errorf("schema cannot be nil")
	}

	// Build provider lookup map
	providersMap := make(map[string]config.Provider)
	for _, p := range s.Providers {
		providersMap[p.Name] = p
	}

	return &router{
		schema:       s,
		providersMap: providersMap,
	}, nil
}

// Resolve resolves a model name to a route with provider information.
// It first checks for an exact model match in the schema.
// If not found and fallback is enabled, it uses the fallback configuration.
// Returns an error if the model is unknown and no fallback is available.
func (r *router) Resolve(modelName string) (*ResolvedRoute, error) {
	// Check for exact model match
	if modelConfig, ok := r.schema.Models[modelName]; ok {
		provider, ok := r.providersMap[modelConfig.Provider]
		if !ok {
			return nil, fmt.Errorf("provider '%s' not found for model '%s'", modelConfig.Provider, modelName)
		}
		return &ResolvedRoute{
			Provider:          provider,
			Model:             modelConfig.Model,
			OutputProtocol:    provider.Type,
			ToolCallTransform: modelConfig.ToolCallTransform,
		}, nil
	}

	// Model not found, check fallback
	if r.schema.Fallback.Enabled {
		provider, ok := r.providersMap[r.schema.Fallback.Provider]
		if !ok {
			return nil, fmt.Errorf("provider '%s' not found for fallback", r.schema.Fallback.Provider)
		}

		// Replace {model} placeholder with the requested model name
		model := r.schema.Fallback.Model
		model = strings.ReplaceAll(model, "{model}", modelName)

		return &ResolvedRoute{
			Provider:          provider,
			Model:             model,
			OutputProtocol:    provider.Type,
			ToolCallTransform: r.schema.Fallback.ToolCallTransform,
		}, nil
	}

	// No match and no fallback
	return nil, fmt.Errorf("unknown model: '%s'", modelName)
}

// GetProvider retrieves a provider by name.
// Returns the provider and true if found, or an empty provider and false if not.
func (r *router) GetProvider(name string) (config.Provider, bool) {
	provider, ok := r.providersMap[name]
	return provider, ok
}

// ListModels returns all configured model names.
func (r *router) ListModels() []string {
	models := make([]string, 0, len(r.schema.Models))
	for name := range r.schema.Models {
		models = append(models, name)
	}
	return models
}
