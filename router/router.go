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
	// ResolveWithProtocol resolves a model name with incoming protocol context.
	// Used for "auto" type routing to enable passthrough optimization.
	ResolveWithProtocol(modelName, incomingProtocol string) (*ResolvedRoute, error)
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
	// OutputProtocol specifies the protocol to use: "openai", "anthropic", or "auto".
	OutputProtocol string
	// KimiToolCallTransform enables tool call transformation for this route.
	KimiToolCallTransform bool
	// GLM5ToolCallTransform enables GLM-5 XML tool call extraction for this route.
	GLM5ToolCallTransform bool
	// ReasoningSplit enables separate reasoning output for this route.
	ReasoningSplit bool
	// IsPassthrough indicates when no protocol transformation is needed.
	// True when incoming protocol matches output protocol (passthrough mode).
	IsPassthrough bool
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
//
// @pre modelName must not be empty
// @post returned OutputProtocol is determined by modelConfig.Type, fallback.Type, or provider's default
// @post IsPassthrough is false when returned (use ResolveWithProtocol for passthrough detection)
func (r *router) Resolve(modelName string) (*ResolvedRoute, error) {
	// Check for exact model match
	if modelConfig, ok := r.schema.Models[modelName]; ok {
		provider, ok := r.providersMap[modelConfig.Provider]
		if !ok {
			return nil, fmt.Errorf("provider '%s' not found for model '%s'", modelConfig.Provider, modelName)
		}

		// Determine output protocol
		outputProtocol := provider.GetDefaultProtocol() // default to provider's default
		if modelConfig.Type != "" {
			outputProtocol = modelConfig.Type
		}

		return &ResolvedRoute{
			Provider:              provider,
			Model:                 modelConfig.Model,
			OutputProtocol:        outputProtocol,
			KimiToolCallTransform: modelConfig.KimiToolCallTransform,
			GLM5ToolCallTransform: modelConfig.GLM5ToolCallTransform,
			ReasoningSplit:        modelConfig.ReasoningSplit,
			IsPassthrough:         false,
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

		// Determine output protocol for fallback
		outputProtocol := provider.GetDefaultProtocol() // default to provider's default
		if r.schema.Fallback.Type != "" {
			outputProtocol = r.schema.Fallback.Type
		}

		return &ResolvedRoute{
			Provider:              provider,
			Model:                 model,
			OutputProtocol:        outputProtocol,
			KimiToolCallTransform: r.schema.Fallback.KimiToolCallTransform,
			GLM5ToolCallTransform: r.schema.Fallback.GLM5ToolCallTransform,
			ReasoningSplit:        r.schema.Fallback.ReasoningSplit,
			IsPassthrough:         false,
		}, nil
	}

	// No match and no fallback
	return nil, fmt.Errorf("unknown model: '%s'", modelName)
}

// ResolveWithProtocol resolves a model name with incoming protocol context.
// This is used for "auto" type routing to enable passthrough optimization.
// When the model is configured with type "auto", it checks if the provider
// supports the incoming protocol and sets IsPassthrough accordingly.
//
// @pre modelName must not be empty
// @pre incomingProtocol should be "openai", "anthropic", or "responses"
// @post If OutputProtocol is "auto", it will be resolved to a concrete protocol
// @post IsPassthrough will be true when incoming protocol matches output protocol
func (r *router) ResolveWithProtocol(modelName, incomingProtocol string) (*ResolvedRoute, error) {
	// First get base route
	route, err := r.Resolve(modelName)
	if err != nil {
		return nil, err
	}

	// If not auto type, return as-is
	if route.OutputProtocol != "auto" {
		return route, nil
	}

	// Handle auto type - check if provider supports incoming protocol
	if route.Provider.HasProtocol(incomingProtocol) {
		route.OutputProtocol = incomingProtocol
		route.IsPassthrough = true
	} else {
		// Use provider's default protocol
		route.OutputProtocol = route.Provider.GetDefaultProtocol()
		route.IsPassthrough = false
	}

	return route, nil
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
