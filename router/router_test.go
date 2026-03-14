package router

import (
	"testing"

	"ai-proxy/config"
)

func TestNewRouter_NilSchema(t *testing.T) {
	_, err := NewRouter(nil)
	if err == nil {
		t.Error("expected error for nil schema")
	}
}

func TestNewRouter_ValidSchema(t *testing.T) {
	schema := &config.Schema{
		Providers: []config.Provider{
			{Name: "openai", Type: "openai", BaseURL: "https://api.openai.com"},
		},
	}
	_, err := NewRouter(schema)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestResolve_ExactModelMatch(t *testing.T) {
	schema := &config.Schema{
		Providers: []config.Provider{
			{Name: "openai", Type: "openai", BaseURL: "https://api.openai.com"},
			{Name: "anthropic", Type: "anthropic", BaseURL: "https://api.anthropic.com"},
		},
		Models: map[string]config.ModelConfig{
			"gpt-4": {
				Provider:          "openai",
				Model:             "gpt-4-turbo",
				ToolCallTransform: false,
			},
			"claude": {
				Provider:          "anthropic",
				Model:             "claude-3-opus",
				ToolCallTransform: true,
			},
		},
	}

	r, err := NewRouter(schema)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Test exact match for gpt-4
	route, err := r.Resolve("gpt-4")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if route.Provider.Name != "openai" {
		t.Errorf("expected provider 'openai', got '%s'", route.Provider.Name)
	}
	if route.Model != "gpt-4-turbo" {
		t.Errorf("expected model 'gpt-4-turbo', got '%s'", route.Model)
	}
	if route.OutputProtocol != "openai" {
		t.Errorf("expected output protocol 'openai', got '%s'", route.OutputProtocol)
	}
	if route.ToolCallTransform != false {
		t.Error("expected ToolCallTransform to be false")
	}

	// Test exact match for claude
	route, err = r.Resolve("claude")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if route.Provider.Name != "anthropic" {
		t.Errorf("expected provider 'anthropic', got '%s'", route.Provider.Name)
	}
	if route.Model != "claude-3-opus" {
		t.Errorf("expected model 'claude-3-opus', got '%s'", route.Model)
	}
	if route.OutputProtocol != "anthropic" {
		t.Errorf("expected output protocol 'anthropic', got '%s'", route.OutputProtocol)
	}
	if route.ToolCallTransform != true {
		t.Error("expected ToolCallTransform to be true")
	}
}

func TestResolve_FallbackWithPlaceholder(t *testing.T) {
	schema := &config.Schema{
		Providers: []config.Provider{
			{Name: "openai", Type: "openai", BaseURL: "https://api.openai.com"},
			{Name: "fallback-provider", Type: "openai", BaseURL: "https://fallback.example.com"},
		},
		Models: map[string]config.ModelConfig{
			"known-model": {
				Provider: "openai",
				Model:    "gpt-4",
			},
		},
		Fallback: config.FallbackConfig{
			Enabled:           true,
			Provider:          "fallback-provider",
			Model:             "prefix-{model}-suffix",
			ToolCallTransform: true,
		},
	}

	r, err := NewRouter(schema)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Test fallback with unknown model
	route, err := r.Resolve("unknown-model")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if route.Provider.Name != "fallback-provider" {
		t.Errorf("expected provider 'fallback-provider', got '%s'", route.Provider.Name)
	}
	if route.Model != "prefix-unknown-model-suffix" {
		t.Errorf("expected model 'prefix-unknown-model-suffix', got '%s'", route.Model)
	}
	if route.ToolCallTransform != true {
		t.Error("expected ToolCallTransform to be true")
	}
}

func TestResolve_FallbackDisabled(t *testing.T) {
	schema := &config.Schema{
		Providers: []config.Provider{
			{Name: "openai", Type: "openai", BaseURL: "https://api.openai.com"},
		},
		Models: map[string]config.ModelConfig{
			"known-model": {
				Provider: "openai",
				Model:    "gpt-4",
			},
		},
		Fallback: config.FallbackConfig{
			Enabled:  false,
			Provider: "openai",
			Model:    "fallback-model",
		},
	}

	r, err := NewRouter(schema)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Test unknown model with fallback disabled
	_, err = r.Resolve("unknown-model")
	if err == nil {
		t.Error("expected error for unknown model")
	}
	if err.Error() != "unknown model: 'unknown-model'" {
		t.Errorf("expected error 'unknown model: 'unknown-model'', got '%s'", err.Error())
	}
}

func TestResolve_UnknownModelError(t *testing.T) {
	schema := &config.Schema{
		Providers: []config.Provider{
			{Name: "openai", Type: "openai", BaseURL: "https://api.openai.com"},
		},
		Models: map[string]config.ModelConfig{},
		Fallback: config.FallbackConfig{
			Enabled: false,
		},
	}

	r, err := NewRouter(schema)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	_, err = r.Resolve("nonexistent-model")
	if err == nil {
		t.Error("expected error for unknown model")
	}
	if err.Error() != "unknown model: 'nonexistent-model'" {
		t.Errorf("expected error 'unknown model: 'nonexistent-model'', got '%s'", err.Error())
	}
}

func TestResolve_MissingProviderError(t *testing.T) {
	schema := &config.Schema{
		Providers: []config.Provider{
			{Name: "openai", Type: "openai", BaseURL: "https://api.openai.com"},
		},
		Models: map[string]config.ModelConfig{
			"test-model": {
				Provider: "nonexistent-provider",
				Model:    "some-model",
			},
		},
		Fallback: config.FallbackConfig{
			Enabled: false,
		},
	}

	r, err := NewRouter(schema)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	_, err = r.Resolve("test-model")
	if err == nil {
		t.Error("expected error for missing provider")
	}
	expectedErr := "provider 'nonexistent-provider' not found for model 'test-model'"
	if err.Error() != expectedErr {
		t.Errorf("expected error '%s', got '%s'", expectedErr, err.Error())
	}
}

func TestResolve_MissingFallbackProviderError(t *testing.T) {
	schema := &config.Schema{
		Providers: []config.Provider{
			{Name: "openai", Type: "openai", BaseURL: "https://api.openai.com"},
		},
		Models: map[string]config.ModelConfig{},
		Fallback: config.FallbackConfig{
			Enabled:  true,
			Provider: "nonexistent-fallback-provider",
			Model:    "{model}",
		},
	}

	r, err := NewRouter(schema)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	_, err = r.Resolve("unknown-model")
	if err == nil {
		t.Error("expected error for missing fallback provider")
	}
	expectedErr := "provider 'nonexistent-fallback-provider' not found for fallback"
	if err.Error() != expectedErr {
		t.Errorf("expected error '%s', got '%s'", expectedErr, err.Error())
	}
}

func TestGetProvider_Found(t *testing.T) {
	schema := &config.Schema{
		Providers: []config.Provider{
			{Name: "openai", Type: "openai", BaseURL: "https://api.openai.com"},
			{Name: "anthropic", Type: "anthropic", BaseURL: "https://api.anthropic.com"},
		},
		Models: map[string]config.ModelConfig{},
	}

	r, err := NewRouter(schema)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	provider, ok := r.GetProvider("openai")
	if !ok {
		t.Error("expected to find provider 'openai'")
	}
	if provider.Name != "openai" {
		t.Errorf("expected provider name 'openai', got '%s'", provider.Name)
	}
	if provider.Type != "openai" {
		t.Errorf("expected provider type 'openai', got '%s'", provider.Type)
	}
}

func TestGetProvider_NotFound(t *testing.T) {
	schema := &config.Schema{
		Providers: []config.Provider{
			{Name: "openai", Type: "openai", BaseURL: "https://api.openai.com"},
		},
		Models: map[string]config.ModelConfig{},
	}

	r, err := NewRouter(schema)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	_, ok := r.GetProvider("nonexistent")
	if ok {
		t.Error("expected not to find provider 'nonexistent'")
	}
}

func TestListModels(t *testing.T) {
	schema := &config.Schema{
		Providers: []config.Provider{
			{Name: "openai", Type: "openai", BaseURL: "https://api.openai.com"},
		},
		Models: map[string]config.ModelConfig{
			"gpt-4":    {Provider: "openai", Model: "gpt-4-turbo"},
			"gpt-3.5":  {Provider: "openai", Model: "gpt-3.5-turbo"},
			"claude":   {Provider: "openai", Model: "claude-3"},
		},
	}

	r, err := NewRouter(schema)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	models := r.ListModels()
	if len(models) != 3 {
		t.Errorf("expected 3 models, got %d", len(models))
	}

	// Check that all expected models are present
	modelSet := make(map[string]bool)
	for _, m := range models {
		modelSet[m] = true
	}
	for _, expected := range []string{"gpt-4", "gpt-3.5", "claude"} {
		if !modelSet[expected] {
			t.Errorf("expected model '%s' in list", expected)
		}
	}
}

func TestListModels_Empty(t *testing.T) {
	schema := &config.Schema{
		Providers: []config.Provider{},
		Models:    map[string]config.ModelConfig{},
	}

	r, err := NewRouter(schema)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	models := r.ListModels()
	if len(models) != 0 {
		t.Errorf("expected 0 models, got %d", len(models))
	}
}

func TestResolve_FallbackWithModelOnlyPlaceholder(t *testing.T) {
	schema := &config.Schema{
		Providers: []config.Provider{
			{Name: "fallback-provider", Type: "openai", BaseURL: "https://fallback.example.com"},
		},
		Models: map[string]config.ModelConfig{},
		Fallback: config.FallbackConfig{
			Enabled:  true,
			Provider: "fallback-provider",
			Model:    "{model}",
		},
	}

	r, err := NewRouter(schema)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	route, err := r.Resolve("some-random-model")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if route.Model != "some-random-model" {
		t.Errorf("expected model 'some-random-model', got '%s'", route.Model)
	}
}