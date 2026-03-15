package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"ai-proxy/config"
	"ai-proxy/convert"
	"ai-proxy/router"
	"ai-proxy/transform"
	"ai-proxy/transform/toolcall"
	"ai-proxy/types"

	"github.com/gin-gonic/gin"
)

// ResponsesHandler handles OpenAI Responses API requests.
// It can route to either OpenAI or Anthropic upstream based on model configuration.
//
// This handler:
//   - Accepts requests in OpenAI Responses API format
//   - Routes to the appropriate upstream based on model configuration
//   - For OpenAI upstream: converts to Chat Completions format
//   - For Anthropic upstream: converts to Anthropic Messages format
//   - Transforms responses back to OpenAI Responses API format
//
// @note This enables clients using OpenAI Responses API to call multiple providers.
type ResponsesHandler struct {
	// cfg contains the application configuration including providers and models.
	// Must not be nil after construction.
	cfg *config.Config
	// router resolves model names to providers and routes.
	// Must not be nil after construction.
	router router.Router
	// route is the resolved route for the current request.
	// Set during ValidateRequest for use in subsequent methods.
	route *router.ResolvedRoute
	// originalModel is the model name from the original request.
	// Preserved for response transformation.
	originalModel string
}

// NewResponsesHandler creates a Gin handler for the /v1/responses endpoint.
//
// @param cfg - Application configuration. Must not be nil.
// @param r - Router for model resolution. Must not be nil.
// @return Gin handler function that processes responses requests.
//
// @pre cfg != nil
// @pre r != nil
func NewResponsesHandler(cfg *config.Config, r router.Router) gin.HandlerFunc {
	return Handle(&ResponsesHandler{
		cfg:    cfg,
		router: r,
	})
}

// ValidateRequest validates the Responses API request and resolves the route.
// It parses the request to extract the model name and resolves it to a provider.
//
// @param body - Raw request body bytes.
// @return Error if JSON parsing fails, model is missing, or route cannot be resolved.
func (h *ResponsesHandler) ValidateRequest(body []byte) error {
	var req types.ResponsesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return fmt.Errorf("invalid JSON: %w", err)
	}

	if req.Model == "" {
		return fmt.Errorf("model is required")
	}

	// Resolve the model to a route
	route, err := h.router.Resolve(req.Model)
	if err != nil {
		return fmt.Errorf("failed to resolve model '%s': %w", req.Model, err)
	}

	// Store the resolved route for use in other methods
	h.route = route
	h.originalModel = req.Model

	return nil
}

// TransformRequest converts the request body based on the upstream provider type.
// For OpenAI providers, it converts to Chat Completions format.
// For Anthropic providers, it converts to Anthropic Messages format.
//
// @param body - Raw request body in OpenAI Responses API format.
// @return Transformed body in the appropriate upstream format.
// @return Error if transformation fails.
func (h *ResponsesHandler) TransformRequest(body []byte) ([]byte, error) {
	if h.route == nil {
		return nil, fmt.Errorf("route not resolved")
	}

	// Update model in request to the resolved model
	var req map[string]interface{}
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, fmt.Errorf("failed to parse request: %w", err)
	}
	req["model"] = h.route.Model
	updatedBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal updated request: %w", err)
	}

	switch h.route.Provider.Type {
	case "openai":
		// Convert ResponsesRequest to ChatCompletionRequest
		converter := convert.NewResponsesToChatConverter()
		return converter.Convert(updatedBody)
	case "anthropic":
		// Convert ResponsesRequest to Anthropic MessageRequest
		return convert.TransformResponsesToAnthropic(updatedBody)
	default:
		// Unknown provider type - pass through as-is
		return updatedBody, nil
	}
}

// UpstreamURL returns the upstream API URL based on the resolved provider.
//
// @return URL string for the upstream API endpoint.
func (h *ResponsesHandler) UpstreamURL() string {
	if h.route == nil {
		return ""
	}

	// For OpenAI providers, append the chat completions path if needed
	if h.route.Provider.Type == "openai" {
		baseURL := h.route.Provider.BaseURL
		// If the base URL doesn't already end with /chat/completions, append it
		if !strings.HasSuffix(baseURL, "/chat/completions") {
			baseURL = strings.TrimSuffix(baseURL, "/") + "/chat/completions"
		}
		return baseURL
	}

	// For Anthropic providers, use the base URL directly
	return h.route.Provider.BaseURL
}

// ResolveAPIKey returns the API key for the resolved provider.
//
// @param c - Gin context (unused in this implementation).
// @return API key string from the provider configuration.
func (h *ResponsesHandler) ResolveAPIKey(c *gin.Context) string {
	if h.route == nil {
		return ""
	}
	return h.route.Provider.GetAPIKey()
}

// ForwardHeaders copies relevant headers to the upstream request.
// For OpenAI providers, it forwards X-* headers.
// For Anthropic providers, it also forwards Anthropic-specific headers.
//
// @param c - Gin context containing the original request headers.
// @param req - Upstream request to receive forwarded headers.
func (h *ResponsesHandler) ForwardHeaders(c *gin.Context, req *http.Request) {
	if h.route == nil {
		return
	}

	switch h.route.Provider.Type {
	case "openai":
		// Forward custom headers and Extra header
		forwardCustomHeaders(c, req, "X-")
		req.Header.Set("Extra", c.Request.Header.Get("Extra"))
	case "anthropic":
		// Forward Anthropic-specific headers
		for k, v := range c.Request.Header {
			if strings.HasPrefix(k, "X-") || k == "Anthropic-Version" || k == "Anthropic-Beta" {
				req.Header[k] = v
			}
		}
	default:
		// Forward X-* headers by default
		forwardCustomHeaders(c, req, "X-")
	}
}

// CreateTransformer builds an SSE transformer for converting upstream responses.
// For OpenAI providers, it converts Chat Completions to Responses API format.
// For Anthropic providers, it converts Anthropic events to Responses API format.
//
// @param w - Writer to receive transformed output.
// @return Transformer for processing SSE events.
func (h *ResponsesHandler) CreateTransformer(w io.Writer) transform.SSETransformer {
	if h.route == nil {
		return transform.NewPassthroughTransformer(w)
	}

	switch h.route.Provider.Type {
	case "openai":
		return convert.NewChatToResponsesTransformer(w)
	case "anthropic":
		return toolcall.NewResponsesTransformer(w)
	default:
		return transform.NewPassthroughTransformer(w)
	}
}

// WriteError sends an error response in OpenAI Responses API format.
//
// @param c - Gin context for writing the response.
// @param status - HTTP status code for the error.
// @param msg - Human-readable error message.
func (h *ResponsesHandler) WriteError(c *gin.Context, status int, msg string) {
	sendOpenAIResponsesError(c, status, msg)
}

// ResponsesHandlerNoRouter handles OpenAI Responses API requests without using a router.
// It is used for simple configurations where model routing is not needed.
//
// @note This handler uses the first OpenAI or Anthropic provider from the configuration.
type ResponsesHandlerNoRouter struct {
	cfg *config.Config
}

// NewResponsesHandlerNoRouter creates a Gin handler for the /v1/responses endpoint
// without using a router. It uses the first configured provider.
//
// @param cfg - Application configuration. Must not be nil.
// @return Gin handler function that processes responses requests.
//
// @deprecated Use NewResponsesHandler with a router for proper model routing.
func NewResponsesHandlerNoRouter(cfg *config.Config) gin.HandlerFunc {
	return Handle(&ResponsesHandlerNoRouter{cfg: cfg})
}

// ValidateRequest validates the Responses API request.
func (h *ResponsesHandlerNoRouter) ValidateRequest(body []byte) error {
	var req types.ResponsesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return fmt.Errorf("invalid JSON: %w", err)
	}
	if req.Model == "" {
		return fmt.Errorf("model is required")
	}
	return nil
}

// TransformRequest converts the request based on available providers.
func (h *ResponsesHandlerNoRouter) TransformRequest(body []byte) ([]byte, error) {
	// Check for Anthropic provider first (for backwards compatibility)
	if h.cfg.GetAnthropicUpstreamURL() != "" {
		return convert.TransformResponsesToAnthropic(body)
	}
	// Fall back to OpenAI provider
	if h.cfg.GetOpenAIUpstreamURL() != "" {
		converter := convert.NewResponsesToChatConverter()
		return converter.Convert(body)
	}
	return body, nil
}

// UpstreamURL returns the upstream URL based on available providers.
func (h *ResponsesHandlerNoRouter) UpstreamURL() string {
	// Check for Anthropic provider first
	anthropicURL := h.cfg.GetAnthropicUpstreamURL()
	if anthropicURL != "" {
		return anthropicURL
	}
	// Fall back to OpenAI provider
	openaiURL := h.cfg.GetOpenAIUpstreamURL()
	if openaiURL != "" {
		// Append chat completions path if needed
		if !strings.HasSuffix(openaiURL, "/chat/completions") {
			return strings.TrimSuffix(openaiURL, "/") + "/chat/completions"
		}
		return openaiURL
	}
	return ""
}

// ResolveAPIKey returns the API key for the appropriate provider.
func (h *ResponsesHandlerNoRouter) ResolveAPIKey(c *gin.Context) string {
	// Check for Anthropic provider first
	if h.cfg.GetAnthropicUpstreamURL() != "" {
		return h.cfg.GetAnthropicAPIKey()
	}
	// Fall back to OpenAI provider
	return h.cfg.GetOpenAIUpstreamAPIKey()
}

// ForwardHeaders copies headers based on the provider type.
func (h *ResponsesHandlerNoRouter) ForwardHeaders(c *gin.Context, req *http.Request) {
	// Check for Anthropic provider first
	if h.cfg.GetAnthropicUpstreamURL() != "" {
		for k, v := range c.Request.Header {
			if strings.HasPrefix(k, "X-") || k == "Anthropic-Version" || k == "Anthropic-Beta" {
				req.Header[k] = v
			}
		}
		return
	}
	// Fall back to OpenAI provider headers
	forwardCustomHeaders(c, req, "X-")
	req.Header.Set("Extra", c.Request.Header.Get("Extra"))
}

// CreateTransformer builds an SSE transformer based on the provider type.
func (h *ResponsesHandlerNoRouter) CreateTransformer(w io.Writer) transform.SSETransformer {
	if h.cfg.GetAnthropicUpstreamURL() != "" {
		return toolcall.NewResponsesTransformer(w)
	}
	return convert.NewChatToResponsesTransformer(w)
}

// WriteError sends an error response in OpenAI Responses API format.
func (h *ResponsesHandlerNoRouter) WriteError(c *gin.Context, status int, msg string) {
	sendOpenAIResponsesError(c, status, msg)
}
