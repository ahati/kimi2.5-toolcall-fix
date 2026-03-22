package handlers

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"ai-proxy/config"
	"ai-proxy/convert"
	"ai-proxy/router"
	"ai-proxy/transform"
	"ai-proxy/transform/toolcall"

	"github.com/gin-gonic/gin"
)

// CompletionsHandler handles OpenAI-compatible chat completion requests.
// It implements the Handler interface for the /v1/chat/completions endpoint.
//
// This handler:
//   - Accepts requests in OpenAI ChatCompletion format
//   - Routes to the appropriate upstream based on model configuration
//   - For OpenAI providers: passes through requests without transformation
//   - For Anthropic providers: converts OpenAI Chat→Anthropic Messages, transforms responses back
//   - Supports streaming responses with tool call handling
//
// @note This enables clients using OpenAI SDK to call any provider.
type CompletionsHandler struct {
	// cfg contains the application configuration including upstream URL and API key.
	// Must not be nil after construction.
	cfg *config.Config
	// router resolves model names to providers. May be nil for legacy behavior.
	modelRouter router.Router
	// route is the resolved route for the current request.
	// Set during ValidateRequest for use in subsequent methods.
	route *router.ResolvedRoute
	// originalModel is the model name from the original request.
	originalModel string
}

// NewCompletionsHandler creates a Gin handler for the /v1/chat/completions endpoint.
//
// @param cfg - Application configuration. Must not be nil.
// @param r - Router for model resolution. May be nil for legacy behavior.
// @return Gin handler function that processes completion requests.
//
// @pre cfg != nil
func NewCompletionsHandler(cfg *config.Config, r router.Router) gin.HandlerFunc {
	return func(c *gin.Context) {
		h := &CompletionsHandler{
			cfg:         cfg,
			modelRouter: r,
		}
		Handle(h)(c)
	}
}

// ValidateRequest validates the request and resolves the model route.
// It parses the request to extract the model name and resolves it to a provider.
//
// @param body - Raw request body bytes.
// @return Error if JSON parsing fails or model cannot be resolved.
func (h *CompletionsHandler) ValidateRequest(body []byte) error {
	// If no router, use legacy behavior
	if h.modelRouter == nil {
		return nil
	}

	// Parse to get model name
	var req struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return nil // Let upstream handle invalid JSON
	}

	if req.Model == "" {
		return nil // Let upstream handle missing model
	}

	// Resolve the model to a route
	route, err := h.modelRouter.Resolve(req.Model)
	if err != nil {
		return nil // Use fallback behavior
	}

	h.route = route
	h.originalModel = req.Model
	return nil
}

// TransformRequest converts the request body based on the upstream provider type.
// For OpenAI providers: passes through without transformation, adding stream_options.
// For Anthropic providers: converts OpenAI Chat Completions to Anthropic Messages.
//
// @param body - Raw request body in OpenAI ChatCompletion format.
// @return Transformed body in the appropriate upstream format.
// @return Error if transformation fails.
func (h *CompletionsHandler) TransformRequest(body []byte) ([]byte, error) {
	// If no route resolved, pass through (legacy behavior)
	if h.route == nil {
		return body, nil
	}

	// Update model in request to the resolved model
	var req map[string]interface{}
	if err := json.Unmarshal(body, &req); err != nil {
		return body, nil
	}
	req["model"] = h.route.Model

	switch h.route.Provider.Type {
	case "anthropic":
		// Convert OpenAI ChatCompletionRequest to Anthropic MessageRequest
		updatedBody, err := json.Marshal(req)
		if err != nil {
			return nil, err
		}
		converter := convert.NewChatToAnthropicConverter()
		return converter.Convert(updatedBody)
	case "openai":
		// Add stream_options.include_usage for usage statistics in streaming
		if stream, ok := req["stream"].(bool); !ok || stream {
			req["stream"] = true
			req["stream_options"] = map[string]interface{}{
				"include_usage": true,
			}
		}
		return json.Marshal(req)
	default:
		// Unknown provider type - pass through as-is
		return json.Marshal(req)
	}
}

// UpstreamURL returns the upstream API URL based on the resolved provider.
//
// @return URL string for the upstream API endpoint.
func (h *CompletionsHandler) UpstreamURL() string {
	if h.route != nil {
		// Select endpoint based on provider type
		endpoint := "/chat/completions"
		if h.route.Provider.Type == "anthropic" {
			endpoint = "/v1/messages"
		}
		return h.route.Provider.GetUpstreamURL(endpoint)
	}
	// Legacy behavior - use OpenAI upstream
	return h.cfg.GetOpenAIUpstreamURL()
}

// ResolveAPIKey returns the API key for the resolved provider.
//
// @param c - Gin context (unused in this implementation).
// @return API key string from the provider configuration.
func (h *CompletionsHandler) ResolveAPIKey(c *gin.Context) string {
	if h.route != nil {
		return h.route.Provider.GetAPIKey()
	}
	// Legacy behavior
	return h.cfg.GetOpenAIUpstreamAPIKey()
}

// ForwardHeaders copies headers to the upstream request based on provider type.
// For OpenAI providers: forwards X-* headers and Extra header.
// For Anthropic providers: forwards X-*, Anthropic-Version, and Anthropic-Beta headers.
//
// @param c - Gin context containing the original request headers.
// @param req - Upstream request to receive forwarded headers.
func (h *CompletionsHandler) ForwardHeaders(c *gin.Context, req *http.Request) {
	providerType := "openai" // default
	if h.route != nil {
		providerType = h.route.Provider.Type
	}

	switch providerType {
	case "anthropic":
		// Forward headers that are important for Anthropic API
		for k, v := range c.Request.Header {
			if strings.HasPrefix(k, "X-") || k == "Anthropic-Version" || k == "Anthropic-Beta" {
				req.Header[k] = v
			}
		}
	case "openai":
		// Forward custom headers and Extra header
		forwardCustomHeaders(c, req, "X-")
		req.Header.Set("Extra", c.Request.Header.Get("Extra"))
	default:
		// Forward X-* headers by default
		forwardCustomHeaders(c, req, "X-")
	}
}

// CreateTransformer builds an SSE transformer based on the provider type.
// For OpenAI providers: uses OpenAITransformer for tool call handling.
// For Anthropic providers: uses ChatToAnthropicTransformer to convert responses back to OpenAI format.
//
// @param w - Writer to receive transformed output.
// @return Transformer for processing SSE events.
func (h *CompletionsHandler) CreateTransformer(w io.Writer) transform.SSETransformer {
	if h.route == nil {
		// Legacy behavior - use OpenAI transformer
		return toolcall.NewOpenAITransformer(w)
	}

	switch h.route.Provider.Type {
	case "anthropic":
		// Convert Anthropic responses back to OpenAI Chat format
		return convert.NewChatToAnthropicTransformer(w)
	case "openai":
		// Use OpenAI transformer for tool call handling
		if h.route.ToolCallTransform {
			return toolcall.NewOpenAITransformer(w)
		}
		return transform.NewPassthroughTransformer(w)
	default:
		return transform.NewPassthroughTransformer(w)
	}
}

// WriteError sends an error response in OpenAI format.
// Maintains consistency with OpenAI API error responses.
//
// @param c - Gin context for writing the response.
// @param status - HTTP status code for the error.
// @param msg - Human-readable error message.
func (h *CompletionsHandler) WriteError(c *gin.Context, status int, msg string) {
	sendOpenAIError(c, status, msg)
}

// forwardCustomHeaders copies headers matching any of the given prefixes
// from the incoming request to the upstream request.
func forwardCustomHeaders(c *gin.Context, req *http.Request, prefixes ...string) {
	for key, values := range c.Request.Header {
		for _, prefix := range prefixes {
			if strings.HasPrefix(key, prefix) {
				req.Header[key] = values
				break
			}
		}
	}
}

// ModelInfo returns the downstream and upstream model names for logging.
func (h *CompletionsHandler) ModelInfo() (downstreamModel string, upstreamModel string) {
	downstreamModel = h.originalModel
	if h.route != nil {
		upstreamModel = h.route.Model
	}
	return
}
