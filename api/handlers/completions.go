package handlers

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"ai-proxy/convert"
	"ai-proxy/router"
	"ai-proxy/transform"
	"ai-proxy/transform/toolcall"

	"github.com/gin-gonic/gin"
)

// CompletionsHandler handles OpenAI-compatible chat completion requests.
type CompletionsHandler struct {
	router        router.Router
	resolvedRoute *router.ResolvedRoute
}

// NewCompletionsHandler creates a Gin handler for the /v1/chat/completions endpoint.
func NewCompletionsHandler(r router.Router) gin.HandlerFunc {
	return Handle(&CompletionsHandler{router: r})
}

// ValidateRequest performs no additional validation for completions requests.
func (h *CompletionsHandler) ValidateRequest(body []byte) error {
	return nil
}

// TransformRequest extracts the model and resolves the route.
func (h *CompletionsHandler) TransformRequest(body []byte) ([]byte, error) {
	var req struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return body, nil
	}

	route, err := h.router.Resolve(req.Model)
	if err != nil {
		return body, nil
	}

	h.resolvedRoute = route

	if route.OutputProtocol == "anthropic" {
		converter := convert.NewChatToAnthropicConverter()
		return converter.Convert(body)
	}

	return body, nil
}

// UpstreamURL returns the resolved provider's base URL.
func (h *CompletionsHandler) UpstreamURL() string {
	if h.resolvedRoute != nil && h.resolvedRoute.Provider != nil {
		return h.resolvedRoute.Provider.BaseURL
	}
	return ""
}

// ResolveAPIKey returns the resolved provider's API key.
func (h *CompletionsHandler) ResolveAPIKey(c *gin.Context) string {
	if h.resolvedRoute != nil && h.resolvedRoute.Provider != nil {
		return h.resolvedRoute.Provider.GetAPIKey()
	}
	return ""
}

// ForwardHeaders copies custom headers to the upstream request.
func (h *CompletionsHandler) ForwardHeaders(c *gin.Context, req *http.Request) {
	for key, values := range c.Request.Header {
		if strings.HasPrefix(key, "X-") {
			req.Header[key] = values
		}
	}
	req.Header.Set("Extra", c.Request.Header.Get("Extra"))
}

// CreateTransformer builds an SSE transformer for the response stream.
func (h *CompletionsHandler) CreateTransformer(w io.Writer) transform.SSETransformer {
	if h.resolvedRoute != nil && h.resolvedRoute.ToolCallTransform {
		return toolcall.NewOpenAITransformer(w)
	}
	return transform.NewPassthroughTransformer(w)
}

// WriteError sends an error response in OpenAI format.
func (h *CompletionsHandler) WriteError(c *gin.Context, status int, msg string) {
	sendOpenAIError(c, status, msg)
}
