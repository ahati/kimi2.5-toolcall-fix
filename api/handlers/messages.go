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

// MessagesHandler handles native Anthropic messages API requests.
type MessagesHandler struct {
	router        router.Router
	resolvedRoute *router.ResolvedRoute
}

// NewMessagesHandler creates a Gin handler for the /v1/messages endpoint.
func NewMessagesHandler(r router.Router) gin.HandlerFunc {
	return Handle(&MessagesHandler{router: r})
}

// ValidateRequest performs no additional validation for messages requests.
func (h *MessagesHandler) ValidateRequest(body []byte) error {
	return nil
}

// TransformRequest extracts the model and resolves the route.
func (h *MessagesHandler) TransformRequest(body []byte) ([]byte, error) {
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

	if route.OutputProtocol == "openai" {
		converter := convert.NewChatToAnthropicConverter()
		return converter.Convert(body)
	}

	return body, nil
}

// UpstreamURL returns the resolved provider's base URL.
func (h *MessagesHandler) UpstreamURL() string {
	if h.resolvedRoute != nil && h.resolvedRoute.Provider != nil {
		return h.resolvedRoute.Provider.BaseURL
	}
	return ""
}

// ResolveAPIKey returns the resolved provider's API key.
func (h *MessagesHandler) ResolveAPIKey(c *gin.Context) string {
	if h.resolvedRoute != nil && h.resolvedRoute.Provider != nil {
		return h.resolvedRoute.Provider.GetAPIKey()
	}
	return ""
}

// ForwardHeaders copies relevant headers to the upstream request.
func (h *MessagesHandler) ForwardHeaders(c *gin.Context, req *http.Request) {
	for key, values := range c.Request.Header {
		if strings.HasPrefix(key, "X-") || key == "Anthropic-Version" || key == "Anthropic-Beta" {
			req.Header[key] = values
		}
	}
}

// CreateTransformer builds an SSE transformer for the response stream.
func (h *MessagesHandler) CreateTransformer(w io.Writer) transform.SSETransformer {
	if h.resolvedRoute != nil && h.resolvedRoute.ToolCallTransform {
		return toolcall.NewAnthropicTransformer(w)
	}
	return transform.NewPassthroughTransformer(w)
}

// WriteError sends an error response in Anthropic format.
func (h *MessagesHandler) WriteError(c *gin.Context, status int, msg string) {
	sendAnthropicError(c, status, msg)
}
