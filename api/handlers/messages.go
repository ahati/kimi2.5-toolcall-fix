package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"ai-proxy/router"
	"ai-proxy/transform"
	"ai-proxy/transform/toolcall"

	"github.com/gin-gonic/gin"
)

type MessagesHandler struct{}

func NewMessagesHandler(r router.Router) gin.HandlerFunc {
	return HandleWithRouter(&MessagesHandler{}, r)
}

func (h *MessagesHandler) ValidateRequest(body []byte) error {
	return nil
}

func (h *MessagesHandler) ExtractModel(body []byte) (string, error) {
	var req struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return "", fmt.Errorf("invalid JSON: %w", err)
	}
	return req.Model, nil
}

func (h *MessagesHandler) TransformRequest(body []byte) ([]byte, error) {
	return body, nil
}

func (h *MessagesHandler) TransformRequestWithRoute(body []byte, route *router.ResolvedRoute) ([]byte, error) {
	var req map[string]interface{}
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, fmt.Errorf("failed to parse request: %w", err)
	}
	req["model"] = route.Model
	updatedBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal updated request: %w", err)
	}

	switch route.Provider.Type {
	case "anthropic":
		return updatedBody, nil
	case "openai":
		return transformAnthropicToOpenAI(updatedBody)
	default:
		return updatedBody, nil
	}
}

func (h *MessagesHandler) UpstreamURL() string {
	return ""
}

func (h *MessagesHandler) UpstreamURLWithRoute(route *router.ResolvedRoute) string {
	switch route.Provider.Type {
	case "anthropic":
		return route.Provider.BaseURL
	case "openai":
		baseURL := route.Provider.BaseURL
		if !strings.HasSuffix(baseURL, "/chat/completions") {
			baseURL = strings.TrimSuffix(baseURL, "/") + "/chat/completions"
		}
		return baseURL
	default:
		return route.Provider.BaseURL
	}
}

func (h *MessagesHandler) ResolveAPIKey(c *gin.Context) string {
	return ""
}

func (h *MessagesHandler) ResolveAPIKeyWithRoute(route *router.ResolvedRoute) string {
	return route.Provider.GetAPIKey()
}

func (h *MessagesHandler) ForwardHeaders(c *gin.Context, req *http.Request) {
}

func (h *MessagesHandler) ForwardHeadersWithRoute(c *gin.Context, req *http.Request, route *router.ResolvedRoute) {
	switch route.Provider.Type {
	case "anthropic":
		for k, v := range c.Request.Header {
			if strings.HasPrefix(k, "X-") || k == "Anthropic-Version" || k == "Anthropic-Beta" {
				req.Header[k] = v
			}
		}
	case "openai":
		forwardCustomHeaders(c, req, "X-")
		req.Header.Set("Extra", c.Request.Header.Get("Extra"))
	default:
		forwardCustomHeaders(c, req, "X-")
	}
}

func (h *MessagesHandler) CreateTransformer(w io.Writer) transform.SSETransformer {
	return transform.NewPassthroughTransformer(w)
}

func (h *MessagesHandler) CreateTransformerWithRoute(w io.Writer, route *router.ResolvedRoute) transform.SSETransformer {
	switch route.Provider.Type {
	case "anthropic":
		if route.ToolCallTransform {
			return toolcall.NewAnthropicTransformer(w)
		}
		return transform.NewPassthroughTransformer(w)
	case "openai":
		return toolcall.NewAnthropicTransformer(w)
	default:
		return transform.NewPassthroughTransformer(w)
	}
}

func (h *MessagesHandler) WriteError(c *gin.Context, status int, msg string) {
	sendAnthropicError(c, status, msg)
}
