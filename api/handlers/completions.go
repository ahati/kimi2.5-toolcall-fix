package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"ai-proxy/convert"
	"ai-proxy/router"
	"ai-proxy/transform"
	"ai-proxy/transform/toolcall"

	"github.com/gin-gonic/gin"
)

type CompletionsHandler struct{}

func NewCompletionsHandler(r router.Router) gin.HandlerFunc {
	return HandleWithRouter(&CompletionsHandler{}, r)
}

func (h *CompletionsHandler) ValidateRequest(body []byte) error {
	return nil
}

func (h *CompletionsHandler) ExtractModel(body []byte) (string, error) {
	var req struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return "", fmt.Errorf("invalid JSON: %w", err)
	}
	return req.Model, nil
}

func (h *CompletionsHandler) TransformRequest(body []byte) ([]byte, error) {
	return body, nil
}

func (h *CompletionsHandler) TransformRequestWithRoute(body []byte, route *router.ResolvedRoute) ([]byte, error) {
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
	case "openai":
		return updatedBody, nil
	case "anthropic":
		converter := convert.NewChatToAnthropicConverter()
		return converter.Convert(updatedBody)
	default:
		return updatedBody, nil
	}
}

func (h *CompletionsHandler) UpstreamURL() string {
	return ""
}

func (h *CompletionsHandler) UpstreamURLWithRoute(route *router.ResolvedRoute) string {
	switch route.Provider.Type {
	case "openai":
		baseURL := route.Provider.BaseURL
		if !strings.HasSuffix(baseURL, "/chat/completions") {
			baseURL = strings.TrimSuffix(baseURL, "/") + "/chat/completions"
		}
		return baseURL
	case "anthropic":
		return route.Provider.BaseURL
	default:
		return route.Provider.BaseURL
	}
}

func (h *CompletionsHandler) ResolveAPIKey(c *gin.Context) string {
	return ""
}

func (h *CompletionsHandler) ResolveAPIKeyWithRoute(route *router.ResolvedRoute) string {
	return route.Provider.GetAPIKey()
}

func (h *CompletionsHandler) ForwardHeaders(c *gin.Context, req *http.Request) {
}

func (h *CompletionsHandler) ForwardHeadersWithRoute(c *gin.Context, req *http.Request, route *router.ResolvedRoute) {
	switch route.Provider.Type {
	case "openai":
		forwardCustomHeaders(c, req, "X-")
		req.Header.Set("Extra", c.Request.Header.Get("Extra"))
	case "anthropic":
		for k, v := range c.Request.Header {
			if strings.HasPrefix(k, "X-") || k == "Anthropic-Version" || k == "Anthropic-Beta" {
				req.Header[k] = v
			}
		}
	default:
		forwardCustomHeaders(c, req, "X-")
	}
}

func (h *CompletionsHandler) CreateTransformer(w io.Writer) transform.SSETransformer {
	return transform.NewPassthroughTransformer(w)
}

func (h *CompletionsHandler) CreateTransformerWithRoute(w io.Writer, route *router.ResolvedRoute) transform.SSETransformer {
	switch route.Provider.Type {
	case "openai":
		if route.ToolCallTransform {
			return toolcall.NewOpenAITransformer(w)
		}
		return transform.NewPassthroughTransformer(w)
	case "anthropic":
		if route.ToolCallTransform {
			return toolcall.NewAnthropicTransformer(w)
		}
		return convert.NewChatToAnthropicTransformer(w)
	default:
		return transform.NewPassthroughTransformer(w)
	}
}

func (h *CompletionsHandler) WriteError(c *gin.Context, status int, msg string) {
	sendOpenAIError(c, status, msg)
}

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
