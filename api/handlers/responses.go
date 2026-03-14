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
	"ai-proxy/types"

	"github.com/gin-gonic/gin"
)

type ResponsesHandler struct{}

func NewResponsesHandler(r router.Router) gin.HandlerFunc {
	return HandleWithRouter(&ResponsesHandler{}, r)
}

func (h *ResponsesHandler) ValidateRequest(body []byte) error {
	var req types.ResponsesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return fmt.Errorf("invalid JSON: %w", err)
	}
	if req.Model == "" {
		return fmt.Errorf("model is required")
	}
	return nil
}

func (h *ResponsesHandler) ExtractModel(body []byte) (string, error) {
	var req struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return "", fmt.Errorf("invalid JSON: %w", err)
	}
	return req.Model, nil
}

func (h *ResponsesHandler) TransformRequest(body []byte) ([]byte, error) {
	return body, nil
}

func (h *ResponsesHandler) TransformRequestWithRoute(body []byte, route *router.ResolvedRoute) ([]byte, error) {
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
		converter := convert.NewResponsesToChatConverter()
		return converter.Convert(updatedBody)
	case "anthropic":
		return transformResponsesRequest(updatedBody)
	default:
		return updatedBody, nil
	}
}

func (h *ResponsesHandler) UpstreamURL() string {
	return ""
}

func (h *ResponsesHandler) UpstreamURLWithRoute(route *router.ResolvedRoute) string {
	if route.Provider.Type == "openai" {
		baseURL := route.Provider.BaseURL
		if !strings.HasSuffix(baseURL, "/chat/completions") {
			baseURL = strings.TrimSuffix(baseURL, "/") + "/chat/completions"
		}
		return baseURL
	}
	return route.Provider.BaseURL
}

func (h *ResponsesHandler) ResolveAPIKey(c *gin.Context) string {
	return ""
}

func (h *ResponsesHandler) ResolveAPIKeyWithRoute(route *router.ResolvedRoute) string {
	return route.Provider.GetAPIKey()
}

func (h *ResponsesHandler) ForwardHeaders(c *gin.Context, req *http.Request) {
}

func (h *ResponsesHandler) ForwardHeadersWithRoute(c *gin.Context, req *http.Request, route *router.ResolvedRoute) {
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

func (h *ResponsesHandler) CreateTransformer(w io.Writer) transform.SSETransformer {
	return transform.NewPassthroughTransformer(w)
}

func (h *ResponsesHandler) CreateTransformerWithRoute(w io.Writer, route *router.ResolvedRoute) transform.SSETransformer {
	switch route.Provider.Type {
	case "openai":
		return toolcall.NewOpenAITransformer(w)
	case "anthropic":
		return toolcall.NewResponsesTransformer(w)
	default:
		return transform.NewPassthroughTransformer(w)
	}
}

func (h *ResponsesHandler) WriteError(c *gin.Context, status int, msg string) {
	sendOpenAIResponsesError(c, status, msg)
}
