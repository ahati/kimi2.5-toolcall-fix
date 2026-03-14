package handlers

import (
	"encoding/json"
	"io"
	"net/http"

	"ai-proxy/convert"
	"ai-proxy/router"
	"ai-proxy/transform"
	"ai-proxy/transform/toolcall"

	"github.com/gin-gonic/gin"
)

type ResponsesHandler struct {
	router            router.Router
	resolvedRoute     *router.ResolvedRoute
	requestConverter  convert.RequestConverter
	responseConverter func(io.Writer) convert.ResponseTransformer
}

func NewResponsesHandler(r router.Router) gin.HandlerFunc {
	h := &ResponsesHandler{router: r}
	return Handle(h)
}

func (h *ResponsesHandler) ValidateRequest(body []byte) error {
	return nil
}

func (h *ResponsesHandler) TransformRequest(body []byte) ([]byte, error) {
	var req struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, err
	}

	route, err := h.router.Resolve(req.Model)
	if err != nil {
		return body, nil
	}

	h.resolvedRoute = route

	if route.OutputProtocol == "anthropic" {
		converter := convert.NewResponsesToChatConverter()
		return converter.Convert(body)
	}

	return body, nil
}

func (h *ResponsesHandler) UpstreamURL() string {
	if h.resolvedRoute != nil && h.resolvedRoute.Provider != nil {
		return h.resolvedRoute.Provider.BaseURL
	}
	return ""
}

func (h *ResponsesHandler) ResolveAPIKey(c *gin.Context) string {
	if h.resolvedRoute != nil && h.resolvedRoute.Provider != nil {
		return h.resolvedRoute.Provider.GetAPIKey()
	}
	return ""
}

func (h *ResponsesHandler) ForwardHeaders(c *gin.Context, req *http.Request) {
	for k, v := range c.Request.Header {
		if len(k) > 2 && k[:2] == "X-" {
			req.Header[k] = v
		}
	}
}

func (h *ResponsesHandler) CreateTransformer(w io.Writer) transform.SSETransformer {
	return toolcall.NewResponsesTransformer(w)
}

func (h *ResponsesHandler) WriteError(c *gin.Context, status int, msg string) {
	sendOpenAIError(c, status, msg)
}
