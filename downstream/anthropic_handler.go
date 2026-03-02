package downstream

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"ai-proxy/config"
	"ai-proxy/logging"
	"ai-proxy/upstream"

	"github.com/gin-gonic/gin"
	"github.com/tmaxmax/go-sse"
)

type AnthropicHandler struct {
	cfg *config.Config
}

func NewAnthropicHandler(cfg *config.Config) gin.HandlerFunc {
	handler := &AnthropicHandler{
		cfg: cfg,
	}
	return handler.Handle
}

func (h *AnthropicHandler) Handle(c *gin.Context) {
	body, err := h.readBody(c)
	if err != nil {
		h.sendError(c, http.StatusBadRequest, "Failed to read request body")
		return
	}

	if !h.isStreamingRequest(body) {
		h.sendError(c, http.StatusBadRequest, "Non-streaming requests not supported")
		return
	}

	h.proxyAndStream(c, body)
}

func (h *AnthropicHandler) readBody(c *gin.Context) ([]byte, error) {
	return io.ReadAll(c.Request.Body)
}

func (h *AnthropicHandler) isStreamingRequest(body []byte) bool {
	var req struct {
		Stream bool `json:"stream"`
	}
	json.Unmarshal(body, &req)
	return req.Stream
}

func (h *AnthropicHandler) proxyAndStream(c *gin.Context, body []byte) {
	apiKey := h.resolveAPIKey(c)
	client := upstream.NewClient(h.cfg.AnthropicUpstreamURL, apiKey)
	defer client.Close()

	req, err := client.BuildRequest(c.Request.Context(), body)
	if err != nil {
		h.sendError(c, http.StatusInternalServerError, "Failed to create upstream request")
		return
	}

	client.SetHeaders(req)

	for k, v := range c.Request.Header {
		if strings.HasPrefix(k, "X-") || k == "Anthropic-Version" || k == "Anthropic-Beta" {
			req.Header[k] = v
		}
	}

	resp, err := client.Do(req)
	if err != nil {
		h.sendError(c, http.StatusBadGateway, "Upstream request failed")
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		h.handleUpstreamError(c, resp)
		return
	}

	h.streamResponse(c, resp.Body)
}

func (h *AnthropicHandler) resolveAPIKey(c *gin.Context) string {
	auth := c.GetHeader("x-api-key")
	if auth == "" {
		auth = c.GetHeader("Authorization")
	}
	if auth == "" {
		return h.cfg.AnthropicAPIKey
	}
	if len(auth) > 7 && auth[:7] == "Bearer " {
		return auth[7:]
	}
	return auth
}

func (h *AnthropicHandler) streamResponse(c *gin.Context, body io.Reader) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("X-Accel-Buffering", "no")

	var loggingTransformer *LoggingTransformer
	if h.cfg.SSELogDir != "" {
		var err error
		loggingTransformer, err = NewLoggingTransformer(h.cfg.SSELogDir)
		if err != nil {
			logging.ErrorMsg("Failed to create logging transformer: %v", err)
		}
	}
	defer func() {
		if loggingTransformer != nil {
			loggingTransformer.Close()
		}
	}()

	transformer := NewAnthropicToolCallTransformer(c.Writer)
	defer transformer.Close()

	c.Stream(func(w io.Writer) bool {
		for ev, err := range sse.Read(body, nil) {
			if err != nil {
				logging.ErrorMsg("SSE read error: %v", err)
				return false
			}
			if loggingTransformer != nil {
				loggingTransformer.Transform(&ev)
			}
			transformer.Transform(&ev)
		}
		return false
	})

	transformer.Flush()
}

func (h *AnthropicHandler) sendError(c *gin.Context, status int, msg string) {
	logging.ErrorMsg("Anthropic handler error: %s", msg)
	c.JSON(status, AnthropicError{
		Type: "error",
		Error: ErrorDetail{
			Type:    "invalid_request_error",
			Message: msg,
		},
	})
}

func (h *AnthropicHandler) handleUpstreamError(c *gin.Context, resp *http.Response) {
	body, _ := io.ReadAll(resp.Body)
	c.Data(resp.StatusCode, resp.Header.Get("Content-Type"), body)
}
