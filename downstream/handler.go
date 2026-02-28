package downstream

import (
	"io"
	"net/http"

	"ai-proxy/config"
	"ai-proxy/logging"
	"ai-proxy/upstream"

	"github.com/gin-gonic/gin"
)

func handler(c *gin.Context, cfg *config.Config) {
	body, err := readBody(c)
	if err != nil {
		sendError(c, http.StatusBadRequest, "Failed to read request body", "")
		return
	}

	proxyAndRespond(c, cfg, body)
}

func resolveAPIKey(c *gin.Context, cfg *config.Config) string {
	auth := c.GetHeader("Authorization")
	if auth == "" {
		return cfg.UpstreamAPIKey
	}
	if len(auth) > 7 && auth[:7] == "Bearer " {
		return auth[7:]
	}
	return cfg.UpstreamAPIKey
}

func proxyAndRespond(c *gin.Context, cfg *config.Config, body []byte) {
	client := upstream.NewClient(cfg.UpstreamURL, cfg.UpstreamAPIKey)
	defer client.Close()

	req, err := client.BuildRequest(c.Request.Context(), body)
	if err != nil {
		sendError(c, http.StatusInternalServerError, "Failed to create upstream request", "")
		return
	}

	client.SetHeaders(req)

	for k, v := range c.Request.Header {
		if len(k) > 1 && k[:2] == "X-" || k == "Extra" {
			req.Header[k] = v
		}
	}

	resp, err := client.Do(req)
	if err != nil {
		sendError(c, http.StatusBadGateway, "Upstream request failed", "")
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		handleUpstreamError(c, resp)
		return
	}

	toolCallTransformer := NewToolCallTransformer(c.Writer)

	var loggingTransformer *LoggingTransformer
	if cfg.SSELogDir != "" {
		var err error
		loggingTransformer, err = NewLoggingTransformer(cfg.SSELogDir)
		if err != nil {
			logging.ErrorMsg("Failed to create logging transformer: %v", err)
		}
	}
	defer func() {
		if loggingTransformer != nil {
			loggingTransformer.Close()
		}
	}()

	streamResponse(c, resp.Body, loggingTransformer, toolCallTransformer)
}

func readBody(c *gin.Context) ([]byte, error) {
	return io.ReadAll(c.Request.Body)
}
