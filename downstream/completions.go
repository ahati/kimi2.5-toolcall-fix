package downstream

import (
	"fmt"
	"io"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/tmaxmax/go-sse"
	"proxy/config"
	"proxy/logging"
)

func Completions(cfg *config.Config) gin.HandlerFunc {
	return func(c *gin.Context) {
		handler(c, cfg)
	}
}

func getEffectiveKey(clientAuth, fallback string) string {
	if len(clientAuth) > 7 && clientAuth[:7] == "Bearer " {
		return clientAuth[7:]
	}
	return fallback
}

func sendError(c *gin.Context, status int, msg, code string) {
	logging.ErrorMsg("%s", msg)
	c.JSON(status, gin.H{
		"error": gin.H{
			"message": msg,
			"type":    "invalid_request_error",
			"code":    code,
		},
	})
}

func handleUpstreamError(c *gin.Context, resp *http.Response) {
	body, _ := io.ReadAll(resp.Body)
	msg := fmt.Sprintf("Upstream error: %s", string(body))
	sendError(c, http.StatusBadGateway, msg, fmt.Sprintf("status_%d", resp.StatusCode))
}

func streamResponse(c *gin.Context, body io.Reader) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("X-Accel-Buffering", "no")

	c.Stream(func(w io.Writer) bool {
		for ev, err := range sse.Read(body, nil) {
			if err != nil {
				return false
			}
			if ev.Data != "" {
				w.Write([]byte("data: " + ev.Data + "\n\n"))
			}
		}
		return false
	})
}
