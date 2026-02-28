package downstream

import (
	"io"
	"net/http"

	"ai-proxy/config"
	"ai-proxy/upstream"
	"github.com/gin-gonic/gin"
)

func ListModels(cfg *config.Config) gin.HandlerFunc {
	return func(c *gin.Context) {
		auth := c.GetHeader("Authorization")
		apiKey := cfg.UpstreamAPIKey

		if len(auth) > 7 && auth[:7] == "Bearer " {
			apiKey = auth[7:]
		}

		if apiKey == "" {
			c.JSON(http.StatusUnauthorized, gin.H{
				"error": gin.H{
					"message": "Missing API key",
					"type":    "invalid_request_error",
					"code":    "missing_api_key",
				},
			})
			return
		}

		client := upstream.NewClient(cfg.UpstreamURL, cfg.UpstreamAPIKey)
		defer client.Close()

		modelsURL := cfg.UpstreamURL
		modelsURL = modelsURL[:len(modelsURL)-len("chat/completions")] + "models"

		req, err := http.NewRequestWithContext(c.Request.Context(), "GET", modelsURL, nil)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": gin.H{"message": "Failed to create request", "type": "internal_error"},
			})
			return
		}

		req.Header.Set("Authorization", "Bearer "+apiKey)

		resp, err := client.Do(req)
		if err != nil {
			c.JSON(http.StatusBadGateway, gin.H{
				"error": gin.H{"message": "Upstream request failed", "type": "upstream_error"},
			})
			return
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		c.Data(resp.StatusCode, resp.Header.Get("Content-Type"), body)
	}
}
