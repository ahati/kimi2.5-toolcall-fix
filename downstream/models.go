// Package downstream provides HTTP handlers for the proxy's client-facing API endpoints.
// It implements a unified stream handler that works with protocol adapters to support
// multiple API formats (OpenAI, Anthropic, Bridge).
package downstream

import (
	"io"
	"net/http"

	"ai-proxy/config"
	"ai-proxy/upstream"

	"github.com/gin-gonic/gin"
)

// ListModels creates a Gin handler for listing available models from the upstream API.
//
// @brief    Creates a handler that proxies model listing requests to the upstream OpenAI API.
// @param    cfg Application configuration containing upstream URL and API key.
// @return   Gin handler function for the models listing endpoint.
//
// @note     Extracts API key from Authorization header if provided, falls back to config.
// @note     Constructs models URL by replacing "chat/completions" with "models" in upstream URL.
// @note     Proxies the response directly without transformation.
//
// @pre      cfg must contain valid OpenAIUpstreamURL and OpenAIUpstreamAPIKey.
// @pre      OpenAIUpstreamURL must end with "chat/completions" for URL construction.
// @post     Response from upstream is forwarded to client with original status and content-type.
func ListModels(cfg *config.Config) gin.HandlerFunc {
	return func(c *gin.Context) {
		auth := c.GetHeader("Authorization")
		apiKey := cfg.OpenAIUpstreamAPIKey

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

		client := upstream.NewClient(cfg.OpenAIUpstreamURL, cfg.OpenAIUpstreamAPIKey)
		defer client.Close()

		modelsURL := cfg.OpenAIUpstreamURL
		modelsURL = modelsURL[:len(modelsURL)-len("chat/completions")] + "models"

		req, err := http.NewRequestWithContext(c.Request.Context(), "GET", modelsURL, nil)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": gin.H{"message": "Failed to create request", "type": "internal_error"},
			})
			return
		}

		req.Header.Set("Authorization", "Bearer "+apiKey)

		resp, err := client.Do(c.Request.Context(), req)
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
