// Package downstream provides HTTP handlers for the proxy's client-facing API endpoints.
// It implements a unified stream handler that works with protocol adapters to support
// multiple API formats (OpenAI, Anthropic, Bridge).
package downstream

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

// HealthCheck handles health check requests for the proxy service.
//
// @brief    Returns a simple JSON response indicating the service is healthy.
// @param    c Gin context for the HTTP request/response.
//
// @note     Returns HTTP 200 with {"status": "ok"} on success.
// @note     This endpoint is typically used by load balancers and monitoring systems.
//
// @post     Response body contains JSON with status field set to "ok".
func HealthCheck(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"status": "ok"})
}
