package handlers

import (
	"github.com/gin-gonic/gin"
)

// HealthCheck returns a simple health status response for monitoring
// and load balancer health checks.
// This endpoint is used by:
//   - Kubernetes liveness/readiness probes
//   - Load balancer health checks
//   - Monitoring systems to verify service availability
//
// @param c - Gin context for the HTTP request.
//
// @pre c != nil
// @post Response body contains JSON with "status": "ok".
// @post HTTP status code is 200 OK.
// @note This endpoint does not check upstream connectivity - it only verifies
// that the proxy service itself is running and able to respond.
func HealthCheck(c *gin.Context) {
	// Return simple OK status - this indicates the service is running
	// Does not verify upstream connectivity or database health
	c.JSON(200, gin.H{"status": "ok"})
}
