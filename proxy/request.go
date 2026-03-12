// Package proxy provides utilities for building and forwarding HTTP requests to upstream APIs.
// This file contains header forwarding utilities for proxying requests.
package proxy

import (
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
)

// ForwardHeaders copies headers from the incoming Gin request to the outgoing HTTP request.
// Only headers matching one of the provided prefixes are forwarded.
// This function is used to selectively forward headers like X-* custom headers.
//
// @param c - the Gin context containing the incoming HTTP request
// @param req - the outgoing HTTP request to copy headers to
// @param prefixes - list of header prefixes to match (case-insensitive)
// @pre c must not be nil
// @pre req must not be nil
// @post Matching headers are added to req (not replaced)
// @note Headers are matched case-insensitively but preserved in original case
// @note Multiple values for the same header are all forwarded
func ForwardHeaders(c *gin.Context, req *http.Request, prefixes ...string) {
	// Iterate over all headers from the incoming request
	for key, values := range c.Request.Header {
		keyLower := strings.ToLower(key)
		// Check each prefix to see if this header should be forwarded
		for _, prefix := range prefixes {
			if strings.HasPrefix(keyLower, strings.ToLower(prefix)) {
				// Add all values for this header (headers can have multiple values)
				for _, value := range values {
					req.Header.Add(key, value)
				}
				break // Stop checking prefixes once a match is found
			}
		}
	}
}

// ForwardConnectionHeaders forwards connection-related headers from the incoming request to the outgoing request.
// This includes Connection, Keep-Alive, Upgrade, and TE headers.
// These headers are critical for proper HTTP connection handling.
//
// @param c - the Gin context containing the incoming HTTP request
// @param req - the outgoing HTTP request to copy headers to
// @pre c must not be nil
// @pre req must not be nil
// @post Connection headers are set on req (overwrites existing values)
// @note Uses Set instead of Add to ensure single values for connection headers
// @note Empty headers are not forwarded
func ForwardConnectionHeaders(c *gin.Context, req *http.Request) {
	// List of headers that affect HTTP connection behavior
	connectionHeaders := []string{"Connection", "Keep-Alive", "Upgrade", "TE"}
	for _, header := range connectionHeaders {
		// Only forward if the header has a value
		if value := c.GetHeader(header); value != "" {
			req.Header.Set(header, value) // Use Set to overwrite, connection headers should have single values
		}
	}
}
