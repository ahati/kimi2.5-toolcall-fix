// Package logging provides structured logging and request/response capture functionality.
// It captures bidirectional traffic between client and upstream LLM API for debugging
// and analysis purposes.
package logging

import (
	"github.com/gin-gonic/gin"
)

// CaptureMiddleware creates a Gin middleware for capturing request/response data.
//
// @brief    Creates middleware for request/response capture.
// @param    storage Storage instance for writing capture files, may be nil.
// @return   Gin middleware handler function.
//
// @note     If storage is nil, capture is disabled but context is still created.
// @note     Capture files are written asynchronously to avoid blocking responses.
//
// @pre      Gin router must be properly configured.
// @post     CaptureContext is attached to request context.
// @post     If storage is not nil, capture is written after response completes.
func CaptureMiddleware(storage *Storage) gin.HandlerFunc {
	return func(c *gin.Context) {
		cc := NewCaptureContext(c.Request)

		ctx := WithCaptureContext(c.Request.Context(), cc)
		c.Request = c.Request.WithContext(ctx)

		c.Next()

		if storage != nil {
			go func() {
				if err := storage.Write(cc.Recorder); err != nil {
					ErrorMsg("Failed to write capture: %v", err)
				}
			}()
		}
	}
}

// RecordDownstreamRequest captures the incoming client request data.
//
// @brief    Records downstream (client-to-proxy) request in capture context.
// @param    c    Gin context for the request.
// @param    body Raw request body bytes.
//
// @pre      CaptureMiddleware must have run before this call.
// @post     DownstreamRequest field is populated in recorder.
// @note     Headers are sanitized before storage.
// @note     Does nothing if no CaptureContext exists.
func RecordDownstreamRequest(c *gin.Context, body []byte) {
	if cc := GetCaptureContext(c.Request.Context()); cc != nil {
		cc.Recorder.DownstreamRequest = &HTTPRequestCapture{
			At:      cc.StartTime,
			Headers: SanitizeHeaders(c.Request.Header),
			Body:    body,
			RawBody: body,
		}
	}
}
