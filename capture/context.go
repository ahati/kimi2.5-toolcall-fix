// Package capture provides request/response recording and persistence for HTTP proxy operations.
// It captures downstream client requests, upstream API requests, and their corresponding
// SSE streaming responses for debugging and analysis.
//
// Thread Safety:
//   - CaptureContext is NOT thread-safe and should only be accessed from a single goroutine
//   - All functions in this file are safe for concurrent use with different CaptureContext instances
//   - context.Context operations are inherently thread-safe as they are immutable
package capture

import (
	"context"
	"net/http"
	"time"
)

// contextKey defines the type used for context keys to prevent collisions.
// Using a private type ensures external packages cannot access or modify context values.
type contextKey string

// captureContextKey is the unique key for storing CaptureContext in context.Context.
// Constant value prevents accidental key collisions.
const captureContextKey contextKey = "capture_context"

// CaptureContext holds the capture state for a single request lifecycle.
// It tracks the request ID, timing information, and the recorder instance.
//
// Lifecycle:
//   - Created at request start via NewCaptureContext
//   - Populated with request ID via SetRequestID
//   - Attached to context via WithCaptureContext
//   - Retrieved via GetCaptureContext for recording operations
//
// Thread Safety: NOT thread-safe. Only access from a single goroutine per request.
type CaptureContext struct {
	// RequestID is the unique identifier for this request.
	// Empty string indicates ID has not been extracted yet.
	// Valid values: non-empty string after extraction, empty string before extraction.
	RequestID string

	// StartTime is the timestamp when this capture context was created.
	// Used for calculating elapsed time throughout the request lifecycle.
	// Valid values: any valid time.Time, typically time.Now() at creation.
	StartTime time.Time

	// Recorder is the data collector for this request lifecycle.
	// Never nil after NewCaptureContext returns.
	// Valid values: pointer to initialized RequestRecorder.
	Recorder *RequestRecorder

	// IDExtracted indicates whether the request ID has been extracted from SSE response.
	// Prevents duplicate extraction attempts which could cause race conditions.
	// Valid values: false initially, true after SetRequestID is called.
	IDExtracted bool
}

// NewCaptureContext creates a new CaptureContext initialized with the current time
// and a new RequestRecorder populated from the incoming HTTP request.
//
// @param r - The incoming HTTP request. Must not be nil.
// @return Pointer to newly allocated CaptureContext, never nil.
//
// @pre r != nil
// @post Returned CaptureContext.RequestID == ""
// @post Returned CaptureContext.Recorder != nil
// @post Returned CaptureContext.IDExtracted == false
// @post Returned CaptureContext.StartTime represents current time
//
// @note This function captures request metadata but not the request body.
// @note The returned context should be attached to the request context via WithCaptureContext.
func NewCaptureContext(r *http.Request) *CaptureContext {
	// Initialize all fields explicitly to avoid zero-value confusion
	// StartTime captures the moment request processing begins for duration tracking
	return &CaptureContext{
		StartTime: time.Now(),
		// Recorder is created fresh for each request to isolate capture data
		Recorder: &RequestRecorder{
			StartedAt: time.Now(),
			Method:    r.Method,
			Path:      r.URL.Path,
			ClientIP:  r.RemoteAddr,
		},
		// IDExtracted starts false to indicate ID extraction is pending
		IDExtracted: false,
	}
}

// SetRequestID assigns the request ID to both the context and its recorder.
// It marks the ID as extracted to prevent duplicate extraction attempts.
//
// @param id - The request ID to assign. Should be non-empty for valid IDs.
//
// @pre cc != nil (receiver must be valid)
// @post cc.RequestID == id
// @post cc.Recorder.RequestID == id
// @post cc.IDExtracted == true
//
// @note This method should only be called once per CaptureContext instance.
// @note Calling multiple times will overwrite the previous ID without warning.
// @note Empty string is accepted but indicates no valid ID was extracted.
func (cc *CaptureContext) SetRequestID(id string) {
	// Assign to both fields to maintain consistency between context and recorder
	// This ensures all downstream code can access ID from either location
	cc.RequestID = id
	cc.Recorder.RequestID = id
	// Mark as extracted to prevent redundant SSE parsing attempts
	// SSE parsing is expensive so we want to avoid doing it multiple times
	cc.IDExtracted = true
}

// WithCaptureContext returns a new context with the CaptureContext attached.
// The CaptureContext can later be retrieved using GetCaptureContext.
//
// @param ctx - Parent context to extend. May be nil (creates context with only CaptureContext).
// @param cc  - CaptureContext to attach. Must not be nil for meaningful operation.
// @return New context containing the CaptureContext, or nil if ctx is nil.
//
// @pre cc != nil for proper functionality
// @post If ctx != nil, returned context.Value(captureContextKey) == cc
// @post Returned context is a child of ctx (inherits cancellation, values, etc.)
//
// @note Context values are immutable; the original context is not modified.
// @note This uses the standard context.Value mechanism; key collision is prevented by private type.
// @note Thread-safe: context.Context is inherently safe for concurrent access.
func WithCaptureContext(ctx context.Context, cc *CaptureContext) context.Context {
	// context.WithValue creates a new context rather than modifying the existing one
	// This ensures thread-safety as contexts are immutable after creation
	return context.WithValue(ctx, captureContextKey, cc)
}

// GetCaptureContext retrieves the CaptureContext from the given context.
// It returns nil if the context is nil or does not contain a CaptureContext.
//
// @param ctx - Context to search for CaptureContext. May be nil.
// @return Pointer to CaptureContext if found, nil otherwise.
//
// @post Returns nil if ctx == nil
// @post Returns nil if context does not contain CaptureContext
// @post Returns cc if context was created via WithCaptureContext(ctx, cc)
//
// @note Safe to call with nil context; returns nil without panic.
// @note Thread-safe: context.Value operations are safe for concurrent access.
func GetCaptureContext(ctx context.Context) *CaptureContext {
	// Early return for nil context prevents panic on nil pointer dereference
	// This defensive check allows callers to pass context values directly
	if ctx == nil {
		return nil
	}
	// Type assertion with ok pattern provides safe extraction without panic
	// If the value is not a *CaptureContext, ok will be false
	if cc, ok := ctx.Value(captureContextKey).(*CaptureContext); ok {
		return cc
	}
	// Return nil explicitly to indicate no CaptureContext was found
	// Callers should always check the return value for nil
	return nil
}

// RecordDownstreamRequest records the incoming client request body in the CaptureContext.
// It is a convenience function that retrieves the context and delegates to the recorder.
//
// @param ctx  - Context containing CaptureContext. May be nil.
// @param r    - HTTP request being recorded. May be nil (only used for headers).
// @param body - Request body bytes to record. May be nil or empty.
//
// @pre None (handles nil inputs gracefully)
// @post If ctx contains valid CaptureContext, downstream request is recorded
// @post If ctx is nil or lacks CaptureContext, no action is taken (no-op)
//
// @note This is a convenience wrapper combining GetCaptureContext and recorder method.
// @note Safe to call with nil context; simply returns without side effects.
// @note The request parameter may be nil if only body needs recording.
func RecordDownstreamRequest(ctx context.Context, r *http.Request, body []byte) {
	// Retrieve CaptureContext from context; nil check handles missing context gracefully
	// This allows callers to always call this function without pre-checking context
	cc := GetCaptureContext(ctx)
	if cc == nil {
		// No CaptureContext available; nothing to record
		// This is not an error condition; capture may be disabled
		return
	}
	// Delegate to recorder; actual recording logic is in RequestRecorder
	// This separation of concerns keeps this file focused on context management
	cc.Recorder.RecordDownstreamRequest(r, body)
}
