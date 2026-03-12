package upstream

import (
	"context"
	"net/http"
)

// MockClient implements the Client interface for testing purposes.
// It allows tests to control the response and error returned by Do,
// and records all requests made for later verification.
//
// @brief Mock client for testing upstream HTTP operations.
type MockClient struct {
	// Response is the HTTP response to return from Do.
	Response *http.Response
	// Error is the error to return from Do.
	Error error
	// Requests records all requests passed to Do for verification.
	Requests []*http.Request
}

// Do records the request and returns the configured response and error.
//
// @brief    Records request and returns mock response.
// @param    ctx Context (unused in mock but required by interface).
// @param    req The HTTP request to record.
// @return   Configured Response value.
// @return   Configured Error value.
//
// @note     Appends request to Requests slice for test verification.
// @note     Does not execute any real HTTP operations.
func (m *MockClient) Do(ctx context.Context, req *http.Request) (*http.Response, error) {
	m.Requests = append(m.Requests, req)
	return m.Response, m.Error
}

// Close is a no-op for the mock client.
//
// @brief    No-op close for mock client.
//
// @note     Satisfies Client interface without releasing resources.
func (m *MockClient) Close() {}
