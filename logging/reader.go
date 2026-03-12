// Package logging provides structured logging and request/response capture functionality.
// It captures bidirectional traffic between client and upstream LLM API for debugging
// and analysis purposes.
package logging

import (
	"io"
	"sync"
)

// capturingReadCloser wraps an io.ReadCloser to capture all read data.
// It implements io.ReadCloser and stores all bytes read for later retrieval.
type capturingReadCloser struct {
	// r is the underlying reader being wrapped.
	r io.ReadCloser
	// mu protects concurrent access to data.
	mu sync.Mutex
	// data accumulates all bytes read from the underlying reader.
	data []byte
	// resp is the SSEResponseCapture to store raw body on close.
	resp *SSEResponseCapture
}

// WrapResponseBody wraps a response body reader to capture its contents.
//
// @brief    Creates capturing wrapper around response body.
// @param    r    Original response body reader.
// @param    resp SSEResponseCapture to store raw body on close.
// @return   io.ReadCloser that captures all read data.
//
// @pre      r must not be nil.
// @pre      resp may be nil (raw body capture disabled).
// @post     All reads are captured in internal buffer.
// @post     On Close, raw body is stored in resp if resp is not nil.
func WrapResponseBody(r io.ReadCloser, resp *SSEResponseCapture) io.ReadCloser {
	return &capturingReadCloser{
		r:    r,
		resp: resp,
		data: []byte{},
	}
}

// Read reads from the underlying reader and captures the data.
//
// @brief    Implements io.Reader interface with data capture.
// @param    p   Buffer to read data into.
// @return   n   Number of bytes read.
// @return   err Error from underlying read, if any.
//
// @pre      p must have capacity > 0.
// @post     Data read is appended to internal buffer.
// @note     Thread-safe via mutex lock on data append.
func (c *capturingReadCloser) Read(p []byte) (n int, err error) {
	n, err = c.r.Read(p)
	if n > 0 {
		c.mu.Lock()
		c.data = append(c.data, p[:n]...)
		c.mu.Unlock()
	}
	return n, err
}

// Close closes the underlying reader and stores captured data.
//
// @brief    Implements io.Closer interface with data storage.
// @return   Error from underlying close, if any.
//
// @pre      None.
// @post     Underlying reader is closed.
// @post     Captured data is stored in resp.RawBody if resp is not nil.
// @note     Thread-safe via mutex lock.
func (c *capturingReadCloser) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.resp != nil && len(c.data) > 0 {
		c.resp.RawBody = c.data
	}

	return c.r.Close()
}
