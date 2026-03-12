package logging

import (
	"bytes"
	"errors"
	"io"
	"testing"
)

type mockReadCloser struct {
	data   []byte
	pos    int
	closed bool
	err    error
}

func (m *mockReadCloser) Read(p []byte) (n int, err error) {
	if m.err != nil {
		return 0, m.err
	}
	if m.pos >= len(m.data) {
		return 0, io.EOF
	}
	n = copy(p, m.data[m.pos:])
	m.pos += n
	return n, nil
}

func (m *mockReadCloser) Close() error {
	m.closed = true
	return nil
}

func TestWrapResponseBody(t *testing.T) {
	data := []byte("test data")
	inner := &mockReadCloser{data: data}
	resp := &SSEResponseCapture{}

	wrapped := WrapResponseBody(inner, resp)

	if wrapped == nil {
		t.Fatal("WrapResponseBody returned nil")
	}

	buf := make([]byte, len(data))
	n, err := wrapped.Read(buf)
	if err != nil && err != io.EOF {
		t.Fatalf("Read error: %v", err)
	}
	if n != len(data) {
		t.Errorf("Expected %d bytes, got %d", len(data), n)
	}
	if string(buf[:n]) != "test data" {
		t.Errorf("Expected 'test data', got %s", string(buf[:n]))
	}
}

func TestCapturingReadCloser_Read(t *testing.T) {
	t.Run("reads and captures data", func(t *testing.T) {
		data := []byte("hello world")
		inner := &mockReadCloser{data: data}
		resp := &SSEResponseCapture{}

		wrapped := WrapResponseBody(inner, resp).(*capturingReadCloser)

		buf := make([]byte, 100)
		n, err := wrapped.Read(buf)
		if err != nil && err != io.EOF {
			t.Fatalf("Read error: %v", err)
		}
		if n != len(data) {
			t.Errorf("Expected %d bytes, got %d", len(data), n)
		}
		if string(wrapped.data) != "hello world" {
			t.Errorf("Captured data mismatch: %s", string(wrapped.data))
		}
	})

	t.Run("reads in chunks", func(t *testing.T) {
		data := []byte("chunk1chunk2chunk3")
		inner := &mockReadCloser{data: data}
		resp := &SSEResponseCapture{}

		wrapped := WrapResponseBody(inner, resp).(*capturingReadCloser)

		buf := make([]byte, 6)
		n1, _ := wrapped.Read(buf)
		n2, _ := wrapped.Read(buf)
		n3, _ := wrapped.Read(buf)

		if n1+n2+n3 != len(data) {
			t.Errorf("Expected total %d bytes, got %d", len(data), n1+n2+n3)
		}
		if string(wrapped.data) != "chunk1chunk2chunk3" {
			t.Errorf("Captured data mismatch: %s", string(wrapped.data))
		}
	})

	t.Run("handles read error", func(t *testing.T) {
		testErr := errors.New("read error")
		inner := &mockReadCloser{err: testErr}
		resp := &SSEResponseCapture{}

		wrapped := WrapResponseBody(inner, resp)

		buf := make([]byte, 100)
		n, err := wrapped.Read(buf)
		if err != testErr {
			t.Errorf("Expected read error, got %v", err)
		}
		if n != 0 {
			t.Errorf("Expected 0 bytes on error, got %d", n)
		}
	})

	t.Run("reads zero bytes", func(t *testing.T) {
		inner := &mockReadCloser{data: []byte{}}
		resp := &SSEResponseCapture{}

		wrapped := WrapResponseBody(inner, resp).(*capturingReadCloser)

		buf := make([]byte, 100)
		n, _ := wrapped.Read(buf)

		if n != 0 {
			t.Errorf("Expected 0 bytes, got %d", n)
		}
		if len(wrapped.data) != 0 {
			t.Errorf("Expected no captured data, got %d bytes", len(wrapped.data))
		}
	})
}

func TestCapturingReadCloser_Close(t *testing.T) {
	t.Run("closes and captures raw body", func(t *testing.T) {
		data := []byte("response data")
		inner := &mockReadCloser{data: data}
		resp := &SSEResponseCapture{}

		wrapped := WrapResponseBody(inner, resp).(*capturingReadCloser)

		buf := make([]byte, 100)
		wrapped.Read(buf)

		err := wrapped.Close()
		if err != nil {
			t.Fatalf("Close error: %v", err)
		}
		if !inner.closed {
			t.Error("Inner reader was not closed")
		}
		if string(resp.RawBody) != "response data" {
			t.Errorf("RawBody mismatch: %s", string(resp.RawBody))
		}
	})

	t.Run("close with no data", func(t *testing.T) {
		inner := &mockReadCloser{data: []byte{}}
		resp := &SSEResponseCapture{}

		wrapped := WrapResponseBody(inner, resp).(*capturingReadCloser)

		err := wrapped.Close()
		if err != nil {
			t.Fatalf("Close error: %v", err)
		}
		if resp.RawBody != nil {
			t.Errorf("RawBody should be nil when no data, got %s", string(resp.RawBody))
		}
	})

	t.Run("close with nil response", func(t *testing.T) {
		inner := &mockReadCloser{data: []byte{}}

		wrapped := WrapResponseBody(inner, nil).(*capturingReadCloser)

		err := wrapped.Close()
		if err != nil {
			t.Fatalf("Close error: %v", err)
		}
		if !inner.closed {
			t.Error("Inner reader was not closed")
		}
	})

	t.Run("close sets raw body even after partial read", func(t *testing.T) {
		data := []byte("partial read test")
		inner := &mockReadCloser{data: data}
		resp := &SSEResponseCapture{}

		wrapped := WrapResponseBody(inner, resp).(*capturingReadCloser)

		buf := make([]byte, 8)
		wrapped.Read(buf)

		err := wrapped.Close()
		if err != nil {
			t.Fatalf("Close error: %v", err)
		}
		if string(resp.RawBody) != "partial " {
			t.Errorf("RawBody mismatch: %s", string(resp.RawBody))
		}
	})
}

func TestCapturingReadCloser_FullFlow(t *testing.T) {
	data := []byte(`{"id":"test","choices":[{"delta":{"content":"hello"}}]}`)
	inner := io.NopCloser(bytes.NewReader(data))
	resp := &SSEResponseCapture{}

	wrapped := WrapResponseBody(inner, resp)

	readData, err := io.ReadAll(wrapped)
	if err != nil {
		t.Fatalf("ReadAll error: %v", err)
	}

	if string(readData) != string(data) {
		t.Errorf("Read data mismatch: %s", string(readData))
	}

	wrapped.Close()

	if string(resp.RawBody) != string(data) {
		t.Errorf("RawBody mismatch: %s", string(resp.RawBody))
	}
}
