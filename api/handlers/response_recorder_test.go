package handlers

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"ai-proxy/capture"
)

func TestResponseRecorder_Write(t *testing.T) {
	tests := []struct {
		name           string
		data           []byte
		capture        capture.CaptureWriter
		expectedChunks int
	}{
		{
			name:           "write without capture",
			data:           []byte("test data"),
			capture:        nil,
			expectedChunks: 0,
		},
		{
			name:           "write with capture - no data line",
			data:           []byte("event: test\n\n"),
			capture:        capture.NewCaptureWriter(time.Now()),
			expectedChunks: 0,
		},
		{
			name:           "write with capture - data line",
			data:           []byte("data: {\"test\": \"value\"}\n\n"),
			capture:        capture.NewCaptureWriter(time.Now()),
			expectedChunks: 1,
		},
		{
			name:           "write with capture - event and data",
			data:           []byte("event: message\ndata: {\"test\": \"value\"}\n\n"),
			capture:        capture.NewCaptureWriter(time.Now()),
			expectedChunks: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			underlying := httptest.NewRecorder()
			recorder := NewResponseRecorder(underlying, tt.capture)

			n, err := recorder.Write(tt.data)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if n != len(tt.data) {
				t.Errorf("expected %d bytes written, got %d", len(tt.data), n)
			}

			if tt.capture != nil {
				chunks := tt.capture.Chunks()
				if len(chunks) != tt.expectedChunks {
					t.Errorf("expected %d chunks, got %d", tt.expectedChunks, len(chunks))
				}
			}
		})
	}
}

func TestResponseRecorder_Header(t *testing.T) {
	underlying := httptest.NewRecorder()
	underlying.Header().Set("Content-Type", "text/event-stream")

	recorder := NewResponseRecorder(underlying, nil)

	headers := recorder.Header()
	if headers.Get("Content-Type") != "text/event-stream" {
		t.Errorf("expected Content-Type 'text/event-stream', got %q", headers.Get("Content-Type"))
	}

	headers.Set("X-Custom", "value")
	if underlying.Header().Get("X-Custom") != "value" {
		t.Errorf("expected X-Custom 'value', got %q", underlying.Header().Get("X-Custom"))
	}
}

func TestResponseRecorder_WriteHeader(t *testing.T) {
	underlying := httptest.NewRecorder()
	recorder := NewResponseRecorder(underlying, nil)

	recorder.WriteHeader(http.StatusOK)

	if underlying.Code != http.StatusOK {
		t.Errorf("expected status %d, got %d", http.StatusOK, underlying.Code)
	}
}

func TestExtractDataPart(t *testing.T) {
	tests := []struct {
		name     string
		data     []byte
		expected []byte
	}{
		{
			name:     "empty data",
			data:     []byte{},
			expected: nil,
		},
		{
			name:     "single data line",
			data:     []byte("data: {\"test\": \"value\"}\n"),
			expected: []byte(`{"test": "value"}`),
		},
		{
			name:     "event and data lines",
			data:     []byte("event: message\ndata: {\"test\": \"value\"}\n"),
			expected: []byte(`{"test": "value"}`),
		},
		{
			name:     "multiple lines without data",
			data:     []byte("event: message\nid: 123\n\n"),
			expected: nil,
		},
		{
			name:     "multiple data lines - returns first",
			data:     []byte("data: first\ndata: second\n"),
			expected: []byte("first"),
		},
		{
			name:     "data without newline",
			data:     []byte("data: test"),
			expected: []byte("test"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractDataPart(tt.data)
			if string(result) != string(tt.expected) {
				t.Errorf("extractDataPart() = %q, want %q", result, tt.expected)
			}
		})
	}
}

func TestExtractEventType(t *testing.T) {
	tests := []struct {
		name     string
		data     []byte
		expected string
	}{
		{
			name:     "empty data",
			data:     []byte{},
			expected: "",
		},
		{
			name:     "event line",
			data:     []byte("event: message\n"),
			expected: "message",
		},
		{
			name:     "event and data lines",
			data:     []byte("event: message_start\ndata: {\"test\": \"value\"}\n"),
			expected: "message_start",
		},
		{
			name:     "no event line",
			data:     []byte("data: {\"test\": \"value\"}\n"),
			expected: "",
		},
		{
			name:     "multiple lines with event",
			data:     []byte("id: 123\nevent: ping\ndata: {}\n"),
			expected: "ping",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractEventType(tt.data)
			if result != tt.expected {
				t.Errorf("extractEventType() = %q, want %q", result, tt.expected)
			}
		})
	}
}

func TestNewResponseRecorder(t *testing.T) {
	underlying := httptest.NewRecorder()
	captureWriter := capture.NewCaptureWriter(time.Now())

	recorder := NewResponseRecorder(underlying, captureWriter)

	if recorder == nil {
		t.Fatal("expected non-nil recorder")
	}
	if recorder.writer != underlying {
		t.Error("writer not set correctly")
	}
	if recorder.capture != captureWriter {
		t.Error("capture not set correctly")
	}
}

func TestResponseRecorder_WriteMultipleTimes(t *testing.T) {
	underlying := httptest.NewRecorder()
	captureWriter := capture.NewCaptureWriter(time.Now())
	recorder := NewResponseRecorder(underlying, captureWriter)

	recorder.Write([]byte("data: {\"chunk\": 1}\n\n"))
	recorder.Write([]byte("data: {\"chunk\": 2}\n\n"))
	recorder.Write([]byte("event: done\ndata: [DONE]\n\n"))

	chunks := captureWriter.Chunks()
	if len(chunks) != 3 {
		t.Errorf("expected 3 chunks, got %d", len(chunks))
	}

	if len(chunks) >= 1 && string(chunks[0].Data) != `{"chunk": 1}` {
		t.Errorf("first chunk data = %s, want %s", chunks[0].Data, `{"chunk": 1}`)
	}
}
