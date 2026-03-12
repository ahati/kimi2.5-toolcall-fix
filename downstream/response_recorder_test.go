package downstream

import (
	"net/http"
	"testing"
	"time"

	"ai-proxy/logging"
)

type mockResponseWriter struct {
	header  http.Header
	body    []byte
	status  int
	written bool
}

func newMockResponseWriter() *mockResponseWriter {
	return &mockResponseWriter{
		header: make(http.Header),
	}
}

func (m *mockResponseWriter) Header() http.Header {
	return m.header
}

func (m *mockResponseWriter) Write(data []byte) (int, error) {
	m.body = append(m.body, data...)
	return len(data), nil
}

func (m *mockResponseWriter) WriteHeader(statusCode int) {
	m.status = statusCode
	m.written = true
}

func TestNewResponseRecorder(t *testing.T) {
	mockWriter := newMockResponseWriter()
	capture := logging.NewCaptureWriter(time.Now())

	recorder := NewResponseRecorder(mockWriter, capture)

	if recorder == nil {
		t.Fatal("Expected recorder to be created")
	}
}

func TestResponseRecorder_Write(t *testing.T) {
	mockWriter := newMockResponseWriter()
	capture := logging.NewCaptureWriter(time.Now())
	recorder := NewResponseRecorder(mockWriter, capture)

	data := []byte("data: {\"test\": \"value\"}\n\n")
	n, err := recorder.Write(data)

	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if n != len(data) {
		t.Errorf("Expected %d bytes written, got %d", len(data), n)
	}

	chunks := capture.Chunks()
	if len(chunks) != 1 {
		t.Fatalf("Expected 1 chunk, got %d", len(chunks))
	}
	if chunks[0].Event != "message" {
		t.Errorf("Expected event 'message', got '%s'", chunks[0].Event)
	}
}

func TestResponseRecorder_WriteWithEventType(t *testing.T) {
	mockWriter := newMockResponseWriter()
	capture := logging.NewCaptureWriter(time.Now())
	recorder := NewResponseRecorder(mockWriter, capture)

	data := []byte("event: custom\ndata: {\"test\": \"value\"}\n\n")
	n, err := recorder.Write(data)

	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if n != len(data) {
		t.Errorf("Expected %d bytes written, got %d", len(data), n)
	}

	chunks := capture.Chunks()
	if len(chunks) != 1 {
		t.Fatalf("Expected 1 chunk, got %d", len(chunks))
	}
	if chunks[0].Event != "custom" {
		t.Errorf("Expected event 'custom', got '%s'", chunks[0].Event)
	}
}

func TestResponseRecorder_WriteEmpty(t *testing.T) {
	mockWriter := newMockResponseWriter()
	capture := logging.NewCaptureWriter(time.Now())
	recorder := NewResponseRecorder(mockWriter, capture)

	n, err := recorder.Write([]byte{})

	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if n != 0 {
		t.Errorf("Expected 0 bytes written, got %d", n)
	}

	chunks := capture.Chunks()
	if len(chunks) != 0 {
		t.Errorf("Expected 0 chunks for empty write, got %d", len(chunks))
	}
}

func TestResponseRecorder_Header(t *testing.T) {
	mockWriter := newMockResponseWriter()
	mockWriter.header.Set("Content-Type", "text/event-stream")
	capture := logging.NewCaptureWriter(time.Now())
	recorder := NewResponseRecorder(mockWriter, capture)

	header := recorder.Header()

	if header.Get("Content-Type") != "text/event-stream" {
		t.Errorf("Expected Content-Type header, got '%s'", header.Get("Content-Type"))
	}
}

func TestResponseRecorder_WriteHeader(t *testing.T) {
	mockWriter := newMockResponseWriter()
	capture := logging.NewCaptureWriter(time.Now())
	recorder := NewResponseRecorder(mockWriter, capture)

	recorder.WriteHeader(http.StatusOK)

	if mockWriter.status != http.StatusOK {
		t.Errorf("Expected status %d, got %d", http.StatusOK, mockWriter.status)
	}
}

func TestExtractDataPart_Basic(t *testing.T) {
	data := []byte("data: {\"id\":\"test\"}\n\n")
	result := extractDataPart(data)

	expected := "{\"id\":\"test\"}"
	if string(result) != expected {
		t.Errorf("Expected '%s', got '%s'", expected, string(result))
	}
}

func TestExtractDataPart_WithEvent(t *testing.T) {
	data := []byte("event: message\ndata: {\"id\":\"test\"}\n\n")
	result := extractDataPart(data)

	expected := "{\"id\":\"test\"}"
	if string(result) != expected {
		t.Errorf("Expected '%s', got '%s'", expected, string(result))
	}
}

func TestExtractDataPart_NoDataPrefix(t *testing.T) {
	data := []byte("{\"id\":\"test\"}")
	result := extractDataPart(data)

	if string(result) != string(data) {
		t.Errorf("Expected original data when no data prefix, got '%s'", string(result))
	}
}

func TestExtractDataPart_NoDoubleNewline(t *testing.T) {
	data := []byte("data: {\"id\":\"test\"}")
	result := extractDataPart(data)

	expected := "{\"id\":\"test\"}"
	if string(result) != expected {
		t.Errorf("Expected '%s', got '%s'", expected, string(result))
	}
}

func TestExtractDataPart_DoneMessage(t *testing.T) {
	data := []byte("data: [DONE]\n\n")
	result := extractDataPart(data)

	expected := "[DONE]"
	if string(result) != expected {
		t.Errorf("Expected '%s', got '%s'", expected, string(result))
	}
}

func TestFindDataPrefix_Found(t *testing.T) {
	s := "event: test\ndata: {\"id\":1}\n\n"
	idx := findDataPrefix(s)

	if idx != 12 {
		t.Errorf("Expected index 12, got %d", idx)
	}
}

func TestFindDataPrefix_NotFound(t *testing.T) {
	s := "event: test\n{\"id\":1}"
	idx := findDataPrefix(s)

	if idx != -1 {
		t.Errorf("Expected -1 when not found, got %d", idx)
	}
}

func TestFindDataPrefix_AtStart(t *testing.T) {
	s := "data: {\"id\":1}\n\n"
	idx := findDataPrefix(s)

	if idx != 0 {
		t.Errorf("Expected index 0, got %d", idx)
	}
}

func TestIndexOfDoubleNewline_Found(t *testing.T) {
	s := "{\"id\":1}\n\n"
	idx := indexOfDoubleNewline(s)

	if idx != 8 {
		t.Errorf("Expected index 8, got %d", idx)
	}
}

func TestIndexOfDoubleNewline_NotFound(t *testing.T) {
	s := "{\"id\":1}\n"
	idx := indexOfDoubleNewline(s)

	if idx != -1 {
		t.Errorf("Expected -1 when not found, got %d", idx)
	}
}

func TestIndexOfDoubleNewline_Empty(t *testing.T) {
	s := ""
	idx := indexOfDoubleNewline(s)

	if idx != -1 {
		t.Errorf("Expected -1 for empty string, got %d", idx)
	}
}

func TestExtractEventType_Message(t *testing.T) {
	tests := []struct {
		name     string
		data     []byte
		expected string
	}{
		{"default message", []byte("data: {\"id\":1}\n\n"), "message"},
		{"custom event", []byte("event: custom\ndata: {\"id\":1}\n\n"), "custom"},
		{"event with spaces", []byte("event: content_block_start\ndata: {}\n\n"), "content_block_start"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractEventType(tt.data)
			if result != tt.expected {
				t.Errorf("Expected '%s', got '%s'", tt.expected, result)
			}
		})
	}
}

func TestExtractEventType_Empty(t *testing.T) {
	result := extractEventType([]byte{})
	if result != "message" {
		t.Errorf("Expected 'message' for empty data, got '%s'", result)
	}
}

func TestExtractEventType_NoEventField(t *testing.T) {
	data := []byte("data: {\"id\":1}\n\n")
	result := extractEventType(data)

	if result != "message" {
		t.Errorf("Expected 'message' when no event field, got '%s'", result)
	}
}
