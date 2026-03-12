package downstream

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"ai-proxy/logging"

	"github.com/tmaxmax/go-sse"
)

func TestNewSSECaptureReader(t *testing.T) {
	reader := strings.NewReader("test")
	capture := logging.NewCaptureWriter(time.Now())

	scr := NewSSECaptureReader(reader, capture)

	if scr == nil {
		t.Fatal("Expected reader to be created")
	}
}

func TestSSECaptureReader_ForEach(t *testing.T) {
	sseData := "event: message\ndata: {\"id\":\"1\"}\n\nevent: message\ndata: {\"id\":\"2\"}\n\n"
	reader := strings.NewReader(sseData)
	capture := logging.NewCaptureWriter(time.Now())
	scr := NewSSECaptureReader(reader, capture)

	count := 0
	err := scr.ForEach(func(ev sse.Event) bool {
		count++
		return true
	})

	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if count != 2 {
		t.Errorf("Expected 2 events, got %d", count)
	}

	chunks := capture.Chunks()
	if len(chunks) != 2 {
		t.Errorf("Expected 2 chunks captured, got %d", len(chunks))
	}
}

func TestSSECaptureReader_ForEach_StopEarly(t *testing.T) {
	sseData := "event: message\ndata: {\"id\":\"1\"}\n\nevent: message\ndata: {\"id\":\"2\"}\n\n"
	reader := strings.NewReader(sseData)
	capture := logging.NewCaptureWriter(time.Now())
	scr := NewSSECaptureReader(reader, capture)

	count := 0
	err := scr.ForEach(func(ev sse.Event) bool {
		count++
		return count < 1
	})

	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if count != 1 {
		t.Errorf("Expected 1 event processed, got %d", count)
	}
}

func TestSSECaptureReader_ForEach_EmptyData(t *testing.T) {
	sseData := "event: message\ndata: \n\n"
	reader := strings.NewReader(sseData)
	capture := logging.NewCaptureWriter(time.Now())
	scr := NewSSECaptureReader(reader, capture)

	count := 0
	err := scr.ForEach(func(ev sse.Event) bool {
		count++
		return true
	})

	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	chunks := capture.Chunks()
	if len(chunks) != 0 {
		t.Errorf("Expected 0 chunks for empty data, got %d", len(chunks))
	}
}

func TestSSECaptureReader_ForEach_CapturesEventType(t *testing.T) {
	sseData := "event: custom\ndata: {\"id\":\"1\"}\n\n"
	reader := strings.NewReader(sseData)
	capture := logging.NewCaptureWriter(time.Now())
	scr := NewSSECaptureReader(reader, capture)

	_ = scr.ForEach(func(ev sse.Event) bool {
		return true
	})

	chunks := capture.Chunks()
	if len(chunks) != 1 {
		t.Fatalf("Expected 1 chunk, got %d", len(chunks))
	}
	if chunks[0].Event != "custom" {
		t.Errorf("Expected event 'custom', got '%s'", chunks[0].Event)
	}
}

func TestSSECaptureReader_ForEach_Done(t *testing.T) {
	sseData := "data: [DONE]\n\n"
	reader := strings.NewReader(sseData)
	capture := logging.NewCaptureWriter(time.Now())
	scr := NewSSECaptureReader(reader, capture)

	count := 0
	err := scr.ForEach(func(ev sse.Event) bool {
		count++
		return true
	})

	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if count != 1 {
		t.Errorf("Expected 1 event, got %d", count)
	}

	chunks := capture.Chunks()
	if len(chunks) != 1 {
		t.Fatalf("Expected 1 chunk, got %d", len(chunks))
	}
	if chunks[0].Raw != "[DONE]" {
		t.Errorf("Expected '[DONE]' in Raw, got '%s'", chunks[0].Raw)
	}
}

func TestSSECaptureReader_ForEach_InvalidReader(t *testing.T) {
	reader := &errorReader{err: bytes.ErrTooLarge}
	capture := logging.NewCaptureWriter(time.Now())
	scr := NewSSECaptureReader(reader, capture)

	err := scr.ForEach(func(ev sse.Event) bool {
		return true
	})

	if err == nil {
		t.Error("Expected error from invalid reader")
	}
}

type errorReader struct {
	err error
}

func (r *errorReader) Read(p []byte) (n int, err error) {
	return 0, r.err
}
