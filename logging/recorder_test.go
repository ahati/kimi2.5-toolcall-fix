package logging

import (
	"net/http"
	"testing"
	"time"
)

func TestNewRecorder(t *testing.T) {
	r := newRecorder("req-123", "POST", "/v1/chat", "192.168.1.1:1234")

	if r.data.RequestID != "req-123" {
		t.Errorf("Expected RequestID req-123, got %s", r.data.RequestID)
	}
	if r.data.Method != "POST" {
		t.Errorf("Expected Method POST, got %s", r.data.Method)
	}
	if r.data.Path != "/v1/chat" {
		t.Errorf("Expected Path /v1/chat, got %s", r.data.Path)
	}
	if r.data.ClientIP != "192.168.1.1:1234" {
		t.Errorf("Expected ClientIP 192.168.1.1:1234, got %s", r.data.ClientIP)
	}
	if r.data.StartedAt.IsZero() {
		t.Error("StartedAt should not be zero")
	}
}

func TestRecorder_RecordDownstreamRequest(t *testing.T) {
	r := newRecorder("test-req", "GET", "/test", "localhost")

	headers := http.Header{}
	headers.Set("Content-Type", "application/json")
	headers.Set("Authorization", "Bearer secret")

	body := []byte(`{"test":"data"}`)
	r.RecordDownstreamRequest(headers, body)

	if r.data.DownstreamRequest == nil {
		t.Fatal("DownstreamRequest should not be nil")
	}
	if r.data.DownstreamRequest.Headers["Authorization"] != "***" {
		t.Errorf("Authorization should be masked, got %s", r.data.DownstreamRequest.Headers["Authorization"])
	}
	if r.data.DownstreamRequest.Headers["Content-Type"] != "application/json" {
		t.Errorf("Content-Type should not be masked, got %s", r.data.DownstreamRequest.Headers["Content-Type"])
	}
	if string(r.data.DownstreamRequest.Body) != `{"test":"data"}` {
		t.Errorf("Body mismatch: %s", string(r.data.DownstreamRequest.Body))
	}
	if r.data.DownstreamRequest.At.IsZero() {
		t.Error("At timestamp should not be zero")
	}
}

func TestRecorder_RecordUpstreamRequest(t *testing.T) {
	r := newRecorder("test-req", "POST", "/upstream", "localhost")

	headers := http.Header{}
	headers.Set("X-Api-Key", "my-api-key")

	body := []byte(`{"model":"gpt-4"}`)
	r.RecordUpstreamRequest(headers, body)

	if r.data.UpstreamRequest == nil {
		t.Fatal("UpstreamRequest should not be nil")
	}
	if r.data.UpstreamRequest.Headers["X-Api-Key"] != "***" {
		t.Errorf("X-Api-Key should be masked, got %s", r.data.UpstreamRequest.Headers["X-Api-Key"])
	}
	if string(r.data.UpstreamRequest.Body) != `{"model":"gpt-4"}` {
		t.Errorf("Body mismatch: %s", string(r.data.UpstreamRequest.Body))
	}
}

func TestRecorder_RecordUpstreamResponse(t *testing.T) {
	r := newRecorder("test-req", "POST", "/test", "localhost")

	headers := http.Header{}
	headers.Set("Content-Type", "text/event-stream")

	rr := r.RecordUpstreamResponse(200, headers)

	if r.data.UpstreamResponse == nil {
		t.Fatal("UpstreamResponse should not be nil")
	}
	if r.data.UpstreamResponse.StatusCode != 200 {
		t.Errorf("Expected StatusCode 200, got %d", r.data.UpstreamResponse.StatusCode)
	}
	if r.data.UpstreamResponse.Headers["Content-Type"] != "text/event-stream" {
		t.Errorf("Content-Type mismatch: %s", r.data.UpstreamResponse.Headers["Content-Type"])
	}
	if rr == nil {
		t.Fatal("responseRecorder should not be nil")
	}
	if rr.capture != r.data.UpstreamResponse {
		t.Error("responseRecorder should reference UpstreamResponse")
	}
}

func TestRecorder_RecordDownstreamResponse(t *testing.T) {
	r := newRecorder("test-req", "POST", "/test", "localhost")

	rr := r.RecordDownstreamResponse()

	if r.data.DownstreamResponse == nil {
		t.Fatal("DownstreamResponse should not be nil")
	}
	if r.data.DownstreamResponse.Chunks == nil {
		t.Error("Chunks should be initialized")
	}
	if rr == nil {
		t.Fatal("responseRecorder should not be nil")
	}
	if rr.capture != r.data.DownstreamResponse {
		t.Error("responseRecorder should reference DownstreamResponse")
	}
}

func TestRecorder_Data(t *testing.T) {
	r := newRecorder("test-req", "GET", "/data", "localhost")

	data := r.Data()

	if data == nil {
		t.Fatal("Data should not return nil")
	}
	if data.RequestID != "test-req" {
		t.Errorf("Expected RequestID test-req, got %s", data.RequestID)
	}
}

func TestResponseRecorder_RecordChunk(t *testing.T) {
	r := newRecorder("test-req", "POST", "/test", "localhost")
	rr := r.RecordUpstreamResponse(200, http.Header{})

	rr.RecordChunk("message", `{"id":"test-id","choices":[{"delta":{"content":"hello"}}]}`)

	if len(r.data.UpstreamResponse.Chunks) != 1 {
		t.Fatalf("Expected 1 chunk, got %d", len(r.data.UpstreamResponse.Chunks))
	}

	chunk := r.data.UpstreamResponse.Chunks[0]
	if chunk.Event != "message" {
		t.Errorf("Expected Event 'message', got %s", chunk.Event)
	}
	if chunk.Raw != `{"id":"test-id","choices":[{"delta":{"content":"hello"}}]}` {
		t.Errorf("Raw mismatch: %s", chunk.Raw)
	}
	if string(chunk.Data) != `{"id":"test-id","choices":[{"delta":{"content":"hello"}}]}` {
		t.Errorf("Data mismatch: %s", string(chunk.Data))
	}
	if chunk.OffsetMS < 0 {
		t.Errorf("OffsetMS should be >= 0, got %d", chunk.OffsetMS)
	}
}

func TestResponseRecorder_RecordChunk_InvalidJSON(t *testing.T) {
	r := newRecorder("test-req", "POST", "/test", "localhost")
	rr := r.RecordUpstreamResponse(200, http.Header{})

	rr.RecordChunk("message", `not valid json`)

	if len(r.data.UpstreamResponse.Chunks) != 1 {
		t.Fatalf("Expected 1 chunk, got %d", len(r.data.UpstreamResponse.Chunks))
	}

	chunk := r.data.UpstreamResponse.Chunks[0]
	if chunk.Raw != "not valid json" {
		t.Errorf("Raw mismatch: %s", chunk.Raw)
	}
	if chunk.Data != nil {
		t.Errorf("Data should be nil for invalid JSON, got %s", string(chunk.Data))
	}
}

func TestResponseRecorder_RecordChunk_NilRecorder(t *testing.T) {
	var rr *responseRecorder
	rr.RecordChunk("message", "test")
}

func TestResponseRecorder_RecordChunk_NilCapture(t *testing.T) {
	rr := &responseRecorder{capture: nil, started: time.Now()}
	rr.RecordChunk("message", "test")
}

func TestResponseRecorder_RecordChunkBytes(t *testing.T) {
	r := newRecorder("test-req", "POST", "/test", "localhost")
	rr := r.RecordUpstreamResponse(200, http.Header{})

	data := []byte(`{"id":"byte-test"}`)
	rr.RecordChunkBytes("message", data)

	if len(r.data.UpstreamResponse.Chunks) != 1 {
		t.Fatalf("Expected 1 chunk, got %d", len(r.data.UpstreamResponse.Chunks))
	}

	chunk := r.data.UpstreamResponse.Chunks[0]
	if chunk.Raw != `{"id":"byte-test"}` {
		t.Errorf("Raw mismatch: %s", chunk.Raw)
	}
}

func TestRecorder_MultipleChunks(t *testing.T) {
	r := newRecorder("test-req", "POST", "/test", "localhost")
	rr := r.RecordUpstreamResponse(200, http.Header{})

	rr.RecordChunk("message", `{"chunk":1}`)
	rr.RecordChunk("message", `{"chunk":2}`)
	rr.RecordChunk("done", `[DONE]`)

	if len(r.data.UpstreamResponse.Chunks) != 3 {
		t.Fatalf("Expected 3 chunks, got %d", len(r.data.UpstreamResponse.Chunks))
	}

	for i, chunk := range r.data.UpstreamResponse.Chunks {
		if i < 2 && chunk.Event != "message" {
			t.Errorf("Chunk %d: Expected Event 'message', got %s", i, chunk.Event)
		}
		if i == 2 && chunk.Event != "done" {
			t.Errorf("Chunk %d: Expected Event 'done', got %s", i, chunk.Event)
		}
	}
}

func TestRecorder_Concurrency(t *testing.T) {
	r := newRecorder("concurrent-req", "POST", "/test", "localhost")

	done := make(chan bool)

	for i := 0; i < 10; i++ {
		go func(id int) {
			headers := http.Header{}
			headers.Set("X-Test", "test")
			r.RecordDownstreamRequest(headers, []byte(`{"id":id}`))
			r.RecordUpstreamRequest(headers, []byte(`{"id":id}`))
			done <- true
		}(i)
	}

	for i := 0; i < 10; i++ {
		<-done
	}

	data := r.Data()
	if data.DownstreamRequest == nil {
		t.Error("DownstreamRequest should not be nil after concurrent writes")
	}
	if data.UpstreamRequest == nil {
		t.Error("UpstreamRequest should not be nil after concurrent writes")
	}
}
