package capture

import (
	"encoding/json"
	"net/http"
	"testing"
	"time"
)

func TestSanitizeHeaders(t *testing.T) {
	tests := []struct {
		name     string
		headers  http.Header
		expected map[string]string
	}{
		{
			name: "authorization header masked",
			headers: http.Header{
				"Authorization": []string{"Bearer secret-token"},
				"Content-Type":  []string{"application/json"},
			},
			expected: map[string]string{
				"Authorization": "***",
				"Content-Type":  "application/json",
			},
		},
		{
			name: "x-api-key header masked",
			headers: http.Header{
				"X-Api-Key": []string{"my-secret-key"},
				"Accept":    []string{"application/json"},
			},
			expected: map[string]string{
				"X-Api-Key": "***",
				"Accept":    "application/json",
			},
		},
		{
			name: "cookie header masked",
			headers: http.Header{
				"Cookie": []string{"session=abc123"},
			},
			expected: map[string]string{
				"Cookie": "***",
			},
		},
		{
			name: "set-cookie header masked",
			headers: http.Header{
				"Set-Cookie": []string{"session=xyz789; HttpOnly"},
			},
			expected: map[string]string{
				"Set-Cookie": "***",
			},
		},
		{
			name: "x-auth-token header masked",
			headers: http.Header{
				"X-Auth-Token": []string{"token-12345"},
			},
			expected: map[string]string{
				"X-Auth-Token": "***",
			},
		},
		{
			name: "case insensitive masking",
			headers: http.Header{
				"AUTHORIZATION": []string{"Bearer token"},
				"X-API-KEY":     []string{"key"},
			},
			expected: map[string]string{
				"AUTHORIZATION": "***",
				"X-API-KEY":     "***",
			},
		},
		{
			name: "multiple values uses first",
			headers: http.Header{
				"Accept": []string{"application/json", "text/html"},
			},
			expected: map[string]string{
				"Accept": "application/json",
			},
		},
		{
			name:     "empty headers",
			headers:  http.Header{},
			expected: map[string]string{},
		},
		{
			name: "normal headers preserved",
			headers: http.Header{
				"Content-Type": []string{"application/json"},
				"User-Agent":   []string{"test-client/1.0"},
				"X-Request-Id": []string{"req-123"},
			},
			expected: map[string]string{
				"Content-Type": "application/json",
				"User-Agent":   "test-client/1.0",
				"X-Request-Id": "req-123",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := SanitizeHeaders(tt.headers)

			if len(result) != len(tt.expected) {
				t.Fatalf("expected %d headers, got %d", len(tt.expected), len(result))
			}

			for key, expectedValue := range tt.expected {
				if result[key] != expectedValue {
					t.Errorf("header %q: expected %q, got %q", key, expectedValue, result[key])
				}
			}
		})
	}
}

func TestOffsetMS(t *testing.T) {
	tests := []struct {
		name  string
		delay time.Duration
		minMS int64
		maxMS int64
	}{
		{
			name:  "immediate",
			delay: 0,
			minMS: 0,
			maxMS: 10,
		},
		{
			name:  "small delay",
			delay: 5 * time.Millisecond,
			minMS: 5,
			maxMS: 15,
		},
		{
			name:  "medium delay",
			delay: 50 * time.Millisecond,
			minMS: 50,
			maxMS: 60,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			start := time.Now()
			if tt.delay > 0 {
				time.Sleep(tt.delay)
			}
			offset := OffsetMS(start)

			if offset < tt.minMS || offset > tt.maxMS {
				t.Errorf("offset %d not in expected range [%d, %d]", offset, tt.minMS, tt.maxMS)
			}
		})
	}
}

func TestOffsetMS_NegativeTime(t *testing.T) {
	future := time.Now().Add(1 * time.Hour)
	offset := OffsetMS(future)

	if offset > 0 {
		t.Errorf("expected negative or zero offset for future time, got %d", offset)
	}
}

func TestRequestRecorder_RecordDownstreamRequest(t *testing.T) {
	recorder := &RequestRecorder{
		StartedAt: time.Now(),
		Method:    "POST",
		Path:      "/test",
		ClientIP:  "localhost:8080",
	}

	headers := http.Header{
		"Content-Type":  []string{"application/json"},
		"Authorization": []string{"Bearer secret"},
	}
	body := []byte(`{"test": true}`)

	recorder.RecordDownstreamRequest(&http.Request{Header: headers}, body)

	if recorder.DownstreamRequest == nil {
		t.Fatal("DownstreamRequest should not be nil")
	}

	if recorder.DownstreamRequest.Headers["Authorization"] != "***" {
		t.Error("Authorization header should be masked")
	}

	if recorder.DownstreamRequest.Headers["Content-Type"] != "application/json" {
		t.Error("Content-Type header should be preserved")
	}

	if string(recorder.DownstreamRequest.Body) != string(body) {
		t.Error("Body should match")
	}

	if recorder.DownstreamRequest.At.IsZero() {
		t.Error("At should be set")
	}
}

func TestRequestRecorder_OverwriteDownstreamRequest(t *testing.T) {
	recorder := &RequestRecorder{
		StartedAt: time.Now(),
		Method:    "POST",
		Path:      "/test",
	}

	recorder.RecordDownstreamRequest(&http.Request{Header: http.Header{}}, []byte("first"))
	recorder.RecordDownstreamRequest(&http.Request{Header: http.Header{}}, []byte("second"))

	if string(recorder.DownstreamRequest.Body) != "second" {
		t.Error("DownstreamRequest should be overwritten")
	}
}

func TestHTTPRequestCapture(t *testing.T) {
	tests := []struct {
		name    string
		at      time.Time
		headers map[string]string
		body    json.RawMessage
	}{
		{
			name:    "full capture",
			at:      time.Now(),
			headers: map[string]string{"Content-Type": "application/json"},
			body:    json.RawMessage(`{"key": "value"}`),
		},
		{
			name:    "empty capture",
			at:      time.Now(),
			headers: nil,
			body:    nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			capture := &HTTPRequestCapture{
				At:      tt.at,
				Headers: tt.headers,
				Body:    tt.body,
			}

			data, err := json.Marshal(capture)
			if err != nil {
				t.Fatalf("failed to marshal: %v", err)
			}

			var parsed HTTPRequestCapture
			if err := json.Unmarshal(data, &parsed); err != nil {
				t.Fatalf("failed to unmarshal: %v", err)
			}

			if !parsed.At.Equal(tt.at) {
				t.Error("At should match")
			}
		})
	}
}

func TestHTTPRequestCapture_RawBody(t *testing.T) {
	body := []byte(`{"test": "data"}`)
	capture := &HTTPRequestCapture{
		At:      time.Now(),
		Body:    body,
		RawBody: body,
	}

	if string(capture.RawBody) != string(capture.Body) {
		t.Error("RawBody should match Body")
	}
}

func TestSSEResponseCapture(t *testing.T) {
	capture := &SSEResponseCapture{
		StatusCode: 200,
		Headers:    map[string]string{"Content-Type": "text/event-stream"},
		Chunks: []SSEChunk{
			{OffsetMS: 0, Event: "message", Data: json.RawMessage(`{"id": "1"}`)},
			{OffsetMS: 100, Event: "message", Data: json.RawMessage(`{"id": "2"}`)},
		},
	}

	data, err := json.Marshal(capture)
	if err != nil {
		t.Fatalf("failed to marshal: %v", err)
	}

	var parsed SSEResponseCapture
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	if parsed.StatusCode != 200 {
		t.Errorf("expected StatusCode 200, got %d", parsed.StatusCode)
	}

	if len(parsed.Chunks) != 2 {
		t.Errorf("expected 2 chunks, got %d", len(parsed.Chunks))
	}
}

func TestSSEResponseCapture_Empty(t *testing.T) {
	capture := &SSEResponseCapture{}

	data, err := json.Marshal(capture)
	if err != nil {
		t.Fatalf("failed to marshal: %v", err)
	}

	var parsed SSEResponseCapture
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	if parsed.StatusCode != 0 {
		t.Error("StatusCode should be 0 by default")
	}
}

func TestRequestRecorder_AllFields(t *testing.T) {
	now := time.Now()
	recorder := &RequestRecorder{
		RequestID:          "req-123",
		StartedAt:          now,
		Method:             "POST",
		Path:               "/v1/chat/completions",
		ClientIP:           "192.168.1.1:8080",
		DownstreamRequest:  &HTTPRequestCapture{At: now},
		UpstreamRequest:    &HTTPRequestCapture{At: now},
		UpstreamResponse:   &SSEResponseCapture{StatusCode: 200},
		DownstreamResponse: &SSEResponseCapture{StatusCode: 200},
	}

	if recorder.RequestID != "req-123" {
		t.Error("RequestID should match")
	}

	if recorder.Method != "POST" {
		t.Error("Method should match")
	}

	if recorder.Path != "/v1/chat/completions" {
		t.Error("Path should match")
	}

	if recorder.DownstreamRequest == nil {
		t.Error("DownstreamRequest should not be nil")
	}

	if recorder.UpstreamRequest == nil {
		t.Error("UpstreamRequest should not be nil")
	}

	if recorder.UpstreamResponse == nil {
		t.Error("UpstreamResponse should not be nil")
	}

	if recorder.DownstreamResponse == nil {
		t.Error("DownstreamResponse should not be nil")
	}
}

func TestNewRecorder(t *testing.T) {
	rec := newRecorder("req-id", "GET", "/test", "localhost:8080")

	if rec == nil {
		t.Fatal("newRecorder returned nil")
	}

	if rec.data.RequestID != "req-id" {
		t.Errorf("expected RequestID 'req-id', got %q", rec.data.RequestID)
	}

	if rec.data.Method != "GET" {
		t.Errorf("expected Method 'GET', got %q", rec.data.Method)
	}

	if rec.data.Path != "/test" {
		t.Errorf("expected Path '/test', got %q", rec.data.Path)
	}

	if rec.data.ClientIP != "localhost:8080" {
		t.Errorf("expected ClientIP 'localhost:8080', got %q", rec.data.ClientIP)
	}

	if rec.data.StartedAt.IsZero() {
		t.Error("StartedAt should be set")
	}
}

func TestRecorder_RecordDownstreamRequest(t *testing.T) {
	rec := newRecorder("req-id", "POST", "/test", "localhost")
	headers := http.Header{
		"Content-Type":  []string{"application/json"},
		"Authorization": []string{"Bearer token"},
	}
	body := []byte(`{"test": true}`)

	rec.RecordDownstreamRequest(headers, body)

	if rec.data.DownstreamRequest == nil {
		t.Fatal("DownstreamRequest should not be nil")
	}

	if rec.data.DownstreamRequest.Headers["Authorization"] != "***" {
		t.Error("Authorization should be masked")
	}
}

func TestRecorder_RecordUpstreamRequest(t *testing.T) {
	rec := newRecorder("req-id", "POST", "/test", "localhost")
	headers := http.Header{
		"X-Api-Key":    []string{"upstream-key"},
		"Content-Type": []string{"application/json"},
	}
	body := []byte(`{"model": "test"}`)

	rec.RecordUpstreamRequest(headers, body)

	if rec.data.UpstreamRequest == nil {
		t.Fatal("UpstreamRequest should not be nil")
	}

	if rec.data.UpstreamRequest.Headers["X-Api-Key"] != "***" {
		t.Error("X-Api-Key should be masked")
	}

	if rec.data.UpstreamRequest.Headers["Content-Type"] != "application/json" {
		t.Error("Content-Type should be preserved")
	}
}

func TestRecorder_RecordUpstreamResponse(t *testing.T) {
	rec := newRecorder("req-id", "POST", "/test", "localhost")
	headers := http.Header{
		"Content-Type": []string{"text/event-stream"},
	}

	rr := rec.RecordUpstreamResponse(200, headers)

	if rec.data.UpstreamResponse == nil {
		t.Fatal("UpstreamResponse should not be nil")
	}

	if rec.data.UpstreamResponse.StatusCode != 200 {
		t.Errorf("expected StatusCode 200, got %d", rec.data.UpstreamResponse.StatusCode)
	}

	if rr == nil {
		t.Fatal("responseRecorder should not be nil")
	}
}

func TestRecorder_RecordDownstreamResponse(t *testing.T) {
	rec := newRecorder("req-id", "POST", "/test", "localhost")

	rr := rec.RecordDownstreamResponse()

	if rec.data.DownstreamResponse == nil {
		t.Fatal("DownstreamResponse should not be nil")
	}

	if rr == nil {
		t.Fatal("responseRecorder should not be nil")
	}
}

func TestRecorder_Data(t *testing.T) {
	rec := newRecorder("req-id", "GET", "/test", "localhost")

	data := rec.Data()

	if data == nil {
		t.Fatal("Data() returned nil")
	}

	if data.RequestID != "req-id" {
		t.Error("RequestID should match")
	}
}

func TestResponseRecorder_RecordChunk(t *testing.T) {
	rec := newRecorder("req-id", "POST", "/test", "localhost")
	rr := rec.RecordUpstreamResponse(200, http.Header{})

	rr.RecordChunk("message", `{"id": "chunk-1", "content": "hello"}`)
	rr.RecordChunk("message", `{"id": "chunk-2", "content": "world"}`)

	if len(rec.data.UpstreamResponse.Chunks) != 2 {
		t.Fatalf("expected 2 chunks, got %d", len(rec.data.UpstreamResponse.Chunks))
	}

	if rec.data.UpstreamResponse.Chunks[0].Event != "message" {
		t.Error("Event should be 'message'")
	}
}

func TestResponseRecorder_RecordChunk_InvalidJSON(t *testing.T) {
	rec := newRecorder("req-id", "POST", "/test", "localhost")
	rr := rec.RecordUpstreamResponse(200, http.Header{})

	rr.RecordChunk("error", "not valid json")

	if len(rec.data.UpstreamResponse.Chunks) != 1 {
		t.Fatal("expected 1 chunk")
	}

	chunk := rec.data.UpstreamResponse.Chunks[0]
	if chunk.Raw != "not valid json" {
		t.Error("Raw field should contain the invalid JSON")
	}
}

func TestResponseRecorder_RecordChunk_NilReceiver(t *testing.T) {
	var rr *responseRecorder
	rr.RecordChunk("test", "data")
}

func TestResponseRecorder_RecordChunkBytes(t *testing.T) {
	rec := newRecorder("req-id", "POST", "/test", "localhost")
	rr := rec.RecordUpstreamResponse(200, http.Header{})

	rr.RecordChunkBytes("message", []byte(`{"test": true}`))

	if len(rec.data.UpstreamResponse.Chunks) != 1 {
		t.Fatal("expected 1 chunk")
	}
}
