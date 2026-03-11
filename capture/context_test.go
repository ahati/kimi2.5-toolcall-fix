package capture

import (
	"context"
	"net/http"
	"net/url"
	"testing"
	"time"
)

func TestNewCaptureContext(t *testing.T) {
	req := &http.Request{
		Method:     "POST",
		URL:        &url.URL{Path: "/v1/chat/completions"},
		RemoteAddr: "192.168.1.1:12345",
	}

	cc := NewCaptureContext(req)

	if cc == nil {
		t.Fatal("NewCaptureContext returned nil")
	}

	if cc.RequestID != "" {
		t.Errorf("expected empty RequestID, got %q", cc.RequestID)
	}

	if cc.StartTime.IsZero() {
		t.Error("StartTime should be set")
	}

	if cc.Recorder == nil {
		t.Fatal("Recorder should not be nil")
	}

	if cc.IDExtracted {
		t.Error("IDExtracted should be false initially")
	}
}

func TestNewCaptureContext_RecorderFields(t *testing.T) {
	tests := []struct {
		name       string
		method     string
		path       string
		remoteAddr string
	}{
		{
			name:       "POST request",
			method:     "POST",
			path:       "/v1/chat/completions",
			remoteAddr: "10.0.0.1:8080",
		},
		{
			name:       "GET request",
			method:     "GET",
			path:       "/v1/models",
			remoteAddr: "localhost:3000",
		},
		{
			name:       "empty path",
			method:     "GET",
			path:       "/",
			remoteAddr: "127.0.0.1:443",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &http.Request{
				Method:     tt.method,
				URL:        &url.URL{Path: tt.path},
				RemoteAddr: tt.remoteAddr,
			}

			cc := NewCaptureContext(req)

			if cc.Recorder.Method != tt.method {
				t.Errorf("expected Method %q, got %q", tt.method, cc.Recorder.Method)
			}

			if cc.Recorder.Path != tt.path {
				t.Errorf("expected Path %q, got %q", tt.path, cc.Recorder.Path)
			}

			if cc.Recorder.ClientIP != tt.remoteAddr {
				t.Errorf("expected ClientIP %q, got %q", tt.remoteAddr, cc.Recorder.ClientIP)
			}
		})
	}
}

func TestCaptureContext_SetRequestID(t *testing.T) {
	req := &http.Request{
		Method:     "POST",
		URL:        &url.URL{Path: "/test"},
		RemoteAddr: "localhost:8080",
	}

	cc := NewCaptureContext(req)

	if cc.IDExtracted {
		t.Error("IDExtracted should be false initially")
	}

	cc.SetRequestID("test-request-id-123")

	if cc.RequestID != "test-request-id-123" {
		t.Errorf("expected RequestID %q, got %q", "test-request-id-123", cc.RequestID)
	}

	if cc.Recorder.RequestID != "test-request-id-123" {
		t.Errorf("expected Recorder.RequestID %q, got %q", "test-request-id-123", cc.Recorder.RequestID)
	}

	if !cc.IDExtracted {
		t.Error("IDExtracted should be true after SetRequestID")
	}
}

func TestCaptureContext_SetRequestID_Overwrite(t *testing.T) {
	req := &http.Request{
		Method:     "POST",
		URL:        &url.URL{Path: "/test"},
		RemoteAddr: "localhost:8080",
	}

	cc := NewCaptureContext(req)
	cc.SetRequestID("first-id")
	cc.SetRequestID("second-id")

	if cc.RequestID != "second-id" {
		t.Errorf("expected RequestID %q, got %q", "second-id", cc.RequestID)
	}

	if cc.Recorder.RequestID != "second-id" {
		t.Errorf("expected Recorder.RequestID %q, got %q", "second-id", cc.Recorder.RequestID)
	}
}

func TestWithCaptureContext(t *testing.T) {
	ctx := context.Background()
	req := &http.Request{
		Method:     "POST",
		URL:        &url.URL{Path: "/test"},
		RemoteAddr: "localhost:8080",
	}
	cc := NewCaptureContext(req)

	newCtx := WithCaptureContext(ctx, cc)

	if newCtx == nil {
		t.Fatal("WithCaptureContext returned nil")
	}

	retrieved := GetCaptureContext(newCtx)
	if retrieved == nil {
		t.Fatal("GetCaptureContext returned nil")
	}

	if retrieved != cc {
		t.Error("retrieved context is not the same as the original")
	}
}

func TestGetCaptureContext_NilContext(t *testing.T) {
	cc := GetCaptureContext(nil)
	if cc != nil {
		t.Error("expected nil for nil context")
	}
}

func TestGetCaptureContext_EmptyContext(t *testing.T) {
	ctx := context.Background()
	cc := GetCaptureContext(ctx)
	if cc != nil {
		t.Error("expected nil for context without capture context")
	}
}

func TestGetCaptureContext_WrongType(t *testing.T) {
	ctx := context.WithValue(context.Background(), captureContextKey, "not a capture context")
	cc := GetCaptureContext(ctx)
	if cc != nil {
		t.Error("expected nil for context with wrong value type")
	}
}

func TestWithCaptureContext_NilValue(t *testing.T) {
	ctx := context.Background()
	newCtx := WithCaptureContext(ctx, nil)

	cc := GetCaptureContext(newCtx)
	if cc != nil {
		t.Error("expected nil for context with nil capture context value")
	}
}

func TestRecordDownstreamRequest(t *testing.T) {
	ctx := context.Background()
	req := &http.Request{
		Method:     "POST",
		URL:        &url.URL{Path: "/v1/chat/completions"},
		RemoteAddr: "localhost:8080",
		Header: http.Header{
			"Content-Type": []string{"application/json"},
		},
	}
	cc := NewCaptureContext(req)
	ctx = WithCaptureContext(ctx, cc)

	body := []byte(`{"model": "test", "messages": []}`)
	RecordDownstreamRequest(ctx, req, body)

	if cc.Recorder.DownstreamRequest == nil {
		t.Fatal("DownstreamRequest should not be nil")
	}

	if string(cc.Recorder.DownstreamRequest.Body) != string(body) {
		t.Errorf("expected Body %q, got %q", body, cc.Recorder.DownstreamRequest.Body)
	}

	if cc.Recorder.DownstreamRequest.Headers["Content-Type"] != "application/json" {
		t.Error("Content-Type header should be recorded")
	}
}

func TestRecordDownstreamRequest_NoContext(t *testing.T) {
	req := &http.Request{
		Method:     "POST",
		URL:        &url.URL{Path: "/test"},
		RemoteAddr: "localhost:8080",
	}

	ctx := context.Background()
	RecordDownstreamRequest(ctx, req, []byte("test"))

}

func TestRecordDownstreamRequest_NilContext(t *testing.T) {
	RecordDownstreamRequest(nil, nil, nil)
}

func TestCaptureContext_StartTime(t *testing.T) {
	before := time.Now()
	req := &http.Request{
		Method:     "GET",
		URL:        &url.URL{Path: "/"},
		RemoteAddr: "localhost:8080",
	}
	cc := NewCaptureContext(req)
	after := time.Now()

	if cc.StartTime.Before(before) {
		t.Error("StartTime should not be before creation")
	}

	if cc.StartTime.After(after) {
		t.Error("StartTime should not be after creation")
	}

	if cc.Recorder.StartedAt.Before(before) || cc.Recorder.StartedAt.After(after) {
		t.Error("Recorder.StartedAt should be within the same time window")
	}
}

func TestCaptureContext_MultipleContexts(t *testing.T) {
	req1 := &http.Request{
		Method:     "POST",
		URL:        &url.URL{Path: "/test1"},
		RemoteAddr: "client1:8080",
	}
	req2 := &http.Request{
		Method:     "GET",
		URL:        &url.URL{Path: "/test2"},
		RemoteAddr: "client2:8080",
	}

	cc1 := NewCaptureContext(req1)
	cc2 := NewCaptureContext(req2)

	ctx1 := WithCaptureContext(context.Background(), cc1)
	ctx2 := WithCaptureContext(context.Background(), cc2)

	retrieved1 := GetCaptureContext(ctx1)
	retrieved2 := GetCaptureContext(ctx2)

	if retrieved1.Recorder.Path == retrieved2.Recorder.Path {
		t.Error("contexts should be independent")
	}

	if retrieved1.Recorder.Path != "/test1" {
		t.Errorf("expected path /test1, got %s", retrieved1.Recorder.Path)
	}

	if retrieved2.Recorder.Path != "/test2" {
		t.Errorf("expected path /test2, got %s", retrieved2.Recorder.Path)
	}
}
