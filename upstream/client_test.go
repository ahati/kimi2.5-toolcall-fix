package upstream

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"ai-proxy/logging"
)

func TestNewClient_CreatesClientWithFields(t *testing.T) {
	url := "https://api.example.com/v1/chat"
	apiKey := "test-api-key-12345"

	client := NewClient(url, apiKey)

	if client.URL != url {
		t.Errorf("expected URL %s, got %s", url, client.URL)
	}
	if client.APIKey != apiKey {
		t.Errorf("expected APIKey %s, got %s", apiKey, client.APIKey)
	}
	if client.Client == nil {
		t.Error("expected Client to be initialized, got nil")
	}
}

func TestBuildRequest_CreatesPostRequest(t *testing.T) {
	client := NewClient("https://api.example.com/v1/chat", "test-key")
	body := []byte(`{"model":"test","messages":[]}`)

	req, err := client.BuildRequest(context.Background(), body)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if req.Method != "POST" {
		t.Errorf("expected POST method, got %s", req.Method)
	}
	if req.URL.String() != client.URL {
		t.Errorf("expected URL %s, got %s", client.URL, req.URL.String())
	}

	reqBody, _ := io.ReadAll(req.Body)
	if !bytes.Equal(reqBody, body) {
		t.Errorf("expected body %s, got %s", string(body), string(reqBody))
	}
}

func TestBuildRequest_ReturnsErrorForInvalidURL(t *testing.T) {
	client := NewClient("://invalid-url", "test-key")
	body := []byte(`{}`)

	req, err := client.BuildRequest(context.Background(), body)

	if err == nil {
		t.Error("expected error for invalid URL, got nil")
	}
	if req != nil {
		t.Errorf("expected nil request, got %v", req)
	}
}

func TestBuildRequest_StoresRequestInCaptureContext(t *testing.T) {
	client := NewClient("https://api.example.com/v1/chat", "test-key")
	body := []byte(`{"model":"test"}`)

	cc := &logging.CaptureContext{
		StartTime: time.Now(),
		Recorder:  &logging.RequestRecorder{},
	}
	ctx := logging.WithCaptureContext(context.Background(), cc)

	_, err := client.BuildRequest(ctx, body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if cc.Recorder.UpstreamRequest == nil {
		t.Fatal("expected UpstreamRequest to be set, got nil")
	}
	if !bytes.Equal(cc.Recorder.UpstreamRequest.Body, body) {
		t.Errorf("expected body %s, got %s", string(body), string(cc.Recorder.UpstreamRequest.Body))
	}
	if !bytes.Equal(cc.Recorder.UpstreamRequest.RawBody, body) {
		t.Errorf("expected RawBody %s, got %s", string(body), string(cc.Recorder.UpstreamRequest.RawBody))
	}
}

func TestBuildRequest_NoCaptureContext(t *testing.T) {
	client := NewClient("https://api.example.com/v1/chat", "test-key")
	body := []byte(`{"model":"test"}`)

	_, err := client.BuildRequest(context.Background(), body)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestSetHeaders_SetsRequiredHeaders(t *testing.T) {
	client := NewClient("https://api.example.com/v1/chat", "secret-key")
	req := httptest.NewRequest(http.MethodPost, "/test", nil)

	client.SetHeaders(req)

	if ct := req.Header.Get("Content-Type"); ct != "application/json" {
		t.Errorf("expected Content-Type 'application/json', got %s", ct)
	}
	if auth := req.Header.Get("Authorization"); auth != "Bearer secret-key" {
		t.Errorf("expected Authorization 'Bearer secret-key', got %s", auth)
	}
	if accept := req.Header.Get("Accept"); accept != "text/event-stream" {
		t.Errorf("expected Accept 'text/event-stream', got %s", accept)
	}
}

func TestSetHeaders_StoresSanitizedHeadersInCaptureContext(t *testing.T) {
	client := NewClient("https://api.example.com/v1/chat", "secret-key")

	cc := &logging.CaptureContext{
		StartTime: time.Now(),
		Recorder: &logging.RequestRecorder{
			UpstreamRequest: &logging.HTTPRequestCapture{},
		},
	}
	ctx := logging.WithCaptureContext(context.Background(), cc)
	req := httptest.NewRequest(http.MethodPost, "/test", nil).WithContext(ctx)

	client.SetHeaders(req)

	if cc.Recorder.UpstreamRequest.Headers == nil {
		t.Fatal("expected Headers to be set, got nil")
	}
	auth, ok := cc.Recorder.UpstreamRequest.Headers["Authorization"]
	if !ok {
		t.Error("expected Authorization header to be present")
	}
	if auth != "***" {
		t.Errorf("expected sanitized Authorization '***', got %s", auth)
	}
	ct := cc.Recorder.UpstreamRequest.Headers["Content-Type"]
	if ct != "application/json" {
		t.Errorf("expected Content-Type 'application/json', got %s", ct)
	}
}

func TestSetHeaders_NoCaptureContext(t *testing.T) {
	client := NewClient("https://api.example.com/v1/chat", "secret-key")
	req := httptest.NewRequest(http.MethodPost, "/test", nil)

	client.SetHeaders(req)

	if req.Header.Get("Content-Type") != "application/json" {
		t.Error("expected Content-Type header to be set")
	}
}

func TestSetHeaders_NoUpstreamRequestInRecorder(t *testing.T) {
	client := NewClient("https://api.example.com/v1/chat", "secret-key")

	cc := &logging.CaptureContext{
		StartTime: time.Now(),
		Recorder:  &logging.RequestRecorder{},
	}
	ctx := logging.WithCaptureContext(context.Background(), cc)
	req := httptest.NewRequest(http.MethodPost, "/test", nil).WithContext(ctx)

	client.SetHeaders(req)

	if cc.Recorder.UpstreamRequest != nil {
		t.Error("expected UpstreamRequest to remain nil")
	}
}

func TestGetAPIKey_ExtractsBearerToken(t *testing.T) {
	client := NewClient("https://api.example.com/v1/chat", "default-key")

	result := client.GetAPIKey("Bearer user-provided-token")

	if result != "user-provided-token" {
		t.Errorf("expected 'user-provided-token', got %s", result)
	}
}

func TestGetAPIKey_ReturnsConfiguredKeyWhenNoBearer(t *testing.T) {
	client := NewClient("https://api.example.com/v1/chat", "default-key")

	result := client.GetAPIKey("")

	if result != "default-key" {
		t.Errorf("expected 'default-key', got %s", result)
	}
}

func TestGetAPIKey_ReturnsConfiguredKeyForNonBearerValue(t *testing.T) {
	client := NewClient("https://api.example.com/v1/chat", "default-key")

	result := client.GetAPIKey("Basic some-credentials")

	if result != "default-key" {
		t.Errorf("expected 'default-key', got %s", result)
	}
}

func TestGetAPIKey_HandlesCaseSensitiveBearer(t *testing.T) {
	client := NewClient("https://api.example.com/v1/chat", "default-key")

	result := client.GetAPIKey("bearer token")

	if result != "default-key" {
		t.Errorf("expected 'default-key' (case sensitive), got %s", result)
	}
}

func TestDo_MakesHTTPRequest(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"result":"ok"}`))
	}))
	defer server.Close()

	client := NewClient(server.URL, "test-key")
	req, _ := http.NewRequest(http.MethodPost, server.URL, strings.NewReader(`{}`))

	resp, err := client.Do(context.Background(), req)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Errorf("expected status 200, got %d", resp.StatusCode)
	}
	resp.Body.Close()
}

func TestDo_StoresResponseInCaptureContext(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`data: test`))
	}))
	defer server.Close()

	client := NewClient(server.URL, "test-key")

	cc := &logging.CaptureContext{
		StartTime: time.Now(),
		Recorder:  &logging.RequestRecorder{},
	}
	ctx := logging.WithCaptureContext(context.Background(), cc)
	req, _ := http.NewRequest(http.MethodPost, server.URL, strings.NewReader(`{}`))

	resp, err := client.Do(ctx, req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer resp.Body.Close()

	if cc.Recorder.UpstreamResponse == nil {
		t.Fatal("expected UpstreamResponse to be set, got nil")
	}
	if cc.Recorder.UpstreamResponse.StatusCode != http.StatusOK {
		t.Errorf("expected StatusCode 200, got %d", cc.Recorder.UpstreamResponse.StatusCode)
	}
	if cc.Recorder.UpstreamResponse.Headers == nil {
		t.Error("expected Headers to be set")
	}
	if cc.Recorder.UpstreamResponse.Chunks == nil {
		t.Error("expected Chunks to be initialized")
	}
}

func TestDo_ReturnsErrorOnNetworkFailure(t *testing.T) {
	client := NewClient("http://localhost:99999/nonexistent", "test-key")
	req, _ := http.NewRequest(http.MethodPost, "http://localhost:99999/nonexistent", strings.NewReader(`{}`))

	resp, err := client.Do(context.Background(), req)

	if err == nil {
		t.Error("expected error for network failure, got nil")
	}
	if resp != nil {
		t.Errorf("expected nil response, got %v", resp)
	}
}

func TestDo_NoCaptureContext(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	client := NewClient(server.URL, "test-key")
	req, _ := http.NewRequest(http.MethodPost, server.URL, strings.NewReader(`{}`))

	resp, err := client.Do(context.Background(), req)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Errorf("expected status 200, got %d", resp.StatusCode)
	}
	resp.Body.Close()
}

func TestClose_ClosesIdleConnections(t *testing.T) {
	client := NewClient("https://api.example.com/v1/chat", "test-key")

	client.Close()

	if client.Client == nil {
		t.Error("expected Client to still exist after Close")
	}
}

func TestClose_MultipleCalls(t *testing.T) {
	client := NewClient("https://api.example.com/v1/chat", "test-key")

	client.Close()
	client.Close()
}

func TestDo_WithContextCancellation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(100 * time.Millisecond)
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	client := NewClient(server.URL, "test-key")

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	req, _ := http.NewRequest(http.MethodPost, server.URL, strings.NewReader(`{}`))

	resp, err := client.Do(ctx, req)

	if err == nil {
		t.Error("expected error for cancelled context, got nil")
		if resp != nil {
			resp.Body.Close()
		}
	}
}

func TestBuildRequest_WithLargeBody(t *testing.T) {
	client := NewClient("https://api.example.com/v1/chat", "test-key")
	largeBody := make([]byte, 1024*1024)
	for i := range largeBody {
		largeBody[i] = 'a'
	}

	req, err := client.BuildRequest(context.Background(), largeBody)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	reqBody, _ := io.ReadAll(req.Body)
	if !bytes.Equal(reqBody, largeBody) {
		t.Error("body mismatch for large request")
	}
}

func TestSetHeaders_OverwritesExistingHeaders(t *testing.T) {
	client := NewClient("https://api.example.com/v1/chat", "new-key")
	req := httptest.NewRequest(http.MethodPost, "/test", nil)
	req.Header.Set("Content-Type", "text/plain")
	req.Header.Set("Authorization", "Bearer old-key")

	client.SetHeaders(req)

	if req.Header.Get("Content-Type") != "application/json" {
		t.Errorf("expected Content-Type to be overwritten")
	}
	if req.Header.Get("Authorization") != "Bearer new-key" {
		t.Errorf("expected Authorization to be overwritten")
	}
}

func TestDo_RecordsSanitizedHeadersInCaptureContext(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("X-Api-Key", "secret-value")
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	client := NewClient(server.URL, "test-key")

	cc := &logging.CaptureContext{
		StartTime: time.Now(),
		Recorder:  &logging.RequestRecorder{},
	}
	ctx := logging.WithCaptureContext(context.Background(), cc)
	req, _ := http.NewRequest(http.MethodPost, server.URL, strings.NewReader(`{}`))

	resp, err := client.Do(ctx, req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer resp.Body.Close()

	headers := cc.Recorder.UpstreamResponse.Headers
	if headers == nil {
		t.Fatal("expected headers to be captured")
	}
	if apiKey, ok := headers["X-Api-Key"]; ok {
		if apiKey != "***" {
			t.Errorf("expected X-Api-Key to be sanitized, got %s", apiKey)
		}
	}
}
