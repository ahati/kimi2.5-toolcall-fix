package proxy

import (
	"context"
	"crypto/tls"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"ai-proxy/capture"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestNewClient_Defaults(t *testing.T) {
	client := NewClient("https://api.example.com", "test-key")

	if client.baseURL != "https://api.example.com" {
		t.Errorf("expected baseURL 'https://api.example.com', got %q", client.baseURL)
	}
	if client.apiKey != "test-key" {
		t.Errorf("expected apiKey 'test-key', got %q", client.apiKey)
	}
	if client.httpClient == nil {
		t.Error("expected httpClient to be non-nil")
	}
	if client.httpClient.Transport == nil {
		t.Error("expected Transport to be non-nil")
	}
}

func TestNewClient_WithTimeout(t *testing.T) {
	client := NewClient("https://api.example.com", "test-key", WithTimeout(5*time.Second))

	if client.httpClient.Timeout != 5*time.Second {
		t.Errorf("expected timeout 5s, got %v", client.httpClient.Timeout)
	}
}

func TestNewClient_WithTransport(t *testing.T) {
	customTransport := &http.Transport{
		MaxIdleConns: 50,
	}
	client := NewClient("https://api.example.com", "test-key", WithTransport(customTransport))

	transport, ok := client.httpClient.Transport.(*http.Transport)
	if !ok {
		t.Error("expected Transport to be *http.Transport")
	}
	if transport.MaxIdleConns != 50 {
		t.Errorf("expected MaxIdleConns 50, got %d", transport.MaxIdleConns)
	}
}

func TestNewClient_MultipleOptions(t *testing.T) {
	customTransport := &http.Transport{
		MaxIdleConns: 25,
	}
	client := NewClient(
		"https://api.example.com",
		"test-key",
		WithTimeout(10*time.Second),
		WithTransport(customTransport),
	)

	if client.httpClient.Timeout != 10*time.Second {
		t.Errorf("expected timeout 10s, got %v", client.httpClient.Timeout)
	}
	transport, ok := client.httpClient.Transport.(*http.Transport)
	if !ok {
		t.Error("expected Transport to be *http.Transport")
	}
	if transport.MaxIdleConns != 25 {
		t.Errorf("expected MaxIdleConns 25, got %d", transport.MaxIdleConns)
	}
}

func TestDefaultTransport(t *testing.T) {
	transport := defaultTransport()

	if transport.Proxy == nil {
		t.Error("expected Proxy to be set")
	}
	if transport.MaxIdleConns != 100 {
		t.Errorf("expected MaxIdleConns 100, got %d", transport.MaxIdleConns)
	}
	if transport.IdleConnTimeout != 90*time.Second {
		t.Errorf("expected IdleConnTimeout 90s, got %v", transport.IdleConnTimeout)
	}
	if transport.TLSHandshakeTimeout != 10*time.Second {
		t.Errorf("expected TLSHandshakeTimeout 10s, got %v", transport.TLSHandshakeTimeout)
	}
	if transport.ExpectContinueTimeout != 1*time.Second {
		t.Errorf("expected ExpectContinueTimeout 1s, got %v", transport.ExpectContinueTimeout)
	}
	if transport.TLSClientConfig == nil {
		t.Error("expected TLSClientConfig to be set")
	}
	if transport.TLSClientConfig.MinVersion != tls.VersionTLS12 {
		t.Errorf("expected MinVersion TLS 1.2, got %v", transport.TLSClientConfig.MinVersion)
	}
}

func TestBuildRequest(t *testing.T) {
	client := NewClient("https://api.example.com/v1/chat", "test-key")
	ctx := context.Background()
	body := []byte(`{"model":"gpt-4","message":"hello"}`)

	req, err := client.BuildRequest(ctx, body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if req.Method != "POST" {
		t.Errorf("expected method POST, got %s", req.Method)
	}
	if req.URL.String() != "https://api.example.com/v1/chat" {
		t.Errorf("expected URL 'https://api.example.com/v1/chat', got %s", req.URL.String())
	}
}

func TestBuildRequest_WithCaptureContext(t *testing.T) {
	client := NewClient("https://api.example.com/v1/chat", "test-key")
	httpReq := httptest.NewRequest("POST", "/test", nil)
	cc := capture.NewCaptureContext(httpReq)
	ctx := capture.WithCaptureContext(context.Background(), cc)
	body := []byte(`{"model":"gpt-4"}`)

	_, err := client.BuildRequest(ctx, body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if cc.Recorder.Data().UpstreamRequest == nil {
		t.Error("expected UpstreamRequest to be set in capture context")
	}
	if string(cc.Recorder.Data().UpstreamRequest.Body) != `{"model":"gpt-4"}` {
		t.Errorf("expected body to be recorded, got %s", cc.Recorder.Data().UpstreamRequest.Body)
	}
}

func TestSetHeaders(t *testing.T) {
	client := NewClient("https://api.example.com", "my-api-key")
	req := httptest.NewRequest("POST", "/test", nil)

	client.SetHeaders(req)

	if req.Header.Get("Content-Type") != "application/json" {
		t.Errorf("expected Content-Type 'application/json', got %s", req.Header.Get("Content-Type"))
	}
	if req.Header.Get("Authorization") != "Bearer my-api-key" {
		t.Errorf("expected Authorization 'Bearer my-api-key', got %s", req.Header.Get("Authorization"))
	}
	if req.Header.Get("Accept") != "text/event-stream" {
		t.Errorf("expected Accept 'text/event-stream', got %s", req.Header.Get("Accept"))
	}
}

func TestSetHeaders_WithCaptureContext(t *testing.T) {
	client := NewClient("https://api.example.com", "secret-key")
	httpReq := httptest.NewRequest("POST", "/test", nil)
	cc := capture.NewCaptureContext(httpReq)
	cc.Recorder.Data().UpstreamRequest = &capture.HTTPRequestCapture{}
	ctx := capture.WithCaptureContext(context.Background(), cc)
	req := httptest.NewRequest("POST", "/test", nil).WithContext(ctx)

	client.SetHeaders(req)

	if cc.Recorder.Data().UpstreamRequest.Headers == nil {
		t.Error("expected headers to be recorded in capture context")
	}
	if cc.Recorder.Data().UpstreamRequest.Headers["Authorization"] != "***" {
		t.Errorf("expected Authorization to be sanitized, got %v", cc.Recorder.Data().UpstreamRequest.Headers["Authorization"])
	}
}

func TestDo(t *testing.T) {
	client := NewClient("https://api.example.com/test", "test-key")
	client.httpClient.Transport = roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if r.Method != "POST" {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("expected Authorization header, got %s", r.Header.Get("Authorization"))
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader(`{"response":"ok"}`)),
			Header:     make(http.Header),
		}, nil
	})
	req, err := http.NewRequest("POST", "https://api.example.com/test", nil)
	if err != nil {
		t.Fatalf("failed to create request: %v", err)
	}
	client.SetHeaders(req)

	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("expected status 200, got %d", resp.StatusCode)
	}
}

func TestDo_WithCaptureContext(t *testing.T) {
	client := NewClient("https://api.example.com/test", "test-key")
	client.httpClient.Transport = roundTripFunc(func(r *http.Request) (*http.Response, error) {
		resp := &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader("")),
			Header:     make(http.Header),
		}
		resp.Header.Set("Content-Type", "text/event-stream")
		return resp, nil
	})
	httpReq := httptest.NewRequest("POST", "/test", nil)
	cc := capture.NewCaptureContext(httpReq)
	ctx := capture.WithCaptureContext(context.Background(), cc)
	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.example.com/test", nil)
	if err != nil {
		t.Fatalf("failed to create request: %v", err)
	}

	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer resp.Body.Close()

	if cc.Recorder.Data().UpstreamResponse == nil {
		t.Error("expected UpstreamResponse to be set in capture context")
	}
	if cc.Recorder.Data().UpstreamResponse.StatusCode != http.StatusOK {
		t.Errorf("expected status 200, got %d", cc.Recorder.Data().UpstreamResponse.StatusCode)
	}
}

func TestDo_Error(t *testing.T) {
	client := NewClient("https://api.example.com/test", "test-key")
	client.httpClient.Transport = roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return nil, fmt.Errorf("dial blocked")
	})
	client.httpClient.Timeout = 100 * time.Millisecond
	req, err := http.NewRequest("POST", "https://api.example.com/test", nil)
	if err != nil {
		t.Fatalf("failed to create request: %v", err)
	}

	_, err = client.Do(req)
	if err == nil {
		t.Error("expected error for unreachable server")
	}
	if !strings.Contains(err.Error(), "upstream request") {
		t.Errorf("expected error to contain 'upstream request', got %v", err)
	}
}

func TestClose(t *testing.T) {
	client := NewClient("https://api.example.com", "test-key")
	client.Close()
}

func TestGetAPIKey_WithBearerPrefix(t *testing.T) {
	client := NewClient("https://api.example.com", "default-key")

	result := client.GetAPIKey("Bearer custom-key")
	if result != "custom-key" {
		t.Errorf("expected 'custom-key', got %q", result)
	}
}

func TestGetAPIKey_WithoutBearerPrefix(t *testing.T) {
	client := NewClient("https://api.example.com", "default-key")

	result := client.GetAPIKey("custom-key")
	if result != "default-key" {
		t.Errorf("expected 'default-key', got %q", result)
	}
}

func TestGetAPIKey_EmptyAuth(t *testing.T) {
	client := NewClient("https://api.example.com", "default-key")

	result := client.GetAPIKey("")
	if result != "default-key" {
		t.Errorf("expected 'default-key', got %q", result)
	}
}

func TestGetAPIKey_BearerOnly(t *testing.T) {
	client := NewClient("https://api.example.com", "default-key")

	result := client.GetAPIKey("Bearer ")
	if result != "" {
		t.Errorf("expected empty string, got %q", result)
	}
}

func TestBuildRequest_InvalidURL(t *testing.T) {
	client := NewClient("://invalid-url", "test-key")
	ctx := context.Background()
	body := []byte(`{}`)

	_, err := client.BuildRequest(ctx, body)
	if err == nil {
		t.Error("expected error for invalid URL")
	}
}
