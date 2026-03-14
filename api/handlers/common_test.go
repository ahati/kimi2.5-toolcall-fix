package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"ai-proxy/capture"
	"ai-proxy/config"
	"ai-proxy/transform"

	"github.com/gin-gonic/gin"
	"github.com/tmaxmax/go-sse"
)

func init() {
	gin.SetMode(gin.TestMode)
}

type mockResponseWriter struct {
	*httptest.ResponseRecorder
}

func (m *mockResponseWriter) CloseNotify() <-chan bool {
	return make(chan bool)
}

func (m *mockResponseWriter) Flush() {}

func newMockResponseWriter() *mockResponseWriter {
	return &mockResponseWriter{
		ResponseRecorder: httptest.NewRecorder(),
	}
}

func TestReadBody(t *testing.T) {
	tests := []struct {
		name        string
		body        string
		expected    []byte
		expectError bool
	}{
		{
			name:        "empty body",
			body:        "",
			expected:    []byte{},
			expectError: false,
		},
		{
			name:        "simple body",
			body:        "hello world",
			expected:    []byte("hello world"),
			expectError: false,
		},
		{
			name:        "json body",
			body:        `{"key": "value"}`,
			expected:    []byte(`{"key": "value"}`),
			expectError: false,
		},
		{
			name:        "multiline body",
			body:        "line1\nline2\nline3",
			expected:    []byte("line1\nline2\nline3"),
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			c.Request = httptest.NewRequest(http.MethodPost, "/", bytes.NewBufferString(tt.body))

			result, err := readBody(c)

			if tt.expectError {
				if err == nil {
					t.Error("expected error, got nil")
				}
			} else {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				if !bytes.Equal(result, tt.expected) {
					t.Errorf("expected %q, got %q", tt.expected, result)
				}
			}
		})
	}
}

func TestValidateStreaming(t *testing.T) {
	tests := []struct {
		name        string
		body        string
		expectError bool
		errorMsg    string
	}{
		{
			name:        "streaming enabled",
			body:        `{"stream": true}`,
			expectError: false,
		},
		{
			name:        "streaming disabled",
			body:        `{"stream": false}`,
			expectError: true,
			errorMsg:    "non-streaming requests not supported",
		},
		{
			name:        "streaming default false",
			body:        `{"model": "test"}`,
			expectError: true,
			errorMsg:    "non-streaming requests not supported",
		},
		{
			name:        "invalid json",
			body:        `{invalid}`,
			expectError: true,
			errorMsg:    "invalid JSON",
		},
		{
			name:        "empty object streaming false",
			body:        `{}`,
			expectError: true,
			errorMsg:    "non-streaming requests not supported",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateStreaming([]byte(tt.body))

			if tt.expectError {
				if err == nil {
					t.Error("expected error, got nil")
				} else if tt.errorMsg != "" && err.Error() != tt.errorMsg && err.Error()[:len(tt.errorMsg)] != tt.errorMsg {
					t.Errorf("expected error containing %q, got %q", tt.errorMsg, err.Error())
				}
			} else {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
			}
		})
	}
}

func TestSetStreamHeaders(t *testing.T) {
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	setStreamHeaders(c)

	headers := map[string]string{
		"Content-Type":      "text/event-stream",
		"Cache-Control":     "no-cache",
		"Connection":        "keep-alive",
		"X-Accel-Buffering": "no",
	}

	for key, expected := range headers {
		actual := c.Writer.Header().Get(key)
		if actual != expected {
			t.Errorf("header %s: expected %q, got %q", key, expected, actual)
		}
	}
}

func TestSendOpenAIError(t *testing.T) {
	tests := []struct {
		name   string
		status int
		msg    string
	}{
		{
			name:   "bad request",
			status: http.StatusBadRequest,
			msg:    "Invalid request",
		},
		{
			name:   "internal server error",
			status: http.StatusInternalServerError,
			msg:    "Something went wrong",
		},
		{
			name:   "bad gateway",
			status: http.StatusBadGateway,
			msg:    "Upstream error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)

			sendOpenAIError(c, tt.status, tt.msg)

			if w.Code != tt.status {
				t.Errorf("expected status %d, got %d", tt.status, w.Code)
			}

			var response map[string]interface{}
			if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
				t.Fatalf("failed to parse response: %v", err)
			}

			errObj, ok := response["error"].(map[string]interface{})
			if !ok {
				t.Fatal("expected error object in response")
			}

			if errObj["message"] != tt.msg {
				t.Errorf("expected message %q, got %q", tt.msg, errObj["message"])
			}

			if errObj["type"] != "invalid_request_error" {
				t.Errorf("expected type %q, got %q", "invalid_request_error", errObj["type"])
			}
		})
	}
}

func TestSendAnthropicError(t *testing.T) {
	tests := []struct {
		name   string
		status int
		msg    string
	}{
		{
			name:   "bad request",
			status: http.StatusBadRequest,
			msg:    "Invalid request",
		},
		{
			name:   "internal server error",
			status: http.StatusInternalServerError,
			msg:    "Something went wrong",
		},
		{
			name:   "bad gateway",
			status: http.StatusBadGateway,
			msg:    "Upstream error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)

			sendAnthropicError(c, tt.status, tt.msg)

			if w.Code != tt.status {
				t.Errorf("expected status %d, got %d", tt.status, w.Code)
			}

			var response map[string]interface{}
			if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
				t.Fatalf("failed to parse response: %v", err)
			}

			if response["type"] != "error" {
				t.Errorf("expected type %q, got %q", "error", response["type"])
			}

			errObj, ok := response["error"].(map[string]interface{})
			if !ok {
				t.Fatal("expected error object in response")
			}

			if errObj["message"] != tt.msg {
				t.Errorf("expected message %q, got %q", tt.msg, errObj["message"])
			}

			if errObj["type"] != "invalid_request_error" {
				t.Errorf("expected type %q, got %q", "invalid_request_error", errObj["type"])
			}
		})
	}
}

func TestHandleUpstreamError(t *testing.T) {
	tests := []struct {
		name         string
		upstreamBody string
		statusCode   int
	}{
		{
			name:         "error with body",
			upstreamBody: "connection refused",
			statusCode:   http.StatusServiceUnavailable,
		},
		{
			name:         "error with empty body",
			upstreamBody: "",
			statusCode:   http.StatusInternalServerError,
		},
		{
			name:         "error with json body",
			upstreamBody: `{"error": "rate limit exceeded"}`,
			statusCode:   http.StatusTooManyRequests,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)

			resp := &http.Response{
				StatusCode: tt.statusCode,
				Body:       io.NopCloser(bytes.NewBufferString(tt.upstreamBody)),
			}

			handleUpstreamError(c, resp)

			if w.Code != http.StatusBadGateway {
				t.Errorf("expected status %d, got %d", http.StatusBadGateway, w.Code)
			}

			var response map[string]interface{}
			if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
				t.Fatalf("failed to parse response: %v", err)
			}

			errObj, ok := response["error"].(map[string]interface{})
			if !ok {
				t.Fatal("expected error object in response")
			}

			expectedMsg := "Upstream error: " + tt.upstreamBody
			if errObj["message"] != expectedMsg {
				t.Errorf("expected message %q, got %q", expectedMsg, errObj["message"])
			}
		})
	}
}

type mockHandler struct {
	validateErr      error
	transformErr     error
	transformedBody  []byte
	upstreamURL      string
	apiKey           string
	forwardHeadersFn func(c *gin.Context, req *http.Request)
	writeErrorFn     func(c *gin.Context, status int, msg string)
}

func (m *mockHandler) ValidateRequest(body []byte) error {
	return m.validateErr
}

func (m *mockHandler) TransformRequest(body []byte) ([]byte, error) {
	if m.transformErr != nil {
		return nil, m.transformErr
	}
	if m.transformedBody != nil {
		return m.transformedBody, nil
	}
	return body, nil
}

func (m *mockHandler) UpstreamURL() string {
	return m.upstreamURL
}

func (m *mockHandler) ResolveAPIKey(c *gin.Context) string {
	return m.apiKey
}

func (m *mockHandler) ForwardHeaders(c *gin.Context, req *http.Request) {
	if m.forwardHeadersFn != nil {
		m.forwardHeadersFn(c, req)
	}
}

func (m *mockHandler) CreateTransformer(w io.Writer) transform.SSETransformer {
	return &mockTransformer{w: w}
}

func (m *mockHandler) WriteError(c *gin.Context, status int, msg string) {
	if m.writeErrorFn != nil {
		m.writeErrorFn(c, status, msg)
	} else {
		sendOpenAIError(c, status, msg)
	}
}

type mockTransformer struct {
	w      io.Writer
	closed bool
}

func (m *mockTransformer) Transform(event *sse.Event) error {
	return nil
}

func (m *mockTransformer) Flush() error {
	return nil
}

func (m *mockTransformer) Close() error {
	m.closed = true
	return nil
}

func TestHandle_ValidateRequestError(t *testing.T) {
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/", bytes.NewBufferString(`{"test": "data"}`))

	h := &mockHandler{
		validateErr: http.ErrHandlerTimeout,
		upstreamURL: "https://example.com",
		apiKey:      "test-key",
	}

	Handle(h)(c)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected status %d, got %d", http.StatusBadRequest, w.Code)
	}
}

func TestHandle_TransformRequestError(t *testing.T) {
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/", bytes.NewBufferString(`{"test": "data"}`))

	h := &mockHandler{
		transformErr: http.ErrHandlerTimeout,
		upstreamURL:  "https://example.com",
		apiKey:       "test-key",
	}

	Handle(h)(c)

	if w.Code != http.StatusInternalServerError {
		t.Errorf("expected status %d, got %d", http.StatusInternalServerError, w.Code)
	}
}

func TestHandle_UpstreamError(t *testing.T) {
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/", bytes.NewBufferString(`{"test": "data"}`))

	h := &mockHandler{
		upstreamURL: "http://127.0.0.1:1", // invalid URL to force error
		apiKey:      "test-key",
	}

	Handle(h)(c)

	if w.Code != http.StatusBadGateway {
		t.Errorf("expected status %d, got %d", http.StatusBadGateway, w.Code)
	}
}

func TestHandle_UpstreamNon200Status(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		w.Write([]byte("rate limit exceeded"))
	}))
	defer upstream.Close()

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/", bytes.NewBufferString(`{"stream": true}`))

	h := &mockHandler{
		upstreamURL: upstream.URL,
		apiKey:      "test-key",
	}

	Handle(h)(c)

	if w.Code != http.StatusBadGateway {
		t.Errorf("expected status %d, got %d", http.StatusBadGateway, w.Code)
	}
}

func TestRecordUpstreamEvent(t *testing.T) {
	tests := []struct {
		name        string
		event       sse.Event
		expected    string
		expectChunk bool
		isRaw       bool
	}{
		{
			name:        "event with data",
			event:       sse.Event{Type: "message", Data: `{"test": "value"}`},
			expected:    `{"test": "value"}`,
			expectChunk: true,
			isRaw:       false,
		},
		{
			name:        "event with empty data",
			event:       sse.Event{Type: "message", Data: ""},
			expected:    "",
			expectChunk: false,
			isRaw:       false,
		},
		{
			name:        "event with done - not JSON",
			event:       sse.Event{Type: "done", Data: "[DONE]"},
			expected:    "[DONE]",
			expectChunk: true,
			isRaw:       true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cw := capture.NewCaptureWriter(time.Now())
			recordUpstreamEvent(cw, tt.event)

			chunks := cw.Chunks()
			if tt.expectChunk {
				if len(chunks) != 1 {
					t.Errorf("expected 1 chunk, got %d", len(chunks))
				} else {
					if tt.isRaw {
						if chunks[0].Raw != tt.expected {
							t.Errorf("expected raw %q, got %q", tt.expected, chunks[0].Raw)
						}
					} else {
						if string(chunks[0].Data) != tt.expected {
							t.Errorf("expected data %q, got %q", tt.expected, chunks[0].Data)
						}
					}
				}
			} else {
				if len(chunks) != 0 {
					t.Errorf("expected no chunks, got %d", len(chunks))
				}
			}
		})
	}
}

func TestFinalizeCapture(t *testing.T) {
	t.Run("basic finalize", func(t *testing.T) {
		startTime := time.Now()
		cc := &capture.CaptureContext{
			StartTime:   startTime,
			Recorder:    capture.NewRecorder("", "POST", "/test", "localhost:8080"),
			IDExtracted: false,
		}
		// Initialize upstream response before finalize (simulates proxy/client.go behavior)
		cc.Recorder.RecordUpstreamResponse(200, nil)
		downstream := capture.NewCaptureWriter(startTime)
		upstream := capture.NewCaptureWriter(startTime)

		downstream.RecordChunk("message", []byte(`{"test": "value"}`))
		upstream.RecordChunk("message", []byte(`{"id": "req-123"}`))

		finalizeCapture(cc, downstream, upstream)

		if cc.Recorder.Data().DownstreamResponse == nil {
			t.Error("expected DownstreamResponse to be set")
		}
		if cc.Recorder.Data().UpstreamResponse == nil {
			t.Error("expected UpstreamResponse to be set")
		}
		if len(cc.Recorder.Data().UpstreamResponse.Chunks) != 1 {
			t.Errorf("expected 1 upstream chunk, got %d", len(cc.Recorder.Data().UpstreamResponse.Chunks))
		}
	})

	t.Run("extract request ID from chunk", func(t *testing.T) {
		startTime := time.Now()
		cc := &capture.CaptureContext{
			StartTime:   startTime,
			Recorder:    capture.NewRecorder("", "POST", "/test", "localhost:8080"),
			IDExtracted: false,
		}
		// Initialize upstream response
		cc.Recorder.RecordUpstreamResponse(200, nil)
		downstream := capture.NewCaptureWriter(startTime)
		upstream := capture.NewCaptureWriter(startTime)

		downstream.RecordChunk("message", []byte(`{"id": "req-456"}`))

		finalizeCapture(cc, downstream, upstream)

		if cc.RequestID != "req-456" {
			t.Errorf("expected RequestID %q, got %q", "req-456", cc.RequestID)
		}
	})

	t.Run("skip ID extraction if already extracted", func(t *testing.T) {
		startTime := time.Now()
		cc := &capture.CaptureContext{
			StartTime:   startTime,
			Recorder:    capture.NewRecorder("", "POST", "/test", "localhost:8080"),
			IDExtracted: true,
			RequestID:   "existing-id",
		}
		// Initialize upstream response
		cc.Recorder.RecordUpstreamResponse(200, nil)
		downstream := capture.NewCaptureWriter(startTime)
		upstream := capture.NewCaptureWriter(startTime)

		downstream.RecordChunk("message", []byte(`{"id": "new-id"}`))

		finalizeCapture(cc, downstream, upstream)

		if cc.RequestID != "existing-id" {
			t.Errorf("expected RequestID to remain %q, got %q", "existing-id", cc.RequestID)
		}
	})
}

func TestProxyRequest_BadUpstreamURL(t *testing.T) {
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/", bytes.NewBufferString(`{"stream": true}`))

	h := &mockHandler{
		upstreamURL: "://invalid-url",
		apiKey:      "test-key",
	}

	proxyRequest(c, h, []byte(`{"test": "data"}`))

	if w.Code != http.StatusInternalServerError {
		t.Errorf("expected status %d, got %d", http.StatusInternalServerError, w.Code)
	}
}

func TestNewCompletionsHandler(t *testing.T) {
	cfg := &config.Config{
		AppConfig: &config.Schema{
			Providers: []config.Provider{
				{
					Name:    "openai",
					Type:    "openai",
					BaseURL: "https://api.example.com/v1/chat/completions",
					APIKey:  "test-key",
				},
			},
		},
	}

	handler := NewCompletionsHandler(cfg, nil)
	if handler == nil {
		t.Error("expected non-nil handler")
	}
}

func TestNewMessagesHandler(t *testing.T) {
	cfg := &config.Config{
		AppConfig: &config.Schema{
			Providers: []config.Provider{
				{
					Name:    "anthropic",
					Type:    "anthropic",
					BaseURL: "https://api.anthropic.com/v1/messages",
					APIKey:  "test-key",
				},
			},
		},
	}

	handler := NewMessagesHandler(cfg, nil)
	if handler == nil {
		t.Error("expected non-nil handler")
	}
}

func TestNewModelsHandler(t *testing.T) {
	cfg := &config.Config{
		AppConfig: &config.Schema{
			Providers: []config.Provider{
				{
					Name:    "openai",
					Type:    "openai",
					BaseURL: "https://api.example.com/v1/chat/completions",
					APIKey:  "test-key",
				},
			},
		},
	}

	handler := NewModelsHandler(cfg)
	if handler == nil {
		t.Error("expected non-nil handler")
	}
}

func TestGetCaptureContext_NilContext(t *testing.T) {
	result := capture.GetCaptureContext(nil)
	if result != nil {
		t.Error("expected nil result for nil context")
	}
}

func TestGetCaptureContext_NoCaptureContext(t *testing.T) {
	ctx := context.Background()
	result := capture.GetCaptureContext(ctx)
	if result != nil {
		t.Error("expected nil result for context without capture context")
	}
}

func TestRecordDownstreamRequest_NoCaptureContext(t *testing.T) {
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/", bytes.NewBufferString(`{"test": "data"}`))

	capture.RecordDownstreamRequest(c.Request.Context(), c.Request, []byte(`{"test": "data"}`))
}

func TestHandle_WriteErrorCalled(t *testing.T) {
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/", bytes.NewBufferString(`{"test": "data"}`))

	writeErrorCalled := false
	h := &mockHandler{
		validateErr: http.ErrHandlerTimeout,
		upstreamURL: "https://example.com",
		apiKey:      "test-key",
		writeErrorFn: func(c *gin.Context, status int, msg string) {
			writeErrorCalled = true
			sendOpenAIError(c, status, msg)
		},
	}

	Handle(h)(c)

	if !writeErrorCalled {
		t.Error("expected WriteError to be called")
	}
}

func TestStreamResponse(t *testing.T) {
	sseData := "data: {\"id\":\"test\"}\n\ndata: [DONE]\n\n"
	reader := bytes.NewReader([]byte(sseData))

	mockWriter := newMockResponseWriter()
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(mockWriter)
	c.Request = httptest.NewRequest(http.MethodPost, "/", nil)

	h := &mockHandler{}

	streamResponse(c, reader, h)

	headers := map[string]string{
		"Content-Type":      "text/event-stream",
		"Cache-Control":     "no-cache",
		"Connection":        "keep-alive",
		"X-Accel-Buffering": "no",
	}

	for key, expected := range headers {
		actual := c.Writer.Header().Get(key)
		if actual != expected {
			t.Errorf("header %s: expected %q, got %q", key, expected, actual)
		}
	}

	_ = w
}

func TestStreamWithoutCapture(t *testing.T) {
	sseData := "data: {\"id\":\"test\"}\n\ndata: [DONE]\n\n"
	reader := bytes.NewReader([]byte(sseData))

	mockWriter := newMockResponseWriter()
	c, _ := gin.CreateTestContext(mockWriter)
	c.Request = httptest.NewRequest(http.MethodPost, "/", nil)

	h := &mockHandler{}

	streamWithoutCapture(c, reader, h)

	headers := map[string]string{
		"Content-Type":      "text/event-stream",
		"Cache-Control":     "no-cache",
		"Connection":        "keep-alive",
		"X-Accel-Buffering": "no",
	}

	for key, expected := range headers {
		actual := c.Writer.Header().Get(key)
		if actual != expected {
			t.Errorf("header %s: expected %q, got %q", key, expected, actual)
		}
	}
}

func TestStreamWithCapture(t *testing.T) {
	sseData := "event: message\ndata: {\"id\":\"test-123\"}\n\ndata: [DONE]\n\n"
	reader := bytes.NewReader([]byte(sseData))

	mockWriter := newMockResponseWriter()
	c, _ := gin.CreateTestContext(mockWriter)
	c.Request = httptest.NewRequest(http.MethodPost, "/", nil)

	startTime := time.Now()
	cc := &capture.CaptureContext{
		StartTime:   startTime,
		Recorder:    capture.NewRecorder("", "POST", "/test", "localhost:8080"),
		IDExtracted: false,
	}
	// Initialize upstream response
	cc.Recorder.RecordUpstreamResponse(200, nil)
	h := &mockHandler{}

	streamWithCapture(c, reader, h, cc)

	if cc.Recorder.Data().DownstreamResponse == nil {
		t.Error("expected DownstreamResponse to be set")
	}
	if len(cc.Recorder.Data().UpstreamResponse.Chunks) == 0 {
		t.Error("expected upstream chunks to be recorded")
	}
}

func TestHandle_WithCaptureContext(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("data: {\"id\":\"test-123\"}\n\n"))
		w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer upstream.Close()

	mockWriter := newMockResponseWriter()
	c, _ := gin.CreateTestContext(mockWriter)
	c.Request = httptest.NewRequest(http.MethodPost, "/", bytes.NewBufferString(`{"stream": true}`))

	cc := capture.NewCaptureContext(c.Request)
	ctx := capture.WithCaptureContext(c.Request.Context(), cc)
	c.Request = c.Request.WithContext(ctx)

	h := &mockHandler{
		upstreamURL: upstream.URL,
		apiKey:      "test-key",
	}

	Handle(h)(c)

	if cc.Recorder.Data().DownstreamResponse == nil {
		t.Error("expected DownstreamResponse to be set")
	}
}

func TestHandle_SuccessWithMockUpstream(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("data: {\"id\":\"test\"}\n\n"))
		w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer upstream.Close()

	mockWriter := newMockResponseWriter()
	c, _ := gin.CreateTestContext(mockWriter)
	c.Request = httptest.NewRequest(http.MethodPost, "/", bytes.NewBufferString(`{"stream": true}`))

	h := &mockHandler{
		upstreamURL: upstream.URL,
		apiKey:      "test-key",
	}

	Handle(h)(c)
}

func TestHandle_ForwardHeadersCalled(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Custom-Header") != "custom-value" {
			t.Errorf("expected X-Custom-Header to be forwarded")
		}
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("data: {}\n\n"))
	}))
	defer upstream.Close()

	mockWriter := newMockResponseWriter()
	c, _ := gin.CreateTestContext(mockWriter)
	c.Request = httptest.NewRequest(http.MethodPost, "/", bytes.NewBufferString(`{"stream": true}`))
	c.Request.Header.Set("X-Custom-Header", "custom-value")

	forwarded := false
	h := &mockHandler{
		upstreamURL: upstream.URL,
		apiKey:      "test-key",
		forwardHeadersFn: func(c *gin.Context, req *http.Request) {
			forwarded = true
			req.Header.Set("X-Custom-Header", c.Request.Header.Get("X-Custom-Header"))
		},
	}

	Handle(h)(c)

	if !forwarded {
		t.Error("expected ForwardHeaders to be called")
	}
}

func TestHandle_EmptyBody(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("data: {}\n\n"))
	}))
	defer upstream.Close()

	mockWriter := newMockResponseWriter()
	c, _ := gin.CreateTestContext(mockWriter)
	c.Request = httptest.NewRequest(http.MethodPost, "/", nil)

	h := &mockHandler{
		upstreamURL: upstream.URL,
		apiKey:      "test-key",
	}

	Handle(h)(c)
}
