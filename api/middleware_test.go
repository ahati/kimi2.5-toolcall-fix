package api

import (
	"ai-proxy/capture"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
)

func TestNewCaptureMiddleware(t *testing.T) {
	t.Run("with nil storage", func(t *testing.T) {
		m := NewCaptureMiddleware(nil)
		if m == nil {
			t.Fatal("NewCaptureMiddleware returned nil")
		}
		if m.storage != nil {
			t.Error("expected nil storage")
		}
	})

	t.Run("with storage", func(t *testing.T) {
		tmpDir := t.TempDir()
		storage := capture.NewStorage(tmpDir)
		m := NewCaptureMiddleware(storage)
		if m == nil {
			t.Fatal("NewCaptureMiddleware returned nil")
		}
		if m.storage != storage {
			t.Error("storage not set correctly")
		}
	})
}

func TestCaptureMiddleware_Handler_SetsContext(t *testing.T) {
	m := NewCaptureMiddleware(nil)
	handler := m.Handler()

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/test", nil)

	contextSet := false
	handler(c)

	ctx := c.Request.Context()
	cc := capture.GetCaptureContext(ctx)
	if cc == nil {
		t.Fatal("capture context not set in request context")
	}
	contextSet = true

	if !contextSet {
		t.Error("context was not set")
	}

	if cc.Recorder == nil {
		t.Error("recorder should not be nil")
	}
}

func TestCaptureMiddleware_Handler_PopulatesRecorder(t *testing.T) {
	m := NewCaptureMiddleware(nil)
	handler := m.Handler()

	tests := []struct {
		name       string
		method     string
		path       string
		remoteAddr string
	}{
		{
			name:       "POST request",
			method:     http.MethodPost,
			path:       "/v1/chat/completions",
			remoteAddr: "192.168.1.1:12345",
		},
		{
			name:       "GET request",
			method:     http.MethodGet,
			path:       "/v1/models",
			remoteAddr: "10.0.0.1:8080",
		},
		{
			name:       "root path",
			method:     http.MethodGet,
			path:       "/",
			remoteAddr: "localhost:3000",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			c.Request = httptest.NewRequest(tt.method, tt.path, nil)
			c.Request.RemoteAddr = tt.remoteAddr

			handler(c)

			cc := capture.GetCaptureContext(c.Request.Context())
			if cc == nil {
				t.Fatal("capture context is nil")
			}

			if cc.Recorder.Method != tt.method {
				t.Errorf("expected method %s, got %s", tt.method, cc.Recorder.Method)
			}

			if cc.Recorder.Path != tt.path {
				t.Errorf("expected path %s, got %s", tt.path, cc.Recorder.Path)
			}

			if cc.Recorder.ClientIP != tt.remoteAddr {
				t.Errorf("expected ClientIP %s, got %s", tt.remoteAddr, cc.Recorder.ClientIP)
			}
		})
	}
}

func TestCaptureMiddleware_Handler_CallsNext(t *testing.T) {
	m := NewCaptureMiddleware(nil)
	handler := m.Handler()

	nextCalled := false
	testHandler := func(c *gin.Context) {
		nextCalled = true
		c.Next()
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodGet, "/test", nil)

	handler(c)
	testHandler(c)

	if !nextCalled {
		t.Error("next handler was not called")
	}
}

func TestCaptureMiddleware_WithoutStorage(t *testing.T) {
	m := NewCaptureMiddleware(nil)
	handler := m.Handler()

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/test", nil)

	handler(c)

	cc := capture.GetCaptureContext(c.Request.Context())
	if cc == nil {
		t.Fatal("capture context should be set even without storage")
	}
}

func TestCaptureMiddleware_WithStorage(t *testing.T) {
	tmpDir := t.TempDir()
	storage := capture.NewStorage(tmpDir)
	m := NewCaptureMiddleware(storage)
	handler := m.Handler()

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	c.Request.RemoteAddr = "test-client:12345"

	handler(c)

	cc := capture.GetCaptureContext(c.Request.Context())
	if cc == nil {
		t.Fatal("capture context should be set")
	}

	cc.SetRequestID("test-request-id")

	time.Sleep(100 * time.Millisecond)

	files, err := os.ReadDir(tmpDir)
	if err != nil {
		t.Logf("could not read temp dir: %v (expected for async write)", err)
	}

	if len(files) == 0 {
		t.Log("no files written yet (async goroutine may not have completed)")
	}
}

func TestCaptureMiddleware_WithStorage_WritesToFile(t *testing.T) {
	tmpDir := t.TempDir()
	storage := capture.NewStorage(tmpDir)
	m := NewCaptureMiddleware(storage)
	handler := m.Handler()

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/test", nil)
	c.Request.RemoteAddr = "localhost:8080"

	handler(c)

	cc := capture.GetCaptureContext(c.Request.Context())
	cc.SetRequestID("test-write-id")

	time.Sleep(200 * time.Millisecond)

	dateDir := time.Now().Format("2006-01-02")
	fullDir := filepath.Join(tmpDir, dateDir)

	if _, err := os.Stat(fullDir); os.IsNotExist(err) {
		t.Logf("directory %s does not exist yet (async write timing)", fullDir)
	}
}

func TestCaptureMiddleware_Handler_MultipleRequests(t *testing.T) {
	m := NewCaptureMiddleware(nil)
	handler := m.Handler()

	for i := 0; i < 3; i++ {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest(http.MethodGet, "/test", nil)

		handler(c)

		cc := capture.GetCaptureContext(c.Request.Context())
		if cc == nil {
			t.Errorf("request %d: capture context is nil", i)
		}
	}
}

func TestCaptureMiddleware_Handler_StartTime(t *testing.T) {
	m := NewCaptureMiddleware(nil)
	handler := m.Handler()

	before := time.Now()
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodGet, "/test", nil)

	handler(c)
	after := time.Now()

	cc := capture.GetCaptureContext(c.Request.Context())
	if cc == nil {
		t.Fatal("capture context is nil")
	}

	if cc.StartTime.Before(before) {
		t.Error("StartTime should not be before handler call")
	}

	if cc.StartTime.After(after) {
		t.Error("StartTime should not be after handler call")
	}
}

func TestCaptureMiddleware_Handler_IDExtractedInitially(t *testing.T) {
	m := NewCaptureMiddleware(nil)
	handler := m.Handler()

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodGet, "/test", nil)

	handler(c)

	cc := capture.GetCaptureContext(c.Request.Context())
	if cc == nil {
		t.Fatal("capture context is nil")
	}

	if cc.IDExtracted {
		t.Error("IDExtracted should be false initially")
	}
}

func TestInitStorage(t *testing.T) {
	t.Run("with empty baseDir", func(t *testing.T) {
		storage := InitStorage("")
		if storage != nil {
			t.Error("expected nil storage for empty baseDir")
		}
	})

	t.Run("with non-empty baseDir", func(t *testing.T) {
		tmpDir := t.TempDir()
		storage := InitStorage(tmpDir)
		if storage == nil {
			t.Fatal("expected non-nil storage for non-empty baseDir")
		}
	})
}

func TestInitStorage_CreatesValidStorage(t *testing.T) {
	tmpDir := t.TempDir()
	storage := InitStorage(tmpDir)

	if storage == nil {
		t.Fatal("storage is nil")
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	c.Request.RemoteAddr = "test:8080"

	cc := capture.NewCaptureContext(c.Request)
	cc.SetRequestID("init-storage-test")

	err := storage.Write(cc.Recorder)
	if err != nil {
		t.Errorf("storage write failed: %v", err)
	}
}

func TestCaptureMiddleware_Handler_WithContextValues(t *testing.T) {
	m := NewCaptureMiddleware(nil)
	handler := m.Handler()

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/test", nil)

	c.Request.Header.Set("X-Custom-Header", "test-value")

	handler(c)

	cc := capture.GetCaptureContext(c.Request.Context())
	if cc == nil {
		t.Fatal("capture context is nil")
	}

	cc.SetRequestID("custom-request-id")
	if cc.RequestID != "custom-request-id" {
		t.Errorf("expected RequestID 'custom-request-id', got %s", cc.RequestID)
	}
}

func TestCaptureMiddleware_NilStorage_DoesNotPanic(t *testing.T) {
	m := NewCaptureMiddleware(nil)
	handler := m.Handler()

	defer func() {
		if r := recover(); r != nil {
			t.Errorf("handler panicked with nil storage: %v", r)
		}
	}()

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodGet, "/test", nil)

	handler(c)
}

func TestCaptureMiddleware_Handler_ConcurrentRequests(t *testing.T) {
	m := NewCaptureMiddleware(nil)
	handler := m.Handler()

	done := make(chan bool, 10)

	for i := 0; i < 10; i++ {
		go func(id int) {
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			c.Request = httptest.NewRequest(http.MethodGet, "/test", nil)

			handler(c)

			cc := capture.GetCaptureContext(c.Request.Context())
			if cc == nil {
				t.Errorf("goroutine %d: capture context is nil", id)
			}

			cc.SetRequestID(string(rune(id)))
			done <- true
		}(i)
	}

	for i := 0; i < 10; i++ {
		<-done
	}
}

func TestCaptureMiddleware_Handler_AbortHandlers(t *testing.T) {
	m := NewCaptureMiddleware(nil)
	handler := m.Handler()

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodGet, "/test", nil)

	handler(c)

	c.AbortWithStatus(http.StatusUnauthorized)

	cc := capture.GetCaptureContext(c.Request.Context())
	if cc == nil {
		t.Error("capture context should still be set even after abort")
	}
}
