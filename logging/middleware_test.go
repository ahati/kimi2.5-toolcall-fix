package logging

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
)

func init() {
	gin.SetMode(gin.TestMode)
}

func TestCaptureMiddleware_CreatesContext(t *testing.T) {
	router := gin.New()
	router.Use(CaptureMiddleware(nil))
	router.POST("/test", func(c *gin.Context) {
		cc := GetCaptureContext(c.Request.Context())
		if cc == nil {
			t.Error("CaptureContext should not be nil in handler")
			c.JSON(http.StatusInternalServerError, gin.H{"error": "no context"})
			return
		}
		if cc.Recorder == nil {
			t.Error("CaptureContext.Recorder should not be nil")
			c.JSON(http.StatusInternalServerError, gin.H{"error": "no recorder"})
			return
		}
		c.JSON(http.StatusOK, gin.H{"success": true})
	})

	req := httptest.NewRequest("POST", "/test", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}
}

func TestCaptureMiddleware_RecordsRequestInfo(t *testing.T) {
	router := gin.New()
	router.Use(CaptureMiddleware(nil))
	router.GET("/test-path", func(c *gin.Context) {
		cc := GetCaptureContext(c.Request.Context())
		if cc.Recorder.Method != "GET" {
			t.Errorf("Expected Method GET, got %s", cc.Recorder.Method)
		}
		if cc.Recorder.Path != "/test-path" {
			t.Errorf("Expected Path /test-path, got %s", cc.Recorder.Path)
		}
		c.Status(http.StatusOK)
	})

	req := httptest.NewRequest("GET", "/test-path", nil)
	req.RemoteAddr = "10.0.0.1:54321"
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)
}

func TestCaptureMiddleware_WritesToStorage(t *testing.T) {
	tmpDir := t.TempDir()
	storage := NewStorage(tmpDir)

	router := gin.New()
	router.Use(CaptureMiddleware(storage))
	router.POST("/v1/chat/completions", func(c *gin.Context) {
		cc := GetCaptureContext(c.Request.Context())
		cc.SetRequestID("test-write-id")
		c.Status(http.StatusOK)
	})

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	time.Sleep(100 * time.Millisecond)

	files, err := filepath.Glob(filepath.Join(tmpDir, "*", "*test-write-id*.json"))
	if err != nil {
		t.Fatalf("Failed to find log files: %v", err)
	}
	if len(files) == 0 {
		t.Error("Expected log file to be created")
	}
}

func TestCaptureMiddleware_StorageNil(t *testing.T) {
	router := gin.New()
	router.Use(CaptureMiddleware(nil))
	router.GET("/test", func(c *gin.Context) {
		cc := GetCaptureContext(c.Request.Context())
		if cc == nil {
			t.Error("CaptureContext should exist even with nil storage")
		}
		c.Status(http.StatusOK)
	})

	req := httptest.NewRequest("GET", "/test", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}
}

func TestRecordDownstreamRequest(t *testing.T) {
	gin.SetMode(gin.TestMode)

	router := gin.New()
	router.Use(CaptureMiddleware(nil))
	router.POST("/test", func(c *gin.Context) {
		body := []byte(`{"model":"test","messages":[]}`)
		RecordDownstreamRequest(c, body)

		cc := GetCaptureContext(c.Request.Context())
		if cc.Recorder.DownstreamRequest == nil {
			t.Error("DownstreamRequest should not be nil")
			c.Status(http.StatusInternalServerError)
			return
		}
		if string(cc.Recorder.DownstreamRequest.Body) != `{"model":"test","messages":[]}` {
			t.Errorf("Body mismatch: %s", string(cc.Recorder.DownstreamRequest.Body))
		}
		if cc.Recorder.DownstreamRequest.Headers == nil {
			t.Error("Headers should not be nil")
		}
		c.Status(http.StatusOK)
	})

	req := httptest.NewRequest("POST", "/test", bytes.NewReader([]byte(`{"model":"test","messages":[]}`)))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer secret")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}
}

func TestRecordDownstreamRequest_NoContext(t *testing.T) {
	gin.SetMode(gin.TestMode)

	router := gin.New()
	router.POST("/test", func(c *gin.Context) {
		RecordDownstreamRequest(c, []byte("test"))
		c.Status(http.StatusOK)
	})

	req := httptest.NewRequest("POST", "/test", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}
}

func TestCaptureMiddleware_FullFlow(t *testing.T) {
	tmpDir := t.TempDir()
	storage := NewStorage(tmpDir)

	router := gin.New()
	router.Use(CaptureMiddleware(storage))
	router.POST("/v1/chat/completions", func(c *gin.Context) {
		body := []byte(`{"model":"test-model","stream":true}`)
		RecordDownstreamRequest(c, body)

		cc := GetCaptureContext(c.Request.Context())
		cc.SetRequestID("full-flow-test-id")

		c.JSON(http.StatusOK, gin.H{"status": "ok"})
	})

	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader([]byte(`{"model":"test-model","stream":true}`)))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer test-key")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	time.Sleep(150 * time.Millisecond)

	files, err := filepath.Glob(filepath.Join(tmpDir, "*", "*full-flow-test-id*.json"))
	if err != nil {
		t.Fatalf("Failed to find log files: %v", err)
	}
	if len(files) == 0 {
		t.Error("Expected log file to be created")
		return
	}

	data, err := os.ReadFile(files[0])
	if err != nil {
		t.Fatalf("Failed to read log file: %v", err)
	}

	if !bytes.Contains(data, []byte("full-flow-test-id")) {
		t.Error("Log should contain request ID")
	}
	if !bytes.Contains(data, []byte("test-model")) {
		t.Error("Log should contain model name")
	}
	if bytes.Contains(data, []byte("test-key")) {
		t.Error("Log should NOT contain raw API key")
	}
	if bytes.Contains(data, []byte("***")) {
		t.Log("Authorization header properly masked")
	}
}
