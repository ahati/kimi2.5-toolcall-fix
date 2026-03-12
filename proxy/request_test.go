package proxy

import (
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
)

func init() {
	gin.SetMode(gin.TestMode)
}

func TestForwardHeaders_MatchingPrefixes(t *testing.T) {
	tests := []struct {
		name           string
		requestHeaders map[string]string
		prefixes       []string
		expectedKeys   []string
	}{
		{
			name: "X- prefix match",
			requestHeaders: map[string]string{
				"X-Custom-Header":  "value1",
				"X-Another-Header": "value2",
				"Content-Type":     "application/json",
			},
			prefixes:     []string{"X-"},
			expectedKeys: []string{"X-Custom-Header", "X-Another-Header"},
		},
		{
			name: "Multiple prefixes",
			requestHeaders: map[string]string{
				"X-Custom":     "value1",
				"Content-Type": "application/json",
				"Accept":       "text/plain",
			},
			prefixes:     []string{"X-", "Content-"},
			expectedKeys: []string{"X-Custom", "Content-Type"},
		},
		{
			name: "Case insensitive matching",
			requestHeaders: map[string]string{
				"x-lowercase": "value1",
				"X-Uppercase": "value2",
			},
			prefixes:     []string{"X-"},
			expectedKeys: []string{"x-lowercase", "X-Uppercase"},
		},
		{
			name: "Multiple values for same header",
			requestHeaders: map[string]string{
				"X-Multi": "value1",
			},
			prefixes:     []string{"X-"},
			expectedKeys: []string{"X-Multi"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			req := httptest.NewRequest("POST", "/test", nil)
			for k, v := range tt.requestHeaders {
				req.Header.Set(k, v)
			}
			c.Request = req

			upstreamReq := httptest.NewRequest("POST", "http://upstream/test", nil)

			ForwardHeaders(c, upstreamReq, tt.prefixes...)

			for _, key := range tt.expectedKeys {
				if upstreamReq.Header.Get(key) == "" {
					t.Errorf("expected header %q to be forwarded", key)
				}
				if upstreamReq.Header.Get(key) != tt.requestHeaders[key] {
					t.Errorf("expected header %q value %q, got %q", key, tt.requestHeaders[key], upstreamReq.Header.Get(key))
				}
			}
		})
	}
}

func TestForwardHeaders_NonMatchingPrefixes(t *testing.T) {
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	req := httptest.NewRequest("POST", "/test", nil)
	req.Header.Set("X-Custom", "value1")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/plain")
	c.Request = req

	upstreamReq := httptest.NewRequest("POST", "http://upstream/test", nil)

	ForwardHeaders(c, upstreamReq, "Y-")

	if upstreamReq.Header.Get("X-Custom") != "" {
		t.Error("expected X-Custom not to be forwarded")
	}
	if upstreamReq.Header.Get("Content-Type") != "" {
		t.Error("expected Content-Type not to be forwarded")
	}
	if upstreamReq.Header.Get("Accept") != "" {
		t.Error("expected Accept not to be forwarded")
	}
}

func TestForwardHeaders_EmptyPrefixes(t *testing.T) {
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	req := httptest.NewRequest("POST", "/test", nil)
	req.Header.Set("X-Custom", "value1")
	c.Request = req

	upstreamReq := httptest.NewRequest("POST", "http://upstream/test", nil)

	ForwardHeaders(c, upstreamReq)

	if upstreamReq.Header.Get("X-Custom") != "" {
		t.Error("expected no headers to be forwarded with empty prefixes")
	}
}

func TestForwardHeaders_EmptyRequestHeaders(t *testing.T) {
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	req := httptest.NewRequest("POST", "/test", nil)
	c.Request = req

	upstreamReq := httptest.NewRequest("POST", "http://upstream/test", nil)

	ForwardHeaders(c, upstreamReq, "X-")

	if len(upstreamReq.Header) > 0 {
		t.Error("expected no headers in upstream request")
	}
}

func TestForwardHeaders_MultipleValuesForHeader(t *testing.T) {
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	req := httptest.NewRequest("POST", "/test", nil)
	req.Header.Add("X-Custom", "value1")
	req.Header.Add("X-Custom", "value2")
	c.Request = req

	upstreamReq := httptest.NewRequest("POST", "http://upstream/test", nil)

	ForwardHeaders(c, upstreamReq, "X-")

	values := upstreamReq.Header.Values("X-Custom")
	if len(values) != 2 {
		t.Errorf("expected 2 values, got %d", len(values))
	}
	if values[0] != "value1" || values[1] != "value2" {
		t.Errorf("expected values ['value1', 'value2'], got %v", values)
	}
}

func TestForwardHeaders_PreserveOriginalHeaderCase(t *testing.T) {
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	req := httptest.NewRequest("POST", "/test", nil)
	req.Header.Set("X-Custom-Header", "value")
	c.Request = req

	upstreamReq := httptest.NewRequest("POST", "http://upstream/test", nil)

	ForwardHeaders(c, upstreamReq, "X-")

	if upstreamReq.Header.Get("X-Custom-Header") != "value" {
		t.Errorf("expected header value 'value', got %q", upstreamReq.Header.Get("X-Custom-Header"))
	}
}

func TestForwardConnectionHeaders_AllHeaders(t *testing.T) {
	tests := []struct {
		name           string
		headers        map[string]string
		expectedValues map[string]string
	}{
		{
			name: "Connection header",
			headers: map[string]string{
				"Connection": "keep-alive",
			},
			expectedValues: map[string]string{
				"Connection": "keep-alive",
			},
		},
		{
			name: "Keep-Alive header",
			headers: map[string]string{
				"Keep-Alive": "timeout=5",
			},
			expectedValues: map[string]string{
				"Keep-Alive": "timeout=5",
			},
		},
		{
			name: "Upgrade header",
			headers: map[string]string{
				"Upgrade": "websocket",
			},
			expectedValues: map[string]string{
				"Upgrade": "websocket",
			},
		},
		{
			name: "TE header",
			headers: map[string]string{
				"TE": "trailers",
			},
			expectedValues: map[string]string{
				"TE": "trailers",
			},
		},
		{
			name: "All connection headers",
			headers: map[string]string{
				"Connection": "keep-alive",
				"Keep-Alive": "timeout=5",
				"Upgrade":    "websocket",
				"TE":         "trailers",
			},
			expectedValues: map[string]string{
				"Connection": "keep-alive",
				"Keep-Alive": "timeout=5",
				"Upgrade":    "websocket",
				"TE":         "trailers",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			req := httptest.NewRequest("POST", "/test", nil)
			for k, v := range tt.headers {
				req.Header.Set(k, v)
			}
			c.Request = req

			upstreamReq := httptest.NewRequest("POST", "http://upstream/test", nil)

			ForwardConnectionHeaders(c, upstreamReq)

			for key, expected := range tt.expectedValues {
				got := upstreamReq.Header.Get(key)
				if got != expected {
					t.Errorf("expected header %q to be %q, got %q", key, expected, got)
				}
			}
		})
	}
}

func TestForwardConnectionHeaders_NoHeaders(t *testing.T) {
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	req := httptest.NewRequest("POST", "/test", nil)
	c.Request = req

	upstreamReq := httptest.NewRequest("POST", "http://upstream/test", nil)

	ForwardConnectionHeaders(c, upstreamReq)

	connectionHeaders := []string{"Connection", "Keep-Alive", "Upgrade", "TE"}
	for _, header := range connectionHeaders {
		if upstreamReq.Header.Get(header) != "" {
			t.Errorf("expected header %q to be empty", header)
		}
	}
}

func TestForwardConnectionHeaders_OnlySomeHeadersPresent(t *testing.T) {
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	req := httptest.NewRequest("POST", "/test", nil)
	req.Header.Set("Connection", "upgrade")
	req.Header.Set("Upgrade", "websocket")
	c.Request = req

	upstreamReq := httptest.NewRequest("POST", "http://upstream/test", nil)

	ForwardConnectionHeaders(c, upstreamReq)

	if upstreamReq.Header.Get("Connection") != "upgrade" {
		t.Errorf("expected Connection 'upgrade', got %q", upstreamReq.Header.Get("Connection"))
	}
	if upstreamReq.Header.Get("Upgrade") != "websocket" {
		t.Errorf("expected Upgrade 'websocket', got %q", upstreamReq.Header.Get("Upgrade"))
	}
	if upstreamReq.Header.Get("Keep-Alive") != "" {
		t.Error("expected Keep-Alive to be empty")
	}
	if upstreamReq.Header.Get("TE") != "" {
		t.Error("expected TE to be empty")
	}
}

func TestForwardConnectionHeaders_OverwritesExisting(t *testing.T) {
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	req := httptest.NewRequest("POST", "/test", nil)
	req.Header.Set("Connection", "close")
	c.Request = req

	upstreamReq := httptest.NewRequest("POST", "http://upstream/test", nil)
	upstreamReq.Header.Set("Connection", "keep-alive")

	ForwardConnectionHeaders(c, upstreamReq)

	if upstreamReq.Header.Get("Connection") != "close" {
		t.Errorf("expected Connection to be overwritten to 'close', got %q", upstreamReq.Header.Get("Connection"))
	}
}

func TestForwardHeaders_CaseInsensitivePrefix(t *testing.T) {
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	req := httptest.NewRequest("POST", "/test", nil)
	req.Header.Set("X-Custom", "value1")
	req.Header.Set("x-lowercase", "value2")
	req.Header.Set("X-UPPERCASE", "value3")
	c.Request = req

	upstreamReq := httptest.NewRequest("POST", "http://upstream/test", nil)

	ForwardHeaders(c, upstreamReq, "x-")

	if upstreamReq.Header.Get("X-Custom") != "value1" {
		t.Errorf("expected X-Custom to be forwarded")
	}
	if upstreamReq.Header.Get("x-lowercase") != "value2" {
		t.Errorf("expected x-lowercase to be forwarded")
	}
	if upstreamReq.Header.Get("X-UPPERCASE") != "value3" {
		t.Errorf("expected X-UPPERCASE to be forwarded")
	}
}

func TestForwardHeaders_StopsAtFirstMatchingPrefix(t *testing.T) {
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	req := httptest.NewRequest("POST", "/test", nil)
	req.Header.Set("X-Custom", "value")
	c.Request = req

	upstreamReq := httptest.NewRequest("POST", "http://upstream/test", nil)

	ForwardHeaders(c, upstreamReq, "X-", "X-")

	values := upstreamReq.Header.Values("X-Custom")
	if len(values) != 1 {
		t.Errorf("expected 1 value, got %d", len(values))
	}
}

func TestForwardHeaders_LongestPrefixDoesNotMatch(t *testing.T) {
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	req := httptest.NewRequest("POST", "/test", nil)
	req.Header.Set("Accept-Encoding", "gzip")
	req.Header.Set("X-Custom", "value")
	c.Request = req

	upstreamReq := httptest.NewRequest("POST", "http://upstream/test", nil)

	ForwardHeaders(c, upstreamReq, "X-")

	if upstreamReq.Header.Get("Accept-Encoding") != "" {
		t.Error("expected Accept-Encoding not to be forwarded")
	}
	if upstreamReq.Header.Get("X-Custom") != "value" {
		t.Error("expected X-Custom to be forwarded")
	}
}
