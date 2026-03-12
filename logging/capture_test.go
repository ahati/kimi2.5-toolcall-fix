package logging

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestNewSSEChunk_RawMessageReference(t *testing.T) {
	originalData := []byte(`{"choices":[{"delta":{"reasoning":"<|tool_calls_section_begin|><|tool_call_begin|>bash<|tool_call_argument_begin|>{\"command\":\"ls\"}<|tool_call_end|>"}}]}`)

	chunk := NewSSEChunk(0, "message", originalData)

	originalDataCopy := make([]byte, len(originalData))
	copy(originalDataCopy, originalData)

	for i := range originalData {
		originalData[i] = 'X'
	}

	t.Logf("Original (modified): %s", string(originalData))
	t.Logf("Original (copy): %s", string(originalDataCopy))
	t.Logf("Chunk.Data: %s", string(chunk.Data))

	if string(chunk.Data) != string(originalDataCopy) {
		t.Errorf("Chunk.Data was corrupted after modifying original slice!")
		t.Errorf("Expected: %s", string(originalDataCopy))
		t.Errorf("Got: %s", string(chunk.Data))
	}
}

func TestNewSSEChunk_TransformationScenario(t *testing.T) {
	upstreamData := []byte(`{"id":"chatcmpl-1","choices":[{"delta":{"reasoning":"<|tool_calls_section_begin|><|tool_call_begin|>bash<|tool_call_argument_begin|>{\"command\":\"ls\"}<|tool_call_end|>"}}]}`)

	chunk := NewSSEChunk(0, "message", upstreamData)

	var parsed map[string]interface{}
	if err := json.Unmarshal(chunk.Data, &parsed); err != nil {
		t.Fatalf("Failed to parse chunk data: %v", err)
	}

	choices := parsed["choices"].([]interface{})
	delta := choices[0].(map[string]interface{})["delta"].(map[string]interface{})
	reasoning := delta["reasoning"].(string)

	hasToolCallTokens := containsToolCallTokens(reasoning)
	t.Logf("Reasoning field: %s", reasoning)
	t.Logf("Has tool call tokens: %v", hasToolCallTokens)

	if !hasToolCallTokens {
		t.Error("Expected reasoning to contain tool call tokens, but it doesn't!")
	}
}

func TestCaptureWriter_StreamSimulation(t *testing.T) {
	start := time.Now()
	cw := NewCaptureWriter(start)

	upstreamChunks := [][]byte{
		[]byte(`{"id":"chatcmpl-1","choices":[{"delta":{"reasoning":"Some text <|tool_calls_section_begin|>"}}]}`),
		[]byte(`{"id":"chatcmpl-1","choices":[{"delta":{"reasoning":"<|tool_call_begin|>bash<|tool_call_argument_begin|>{\"command\":\"ls\"}<|tool_call_end|>"}}]}`),
		[]byte(`{"id":"chatcmpl-1","choices":[{"delta":{"reasoning":"<|tool_calls_section_end|> more text"}}]}`),
	}

	for i, data := range upstreamChunks {
		dataCopy := make([]byte, len(data))
		copy(dataCopy, data)

		cw.RecordChunk("message", data)

		for j := range data {
			data[j] = 'X'
		}

		t.Logf("Chunk %d: modified original=%s", i, string(data))
		t.Logf("Chunk %d: original copy=%s", i, string(dataCopy))
	}

	captured := cw.Chunks()
	t.Logf("\nCaptured %d chunks", len(captured))

	for i, chunk := range captured {
		t.Logf("Chunk %d Data: %s", i, string(chunk.Data))

		var parsed map[string]interface{}
		if err := json.Unmarshal(chunk.Data, &parsed); err != nil {
			t.Errorf("Chunk %d: Failed to parse - %v", i, err)
			continue
		}

		choices, ok := parsed["choices"].([]interface{})
		if !ok || len(choices) == 0 {
			t.Errorf("Chunk %d: No choices", i)
			continue
		}

		delta, ok := choices[0].(map[string]interface{})["delta"].(map[string]interface{})
		if !ok {
			t.Errorf("Chunk %d: No delta", i)
			continue
		}

		if reasoning, ok := delta["reasoning"].(string); ok {
			hasTokens := containsToolCallTokens(reasoning)
			t.Logf("Chunk %d reasoning: %q (has tokens: %v)", i, reasoning, hasTokens)

			if i < 2 && !hasTokens {
				t.Errorf("Chunk %d: Expected tool call tokens in reasoning but found none!", i)
			}
		}
	}
}

func TestRawMessageBehavior(t *testing.T) {
	t.Run("RawMessage Unmarshal already copies", func(t *testing.T) {
		original := []byte(`{"test":"value"}`)

		var raw json.RawMessage
		err := json.Unmarshal(original, &raw)
		if err != nil {
			t.Fatalf("Unmarshal failed: %v", err)
		}

		original[0] = 'X'

		if raw[0] == 'X' {
			t.Error("RawMessage was corrupted - json.Unmarshal should copy but didn't")
		} else {
			t.Logf("json.Unmarshal correctly copies: raw=%s (independent of modified original)", string(raw))
		}
	})
}

func containsToolCallTokens(s string) bool {
	tokens := []string{
		"<|tool_calls_section_begin|>",
		"<|tool_call_begin|>",
		"<|tool_call_argument_begin|>",
		"<|tool_call_end|>",
		"<|tool_calls_section_end|>",
	}
	for _, tok := range tokens {
		if containsStr(s, tok) {
			return true
		}
	}
	return false
}

func containsStr(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func TestNewCaptureContext(t *testing.T) {
	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	req.RemoteAddr = "192.168.1.1:12345"

	cc := NewCaptureContext(req)

	if cc == nil {
		t.Fatal("NewCaptureContext returned nil")
	}
	if cc.Recorder == nil {
		t.Fatal("CaptureContext.Recorder is nil")
	}
	if cc.IDExtracted {
		t.Error("IDExtracted should be false initially")
	}
	if cc.Recorder.Method != "POST" {
		t.Errorf("Expected Method POST, got %s", cc.Recorder.Method)
	}
	if cc.Recorder.Path != "/v1/chat/completions" {
		t.Errorf("Expected Path /v1/chat/completions, got %s", cc.Recorder.Path)
	}
	if cc.Recorder.ClientIP != "192.168.1.1:12345" {
		t.Errorf("Expected ClientIP 192.168.1.1:12345, got %s", cc.Recorder.ClientIP)
	}
}

func TestSetRequestID(t *testing.T) {
	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	cc := NewCaptureContext(req)

	cc.SetRequestID("test-request-123")

	if cc.RequestID != "test-request-123" {
		t.Errorf("Expected RequestID test-request-123, got %s", cc.RequestID)
	}
	if cc.Recorder.RequestID != "test-request-123" {
		t.Errorf("Expected Recorder.RequestID test-request-123, got %s", cc.Recorder.RequestID)
	}
	if !cc.IDExtracted {
		t.Error("IDExtracted should be true after SetRequestID")
	}
}

func TestWithCaptureContextAndGetCaptureContext(t *testing.T) {
	req := httptest.NewRequest("GET", "/test", nil)
	cc := NewCaptureContext(req)
	cc.SetRequestID("ctx-test-id")

	ctx := context.Background()
	ctxWithCC := WithCaptureContext(ctx, cc)

	retrieved := GetCaptureContext(ctxWithCC)
	if retrieved == nil {
		t.Fatal("GetCaptureContext returned nil")
	}
	if retrieved.RequestID != "ctx-test-id" {
		t.Errorf("Expected RequestID ctx-test-id, got %s", retrieved.RequestID)
	}

	emptyCtx := GetCaptureContext(ctx)
	if emptyCtx != nil {
		t.Error("GetCaptureContext should return nil for context without CaptureContext")
	}
}

func TestExtractRequestID(t *testing.T) {
	t.Run("X-Request-ID header", func(t *testing.T) {
		req := httptest.NewRequest("POST", "/test", nil)
		req.Header.Set("X-Request-ID", "header-request-id")
		id := extractRequestID(req)
		if id != "header-request-id" {
			t.Errorf("Expected header-request-id, got %s", id)
		}
	})

	t.Run("lowercase x-request-id header", func(t *testing.T) {
		req := httptest.NewRequest("POST", "/test", nil)
		req.Header.Set("x-request-id", "lowercase-request-id")
		id := extractRequestID(req)
		if id != "lowercase-request-id" {
			t.Errorf("Expected lowercase-request-id, got %s", id)
		}
	})

	t.Run("no request id header POST with body", func(t *testing.T) {
		req := httptest.NewRequest("POST", "/test", bytes.NewReader([]byte("test")))
		id := extractRequestID(req)
		if id != "" {
			t.Errorf("Expected empty string, got %s", id)
		}
	})

	t.Run("no request id header GET", func(t *testing.T) {
		req := httptest.NewRequest("GET", "/test", nil)
		id := extractRequestID(req)
		if id != "" {
			t.Errorf("Expected empty string, got %s", id)
		}
	})
}

func TestExtractRequestIDFromSSEChunk(t *testing.T) {
	t.Run("OpenAI format", func(t *testing.T) {
		body := []byte(`{"id":"chatcmpl-abc123","choices":[{"delta":{"content":"hello"}}]}`)
		id := ExtractRequestIDFromSSEChunk(body)
		if id != "chatcmpl-abc123" {
			t.Errorf("Expected chatcmpl-abc123, got %s", id)
		}
	})

	t.Run("Anthropic message_start format", func(t *testing.T) {
		body := []byte(`{"type":"message_start","message":{"id":"msg-xyz789"}}`)
		id := ExtractRequestIDFromSSEChunk(body)
		if id != "msg-xyz789" {
			t.Errorf("Expected msg-xyz789, got %s", id)
		}
	})

	t.Run("Anthropic non-message_start format", func(t *testing.T) {
		body := []byte(`{"type":"content_block_delta","index":0}`)
		id := ExtractRequestIDFromSSEChunk(body)
		if id != "" {
			t.Errorf("Expected empty string for non-message_start, got %s", id)
		}
	})

	t.Run("invalid JSON", func(t *testing.T) {
		body := []byte(`not valid json`)
		id := ExtractRequestIDFromSSEChunk(body)
		if id != "" {
			t.Errorf("Expected empty string for invalid JSON, got %s", id)
		}
	})

	t.Run("empty ID field", func(t *testing.T) {
		body := []byte(`{"id":"","choices":[]}`)
		id := ExtractRequestIDFromSSEChunk(body)
		if id != "" {
			t.Errorf("Expected empty string for empty ID, got %s", id)
		}
	})
}

func TestSanitizeHeaders(t *testing.T) {
	headers := http.Header{}
	headers.Set("Content-Type", "application/json")
	headers.Set("Authorization", "Bearer secret-token")
	headers.Set("X-Api-Key", "api-key-123")
	headers.Set("Cookie", "session=abc")
	headers.Set("Set-Cookie", "session=def")
	headers.Set("X-Auth-Token", "token-xyz")
	headers.Set("X-Custom-Header", "custom-value")

	sanitized := SanitizeHeaders(headers)

	if sanitized["Content-Type"] != "application/json" {
		t.Errorf("Content-Type should not be masked, got %s", sanitized["Content-Type"])
	}
	if sanitized["Authorization"] != "***" {
		t.Errorf("Authorization should be masked, got %s", sanitized["Authorization"])
	}
	if sanitized["X-Api-Key"] != "***" {
		t.Errorf("X-Api-Key should be masked, got %s", sanitized["X-Api-Key"])
	}
	if sanitized["Cookie"] != "***" {
		t.Errorf("Cookie should be masked, got %s", sanitized["Cookie"])
	}
	if sanitized["Set-Cookie"] != "***" {
		t.Errorf("Set-Cookie should be masked, got %s", sanitized["Set-Cookie"])
	}
	if sanitized["X-Auth-Token"] != "***" {
		t.Errorf("X-Auth-Token should be masked, got %s", sanitized["X-Auth-Token"])
	}
	if sanitized["X-Custom-Header"] != "custom-value" {
		t.Errorf("X-Custom-Header should not be masked, got %s", sanitized["X-Custom-Header"])
	}
}

func TestSanitizeHeaders_MultipleValues(t *testing.T) {
	headers := http.Header{}
	headers.Add("Accept", "application/json")
	headers.Add("Accept", "text/html")

	sanitized := SanitizeHeaders(headers)

	if sanitized["Accept"] != "application/json" {
		t.Errorf("Expected first value application/json, got %s", sanitized["Accept"])
	}
}

func TestSanitizeHeaders_Empty(t *testing.T) {
	headers := http.Header{}
	sanitized := SanitizeHeaders(headers)

	if len(sanitized) != 0 {
		t.Errorf("Expected empty map, got %d items", len(sanitized))
	}
}

func TestOffsetMS(t *testing.T) {
	start := time.Now()
	time.Sleep(10 * time.Millisecond)
	offset := OffsetMS(start)

	if offset < 10 {
		t.Errorf("Expected offset >= 10ms, got %dms", offset)
	}
}

func TestCaptureWriter_RecordChunk_EmptyData(t *testing.T) {
	start := time.Now()
	cw := NewCaptureWriter(start)

	cw.RecordChunk("message", []byte{})

	if len(cw.Chunks()) != 0 {
		t.Errorf("Expected 0 chunks for empty data, got %d", len(cw.Chunks()))
	}

	cw.RecordChunk("message", nil)

	if len(cw.Chunks()) != 0 {
		t.Errorf("Expected 0 chunks for nil data, got %d", len(cw.Chunks()))
	}
}
