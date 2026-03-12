package logging

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestStorageFullFlow(t *testing.T) {
	tmpDir := t.TempDir()

	storage := NewStorage(tmpDir)

	recorder := &RequestRecorder{
		RequestID: "chatcmpl-test123",
		StartedAt: time.Now(),
		Method:    "POST",
		Path:      "/v1/chat/completions",
		ClientIP:  "127.0.0.1",
		DownstreamRequest: &HTTPRequestCapture{
			At:      time.Now(),
			Headers: map[string]string{"content-type": "application/json"},
			Body:    json.RawMessage(`{"model":"test","messages":[{"role":"user","content":"hello"}],"stream":true}`),
		},
		UpstreamRequest: &HTTPRequestCapture{
			At:      time.Now(),
			Headers: map[string]string{"authorization": "***"},
			Body:    json.RawMessage(`{"model":"test","messages":[{"role":"user","content":"hello"}],"stream":true}`),
		},
		UpstreamResponse: &SSEResponseCapture{
			StatusCode: 200,
			Headers:    map[string]string{"content-type": "text/event-stream"},
			Chunks: []SSEChunk{
				{
					OffsetMS: 100,
					Event:    "message",
					Data:     json.RawMessage(`{"id":"chatcmpl-test123","choices":[{"delta":{"reasoning":"<|tool_calls_section_begin|><|tool_call_begin|>bash<|tool_call_argument_begin|>{\"cmd\":\"ls\"}<|tool_call_end|>"}}]}`),
				},
			},
		},
		DownstreamResponse: &SSEResponseCapture{
			Chunks: []SSEChunk{
				{
					OffsetMS: 105,
					Event:    "message",
					Data:     json.RawMessage(`{"id":"chatcmpl-test123","choices":[{"delta":{"tool_calls":[{"id":"call_0","type":"function","function":{"name":"bash"}}]}}]}`),
				},
			},
		},
	}

	err := storage.Write(recorder)
	if err != nil {
		t.Fatalf("Failed to write: %v", err)
	}

	files, _ := filepath.Glob(filepath.Join(tmpDir, "*", "*.json"))
	if len(files) == 0 {
		t.Fatal("No log file created")
	}

	t.Logf("Log file: %s", files[0])

	data, err := os.ReadFile(files[0])
	if err != nil {
		t.Fatalf("Failed to read log: %v", err)
	}

	t.Logf("\n=== LOG CONTENT ===\n%s", string(data))

	var logged logData
	if err := json.Unmarshal(data, &logged); err != nil {
		t.Fatalf("Failed to parse log: %v", err)
	}

	t.Log("\n=== VERIFICATION ===")

	upstreamReasoning := ""
	if len(logged.UpstreamResponse.Chunks) > 0 {
		chunk := logged.UpstreamResponse.Chunks[0]
		t.Logf("Upstream chunk data: %s", string(chunk.Data))

		var parsed map[string]interface{}
		json.Unmarshal(chunk.Data, &parsed)
		if choices, ok := parsed["choices"].([]interface{}); ok && len(choices) > 0 {
			if choice, ok := choices[0].(map[string]interface{}); ok {
				if delta, ok := choice["delta"].(map[string]interface{}); ok {
					upstreamReasoning, _ = delta["reasoning"].(string)
				}
			}
		}
	}

	hasRawTokens := containsToolCallTokensInStorage(upstreamReasoning)
	t.Logf("Upstream reasoning: %q", upstreamReasoning)
	t.Logf("Has raw tool_call tokens: %v", hasRawTokens)

	if !hasRawTokens {
		t.Error("BUG: Upstream response should contain raw <|tool_call tokens!")
	}

	downstreamHasToolCalls := false
	if len(logged.DownstreamResponse.Chunks) > 0 {
		chunk := logged.DownstreamResponse.Chunks[0]
		t.Logf("Downstream chunk data: %s", string(chunk.Data))

		if string(chunk.Data) != "" && containsStrInStorage(string(chunk.Data), `"tool_calls"`) {
			downstreamHasToolCalls = true
		}
	}

	t.Logf("Downstream has tool_calls: %v", downstreamHasToolCalls)

	if !downstreamHasToolCalls {
		t.Error("Downstream should have transformed tool_calls!")
	}
}

func containsToolCallTokensInStorage(s string) bool {
	tokens := []string{
		"<|tool_calls_section_begin|>",
		"<|tool_call_begin|>",
		"<|tool_call_argument_begin|>",
		"<|tool_call_end|>",
		"<|tool_calls_section_end|>",
	}
	for _, tok := range tokens {
		if containsStrInStorage(s, tok) {
			return true
		}
	}
	return false
}

func containsStrInStorage(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func TestStorage_WriteExistingFile(t *testing.T) {
	tmpDir := t.TempDir()
	storage := NewStorage(tmpDir)

	recorder := &RequestRecorder{
		RequestID: "duplicate-id",
		StartedAt: time.Now(),
		Method:    "POST",
		Path:      "/test",
		ClientIP:  "127.0.0.1",
	}

	err := storage.Write(recorder)
	if err != nil {
		t.Fatalf("First write failed: %v", err)
	}

	time.Sleep(10 * time.Millisecond)

	err = storage.Write(recorder)
	if err != nil {
		t.Fatalf("Second write failed: %v", err)
	}

	files, _ := filepath.Glob(filepath.Join(tmpDir, "*", "*duplicate-id*.json"))
	if len(files) != 2 {
		t.Errorf("Expected 2 files, got %d", len(files))
	}
}
