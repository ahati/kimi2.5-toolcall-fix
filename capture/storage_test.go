package capture

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestNewStorage(t *testing.T) {
	tests := []struct {
		name    string
		baseDir string
	}{
		{
			name:    "with path",
			baseDir: "/var/log/ai-proxy",
		},
		{
			name:    "empty path",
			baseDir: "",
		},
		{
			name:    "relative path",
			baseDir: "./logs",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			storage := NewStorage(tt.baseDir)

			if storage == nil {
				t.Fatal("NewStorage returned nil")
			}

			if storage.baseDir != tt.baseDir {
				t.Errorf("expected baseDir %q, got %q", tt.baseDir, storage.baseDir)
			}
		})
	}
}

func TestStorage_Write(t *testing.T) {
	tmpDir := t.TempDir()
	storage := NewStorage(tmpDir)

	recorder := &RequestRecorder{
		RequestID: "test-req-123",
		StartedAt: time.Now(),
		Method:    "POST",
		Path:      "/v1/chat/completions",
		ClientIP:  "localhost:8080",
		DownstreamRequest: &HTTPRequestCapture{
			At:      time.Now(),
			Headers: map[string]string{"Content-Type": "application/json"},
			Body:    json.RawMessage(`{"model": "test"}`),
		},
	}

	err := storage.Write(recorder)
	if err != nil {
		t.Fatalf("Write failed: %v", err)
	}

	dateDir := filepath.Join(tmpDir, time.Now().Format("2006-01-02"))
	entries, err := os.ReadDir(dateDir)
	if err != nil {
		t.Fatalf("failed to read date directory: %v", err)
	}

	if len(entries) == 0 {
		t.Fatal("no files written")
	}

	found := false
	for _, entry := range entries {
		if strings.Contains(entry.Name(), "test-req-123") {
			found = true
			break
		}
	}

	if !found {
		t.Error("expected file with request ID in name")
	}
}

func TestStorage_Write_CreatesDateDirectory(t *testing.T) {
	tmpDir := t.TempDir()
	storage := NewStorage(tmpDir)

	recorder := &RequestRecorder{
		RequestID: "test-req",
		StartedAt: time.Now(),
		Method:    "GET",
		Path:      "/test",
	}

	err := storage.Write(recorder)
	if err != nil {
		t.Fatalf("Write failed: %v", err)
	}

	dateDir := filepath.Join(tmpDir, time.Now().Format("2006-01-02"))
	if _, err := os.Stat(dateDir); os.IsNotExist(err) {
		t.Fatal("date directory was not created")
	}
}

func TestStorage_Write_NestedDirectory(t *testing.T) {
	tmpDir := t.TempDir()
	nestedPath := filepath.Join(tmpDir, "nested", "deep", "path")
	storage := NewStorage(nestedPath)

	recorder := &RequestRecorder{
		RequestID: "test-req",
		StartedAt: time.Now(),
		Method:    "GET",
		Path:      "/test",
	}

	err := storage.Write(recorder)
	if err != nil {
		t.Fatalf("Write failed: %v", err)
	}

	dateDir := filepath.Join(nestedPath, time.Now().Format("2006-01-02"))
	if _, err := os.Stat(dateDir); os.IsNotExist(err) {
		t.Fatal("nested date directory was not created")
	}
}

func TestStorage_Write_ValidJSON(t *testing.T) {
	tmpDir := t.TempDir()
	storage := NewStorage(tmpDir)

	recorder := &RequestRecorder{
		RequestID: "json-test",
		StartedAt: time.Now(),
		Method:    "POST",
		Path:      "/v1/chat/completions",
		DownstreamRequest: &HTTPRequestCapture{
			At:   time.Now(),
			Body: json.RawMessage(`{"messages": [{"role": "user", "content": "hello"}]}`),
		},
		UpstreamResponse: &SSEResponseCapture{
			StatusCode: 200,
			Chunks: []SSEChunk{
				{OffsetMS: 0, Event: "message", Data: json.RawMessage(`{"id": "1"}`)},
				{OffsetMS: 100, Event: "done", Raw: "[DONE]"},
			},
		},
	}

	err := storage.Write(recorder)
	if err != nil {
		t.Fatalf("Write failed: %v", err)
	}

	dateDir := filepath.Join(tmpDir, time.Now().Format("2006-01-02"))
	entries, err := os.ReadDir(dateDir)
	if err != nil {
		t.Fatalf("failed to read directory: %v", err)
	}

	var fileContent []byte
	for _, entry := range entries {
		if strings.Contains(entry.Name(), "json-test") {
			fileContent, err = os.ReadFile(filepath.Join(dateDir, entry.Name()))
			if err != nil {
				t.Fatalf("failed to read file: %v", err)
			}
			break
		}
	}

	if fileContent == nil {
		t.Fatal("file content not found")
	}

	var parsed logData
	if err := json.Unmarshal(fileContent, &parsed); err != nil {
		t.Fatalf("failed to parse JSON: %v", err)
	}

	if parsed.RequestID != "json-test" {
		t.Errorf("expected RequestID 'json-test', got %q", parsed.RequestID)
	}

	if parsed.Method != "POST" {
		t.Errorf("expected Method 'POST', got %q", parsed.Method)
	}
}

func TestStorage_Write_DuplicateFilename(t *testing.T) {
	tmpDir := t.TempDir()
	storage := NewStorage(tmpDir)

	recorder1 := &RequestRecorder{
		RequestID: "dup-test",
		StartedAt: time.Now(),
		Method:    "GET",
		Path:      "/test1",
	}

	err := storage.Write(recorder1)
	if err != nil {
		t.Fatalf("first Write failed: %v", err)
	}

	time.Sleep(10 * time.Millisecond)

	recorder2 := &RequestRecorder{
		RequestID: "dup-test",
		StartedAt: time.Now(),
		Method:    "GET",
		Path:      "/test2",
	}

	err = storage.Write(recorder2)
	if err != nil {
		t.Fatalf("second Write failed: %v", err)
	}

	dateDir := filepath.Join(tmpDir, time.Now().Format("2006-01-02"))
	entries, err := os.ReadDir(dateDir)
	if err != nil {
		t.Fatalf("failed to read directory: %v", err)
	}

	count := 0
	for _, entry := range entries {
		if strings.Contains(entry.Name(), "dup-test") {
			count++
		}
	}

	if count != 2 {
		t.Errorf("expected 2 files, got %d", count)
	}
}

func TestStorage_Filename(t *testing.T) {
	storage := NewStorage("/tmp")

	tests := []struct {
		name       string
		requestID  string
		wantSuffix string
	}{
		{
			name:       "simple ID",
			requestID:  "req-123",
			wantSuffix: "_req-123.json",
		},
		{
			name:       "empty ID",
			requestID:  "",
			wantSuffix: "_.json",
		},
		{
			name:       "ID with special chars",
			requestID:  "req/test:123",
			wantSuffix: "_req_test_123.json",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			filename := storage.filename(tt.requestID)

			if !strings.HasSuffix(filename, tt.wantSuffix) {
				t.Errorf("expected filename to end with %q, got %q", tt.wantSuffix, filename)
			}

			if !strings.HasSuffix(filename, ".json") {
				t.Error("filename should end with .json")
			}
		})
	}
}

func TestStorage_FilenameWithTimestamp(t *testing.T) {
	storage := NewStorage("/tmp")

	filename := storage.filenameWithTimestamp("test-id")

	if !strings.HasSuffix(filename, "_test-id_1.json") {
		t.Errorf("unexpected filename format: %s", filename)
	}
}

func TestSanitizeFilename(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "no special chars",
			input:    "simple-id-123",
			expected: "simple-id-123",
		},
		{
			name:     "forward slash",
			input:    "path/to/file",
			expected: "path_to_file",
		},
		{
			name:     "backslash",
			input:    "path\\to\\file",
			expected: "path_to_file",
		},
		{
			name:     "colon",
			input:    "time:12:30",
			expected: "time_12_30",
		},
		{
			name:     "asterisk",
			input:    "wild*card",
			expected: "wild_card",
		},
		{
			name:     "question mark",
			input:    "what?",
			expected: "what_",
		},
		{
			name:     "double quote",
			input:    `quote"here`,
			expected: "quote_here",
		},
		{
			name:     "less than",
			input:    "a<b",
			expected: "a_b",
		},
		{
			name:     "greater than",
			input:    "a>b",
			expected: "a_b",
		},
		{
			name:     "pipe",
			input:    "a|b",
			expected: "a_b",
		},
		{
			name:     "space",
			input:    "hello world",
			expected: "hello_world",
		},
		{
			name:     "multiple special chars",
			input:    "a/b:c*d?e\"f<g>h|i j",
			expected: "a_b_c_d_e_f_g_h_i_j",
		},
		{
			name:     "empty string",
			input:    "",
			expected: "",
		},
		{
			name:     "only special chars",
			input:    "/:\\*?\"<>| ",
			expected: "__________",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := sanitizeFilename(tt.input)

			if result != tt.expected {
				t.Errorf("expected %q, got %q", tt.expected, result)
			}
		})
	}
}

func TestStorage_LogDir(t *testing.T) {
	tests := []struct {
		name    string
		baseDir string
	}{
		{
			name:    "simple path",
			baseDir: "/var/log",
		},
		{
			name:    "relative path",
			baseDir: "./logs",
		},
		{
			name:    "empty path",
			baseDir: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			storage := NewStorage(tt.baseDir)
			logDir := storage.logDir()

			expectedDate := time.Now().Format("2006-01-02")

			if !strings.Contains(logDir, expectedDate) {
				t.Errorf("logDir should contain date %q, got %q", expectedDate, logDir)
			}
		})
	}
}

func TestStorage_Serialize(t *testing.T) {
	storage := NewStorage("/tmp")

	startTime := time.Now()
	recorder := &RequestRecorder{
		RequestID: "serialize-test",
		StartedAt: startTime,
		Method:    "POST",
		Path:      "/v1/test",
		ClientIP:  "192.168.1.1",
		DownstreamRequest: &HTTPRequestCapture{
			At:      startTime,
			Headers: map[string]string{"Content-Type": "application/json"},
			Body:    json.RawMessage(`{"test": true}`),
		},
		UpstreamResponse: &SSEResponseCapture{
			StatusCode: 200,
			Chunks:     []SSEChunk{{OffsetMS: 0, Event: "test"}},
		},
	}

	data := storage.serialize(recorder)

	if data.RequestID != "serialize-test" {
		t.Errorf("expected RequestID 'serialize-test', got %q", data.RequestID)
	}

	if data.Method != "POST" {
		t.Errorf("expected Method 'POST', got %q", data.Method)
	}

	if data.Path != "/v1/test" {
		t.Errorf("expected Path '/v1/test', got %q", data.Path)
	}

	if data.ClientIP != "192.168.1.1" {
		t.Errorf("expected ClientIP '192.168.1.1', got %q", data.ClientIP)
	}

	if data.DownstreamRequest == nil {
		t.Error("DownstreamRequest should not be nil")
	}

	if data.UpstreamResponse == nil {
		t.Error("UpstreamResponse should not be nil")
	}

	if data.DurationMS < 0 {
		t.Error("DurationMS should not be negative")
	}
}

func TestStorage_Serialize_EmptyRecorder(t *testing.T) {
	storage := NewStorage("/tmp")

	recorder := &RequestRecorder{
		RequestID: "empty-test",
		StartedAt: time.Now(),
		Method:    "GET",
		Path:      "/test",
	}

	data := storage.serialize(recorder)

	if data.RequestID != "empty-test" {
		t.Errorf("expected RequestID 'empty-test', got %q", data.RequestID)
	}

	if data.DownstreamRequest != nil {
		t.Error("DownstreamRequest should be nil")
	}

	if data.UpstreamRequest != nil {
		t.Error("UpstreamRequest should be nil")
	}

	if data.UpstreamResponse != nil {
		t.Error("UpstreamResponse should be nil")
	}

	if data.DownstreamResponse != nil {
		t.Error("DownstreamResponse should be nil")
	}
}

func TestStorage_Write_InvalidPath(t *testing.T) {
	if os.Getuid() == 0 {
		t.Skip("skipping test when running as root")
	}

	storage := NewStorage("/nonexistent/path/that/cannot/be/created")

	recorder := &RequestRecorder{
		RequestID: "fail-test",
		StartedAt: time.Now(),
		Method:    "GET",
		Path:      "/test",
	}

	err := storage.Write(recorder)
	if err == nil {
		t.Error("expected error for invalid path")
	}
}

func TestInitStorage(t *testing.T) {
	tmpDir := t.TempDir()

	InitStorage(tmpDir)

	storage := GetStorage()
	if storage == nil {
		t.Fatal("GetStorage returned nil after InitStorage")
	}

	if storage.baseDir != tmpDir {
		t.Errorf("expected baseDir %q, got %q", tmpDir, storage.baseDir)
	}
}

func TestInitStorage_Empty(t *testing.T) {
	originalStorage := globalStorage
	defer func() { globalStorage = originalStorage }()

	globalStorage = nil
	InitStorage("")

	if globalStorage != nil {
		t.Error("globalStorage should remain nil for empty path")
	}
}

func TestGetStorage_BeforeInit(t *testing.T) {
	originalStorage := globalStorage
	defer func() { globalStorage = originalStorage }()

	globalStorage = nil
	storage := GetStorage()

	if storage != nil {
		t.Error("GetStorage should return nil before InitStorage")
	}
}

func TestLogData_JSONSerialization(t *testing.T) {
	now := time.Now()
	data := logData{
		RequestID:  "json-test",
		StartedAt:  now,
		DurationMS: 123,
		Method:     "POST",
		Path:       "/v1/chat/completions",
		ClientIP:   "localhost:8080",
		DownstreamRequest: &HTTPRequestCapture{
			At:      now,
			Headers: map[string]string{"Content-Type": "application/json"},
			Body:    json.RawMessage(`{"test": true}`),
		},
		UpstreamResponse: &SSEResponseCapture{
			StatusCode: 200,
			Chunks:     []SSEChunk{{OffsetMS: 0, Event: "message"}},
		},
	}

	jsonData, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		t.Fatalf("failed to marshal: %v", err)
	}

	var parsed logData
	if err := json.Unmarshal(jsonData, &parsed); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	if parsed.RequestID != "json-test" {
		t.Errorf("expected RequestID 'json-test', got %q", parsed.RequestID)
	}

	if parsed.DurationMS != 123 {
		t.Errorf("expected DurationMS 123, got %d", parsed.DurationMS)
	}
}
