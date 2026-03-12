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

	recorder := NewRecorder("test-req-123", "POST", "/v1/chat/completions", "localhost:8080")
	recorder.RecordDownstreamRequest(nil, json.RawMessage(`{"model": "test"}`))

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

	recorder := NewRecorder("test-req", "GET", "/test", "localhost:8080")

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

	recorder := NewRecorder("test-req", "GET", "/test", "localhost:8080")

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

	recorder := NewRecorder("json-test", "POST", "/v1/chat/completions", "localhost:8080")
	recorder.RecordDownstreamRequest(nil, json.RawMessage(`{"messages": [{"role": "user", "content": "hello"}]}`))
	respRecorder := recorder.RecordUpstreamResponse(200, nil)
	respRecorder.RecordChunk("message", `{"id": "1"}`)
	respRecorder.RecordChunk("done", "[DONE]")

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
		t.Errorf("expected RequestID %q, got %q", "json-test", parsed.RequestID)
	}

	if parsed.Method != "POST" {
		t.Errorf("expected Method %q, got %q", "POST", parsed.Method)
	}
}

func TestStorage_Write_DuplicateRequestID(t *testing.T) {
	tmpDir := t.TempDir()
	storage := NewStorage(tmpDir)

	recorder1 := NewRecorder("dup-test", "GET", "/test1", "localhost:8080")

	err := storage.Write(recorder1)
	if err != nil {
		t.Fatalf("first Write failed: %v", err)
	}

	recorder2 := NewRecorder("dup-test", "GET", "/test2", "localhost:8080")

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
			input:    "id:123",
			expected: "id_123",
		},
		{
			name:     "space",
			input:    "id with spaces",
			expected: "id_with_spaces",
		},
		{
			name:     "multiple special chars",
			input:    "path/to:file*name?",
			expected: "path_to_file_name_",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := sanitizeFilename(tt.input)
			if result != tt.expected {
				t.Errorf("sanitizeFilename(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}

func TestStorage_Serialize(t *testing.T) {
	storage := NewStorage("/tmp")
	startTime := time.Now()

	recorder := NewRecorder("serialize-test", "POST", "/v1/chat/completions", "localhost:8080")
	recorder.RecordDownstreamRequest(nil, json.RawMessage(`{"model": "test"}`))
	respRecorder := recorder.RecordUpstreamResponse(200, nil)
	respRecorder.RecordChunk("message", `{"id": "123"}`)

	data := storage.serialize(recorder.Data())

	if data.RequestID != "serialize-test" {
		t.Errorf("expected RequestID %q, got %q", "serialize-test", data.RequestID)
	}

	if data.Method != "POST" {
		t.Errorf("expected Method %q, got %q", "POST", data.Method)
	}

	if data.Path != "/v1/chat/completions" {
		t.Errorf("expected Path %q, got %q", "/v1/chat/completions", data.Path)
	}

	if data.ClientIP != "localhost:8080" {
		t.Errorf("expected ClientIP %q, got %q", "localhost:8080", data.ClientIP)
	}

	if data.DownstreamRequest == nil {
		t.Error("expected DownstreamRequest to be set")
	}

	if data.UpstreamResponse == nil {
		t.Error("expected UpstreamResponse to be set")
	}

	if data.DurationMS < 0 {
		t.Error("DurationMS should be non-negative")
	}

	if data.StartedAt.Before(startTime) || data.StartedAt.After(time.Now()) {
		t.Error("StartedAt should be within valid time range")
	}
}

func TestStorage_Serialize_EmptyRecorder(t *testing.T) {
	storage := NewStorage("/tmp")

	recorder := NewRecorder("empty-test", "GET", "/test", "localhost:8080")

	data := storage.serialize(recorder.Data())

	if data.RequestID != "empty-test" {
		t.Errorf("expected RequestID %q, got %q", "empty-test", data.RequestID)
	}

	if data.DownstreamRequest != nil {
		t.Error("expected DownstreamRequest to be nil")
	}

	if data.UpstreamResponse != nil {
		t.Error("expected UpstreamResponse to be nil")
	}
}

func TestStorage_Write_ReadOnlyPermissions(t *testing.T) {
	tmpDir := t.TempDir()
	readOnlyDir := filepath.Join(tmpDir, "readonly")
	if err := os.MkdirAll(readOnlyDir, 0755); err != nil {
		t.Fatalf("failed to create directory: %v", err)
	}

	if err := os.Chmod(readOnlyDir, 0555); err != nil {
		t.Fatalf("failed to set permissions: %v", err)
	}
	defer os.Chmod(readOnlyDir, 0755)

	storage := NewStorage(readOnlyDir)
	recorder := NewRecorder("fail-test", "GET", "/test", "localhost:8080")

	err := storage.Write(recorder)
	if err == nil {
		t.Error("expected error when writing to read-only directory")
	}
}
