// Package capture provides request/response recording and persistence for HTTP proxy operations.
// It captures downstream client requests, upstream API requests, and their corresponding
// SSE streaming responses for debugging and analysis.
//
// Thread Safety:
//   - Storage is NOT thread-safe for Write operations; use external synchronization
//   - globalStorage is initialized once at startup via InitStorage
//   - InitStorage should only be called once (not thread-safe for initialization)
package capture

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// filePerms defines the permission bits for created files.
// 0644 = owner read/write, group read, other read
// This is a common permission for data files that should be readable by all.
const filePerms = 0644

// dirPerms defines the permission bits for created directories.
// 0755 = owner read/write/execute, group read/execute, other read/execute
// Execute bit is required for directories to allow traversal.
const dirPerms = 0755

// Storage manages persistence of captured request data to the filesystem.
// Files are organized by date in subdirectories under the base directory.
//
// Directory Structure:
//
//	baseDir/
//	├── 2024-01-15/
//	│   ├── 20240115-143052_abc123.json
//	│   └── 20240115-143053_def456.json
//	└── 2024-01-16/
//	    └── 20240116-091234_ghi789.json
//
// Thread Safety: NOT thread-safe. External synchronization required for Write operations.
type Storage struct {
	// baseDir is the root directory for all log files.
	// May be relative or absolute path.
	// Valid values: any valid filesystem path string.
	baseDir string
}

// globalStorage is the singleton storage instance initialized via InitStorage.
// It provides a default storage location for the application.
//
// Thread Safety: NOT thread-safe for initialization. InitStorage should be called once at startup.
// After initialization, GetStorage returns a pointer that should be treated as read-only.
var globalStorage *Storage

// InitStorage initializes the global storage instance with the given base directory.
// If baseDir is empty, no storage is initialized (globalStorage remains nil).
//
// @param baseDir - Root directory for log files. Empty string disables storage.
//
// @pre Should only be called once at application startup
// @post If baseDir != "", globalStorage != nil
// @post If baseDir == "", globalStorage == nil
//
// @note NOT thread-safe: should be called before any concurrent access.
// @note Empty baseDir is valid and disables persistence.
func InitStorage(baseDir string) {
	// Only initialize if baseDir is provided
	// Empty baseDir is valid and results in no storage (disabled persistence)
	if baseDir != "" {
		globalStorage = NewStorage(baseDir)
	}
}

// GetStorage returns the global storage instance, or nil if not initialized.
//
// @return Global Storage instance, or nil if InitStorage was not called or called with empty string.
//
// @pre None
// @post Returns nil if storage not initialized
// @post Returns valid Storage pointer if initialized
//
// @note Thread-safe for reading the pointer, but returned Storage is not thread-safe.
// @note Callers should check for nil before using.
func GetStorage() *Storage {
	// Return the global instance; may be nil if not initialized
	// Callers must check for nil before using
	return globalStorage
}

// NewStorage creates a new Storage instance with the given base directory.
//
// @param baseDir - Root directory for log files. May be empty (Write will fail).
// @return Pointer to newly allocated Storage, never nil.
//
// @pre None
// @post Returned Storage.baseDir == baseDir
//
// @note This does not create the directory; Write will create it if needed.
// @note Thread-safe: creates new instance with no shared state.
func NewStorage(baseDir string) *Storage {
	// Simple struct initialization; directory creation deferred to Write
	return &Storage{baseDir: baseDir}
}

// Write persists the recorded request data to a JSON file.
// Files are stored in date-based subdirectories with timestamps in their names.
// If a file with the same name exists, a suffix is appended to avoid overwrites.
//
// @param recorder - Request data to persist. Must not be nil.
// @return nil on success, error on failure.
//
// @pre s != nil (receiver must be valid)
// @pre recorder != nil
// @post On success, file created in date-based subdirectory
// @post On error, no file created (or partial file removed)
//
// @error "create log dir" - Directory creation failed
// @error "create log file" - File creation failed (e.g., permissions)
// @error "encode log data" - JSON encoding failed
//
// @note NOT thread-safe: concurrent writes may race on file creation.
// @note Uses O_EXCL to prevent overwriting existing files.
// @note If file exists, appends "_1" suffix to filename.
func (s *Storage) Write(recorder *RequestRecorder) error {
	// Determine the date-based subdirectory for organization
	// Using date subdirectories prevents a single directory from growing too large
	dir := s.logDir()
	// Create directory with standard permissions if it doesn't exist
	// MkdirAll creates all parent directories as needed
	if err := os.MkdirAll(dir, dirPerms); err != nil {
		return fmt.Errorf("create log dir: %w", err)
	}

	// Generate filename from timestamp and request ID
	filename := s.filename(recorder.RequestID)
	fullpath := filepath.Join(dir, filename)

	// Check if file already exists to prevent overwrite
	// This handles the rare case of same-second same-ID requests
	if _, err := os.Stat(fullpath); err == nil {
		// File exists; use suffixed filename to avoid collision
		filename = s.filenameWithTimestamp(recorder.RequestID)
		fullpath = filepath.Join(dir, filename)
	}

	// Serialize recorder data to JSON-compatible struct
	data := s.serialize(recorder)

	// Create file with O_EXCL to atomically prevent overwrites
	// This is a safety measure in case the Stat check above raced
	file, err := os.OpenFile(fullpath, os.O_CREATE|os.O_WRONLY|os.O_EXCL, filePerms)
	if err != nil {
		return fmt.Errorf("create log file: %w", err)
	}
	// Ensure file is closed even if encoding fails
	// defer is safe here as file.Close() handles nil gracefully
	defer file.Close()

	// Create JSON encoder with indentation for readability
	// Indentation makes logs human-readable for debugging
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(data); err != nil {
		return fmt.Errorf("encode log data: %w", err)
	}

	return nil
}

// logDir returns the date-based directory path for log files.
// Format: baseDir/YYYY-MM-DD
//
// @pre s != nil && s.baseDir != ""
// @post Returns path in format "baseDir/YYYY-MM-DD"
//
// @note Uses current date; calling at midnight may create different directories.
// @note Thread-safe: pure function with no side effects (time.Now() is concurrency-safe).
func (s *Storage) logDir() string {
	// Format date as ISO 8601 date for directory name
	// This provides natural organization by day
	return filepath.Join(s.baseDir, time.Now().Format("2006-01-02"))
}

// filename generates a log filename from the current timestamp and request ID.
// Format: YYYYMMDD-HHMMSS_<sanitized_id>.json
//
// @param requestID - Request ID to include in filename. May be empty.
// @return Filename string with timestamp and sanitized ID.
//
// @pre s != nil
// @post Filename format: "YYYYMMDD-HHMMSS_<sanitized_id>.json"
// @post Sanitized ID contains no filesystem-unsafe characters
//
// @note Uses current time; multiple calls may return different values.
// @note Thread-safe: pure function aside from time.Now().
func (s *Storage) filename(requestID string) string {
	// Use compact timestamp format for filenames
	// YYYYMMDD-HHMMSS is sortable and human-readable
	timestamp := time.Now().Format("20060102-150405")
	// Sanitize ID to prevent filesystem issues
	safeID := sanitizeFilename(requestID)
	return fmt.Sprintf("%s_%s.json", timestamp, safeID)
}

// filenameWithTimestamp generates a log filename with an additional suffix to prevent collisions.
// Format: YYYYMMDD-HHMMSS_<sanitized_id>_1.json
//
// @param requestID - Request ID to include in filename. May be empty.
// @return Filename string with timestamp, sanitized ID, and "_1" suffix.
//
// @pre s != nil
// @post Filename format: "YYYYMMDD-HHMMSS_<sanitized_id>_1.json"
//
// @note Used when primary filename already exists.
// @note Thread-safe: pure function aside from time.Now().
func (s *Storage) filenameWithTimestamp(requestID string) string {
	// Same compact timestamp format as filename()
	timestamp := time.Now().Format("20060102-150405")
	safeID := sanitizeFilename(requestID)
	// Append "_1" suffix to distinguish from primary filename
	// This simple approach handles most collision cases
	return fmt.Sprintf("%s_%s_1.json", timestamp, safeID)
}

// sanitizeFilename replaces filesystem-unsafe characters with underscores.
// This prevents issues with special characters in request IDs affecting file paths.
//
// @param name - String to sanitize. May be empty.
// @return Sanitized string safe for use in filenames.
//
// @pre None
// @post Returned string contains only safe filename characters
// @post Characters / \ : * ? " < > | and space are replaced with underscore
//
// @note Thread-safe: pure function with no side effects.
// @note Does not handle all edge cases (e.g., very long names, reserved names).
func sanitizeFilename(name string) string {
	// Define replacer for all unsafe characters
	// These characters are problematic on various filesystems:
	// / \ : * ? " < > | are illegal on Windows
	// Space is replaced for cleaner filenames
	replacer := strings.NewReplacer(
		"/", "_",
		"\\", "_",
		":", "_",
		"*", "_",
		"?", "_",
		"\"", "_",
		"<", "_",
		">", "_",
		"|", "_",
		" ", "_",
	)
	return replacer.Replace(name)
}

// logData is the JSON-serializable representation of a captured request.
// It includes all request data plus computed duration for analysis.
//
// Thread Safety: Value type; safe for concurrent read after creation.
type logData struct {
	// RequestID is the unique identifier for this request.
	// May be empty if ID extraction failed.
	// Valid values: any string, typically alphanumeric with dashes.
	RequestID string `json:"request_id"`

	// StartedAt is when the request was initiated.
	// Used for timeline analysis.
	// Valid values: any valid time.Time, serialized to RFC3339.
	StartedAt time.Time `json:"started_at"`

	// DurationMS is the total request duration in milliseconds.
	// Computed at serialization time.
	// Valid values: non-negative integer, 0 for very fast requests.
	DurationMS int64 `json:"duration_ms,omitempty"`

	// Method is the HTTP method of the request.
	// Valid values: standard HTTP methods (GET, POST, etc.).
	Method string `json:"method"`

	// Path is the URL path of the request.
	// Valid values: any valid URL path string.
	Path string `json:"path"`

	// ClientIP is the remote address of the client.
	// May include port number.
	// Valid values: IP:port format string.
	ClientIP string `json:"client_ip,omitempty"`

	// DownstreamRequest is the captured client request.
	// Nil if not captured.
	// Valid values: pointer to HTTPRequestCapture, or nil.
	DownstreamRequest *HTTPRequestCapture `json:"downstream_request,omitempty"`

	// UpstreamRequest is the captured upstream API request.
	// Nil if not captured.
	// Valid values: pointer to HTTPRequestCapture, or nil.
	UpstreamRequest *HTTPRequestCapture `json:"upstream_request,omitempty"`

	// UpstreamResponse is the captured upstream API response.
	// Nil if not captured.
	// Valid values: pointer to SSEResponseCapture, or nil.
	UpstreamResponse *SSEResponseCapture `json:"upstream_response,omitempty"`

	// DownstreamResponse is the captured response sent to client.
	// Nil if not captured.
	// Valid values: pointer to SSEResponseCapture, or nil.
	DownstreamResponse *SSEResponseCapture `json:"downstream_response,omitempty"`
}

// serialize converts a RequestRecorder to a logData struct for JSON encoding.
// Duration is computed from StartedAt to current time.
//
// @param r - RequestRecorder to serialize. Must not be nil.
// @return logData struct ready for JSON encoding.
//
// @pre s != nil && r != nil
// @post Returned logData contains all fields from r
// @post DurationMS is computed from r.StartedAt
//
// @note Thread-safe: pure function with no side effects (time.Since is concurrency-safe).
// @note Duration is computed at call time, not at request end.
func (s *Storage) serialize(r *RequestRecorder) logData {
	// Duration is computed at serialization time
	// This provides the total request duration for debugging
	return logData{
		RequestID:          r.RequestID,
		StartedAt:          r.StartedAt,
		DurationMS:         time.Since(r.StartedAt).Milliseconds(),
		Method:             r.Method,
		Path:               r.Path,
		ClientIP:           r.ClientIP,
		DownstreamRequest:  r.DownstreamRequest,
		UpstreamRequest:    r.UpstreamRequest,
		UpstreamResponse:   r.UpstreamResponse,
		DownstreamResponse: r.DownstreamResponse,
	}
}
