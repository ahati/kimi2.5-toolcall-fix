// Package logging provides structured logging and request/response capture functionality.
// It captures bidirectional traffic between client and upstream LLM API for debugging
// and analysis purposes.
package logging

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// filePerms is the permission mode for created files.
const filePerms = 0644

// dirPerms is the permission mode for created directories.
const dirPerms = 0755

// Storage manages persistent storage of captured request/response data.
type Storage struct {
	// baseDir is the root directory for log files.
	baseDir string
}

// NewStorage creates a new Storage instance.
//
// @brief    Creates storage for writing capture files.
// @param    baseDir Root directory for log files.
// @return   Pointer to new Storage instance.
//
// @pre      baseDir should be a valid filesystem path.
// @post     Storage is ready to write capture files.
// @note     Directory is created on-demand during Write.
func NewStorage(baseDir string) *Storage {
	return &Storage{baseDir: baseDir}
}

// Write persists captured request data to a JSON file.
//
// @brief    Writes request recorder data to timestamped JSON file.
// @param    recorder RequestRecorder containing captured data.
// @return   Error if write fails, nil on success.
//
// @pre      Storage must be initialized with baseDir.
// @post     JSON file created in baseDir/YYYY-MM-DD/ directory.
// @post     Filename format: YYYYMMDD-HHMMSS_requestID.json
// @note     Creates date-based subdirectory if not exists.
// @note     Appends suffix if file already exists.
// @note     File is written atomically with O_EXCL flag.
func (s *Storage) Write(recorder *RequestRecorder) error {
	// Create date-based subdirectory (e.g., ./logs/2024-01-15/)
	// This organizes logs by day for easier navigation and cleanup
	dir := s.logDir()
	if err := os.MkdirAll(dir, dirPerms); err != nil {
		return fmt.Errorf("create log dir: %w", err)
	}

	// Generate filename with timestamp and request ID
	filename := s.filename(recorder.RequestID)
	fullpath := filepath.Join(dir, filename)

	// Check if file already exists to avoid overwriting
	// This can happen if multiple requests with the same ID arrive in the same second
	if _, err := os.Stat(fullpath); err == nil {
		filename = s.filenameWithTimestamp(recorder.RequestID)
		fullpath = filepath.Join(dir, filename)
	}

	data := s.serialize(recorder)

	// Use O_EXCL flag for atomic file creation - fails if file exists
	// This prevents race conditions when multiple goroutines try to write the same file
	file, err := os.OpenFile(fullpath, os.O_CREATE|os.O_WRONLY|os.O_EXCL, filePerms)
	if err != nil {
		return fmt.Errorf("create log file: %w", err)
	}
	defer file.Close()

	// Use streaming JSON encoder for memory efficiency with large payloads
	// SetIndent formats output for human readability during debugging
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(data); err != nil {
		return fmt.Errorf("encode log data: %w", err)
	}

	return nil
}

// logDir returns the date-based directory path for log files.
//
// @brief    Generates directory path with current date.
// @return   Path in format baseDir/YYYY-MM-DD.
//
// @pre      Storage must be initialized.
// @note     Uses current system time for date.
func (s *Storage) logDir() string {
	return filepath.Join(s.baseDir, time.Now().Format("2006-01-02"))
}

// filename generates a log filename from request ID.
//
// @brief    Creates filename with timestamp and sanitized request ID.
// @param    requestID Unique identifier for the request.
// @return   Filename in format YYYYMMDD-HHMMSS_requestID.json.
//
// @pre      requestID should be a valid string.
// @note     Request ID is sanitized for filesystem safety.
func (s *Storage) filename(requestID string) string {
	timestamp := time.Now().Format("20060102-150405")
	safeID := sanitizeFilename(requestID)
	return fmt.Sprintf("%s_%s.json", timestamp, safeID)
}

// filenameWithTimestamp generates a log filename with duplicate suffix.
//
// @brief    Creates filename for handling duplicate files.
// @param    requestID Unique identifier for the request.
// @return   Filename in format YYYYMMDD-HHMMSS_requestID_1.json.
//
// @pre      requestID should be a valid string.
// @note     Used when primary filename already exists.
func (s *Storage) filenameWithTimestamp(requestID string) string {
	timestamp := time.Now().Format("20060102-150405")
	safeID := sanitizeFilename(requestID)
	return fmt.Sprintf("%s_%s_1.json", timestamp, safeID)
}

// sanitizeFilename removes unsafe characters from filenames.
//
// @brief    Replaces filesystem-unsafe characters with underscores.
// @param    name Original filename string.
// @return   Sanitized filename safe for filesystem use.
//
// @pre      None.
// @note     Replaces: / \ : * ? " < > | and spaces.
func sanitizeFilename(name string) string {
	// Replace filesystem-unsafe characters with underscores
	// These characters are forbidden on Windows and could cause issues on Unix
	// - / and \ : path separators that could create unwanted directories
	// - : : used in Windows drive letters and as time separator
	// - * ? " < > | : wildcard and redirect characters, forbidden on Windows
	// - space : replaced for cleaner filenames and shell compatibility
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

// logData is the JSON structure for persisted capture data.
type logData struct {
	// RequestID is the unique identifier for the request.
	RequestID string `json:"request_id"`
	// StartedAt is when the request was first received.
	StartedAt time.Time `json:"started_at"`
	// DurationMS is total request duration in milliseconds.
	DurationMS int64 `json:"duration_ms,omitempty"`
	// Method is the HTTP method (GET, POST, etc.).
	Method string `json:"method"`
	// Path is the request URL path.
	Path string `json:"path"`
	// ClientIP is the client's IP address.
	ClientIP string `json:"client_ip,omitempty"`
	// DownstreamRequest captures the client-to-proxy request.
	DownstreamRequest *HTTPRequestCapture `json:"downstream_request,omitempty"`
	// UpstreamRequest captures the proxy-to-LLM request.
	UpstreamRequest *HTTPRequestCapture `json:"upstream_request,omitempty"`
	// UpstreamResponse captures the LLM-to-proxy response.
	UpstreamResponse *SSEResponseCapture `json:"upstream_response,omitempty"`
	// DownstreamResponse captures the proxy-to-client response.
	DownstreamResponse *SSEResponseCapture `json:"downstream_response,omitempty"`
}

// serialize converts RequestRecorder to logData for JSON output.
//
// @brief    Transforms recorder data for JSON serialization.
// @param    r RequestRecorder containing captured data.
// @return   logData structure ready for JSON encoding.
//
// @pre      r should not be nil.
// @note     Calculates duration from StartedAt to current time.
func (s *Storage) serialize(r *RequestRecorder) logData {
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
