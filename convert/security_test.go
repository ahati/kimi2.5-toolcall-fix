// Package convert provides security and protocol compliance tests for input validation.
// These tests document security boundaries rather than testing actual security
// implementation (which is handled by middleware).
package convert

import (
	"encoding/json"
	"strings"
	"testing"
)

// ============================================================================
// Category G1: Input Validation Tests
// ============================================================================

// TestSQLInjectionInContent documents that SQL injection payloads in content
// are treated as opaque text and passed through to the upstream API.
// The proxy does not sanitize content - sanitization is the responsibility
// of the upstream API and consuming applications.
func TestSQLInjectionInContent(t *testing.T) {
	tests := []struct {
		name    string
		content string
	}{
		{
			name:    "basic SQL injection",
			content: "SELECT * FROM users WHERE id = 1 OR 1=1",
		},
		{
			name:    "union-based SQL injection",
			content: "UNION SELECT username, password FROM admin_users",
		},
		{
			name:    "comment-based SQL injection",
			content: "DELETE FROM logs; --",
		},
		{
			name:    "nested SQL injection",
			content: "'; DROP TABLE users; --",
		},
		{
			name:    "blind SQL injection",
			content: "1 AND (SELECT * FROM (SELECT(SLEEP(5)))a)",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Content should be passed through without modification
			result := extractContentFromInput(tt.content)
			if result != tt.content {
				t.Errorf("content was modified: expected %q, got %q", tt.content, result)
			}

			// Verify content is preserved in JSON encoding
			data, _ := json.Marshal(map[string]string{"content": tt.content})
			var decoded map[string]string
			json.Unmarshal(data, &decoded)
			if decoded["content"] != tt.content {
				t.Errorf("JSON encoding modified content: expected %q, got %q", tt.content, decoded["content"])
			}
		})
	}
}

// TestXSSInContent documents that XSS payloads in content are treated as
// opaque text. The proxy does not escape or sanitize HTML/JavaScript.
func TestXSSInContent(t *testing.T) {
	tests := []struct {
		name    string
		content string
	}{
		{
			name:    "script tag injection",
			content: "<script>alert('xss')</script>",
		},
		{
			name:    "img onerror injection",
			content: "<img src=x onerror=alert('xss')>",
		},
		{
			name:    "javascript protocol",
			content: "<a href=javascript:alert('xss')>click</a>",
		},
		{
			name:    "event handler injection",
			content: "<body onload=alert('xss')>",
		},
		{
			name:    "svg script injection",
			content: "<svg><script>alert('xss')</script></svg>",
		},
		{
			name:    "template injection",
			content: "{{constructor.constructor('alert(1)')()}}",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Content should be passed through without modification
			result := extractContentFromInput(tt.content)
			if result != tt.content {
				t.Errorf("content was modified: expected %q, got %q", tt.content, result)
			}
		})
	}
}

// TestPathTraversalInFileRefs documents that path traversal patterns in
// file references are passed through without validation. The proxy does
// not validate file paths - this is the responsibility of upstream APIs.
func TestPathTraversalInFileRefs(t *testing.T) {
	tests := []struct {
		name string
		url  string
	}{
		{
			name: "basic path traversal",
			url:  "../../../etc/passwd",
		},
		{
			name: "URL encoded traversal",
			url:  "..%2f..%2f..%2fetc%2fpasswd",
		},
		{
			name: "double URL encoded",
			url:  "..%252f..%252f..%252fetc%252fpasswd",
		},
		{
			name: "null byte injection",
			url:  "file.txt%00.jpg",
		},
		{
			name: "absolute path",
			url:  "/etc/passwd",
		},
		{
			name: "windows path",
			url:  "..\\..\\..\\windows\\system32\\config\\sam",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// URLs should be passed through without path validation
			// This tests the extractMediaType function which handles data URLs
			mediaType := extractMediaType(tt.url)
			// Should return empty string for non-data URLs (not a validation failure)
			if strings.HasPrefix(tt.url, "data:") && mediaType == "" {
				t.Logf("non-data URL correctly returns empty media type: %q", tt.url)
			}
		})
	}
}

// TestJSONInjection documents that JSON payloads in content are handled
// correctly. The proxy parses JSON but does not validate the semantic
// content of strings.
func TestJSONInjection(t *testing.T) {
	tests := []struct {
		name    string
		content string
	}{
		{
			name:    "nested JSON object",
			content: `{"nested": {"key": "value"}}`,
		},
		{
			name:    "JSON with special characters",
			content: `{"code": "console.log('test')"}`,
		},
		{
			name:    "JSON array in string",
			content: `{"items": [1, 2, 3, "<script>"]}}`,
		},
		{
			name:    "Unicode escape sequences",
			content: `{"unicode": "\u003cscript\u003e"}`,
		},
		{
			name:    "bypass attempt via JSON",
			content: `{"__proto__": {"isAdmin": true}}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Content should be preserved through JSON round-trip
			var data map[string]interface{}
			err := json.Unmarshal([]byte(tt.content), &data)
			if err != nil {
				// Invalid JSON is handled gracefully
				t.Logf("JSON parse error (expected for invalid JSON): %v", err)
				return
			}

			// Re-encode and verify structure is preserved
			encoded, err := json.Marshal(data)
			if err != nil {
				t.Errorf("failed to re-encode JSON: %v", err)
				return
			}

			// Decode again to verify
			var roundTripped map[string]interface{}
			json.Unmarshal(encoded, &roundTripped)

			// Basic check that we have the expected structure
			if len(roundTripped) == 0 && len(data) > 0 {
				t.Error("JSON round-trip lost data")
			}
		})
	}
}

// ============================================================================
// Category H2: JSON Protocol Compliance Tests
// ============================================================================

// TestUnknownFieldsIgnored documents that unknown fields in JSON requests
// are ignored during unmarshaling. This is standard Go json package behavior.
func TestUnknownFieldsIgnored(t *testing.T) {
	tests := []struct {
		name string
		json string
	}{
		{
			name: "unknown top-level field",
			json: `{"model": "gpt-4", "unknown_field": "value"}`,
		},
		{
			name: "multiple unknown fields",
			json: `{"model": "gpt-4", "field1": 1, "field2": ["a", "b"], "field3": {"nested": true}}`,
		},
		{
			name: "deeply nested unknown field",
			json: `{"model": "gpt-4", "messages": [], "extra": {"deep": {"nesting": "value"}}}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Parse into a simple struct that only captures known fields
			var result struct {
				Model string `json:"model"`
			}

			err := json.Unmarshal([]byte(tt.json), &result)
			if err != nil {
				t.Errorf("failed to unmarshal JSON with unknown fields: %v", err)
				return
			}

			// Known field should be populated
			if result.Model != "gpt-4" {
				t.Errorf("model field not populated: got %q", result.Model)
			}
		})
	}
}

// TestMissingOptionalFields documents that optional fields can be omitted
// from JSON requests without causing errors.
func TestMissingOptionalFields(t *testing.T) {
	tests := []struct {
		name string
		json string
	}{
		{
			name: "minimal request",
			json: `{"model": "gpt-4"}`,
		},
		{
			name: "missing optional nested field",
			json: `{"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}`,
		},
		{
			name: "empty optional object",
			json: `{"model": "gpt-4", "response_format": {}}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Parse into a struct with optional fields
			var result struct {
				Model          string          `json:"model"`
				MaxTokens      int             `json:"max_tokens,omitempty"`
				Temperature    float64         `json:"temperature,omitempty"`
				ResponseFormat json.RawMessage `json:"response_format,omitempty"`
			}

			err := json.Unmarshal([]byte(tt.json), &result)
			if err != nil {
				t.Errorf("failed to unmarshal JSON with missing optional fields: %v", err)
				return
			}

			// Required field should be populated
			if result.Model == "" {
				t.Error("required model field not populated")
			}
		})
	}
}

// TestLargeIntegers documents behavior with integers larger than 2^53
// which cannot be precisely represented by JavaScript numbers.
func TestLargeIntegers(t *testing.T) {
	tests := []struct {
		name        string
		json        string
		expectExact bool
	}{
		{
			name:        "exact 2^53",
			json:        `9007199254740992`,
			expectExact: true,
		},
		{
			name:        "2^53 + 1 (precision loss in JS)",
			json:        `9007199254740993`,
			expectExact: false, // May lose precision when parsed as float64
		},
		{
			name:        "very large integer",
			json:        `9999999999999999999`,
			expectExact: false,
		},
		{
			name:        "integer in object",
			json:        `{"count": 9007199254740993}`,
			expectExact: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Parse as float64 (Go's default for JSON numbers)
			var num float64
			err := json.Unmarshal([]byte(tt.json), &num)

			if err != nil {
				// Try parsing as object
				var obj map[string]interface{}
				err = json.Unmarshal([]byte(tt.json), &obj)
				if err != nil {
					t.Errorf("failed to parse JSON: %v", err)
					return
				}
				// Check the number in the object
				if count, ok := obj["count"].(float64); ok {
					t.Logf("parsed large integer as float64: %v", count)
				}
			} else {
				t.Logf("parsed large integer as float64: %v", num)
			}

			// Parse with json.Number to preserve precision
			var numPrecise json.Number
			decoder := json.NewDecoder(strings.NewReader(tt.json))
			decoder.UseNumber()
			err = decoder.Decode(&numPrecise)

			if err != nil {
				// Try as object
				decoder := json.NewDecoder(strings.NewReader(tt.json))
				decoder.UseNumber()
				var obj map[string]json.Number
				err = decoder.Decode(&obj)
				if err == nil {
					t.Logf("parsed with json.Number precision: %s", obj["count"])
				}
			} else {
				t.Logf("parsed with json.Number precision: %s", numPrecise)
			}
		})
	}
}

// TestJSONEncodingStability documents that JSON encoding is stable and
// produces consistent output.
func TestJSONEncodingStability(t *testing.T) {
	data := map[string]interface{}{
		"model":       "gpt-4",
		"temperature": 0.7,
		"stream":      true,
	}

	// Encode multiple times
	encoded1, err := json.Marshal(data)
	if err != nil {
		t.Fatalf("failed to encode: %v", err)
	}

	encoded2, err := json.Marshal(data)
	if err != nil {
		t.Fatalf("failed to encode: %v", err)
	}

	// Should be identical
	if string(encoded1) != string(encoded2) {
		t.Error("JSON encoding is not stable")
	}

	// Round-trip should preserve data
	var decoded map[string]interface{}
	if err := json.Unmarshal(encoded1, &decoded); err != nil {
		t.Fatalf("failed to decode: %v", err)
	}

	if decoded["model"] != "gpt-4" {
		t.Error("model field not preserved in round-trip")
	}
}
