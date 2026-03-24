package capture

import (
	"encoding/json"
	"testing"
	"time"
)

func TestNewCaptureWriter(t *testing.T) {
	start := time.Now()
	cw := NewCaptureWriter(start)

	if cw == nil {
		t.Fatal("NewCaptureWriter returned nil")
	}

	chunks := cw.Chunks()
	if chunks == nil {
		t.Fatal("Chunks() returned nil")
	}
	if len(chunks) != 0 {
		t.Fatalf("expected 0 chunks, got %d", len(chunks))
	}
}

func TestCaptureWriter_RecordChunk(t *testing.T) {
	tests := []struct {
		name       string
		event      string
		data       []byte
		wantChunks int
	}{
		{
			name:       "valid JSON data",
			event:      "message",
			data:       []byte(`{"id": "test-123", "content": "hello"}`),
			wantChunks: 1,
		},
		{
			name:       "empty data",
			event:      "message",
			data:       []byte{},
			wantChunks: 0,
		},
		{
			name:       "nil data",
			event:      "message",
			data:       nil,
			wantChunks: 0,
		},
		{
			name:       "invalid JSON data",
			event:      "message",
			data:       []byte(`not valid json`),
			wantChunks: 1,
		},
		{
			name:       "multiple records",
			event:      "message",
			data:       []byte(`{"a": 1}`),
			wantChunks: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cw := NewCaptureWriter(time.Now())
			cw.RecordChunk(tt.event, tt.data)

			chunks := cw.Chunks()
			if len(chunks) != tt.wantChunks {
				t.Fatalf("expected %d chunks, got %d", tt.wantChunks, len(chunks))
			}
		})
	}
}

func TestCaptureWriter_RecordChunk_Multiple(t *testing.T) {
	cw := NewCaptureWriter(time.Now())

	cw.RecordChunk("message", []byte(`{"id": "1"}`))
	cw.RecordChunk("message", []byte(`{"id": "2"}`))
	cw.RecordChunk("ping", []byte(`{}`))

	chunks := cw.Chunks()
	if len(chunks) != 3 {
		t.Fatalf("expected 3 chunks, got %d", len(chunks))
	}

	if chunks[0].Event != "message" {
		t.Errorf("expected event 'message', got %q", chunks[0].Event)
	}
	if chunks[1].Event != "message" {
		t.Errorf("expected event 'message', got %q", chunks[1].Event)
	}
	if chunks[2].Event != "ping" {
		t.Errorf("expected event 'ping', got %q", chunks[2].Event)
	}
}

func TestCaptureWriter_Chunks(t *testing.T) {
	cw := NewCaptureWriter(time.Now())

	chunks1 := cw.Chunks()
	chunks1 = append(chunks1, SSEChunk{})

	chunks2 := cw.Chunks()
	if len(chunks2) != 0 {
		t.Fatal("Chunks() should return a copy or the internal slice should be immutable")
	}
}

func TestNewSSEChunk(t *testing.T) {
	tests := []struct {
		name     string
		offsetMS int64
		event    string
		data     []byte
		wantData bool
		wantRaw  bool
	}{
		{
			name:     "valid JSON object",
			offsetMS: 100,
			event:    "message",
			data:     []byte(`{"key": "value"}`),
			wantData: true,
			wantRaw:  false,
		},
		{
			name:     "valid JSON array",
			offsetMS: 200,
			event:    "data",
			data:     []byte(`[1, 2, 3]`),
			wantData: true,
			wantRaw:  false,
		},
		{
			name:     "valid JSON string",
			offsetMS: 300,
			event:    "message",
			data:     []byte(`"hello"`),
			wantData: true,
			wantRaw:  false,
		},
		{
			name:     "invalid JSON",
			offsetMS: 400,
			event:    "error",
			data:     []byte(`not json at all`),
			wantData: false,
			wantRaw:  true,
		},
		{
			name:     "empty data",
			offsetMS: 500,
			event:    "empty",
			data:     []byte{},
			wantData: false,
			wantRaw:  false,
		},
		{
			name:     "partial JSON",
			offsetMS: 600,
			event:    "partial",
			data:     []byte(`{"incomplete": `),
			wantData: false,
			wantRaw:  true,
		},
		{
			name:     "empty event",
			offsetMS: 700,
			event:    "",
			data:     []byte(`{}`),
			wantData: true,
			wantRaw:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chunk := NewSSEChunk(tt.offsetMS, tt.event, tt.data)

			if chunk.OffsetMS != tt.offsetMS {
				t.Errorf("expected OffsetMS %d, got %d", tt.offsetMS, chunk.OffsetMS)
			}

			if chunk.Event != tt.event {
				t.Errorf("expected Event %q, got %q", tt.event, chunk.Event)
			}

			if tt.wantData && len(chunk.Data) == 0 {
				t.Error("expected Data to be populated, got empty")
			}

			if !tt.wantData && len(chunk.Data) > 0 {
				t.Errorf("expected Data to be empty, got %s", chunk.Data)
			}

			if tt.wantRaw && chunk.Raw == "" {
				t.Error("expected Raw to be populated, got empty")
			}

			if !tt.wantRaw && chunk.Raw != "" {
				t.Errorf("expected Raw to be empty, got %q", chunk.Raw)
			}
		})
	}
}

func TestNewSSEChunk_DataPreservation(t *testing.T) {
	originalData := []byte(`{"id": "test-123", "nested": {"key": "value"}}`)
	chunk := NewSSEChunk(0, "test", originalData)

	var parsed map[string]interface{}
	if err := json.Unmarshal(chunk.Data, &parsed); err != nil {
		t.Fatalf("failed to parse chunk data: %v", err)
	}

	if parsed["id"] != "test-123" {
		t.Errorf("expected id 'test-123', got %v", parsed["id"])
	}
}

func TestSSEChunk_JSONSerialization(t *testing.T) {
	tests := []struct {
		name  string
		chunk SSEChunk
	}{
		{
			name: "with data",
			chunk: SSEChunk{
				OffsetMS: 123,
				Event:    "message",
				Data:     json.RawMessage(`{"test": true}`),
			},
		},
		{
			name: "with raw",
			chunk: SSEChunk{
				OffsetMS: 456,
				Event:    "error",
				Raw:      "raw text data",
			},
		},
		{
			name: "minimal",
			chunk: SSEChunk{
				OffsetMS: 789,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.chunk)
			if err != nil {
				t.Fatalf("failed to marshal chunk: %v", err)
			}

			var parsed SSEChunk
			if err := json.Unmarshal(data, &parsed); err != nil {
				t.Fatalf("failed to unmarshal chunk: %v", err)
			}

			if parsed.OffsetMS != tt.chunk.OffsetMS {
				t.Errorf("expected OffsetMS %d, got %d", tt.chunk.OffsetMS, parsed.OffsetMS)
			}
		})
	}
}

func TestExtractRequestIDFromSSEChunk(t *testing.T) {
	tests := []struct {
		name   string
		data   json.RawMessage
		wantID string
	}{
		{
			name:   "valid top-level ID",
			data:   json.RawMessage(`{"id": "req-123", "content": "test"}`),
			wantID: "req-123",
		},
		{
			name:   "empty top-level ID",
			data:   json.RawMessage(`{"id": "", "content": "test"}`),
			wantID: "",
		},
		{
			name:   "missing ID field",
			data:   json.RawMessage(`{"content": "test"}`),
			wantID: "",
		},
		{
			name:   "invalid JSON",
			data:   json.RawMessage(`not json`),
			wantID: "",
		},
		{
			name:   "empty data",
			data:   json.RawMessage{},
			wantID: "",
		},
		{
			name:   "nil data",
			data:   nil,
			wantID: "",
		},
		{
			name:   "ID with nested structure",
			data:   json.RawMessage(`{"id": "nested-test", "data": {"nested": "value"}}`),
			wantID: "nested-test",
		},
		{
			name:   "Anthropic message_start format",
			data:   json.RawMessage(`{"type": "message_start", "message": {"id": "msg_abc123", "role": "assistant"}}`),
			wantID: "msg_abc123",
		},
		{
			name:   "Anthropic message_start with empty message ID",
			data:   json.RawMessage(`{"type": "message_start", "message": {"id": "", "role": "assistant"}}`),
			wantID: "",
		},
		{
			name:   "Anthropic message_start missing message.id",
			data:   json.RawMessage(`{"type": "message_start", "message": {"role": "assistant"}}`),
			wantID: "",
		},
		{
			name:   "top-level ID takes precedence over nested",
			data:   json.RawMessage(`{"id": "top-level-id", "type": "message_start", "message": {"id": "nested-id"}}`),
			wantID: "top-level-id",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotID := ExtractRequestIDFromSSEChunk(tt.data)
			if gotID != tt.wantID {
				t.Errorf("expected ID %q, got %q", tt.wantID, gotID)
			}
		})
	}
}

func TestExtractTokenUsageFromChunks(t *testing.T) {
	tests := []struct {
		name         string
		chunks       []SSEChunk
		wantInput    int
		wantOutput   int
		wantCache    int
		wantCacheCrt int
	}{
		{
			name:   "empty chunks",
			chunks: []SSEChunk{},
		},
		{
			name: "OpenAI format with usage",
			chunks: []SSEChunk{
				{Data: json.RawMessage(`{"id":"1","choices":[{"delta":{"content":"hi"}}]}`)},
				{Data: json.RawMessage(`{"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`)},
			},
			wantInput:  10,
			wantOutput: 5,
		},
		{
			name: "Anthropic format with usage",
			chunks: []SSEChunk{
				{Data: json.RawMessage(`{"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":25,"output_tokens":0}}}`)},
				{Data: json.RawMessage(`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":15}}`)},
			},
			wantInput:  25,
			wantOutput: 15,
		},
		{
			name: "Anthropic format with cache tokens",
			chunks: []SSEChunk{
				{Data: json.RawMessage(`{"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":100,"output_tokens":0,"cache_read_input_tokens":50,"cache_creation_input_tokens":10}}}`)},
				{Data: json.RawMessage(`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":20}}`)},
			},
			wantInput:    100,
			wantOutput:   20,
			wantCache:    50,
			wantCacheCrt: 10,
		},
		{
			name: "Responses API format with usage",
			chunks: []SSEChunk{
				{Data: json.RawMessage(`{"type":"response.completed","response":{"id":"resp_1","usage":{"input_tokens":30,"output_tokens":10,"total_tokens":40}}}`)},
			},
			wantInput:  30,
			wantOutput: 10,
		},
		{
			name: "OpenAI format with cache tokens",
			chunks: []SSEChunk{
				{Data: json.RawMessage(`{"usage":{"prompt_tokens":100,"completion_tokens":20,"total_tokens":120,"prompt_tokens_details":{"cached_tokens":40}}}`)},
			},
			wantInput: 100,
			wantOutput: 20,
			wantCache: 40,
		},
		{
			name: "chunk with raw data only",
			chunks: []SSEChunk{
				{Raw: "some raw data"},
			},
		},
		{
			name: "invalid JSON",
			chunks: []SSEChunk{
				{Data: json.RawMessage(`not json`)},
			},
		},
		{
			name: "no usage field",
			chunks: []SSEChunk{
				{Data: json.RawMessage(`{"id":"1","choices":[]}`)},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ExtractTokenUsageFromChunks(tt.chunks)
			if got.InputTokens != tt.wantInput {
				t.Errorf("expected InputTokens %d, got %d", tt.wantInput, got.InputTokens)
			}
			if got.OutputTokens != tt.wantOutput {
				t.Errorf("expected OutputTokens %d, got %d", tt.wantOutput, got.OutputTokens)
			}
			if got.CacheReadTokens != tt.wantCache {
				t.Errorf("expected CacheReadTokens %d, got %d", tt.wantCache, got.CacheReadTokens)
			}
			if got.CacheCreationTokens != tt.wantCacheCrt {
				t.Errorf("expected CacheCreationTokens %d, got %d", tt.wantCacheCrt, got.CacheCreationTokens)
			}
		})
	}
}
