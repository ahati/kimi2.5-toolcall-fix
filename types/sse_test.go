package types

import (
	"encoding/json"
	"testing"
)

func TestSSEChunk(t *testing.T) {
	tests := []struct {
		name     string
		input    SSEChunk
		wantJSON string
	}{
		{
			name:     "minimal chunk",
			input:    SSEChunk{OffsetMS: 100},
			wantJSON: `{"offset_ms":100}`,
		},
		{
			name:     "with event",
			input:    SSEChunk{OffsetMS: 200, Event: "message"},
			wantJSON: `{"offset_ms":200,"event":"message"}`,
		},
		{
			name:     "with data",
			input:    SSEChunk{OffsetMS: 300, Data: json.RawMessage(`{"type":"text","content":"hello"}`)},
			wantJSON: `{"offset_ms":300,"data":{"type":"text","content":"hello"}}`,
		},
		{
			name:     "with raw",
			input:    SSEChunk{OffsetMS: 400, Raw: "data: test\n\n"},
			wantJSON: `{"offset_ms":400,"raw":"data: test\n\n"}`,
		},
		{
			name: "full chunk",
			input: SSEChunk{
				OffsetMS: 500,
				Event:    "content_block_delta",
				Data:     json.RawMessage(`{"type":"text_delta","text":"world"}`),
				Raw:      "data: {\"type\":\"text_delta\"}\n\n",
			},
			wantJSON: `{"offset_ms":500,"event":"content_block_delta","data":{"type":"text_delta","text":"world"},"raw":"data: {\"type\":\"text_delta\"}\n\n"}`,
		},
		{
			name:     "zero offset",
			input:    SSEChunk{OffsetMS: 0},
			wantJSON: `{"offset_ms":0}`,
		},
		{
			name:     "empty strings",
			input:    SSEChunk{OffsetMS: 100, Event: "", Data: nil, Raw: ""},
			wantJSON: `{"offset_ms":100}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.input)
			if err != nil {
				t.Fatalf("marshal error: %v", err)
			}
			if string(data) != tt.wantJSON {
				t.Errorf("got %s, want %s", data, tt.wantJSON)
			}

			var unmarshaled SSEChunk
			if err := json.Unmarshal(data, &unmarshaled); err != nil {
				t.Fatalf("unmarshal error: %v", err)
			}
			if unmarshaled.OffsetMS != tt.input.OffsetMS {
				t.Errorf("offset_ms mismatch: got %d, want %d", unmarshaled.OffsetMS, tt.input.OffsetMS)
			}
		})
	}
}

func TestSSEChunkUnmarshal(t *testing.T) {
	tests := []struct {
		name    string
		jsonStr string
		want    SSEChunk
		wantErr bool
	}{
		{
			name:    "minimal",
			jsonStr: `{"offset_ms":123}`,
			want:    SSEChunk{OffsetMS: 123},
		},
		{
			name:    "with event and data",
			jsonStr: `{"offset_ms":456,"event":"ping","data":{"key":"value"}}`,
			want: SSEChunk{
				OffsetMS: 456,
				Event:    "ping",
				Data:     json.RawMessage(`{"key":"value"}`),
			},
		},
		{
			name:    "with raw",
			jsonStr: `{"offset_ms":789,"raw":"raw data"}`,
			want: SSEChunk{
				OffsetMS: 789,
				Raw:      "raw data",
			},
		},
		{
			name:    "all fields",
			jsonStr: `{"offset_ms":1000,"event":"test","data":[1,2,3],"raw":"line1\nline2"}`,
			want: SSEChunk{
				OffsetMS: 1000,
				Event:    "test",
				Data:     json.RawMessage(`[1,2,3]`),
				Raw:      "line1\nline2",
			},
		},
		{
			name:    "empty object",
			jsonStr: `{}`,
			want:    SSEChunk{},
		},
		{
			name:    "invalid json",
			jsonStr: `{invalid}`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got SSEChunk
			err := json.Unmarshal([]byte(tt.jsonStr), &got)
			if (err != nil) != tt.wantErr {
				t.Errorf("unmarshal error: %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if got.OffsetMS != tt.want.OffsetMS {
					t.Errorf("offset_ms: got %d, want %d", got.OffsetMS, tt.want.OffsetMS)
				}
				if got.Event != tt.want.Event {
					t.Errorf("event: got %s, want %s", got.Event, tt.want.Event)
				}
				if got.Raw != tt.want.Raw {
					t.Errorf("raw: got %s, want %s", got.Raw, tt.want.Raw)
				}
				if tt.want.Data != nil {
					if string(got.Data) != string(tt.want.Data) {
						t.Errorf("data: got %s, want %s", got.Data, tt.want.Data)
					}
				}
			}
		})
	}
}

func TestSSEChunkDataTypes(t *testing.T) {
	t.Run("data as object", func(t *testing.T) {
		chunk := SSEChunk{
			OffsetMS: 100,
			Data:     json.RawMessage(`{"name":"test","value":42}`),
		}
		data, err := json.Marshal(chunk)
		if err != nil {
			t.Fatalf("marshal error: %v", err)
		}
		var unmarshaled SSEChunk
		if err := json.Unmarshal(data, &unmarshaled); err != nil {
			t.Fatalf("unmarshal error: %v", err)
		}
		var obj map[string]interface{}
		if err := json.Unmarshal(unmarshaled.Data, &obj); err != nil {
			t.Fatalf("unmarshal data error: %v", err)
		}
		if obj["name"] != "test" {
			t.Errorf("name: got %v, want test", obj["name"])
		}
	})

	t.Run("data as array", func(t *testing.T) {
		chunk := SSEChunk{
			OffsetMS: 200,
			Data:     json.RawMessage(`[1,2,3]`),
		}
		data, err := json.Marshal(chunk)
		if err != nil {
			t.Fatalf("marshal error: %v", err)
		}
		var unmarshaled SSEChunk
		if err := json.Unmarshal(data, &unmarshaled); err != nil {
			t.Fatalf("unmarshal error: %v", err)
		}
		var arr []int
		if err := json.Unmarshal(unmarshaled.Data, &arr); err != nil {
			t.Fatalf("unmarshal data error: %v", err)
		}
		if len(arr) != 3 {
			t.Errorf("array length: got %d, want 3", len(arr))
		}
	})

	t.Run("data as string", func(t *testing.T) {
		chunk := SSEChunk{
			OffsetMS: 300,
			Data:     json.RawMessage(`"hello world"`),
		}
		data, err := json.Marshal(chunk)
		if err != nil {
			t.Fatalf("marshal error: %v", err)
		}
		var unmarshaled SSEChunk
		if err := json.Unmarshal(data, &unmarshaled); err != nil {
			t.Fatalf("unmarshal error: %v", err)
		}
		var str string
		if err := json.Unmarshal(unmarshaled.Data, &str); err != nil {
			t.Fatalf("unmarshal data error: %v", err)
		}
		if str != "hello world" {
			t.Errorf("string: got %s, want hello world", str)
		}
	})

	t.Run("data as null", func(t *testing.T) {
		chunk := SSEChunk{
			OffsetMS: 400,
			Data:     json.RawMessage(`null`),
		}
		data, err := json.Marshal(chunk)
		if err != nil {
			t.Fatalf("marshal error: %v", err)
		}
		var unmarshaled SSEChunk
		if err := json.Unmarshal(data, &unmarshaled); err != nil {
			t.Fatalf("unmarshal error: %v", err)
		}
		if string(unmarshaled.Data) != "null" {
			t.Errorf("data: got %s, want null", unmarshaled.Data)
		}
	})

	t.Run("data with nested json", func(t *testing.T) {
		nested := `{"outer":{"inner":{"value":123}}}`
		chunk := SSEChunk{
			OffsetMS: 500,
			Data:     json.RawMessage(nested),
		}
		data, err := json.Marshal(chunk)
		if err != nil {
			t.Fatalf("marshal error: %v", err)
		}
		var unmarshaled SSEChunk
		if err := json.Unmarshal(data, &unmarshaled); err != nil {
			t.Fatalf("unmarshal error: %v", err)
		}
		if string(unmarshaled.Data) != nested {
			t.Errorf("data: got %s, want %s", unmarshaled.Data, nested)
		}
	})
}

func TestSSEChunkEdgeCases(t *testing.T) {
	t.Run("large offset", func(t *testing.T) {
		chunk := SSEChunk{OffsetMS: 9223372036854775807}
		data, err := json.Marshal(chunk)
		if err != nil {
			t.Fatalf("marshal error: %v", err)
		}
		var unmarshaled SSEChunk
		if err := json.Unmarshal(data, &unmarshaled); err != nil {
			t.Fatalf("unmarshal error: %v", err)
		}
		if unmarshaled.OffsetMS != chunk.OffsetMS {
			t.Errorf("offset_ms: got %d, want %d", unmarshaled.OffsetMS, chunk.OffsetMS)
		}
	})

	t.Run("negative offset", func(t *testing.T) {
		chunk := SSEChunk{OffsetMS: -1}
		data, err := json.Marshal(chunk)
		if err != nil {
			t.Fatalf("marshal error: %v", err)
		}
		var unmarshaled SSEChunk
		if err := json.Unmarshal(data, &unmarshaled); err != nil {
			t.Fatalf("unmarshal error: %v", err)
		}
		if unmarshaled.OffsetMS != -1 {
			t.Errorf("offset_ms: got %d, want -1", unmarshaled.OffsetMS)
		}
	})

	t.Run("special characters in raw", func(t *testing.T) {
		chunk := SSEChunk{
			OffsetMS: 100,
			Raw:      "data: \"quoted\"\n\ttabbed\r\nCRLF",
		}
		data, err := json.Marshal(chunk)
		if err != nil {
			t.Fatalf("marshal error: %v", err)
		}
		var unmarshaled SSEChunk
		if err := json.Unmarshal(data, &unmarshaled); err != nil {
			t.Fatalf("unmarshal error: %v", err)
		}
		if unmarshaled.Raw != chunk.Raw {
			t.Errorf("raw: got %s, want %s", unmarshaled.Raw, chunk.Raw)
		}
	})

	t.Run("unicode in event name", func(t *testing.T) {
		chunk := SSEChunk{
			OffsetMS: 100,
			Event:    "测试-event-🎉",
		}
		data, err := json.Marshal(chunk)
		if err != nil {
			t.Fatalf("marshal error: %v", err)
		}
		var unmarshaled SSEChunk
		if err := json.Unmarshal(data, &unmarshaled); err != nil {
			t.Fatalf("unmarshal error: %v", err)
		}
		if unmarshaled.Event != chunk.Event {
			t.Errorf("event: got %s, want %s", unmarshaled.Event, chunk.Event)
		}
	})
}

func TestSSEChunkRoundTrip(t *testing.T) {
	chunks := []SSEChunk{
		{OffsetMS: 0},
		{OffsetMS: 100, Event: "message"},
		{OffsetMS: 200, Data: json.RawMessage(`{"test":"data"}`)},
		{OffsetMS: 300, Raw: "raw content"},
		{OffsetMS: 400, Event: "error", Data: json.RawMessage(`{"error":"timeout"}`), Raw: "error line"},
	}

	for i, original := range chunks {
		t.Run("roundtrip", func(t *testing.T) {
			data, err := json.Marshal(original)
			if err != nil {
				t.Fatalf("marshal error at index %d: %v", i, err)
			}
			var unmarshaled SSEChunk
			if err := json.Unmarshal(data, &unmarshaled); err != nil {
				t.Fatalf("unmarshal error at index %d: %v", i, err)
			}
			if unmarshaled.OffsetMS != original.OffsetMS {
				t.Errorf("offset_ms mismatch at index %d: got %d, want %d", i, unmarshaled.OffsetMS, original.OffsetMS)
			}
			if unmarshaled.Event != original.Event {
				t.Errorf("event mismatch at index %d: got %s, want %s", i, unmarshaled.Event, original.Event)
			}
			if unmarshaled.Raw != original.Raw {
				t.Errorf("raw mismatch at index %d: got %s, want %s", i, unmarshaled.Raw, original.Raw)
			}
			if string(unmarshaled.Data) != string(original.Data) {
				t.Errorf("data mismatch at index %d: got %s, want %s", i, unmarshaled.Data, original.Data)
			}
		})
	}
}
