package transform

import (
	"bytes"
	"testing"

	"github.com/tmaxmax/go-sse"
)

func TestPassthroughTransformer_Transform(t *testing.T) {
	tests := []struct {
		name     string
		events   []sse.Event
		expected string
	}{
		{
			name: "single event",
			events: []sse.Event{
				{Data: `{"test": "value"}`},
			},
			expected: "data: {\"test\": \"value\"}\n\n",
		},
		{
			name: "multiple events",
			events: []sse.Event{
				{Data: `{"id": "1"}`},
				{Data: `{"id": "2"}`},
				{Data: `{"id": "3"}`},
			},
			expected: "data: {\"id\": \"1\"}\n\ndata: {\"id\": \"2\"}\n\ndata: {\"id\": \"3\"}\n\n",
		},
		{
			name: "empty event",
			events: []sse.Event{
				{Data: ""},
			},
			expected: "",
		},
		{
			name: "[DONE] marker",
			events: []sse.Event{
				{Data: "[DONE]"},
			},
			expected: "data: [DONE]\n\n",
		},
		{
			name: "mixed empty and data",
			events: []sse.Event{
				{Data: ""},
				{Data: `{"content": "hello"}`},
				{Data: ""},
			},
			expected: "data: {\"content\": \"hello\"}\n\n",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			transformer := NewPassthroughTransformer(&buf)

			for _, event := range tt.events {
				if err := transformer.Transform(&event); err != nil {
					t.Errorf("Transform() error = %v", err)
				}
			}

			if got := buf.String(); got != tt.expected {
				t.Errorf("Transform() = %q, want %q", got, tt.expected)
			}
		})
	}
}

func TestPassthroughTransformer_Flush(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewPassthroughTransformer(&buf)

	if err := transformer.Flush(); err != nil {
		t.Errorf("Flush() error = %v", err)
	}
}

func TestPassthroughTransformer_Close(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewPassthroughTransformer(&buf)

	if err := transformer.Close(); err != nil {
		t.Errorf("Close() error = %v", err)
	}
}

func TestPassthroughTransformer_FullFlow(t *testing.T) {
	var buf bytes.Buffer
	transformer := NewPassthroughTransformer(&buf)

	events := []sse.Event{
		{Data: `{"type": "message_start"}`},
		{Data: `{"type": "content_block_start"}`},
		{Data: `{"delta": {"text": "Hello"}}`},
		{Data: `[DONE]`},
	}

	for _, event := range events {
		if err := transformer.Transform(&event); err != nil {
			t.Errorf("Transform() error = %v", err)
		}
	}

	if err := transformer.Flush(); err != nil {
		t.Errorf("Flush() error = %v", err)
	}

	if err := transformer.Close(); err != nil {
		t.Errorf("Close() error = %v", err)
	}

	expected := "data: {\"type\": \"message_start\"}\n\ndata: {\"type\": \"content_block_start\"}\n\ndata: {\"delta\": {\"text\": \"Hello\"}}\n\ndata: [DONE]\n\n"
	if got := buf.String(); got != expected {
		t.Errorf("Full flow output = %q, want %q", got, expected)
	}
}
