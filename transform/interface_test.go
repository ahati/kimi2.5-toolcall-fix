package transform

import (
	"errors"
	"io"
	"testing"

	"github.com/tmaxmax/go-sse"
)

type mockSSETransformer struct {
	transformErr error
	flushErr     error
	closeErr     error
	events       []*sse.Event
}

func (m *mockSSETransformer) Transform(event *sse.Event) error {
	m.events = append(m.events, event)
	return m.transformErr
}

func (m *mockSSETransformer) Flush() error {
	return m.flushErr
}

func (m *mockSSETransformer) Close() error {
	return m.closeErr
}

type mockFlushingWriter struct {
	data     []byte
	flushErr error
}

func (m *mockFlushingWriter) Write(p []byte) (n int, err error) {
	m.data = append(m.data, p...)
	return len(p), nil
}

func (m *mockFlushingWriter) Flush() error {
	return m.flushErr
}

func TestSSETransformer_Interface(t *testing.T) {
	var _ SSETransformer = &mockSSETransformer{}
}

func TestFlushingWriter_Interface(t *testing.T) {
	var _ FlushingWriter = &mockFlushingWriter{}
	var _ io.Writer = &mockFlushingWriter{}
}

func TestSSETransformer_Transform(t *testing.T) {
	tests := []struct {
		name    string
		event   *sse.Event
		wantErr error
	}{
		{
			name:  "successful transform",
			event: &sse.Event{Type: "message", Data: "test data"},
		},
		{
			name:  "empty event",
			event: &sse.Event{},
		},
		{
			name:    "transform error",
			event:   &sse.Event{Type: "error", Data: "error data"},
			wantErr: errors.New("transform failed"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mock := &mockSSETransformer{transformErr: tt.wantErr}
			err := mock.Transform(tt.event)

			if tt.wantErr != nil {
				if err == nil {
					t.Errorf("Transform() expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Errorf("Transform() unexpected error: %v", err)
			}
			if len(mock.events) != 1 {
				t.Errorf("Transform() expected 1 event, got %d", len(mock.events))
			}
		})
	}
}

func TestSSETransformer_Flush(t *testing.T) {
	tests := []struct {
		name     string
		flushErr error
	}{
		{"successful flush", nil},
		{"flush error", errors.New("flush failed")},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mock := &mockSSETransformer{flushErr: tt.flushErr}
			err := mock.Flush()

			if tt.flushErr != nil && err == nil {
				t.Errorf("Flush() expected error, got nil")
			}
			if tt.flushErr == nil && err != nil {
				t.Errorf("Flush() unexpected error: %v", err)
			}
		})
	}
}

func TestSSETransformer_Close(t *testing.T) {
	tests := []struct {
		name     string
		closeErr error
	}{
		{"successful close", nil},
		{"close error", errors.New("close failed")},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mock := &mockSSETransformer{closeErr: tt.closeErr}
			err := mock.Close()

			if tt.closeErr != nil && err == nil {
				t.Errorf("Close() expected error, got nil")
			}
			if tt.closeErr == nil && err != nil {
				t.Errorf("Close() unexpected error: %v", err)
			}
		})
	}
}

func TestFlushingWriter_Write(t *testing.T) {
	mock := &mockFlushingWriter{}
	data := []byte("test data")

	n, err := mock.Write(data)
	if err != nil {
		t.Errorf("Write() unexpected error: %v", err)
	}
	if n != len(data) {
		t.Errorf("Write() returned %d, want %d", n, len(data))
	}
	if string(mock.data) != string(data) {
		t.Errorf("Write() data = %s, want %s", mock.data, data)
	}
}

func TestFlushingWriter_Flush(t *testing.T) {
	tests := []struct {
		name     string
		flushErr error
	}{
		{"successful flush", nil},
		{"flush error", errors.New("flush failed")},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mock := &mockFlushingWriter{flushErr: tt.flushErr}
			err := mock.Flush()

			if tt.flushErr != nil && err == nil {
				t.Errorf("Flush() expected error, got nil")
			}
			if tt.flushErr == nil && err != nil {
				t.Errorf("Flush() unexpected error: %v", err)
			}
		})
	}
}
