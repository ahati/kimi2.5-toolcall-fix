package stream

import (
	"context"
	"sync"
	"testing"
	"time"
)

func TestNewRegistry(t *testing.T) {
	r := NewRegistry()
	if r == nil {
		t.Fatal("NewRegistry() returned nil")
	}
	if r.streams == nil {
		t.Error("streams map not initialized")
	}
}

func TestRegistry_Register(t *testing.T) {
	r := NewRegistry()
	_, cancel := context.WithCancel(context.Background())
	defer cancel()

	stream := r.Register("test-id", cancel)

	if stream == nil {
		t.Fatal("Register() returned nil")
	}
	if stream.ID != "test-id" {
		t.Errorf("ID = %q, want %q", stream.ID, "test-id")
	}
	if stream.Cancel == nil {
		t.Error("Cancel function is nil")
	}
	if stream.StartedAt.IsZero() {
		t.Error("StartedAt is zero")
	}

	// Verify stream is stored
	retrieved := r.Get("test-id")
	if retrieved == nil {
		t.Error("Get() returned nil after Register()")
	}
	if retrieved.ID != "test-id" {
		t.Errorf("retrieved ID = %q, want %q", retrieved.ID, "test-id")
	}
}

func TestRegistry_Cancel(t *testing.T) {
	r := NewRegistry()
	ctx, cancel := context.WithCancel(context.Background())

	r.Register("test-id", cancel)

	// Verify context is not cancelled before Cancel
	select {
	case <-ctx.Done():
		t.Error("context cancelled before Cancel()")
	default:
	}

	// Cancel the stream
	result := r.Cancel("test-id")
	if !result {
		t.Error("Cancel() returned false, expected true")
	}

	// Verify context is cancelled
	select {
	case <-ctx.Done():
		// expected
	case <-time.After(100 * time.Millisecond):
		t.Error("context not cancelled after Cancel()")
	}

	// Verify stream is removed
	retrieved := r.Get("test-id")
	if retrieved != nil {
		t.Error("Get() returned non-nil after Cancel()")
	}
}

func TestRegistry_Cancel_NotFound(t *testing.T) {
	r := NewRegistry()

	result := r.Cancel("nonexistent")
	if result {
		t.Error("Cancel() returned true for nonexistent stream")
	}
}

func TestRegistry_Remove(t *testing.T) {
	r := NewRegistry()
	_, cancel := context.WithCancel(context.Background())
	defer cancel()

	r.Register("test-id", cancel)
	r.Remove("test-id")

	retrieved := r.Get("test-id")
	if retrieved != nil {
		t.Error("Get() returned non-nil after Remove()")
	}
}

func TestRegistry_Get(t *testing.T) {
	r := NewRegistry()
	_, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Get nonexistent stream
	retrieved := r.Get("nonexistent")
	if retrieved != nil {
		t.Error("Get() returned non-nil for nonexistent stream")
	}

	// Register and get
	r.Register("test-id", cancel)
	retrieved = r.Get("test-id")
	if retrieved == nil {
		t.Fatal("Get() returned nil for registered stream")
	}
	if retrieved.ID != "test-id" {
		t.Errorf("ID = %q, want %q", retrieved.ID, "test-id")
	}
}

func TestRegistry_List(t *testing.T) {
	r := NewRegistry()
	_, cancel1 := context.WithCancel(context.Background())
	defer cancel1()
	_, cancel2 := context.WithCancel(context.Background())
	defer cancel2()

	// Empty registry
	ids := r.List()
	if len(ids) != 0 {
		t.Errorf("List() = %v, want empty slice", ids)
	}

	// Add streams
	r.Register("id1", cancel1)
	r.Register("id2", cancel2)

	ids = r.List()
	if len(ids) != 2 {
		t.Errorf("List() returned %d ids, want 2", len(ids))
	}

	// Verify both IDs are present (order not guaranteed)
	idSet := make(map[string]bool)
	for _, id := range ids {
		idSet[id] = true
	}
	if !idSet["id1"] || !idSet["id2"] {
		t.Errorf("List() = %v, want [id1, id2]", ids)
	}
}

func TestRegistry_ConcurrentAccess(t *testing.T) {
	r := NewRegistry()
	var wg sync.WaitGroup

	// Concurrent registrations
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			id := string(rune('a'+n%26)) + string(rune('a'+n/26))
			_, cancel := context.WithCancel(context.Background())
			defer cancel()
			r.Register(id, cancel)
		}(i)
	}

	// Concurrent reads
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r.List()
		}()
	}

	// Concurrent removes
	for i := 0; i < 25; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			id := string(rune('a'+n%26)) + string(rune('a'+n/26))
			r.Remove(id)
		}(i)
	}

	wg.Wait()
}
