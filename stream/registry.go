package stream

import (
	"context"
	"sync"
	"time"
)

// ActiveStream represents an in-progress streaming response.
type ActiveStream struct {
	ID        string
	Cancel    context.CancelFunc
	StartedAt time.Time
}

// Registry tracks active streams for cancellation support.
type Registry struct {
	mu      sync.RWMutex
	streams map[string]*ActiveStream
}

// NewRegistry creates a new stream registry.
func NewRegistry() *Registry {
	return &Registry{
		streams: make(map[string]*ActiveStream),
	}
}

// Register adds a new active stream to the registry.
func (r *Registry) Register(id string, cancel context.CancelFunc) *ActiveStream {
	r.mu.Lock()
	defer r.mu.Unlock()

	stream := &ActiveStream{
		ID:        id,
		Cancel:    cancel,
		StartedAt: time.Now(),
	}
	r.streams[id] = stream
	return stream
}

// Cancel attempts to cancel an active stream by ID.
// Returns true if the stream was found and cancelled.
func (r *Registry) Cancel(id string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()

	stream, ok := r.streams[id]
	if !ok {
		return false
	}

	stream.Cancel()
	delete(r.streams, id)
	return true
}

// Remove removes a stream from the registry.
func (r *Registry) Remove(id string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.streams, id)
}

// Get retrieves an active stream by ID.
func (r *Registry) Get(id string) *ActiveStream {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.streams[id]
}

// List returns all active stream IDs.
func (r *Registry) List() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	ids := make([]string, 0, len(r.streams))
	for id := range r.streams {
		ids = append(ids, id)
	}
	return ids
}
