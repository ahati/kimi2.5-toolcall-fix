// Package conversation provides in-memory storage for multi-turn conversations.
// It implements an LRU (Least Recently Used) cache with TTL-based expiration
// to store conversation history for the Responses API's previous_response_id feature.
package conversation

import (
	"container/list"
	"sync"
	"time"

	"ai-proxy/types"
)

// Conversation represents a stored conversation with its history.
// It stores both the input items from the request and the output items
// from the response, allowing the full conversation to be reconstructed.
type Conversation struct {
	// ID is the unique identifier for this conversation (response_id).
	ID string
	// Input contains the original input items from the request.
	Input []types.InputItem
	// Output contains the response output items.
	Output []types.OutputItem
	// CreatedAt is the timestamp when the conversation was created.
	CreatedAt time.Time
	// ExpiresAt is the timestamp when the conversation should be expired.
	ExpiresAt time.Time
}

// Config holds configuration for the conversation store.
type Config struct {
	// MaxSize is the maximum number of conversations to store.
	// When the limit is reached, the least recently used conversations
	// are evicted. Default: 1000.
	MaxSize int
	// TTL is the time-to-live for conversations.
	// Conversations older than this are automatically expired.
	// Default: 24 hours.
	TTL time.Duration
}

// Store provides thread-safe LRU storage for conversations.
// It uses a combination of a map for O(1) lookups and a doubly-linked
// list for O(1) LRU ordering operations.
type Store struct {
	mu     sync.RWMutex
	config Config
	data   map[string]*list.Element // response_id -> list element
	lru    *list.List               // LRU order (front = most recent)
}

// entry represents an element in the LRU list.
type entry struct {
	key         string
	conversation *Conversation
}

// NewStore creates a new conversation store with the given configuration.
// If maxSize is 0, it defaults to 1000.
// If TTL is 0, it defaults to 24 hours.
func NewStore(config Config) *Store {
	if config.MaxSize <= 0 {
		config.MaxSize = 1000
	}
	if config.TTL <= 0 {
		config.TTL = 24 * time.Hour
	}
	return &Store{
		config: config,
		data:   make(map[string]*list.Element),
		lru:    list.New(),
	}
}

// Get retrieves a conversation by ID.
// Returns nil if the conversation is not found or has expired.
// Accessing a conversation moves it to the front of the LRU list.
func (s *Store) Get(id string) *Conversation {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Clean up expired entries lazily
	s.cleanupExpired()

	if elem, ok := s.data[id]; ok {
		conv := elem.Value.(*entry).conversation
		// Check if expired
		if time.Now().After(conv.ExpiresAt) {
			s.deleteElement(elem)
			return nil
		}
		// Move to front (most recently used)
		s.lru.MoveToFront(elem)
		return conv
	}
	return nil
}

// Store saves a conversation, evicting the oldest if at capacity.
// If a conversation with the same ID already exists, it is replaced.
func (s *Store) Store(conv *Conversation) {
	if conv == nil || conv.ID == "" {
		return
	}

	// Set expiration time if not set
	if conv.ExpiresAt.IsZero() {
		conv.ExpiresAt = time.Now().Add(s.config.TTL)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Clean up expired entries lazily
	s.cleanupExpired()

	// If already exists, remove old entry
	if elem, ok := s.data[conv.ID]; ok {
		s.deleteElement(elem)
	}

	// Check capacity and evict if necessary
	for s.lru.Len() >= s.config.MaxSize {
		s.evictOldest()
	}

	// Add new entry at front
	elem := s.lru.PushFront(&entry{
		key:         conv.ID,
		conversation: conv,
	})
	s.data[conv.ID] = elem
}

// Delete removes a conversation by ID.
func (s *Store) Delete(id string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if elem, ok := s.data[id]; ok {
		s.deleteElement(elem)
	}
}

// Size returns the current number of stored conversations.
func (s *Store) Size() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.lru.Len()
}

// Clear removes all conversations from the store.
func (s *Store) Clear() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.data = make(map[string]*list.Element)
	s.lru.Init()
}

// deleteElement removes an element from both the map and the list.
// Must be called with lock held.
func (s *Store) deleteElement(elem *list.Element) {
	if elem == nil {
		return
	}
	ent := elem.Value.(*entry)
	delete(s.data, ent.key)
	s.lru.Remove(elem)
}

// evictOldest removes the least recently used conversation.
// Must be called with lock held.
func (s *Store) evictOldest() {
	elem := s.lru.Back()
	if elem != nil {
		s.deleteElement(elem)
	}
}

// cleanupExpired removes all expired conversations.
// Must be called with lock held.
func (s *Store) cleanupExpired() {
	now := time.Now()
	// Iterate from back (oldest) to front (newest)
	for elem := s.lru.Back(); elem != nil; {
		next := elem.Prev()
		conv := elem.Value.(*entry).conversation
		if now.After(conv.ExpiresAt) {
			s.deleteElement(elem)
		}
		elem = next
	}
}

// DefaultStore is the global conversation store instance.
// It is initialized by the main package at startup.
var DefaultStore *Store

// InitDefaultStore initializes the global conversation store.
func InitDefaultStore(config Config) {
	DefaultStore = NewStore(config)
}

// GetFromDefault retrieves a conversation from the default store.
// Returns nil if the default store is not initialized.
func GetFromDefault(id string) *Conversation {
	if DefaultStore == nil {
		return nil
	}
	return DefaultStore.Get(id)
}

// StoreInDefault saves a conversation to the default store.
// Does nothing if the default store is not initialized.
func StoreInDefault(conv *Conversation) {
	if DefaultStore == nil {
		return
	}
	DefaultStore.Store(conv)
}