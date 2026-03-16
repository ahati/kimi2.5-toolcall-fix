package conversation

import (
	"sync"
	"testing"
	"time"

	"ai-proxy/types"
)

func TestNewStore(t *testing.T) {
	tests := []struct {
		name     string
		config   Config
		wantSize int
		wantTTL  time.Duration
	}{
		{
			name:     "default config",
			config:   Config{},
			wantSize: 1000,
			wantTTL:  24 * time.Hour,
		},
		{
			name: "custom config",
			config: Config{
				MaxSize: 100,
				TTL:     time.Hour,
			},
			wantSize: 100,
			wantTTL:  time.Hour,
		},
		{
			name: "zero values use defaults",
			config: Config{
				MaxSize: 0,
				TTL:     0,
			},
			wantSize: 1000,
			wantTTL:  24 * time.Hour,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			store := NewStore(tt.config)
			if store.config.MaxSize != tt.wantSize {
				t.Errorf("NewStore() MaxSize = %v, want %v", store.config.MaxSize, tt.wantSize)
			}
			if store.config.TTL != tt.wantTTL {
				t.Errorf("NewStore() TTL = %v, want %v", store.config.TTL, tt.wantTTL)
			}
			if store.Size() != 0 {
				t.Errorf("NewStore() initial Size = %v, want 0", store.Size())
			}
		})
	}
}

func TestStore_Get(t *testing.T) {
	store := NewStore(Config{MaxSize: 10, TTL: time.Hour})

	// Test getting non-existent conversation
	if conv := store.Get("nonexistent"); conv != nil {
		t.Errorf("Get(nonexistent) = %v, want nil", conv)
	}

	// Store a conversation
	now := time.Now()
	conv := &Conversation{
		ID:        "resp_123",
		Input:     []types.InputItem{{Type: "message", Role: "user", Content: "hello"}},
		Output:    []types.OutputItem{{Type: "message", ID: "msg_123", Role: "assistant"}},
		CreatedAt: now,
		ExpiresAt: now.Add(time.Hour),
	}
	store.Store(conv)

	// Get the stored conversation
	got := store.Get("resp_123")
	if got == nil {
		t.Fatal("Get(resp_123) = nil, want non-nil")
	}
	if got.ID != conv.ID {
		t.Errorf("Get().ID = %v, want %v", got.ID, conv.ID)
	}
	if len(got.Input) != 1 || got.Input[0].Content != "hello" {
		t.Errorf("Get().Input = %v, want [{Type:message Content:hello}]", got.Input)
	}
}

func TestStore_Store(t *testing.T) {
	store := NewStore(Config{MaxSize: 3, TTL: time.Hour})

	// Store multiple conversations
	for i := 0; i < 5; i++ {
		conv := &Conversation{
			ID:        string(rune('a' + i)),
			CreatedAt: time.Now(),
		}
		store.Store(conv)
	}

	// Should have at most MaxSize conversations
	if store.Size() > 3 {
		t.Errorf("Size() = %v, want at most 3", store.Size())
	}

	// First two should have been evicted (LRU)
	if store.Get("a") != nil {
		t.Error("Expected 'a' to be evicted")
	}
	if store.Get("b") != nil {
		t.Error("Expected 'b' to be evicted")
	}

	// Last three should still exist
	for _, id := range []string{"c", "d", "e"} {
		if store.Get(id) == nil {
			t.Errorf("Expected '%s' to exist", id)
		}
	}
}

func TestStore_LRUEviction(t *testing.T) {
	store := NewStore(Config{MaxSize: 3, TTL: time.Hour})

	// Store 3 conversations
	store.Store(&Conversation{ID: "a", CreatedAt: time.Now()})
	store.Store(&Conversation{ID: "b", CreatedAt: time.Now()})
	store.Store(&Conversation{ID: "c", CreatedAt: time.Now()})

	// Access 'a' to make it most recently used
	store.Get("a")

	// Add a new conversation, should evict 'b' (not 'a' since we accessed it)
	store.Store(&Conversation{ID: "d", CreatedAt: time.Now()})

	if store.Get("a") == nil {
		t.Error("Expected 'a' to still exist (was accessed recently)")
	}
	if store.Get("b") != nil {
		t.Error("Expected 'b' to be evicted (LRU)")
	}
	if store.Get("c") == nil {
		t.Error("Expected 'c' to still exist")
	}
	if store.Get("d") == nil {
		t.Error("Expected 'd' to exist")
	}
}

func TestStore_TTLExpiration(t *testing.T) {
	store := NewStore(Config{MaxSize: 10, TTL: 100 * time.Millisecond})

	// Store a conversation with short TTL
	conv := &Conversation{
		ID:        "expired",
		CreatedAt: time.Now(),
		ExpiresAt: time.Now().Add(50 * time.Millisecond),
	}
	store.Store(conv)

	// Should exist immediately
	if store.Get("expired") == nil {
		t.Error("Expected conversation to exist immediately")
	}

	// Wait for expiration
	time.Sleep(100 * time.Millisecond)

	// Should be expired now
	if store.Get("expired") != nil {
		t.Error("Expected conversation to be expired")
	}
}

func TestStore_Replace(t *testing.T) {
	store := NewStore(Config{MaxSize: 10, TTL: time.Hour})

	// Store initial conversation
	store.Store(&Conversation{
		ID:        "replace_me",
		Input:     []types.InputItem{{Type: "message", Content: "original"}},
		CreatedAt: time.Now(),
	})

	// Replace with new conversation
	store.Store(&Conversation{
		ID:        "replace_me",
		Input:     []types.InputItem{{Type: "message", Content: "replaced"}},
		CreatedAt: time.Now(),
	})

	// Should have the replaced content
	conv := store.Get("replace_me")
	if conv == nil {
		t.Fatal("Get(replace_me) = nil, want non-nil")
	}
	if len(conv.Input) != 1 || conv.Input[0].Content != "replaced" {
		t.Errorf("Input content = %v, want 'replaced'", conv.Input[0].Content)
	}

	// Size should still be 1
	if store.Size() != 1 {
		t.Errorf("Size() = %v, want 1", store.Size())
	}
}

func TestStore_Delete(t *testing.T) {
	store := NewStore(Config{MaxSize: 10, TTL: time.Hour})

	store.Store(&Conversation{ID: "delete_me", CreatedAt: time.Now()})
	if store.Get("delete_me") == nil {
		t.Fatal("Expected conversation to exist before delete")
	}

	store.Delete("delete_me")

	if store.Get("delete_me") != nil {
		t.Error("Expected conversation to be deleted")
	}

	// Delete non-existent should not panic
	store.Delete("nonexistent")
}

func TestStore_Clear(t *testing.T) {
	store := NewStore(Config{MaxSize: 10, TTL: time.Hour})

	// Store multiple conversations
	for i := 0; i < 5; i++ {
		store.Store(&Conversation{
			ID:        string(rune('a' + i)),
			CreatedAt: time.Now(),
		})
	}

	if store.Size() != 5 {
		t.Fatalf("Size() = %v, want 5", store.Size())
	}

	store.Clear()

	if store.Size() != 0 {
		t.Errorf("After Clear(), Size() = %v, want 0", store.Size())
	}
}

func TestStore_NilConversation(t *testing.T) {
	store := NewStore(Config{MaxSize: 10, TTL: time.Hour})

	// Store nil should not panic
	store.Store(nil)

	// Store empty ID should not panic
	store.Store(&Conversation{ID: ""})

	if store.Size() != 0 {
		t.Errorf("Size() = %v, want 0 (nil/empty should not be stored)", store.Size())
	}
}

func TestStore_ConcurrentAccess(t *testing.T) {
	store := NewStore(Config{MaxSize: 100, TTL: time.Hour})
	var wg sync.WaitGroup

	// Concurrent writes
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			store.Store(&Conversation{
				ID:        string(rune('a' + id%26)),
				CreatedAt: time.Now(),
			})
		}(i)
	}

	// Concurrent reads
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			store.Get(string(rune('a' + id%26)))
		}(i)
	}

	wg.Wait()

	// Should not panic and size should be <= MaxSize
	if store.Size() > 100 {
		t.Errorf("Size() = %v, want at most 100", store.Size())
	}
}

func TestStore_AutoExpireTTL(t *testing.T) {
	store := NewStore(Config{MaxSize: 10, TTL: time.Hour})

	// Store conversation without setting ExpiresAt
	conv := &Conversation{
		ID:        "auto_expire",
		CreatedAt: time.Now(),
	}
	store.Store(conv)

	// Should have auto-set ExpiresAt
	got := store.Get("auto_expire")
	if got == nil {
		t.Fatal("Get(auto_expire) = nil")
	}
	if got.ExpiresAt.IsZero() {
		t.Error("ExpiresAt should have been auto-set")
	}
	expectedExpiry := conv.CreatedAt.Add(time.Hour)
	if got.ExpiresAt.Before(expectedExpiry.Add(-time.Second)) || got.ExpiresAt.After(expectedExpiry.Add(time.Second)) {
		t.Errorf("ExpiresAt = %v, want approximately %v", got.ExpiresAt, expectedExpiry)
	}
}

func TestDefaultStore(t *testing.T) {
	// Reset default store
	DefaultStore = nil

	// Get from uninitialized store should return nil
	if GetFromDefault("test") != nil {
		t.Error("GetFromDefault from nil store should return nil")
	}

	// Store to uninitialized store should not panic
	StoreInDefault(&Conversation{ID: "test"})

	// Initialize default store
	InitDefaultStore(Config{MaxSize: 10, TTL: time.Hour})

	if DefaultStore == nil {
		t.Fatal("DefaultStore should be initialized")
	}

	// Store and retrieve
	conv := &Conversation{
		ID:        "default_test",
		Input:     []types.InputItem{{Type: "message", Content: "test"}},
		CreatedAt: time.Now(),
	}
	StoreInDefault(conv)

	got := GetFromDefault("default_test")
	if got == nil {
		t.Fatal("GetFromDefault(default_test) = nil")
	}
	if got.ID != "default_test" {
		t.Errorf("ID = %v, want default_test", got.ID)
	}
}