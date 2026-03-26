package websearch

import (
	"context"
	"testing"

	"github.com/tmaxmax/go-sse"

	"ai-proxy/transform"
	"ai-proxy/types"
)

// mockService implements Service for testing
type mockService struct {
	results []types.WebSearchResult
	err     error
}

func (m *mockService) ExecuteSearch(ctx context.Context, query string, allowedDomains, blockedDomains []string) ([]types.WebSearchResult, error) {
	return m.results, m.err
}

// mockBaseTransformer implements transform.SSETransformer for testing
type mockBaseTransformer struct {
	events []*sse.Event
}

func (m *mockBaseTransformer) Initialize() error   { return nil }
func (m *mockBaseTransformer) HandleCancel() error { return nil }
func (m *mockBaseTransformer) Flush() error        { return nil }
func (m *mockBaseTransformer) Close() error        { return nil }
func (m *mockBaseTransformer) Transform(event *sse.Event) error {
	m.events = append(m.events, event)
	return nil
}

// compile-time check that Transformer implements transform.SSETransformer
var _ transform.SSETransformer = (*Transformer)(nil)

func TestNewTransformer(t *testing.T) {
	mockBase := &mockBaseTransformer{}
	mockSvc := &mockService{}

	tr := NewTransformer(mockBase, mockSvc)
	if tr == nil {
		t.Fatal("NewTransformer returned nil")
	}
	if tr.base == nil {
		t.Error("base transformer not set")
	}
	if tr.service == nil {
		t.Error("service not set")
	}
	if tr.pendingBlocks == nil {
		t.Error("pendingBlocks map not initialized")
	}
	if tr.indexToID == nil {
		t.Error("indexToID map not initialized")
	}
}

func TestTransformPassthrough(t *testing.T) {
	mockBase := &mockBaseTransformer{}
	tr := NewTransformer(mockBase, nil)

	// Non-content_block events should pass through
	event := &sse.Event{
		Type: "message_start",
		Data: `{"type":"message_start","message":{"id":"msg_123"}}`,
	}
	err := tr.Transform(event)
	if err != nil {
		t.Fatalf("Transform failed: %v", err)
	}
	if len(mockBase.events) != 1 {
		t.Errorf("expected 1 event, got %d", len(mockBase.events))
	}
}

func TestTransformWebSearchInterception_ToolUse(t *testing.T) {
	mockBase := &mockBaseTransformer{}
	mockSvc := &mockService{
		results: []types.WebSearchResult{
			{Type: "web_search_result", Title: "Test", URL: "https://example.com", Content: "Test content"},
		},
	}
	tr := NewTransformer(mockBase, mockSvc)

	// Send content_block_start for web_search with tool_use type (what the model outputs)
	startEvent := &sse.Event{
		Type: "content_block_start",
		Data: `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_123","name":"web_search"}}`,
	}
	err := tr.Transform(startEvent)
	if err != nil {
		t.Fatalf("Transform start failed: %v", err)
	}
	// Should pass through to base (we don't block it, just track it)
	if len(mockBase.events) != 1 {
		t.Errorf("expected 1 event (passed through), got %d", len(mockBase.events))
	}

	// Send content_block_delta with input
	deltaEvent := &sse.Event{
		Type: "content_block_delta",
		Data: `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"query\":\"test query\"}"}}`,
	}
	err = tr.Transform(deltaEvent)
	if err != nil {
		t.Fatalf("Transform delta failed: %v", err)
	}
	// Should pass through to base (we buffer it AND pass through)
	if len(mockBase.events) != 2 {
		t.Errorf("expected 2 events (passed through), got %d", len(mockBase.events))
	}

	// Send content_block_stop - should trigger search and emit result
	stopEvent := &sse.Event{
		Type: "content_block_stop",
		Data: `{"type":"content_block_stop","index":0}`,
	}
	err = tr.Transform(stopEvent)
	if err != nil {
		t.Fatalf("Transform stop failed: %v", err)
	}
	// Should emit tool_result in addition to passing through stop
	// Expected: stop event + tool_result start + tool_result stop = 3 more events
	if len(mockBase.events) < 4 {
		t.Errorf("expected at least 4 events (originals + tool_result), got %d", len(mockBase.events))
	}
}

func TestTransformWebSearchInterception_ServerToolUse(t *testing.T) {
	mockBase := &mockBaseTransformer{}
	mockSvc := &mockService{
		results: []types.WebSearchResult{
			{Type: "web_search_result", Title: "Test", URL: "https://example.com", Content: "Test content"},
		},
	}
	tr := NewTransformer(mockBase, mockSvc)

	// Send content_block_start for web_search with server_tool_use type (Anthropic style)
	startEvent := &sse.Event{
		Type: "content_block_start",
		Data: `{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"toolu_123","name":"web_search"}}`,
	}
	err := tr.Transform(startEvent)
	if err != nil {
		t.Fatalf("Transform start failed: %v", err)
	}
	// Should pass through to base (we track it, not block it)
	if len(mockBase.events) != 1 {
		t.Errorf("expected 1 event (passed through), got %d", len(mockBase.events))
	}
}

func TestTransformNonWebSearchTool(t *testing.T) {
	mockBase := &mockBaseTransformer{}
	tr := NewTransformer(mockBase, nil)

	// Send content_block_start for a different tool (WebFetch should pass through)
	startEvent := &sse.Event{
		Type: "content_block_start",
		Data: `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_123","name":"WebFetch"}}`,
	}
	err := tr.Transform(startEvent)
	if err != nil {
		t.Fatalf("Transform failed: %v", err)
	}
	// Should pass through - we do not intercept WebFetch
	if len(mockBase.events) != 1 {
		t.Errorf("expected 1 event (passed through), got %d", len(mockBase.events))
	}
}

func TestTransformNilService(t *testing.T) {
	mockBase := &mockBaseTransformer{}
	tr := NewTransformer(mockBase, nil) // nil service

	// Send web_search tool call
	startEvent := &sse.Event{
		Type: "content_block_start",
		Data: `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_123","name":"web_search"}}`,
	}
	tr.Transform(startEvent)

	deltaEvent := &sse.Event{
		Type: "content_block_delta",
		Data: `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"query\":\"test\"}"}}`,
	}
	tr.Transform(deltaEvent)

	stopEvent := &sse.Event{
		Type: "content_block_stop",
		Data: `{"type":"content_block_stop","index":0}`,
	}
	err := tr.Transform(stopEvent)
	if err != nil {
		t.Fatalf("Transform failed: %v", err)
	}

	// Should emit error result
	if len(mockBase.events) < 2 {
		t.Errorf("expected at least 2 events (error result), got %d", len(mockBase.events))
	}
}

func TestTransformMalformedJSON(t *testing.T) {
	mockBase := &mockBaseTransformer{}
	mockSvc := &mockService{}
	tr := NewTransformer(mockBase, mockSvc)

	// Send web_search tool call with malformed JSON input
	startEvent := &sse.Event{
		Type: "content_block_start",
		Data: `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_123","name":"web_search"}}`,
	}
	tr.Transform(startEvent)

	deltaEvent := &sse.Event{
		Type: "content_block_delta",
		Data: `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{invalid json}"}}`,
	}
	tr.Transform(deltaEvent)

	stopEvent := &sse.Event{
		Type: "content_block_stop",
		Data: `{"type":"content_block_stop","index":0}`,
	}
	err := tr.Transform(stopEvent)
	if err != nil {
		t.Fatalf("Transform failed: %v", err)
	}

	// Should emit error result for malformed JSON
	if len(mockBase.events) < 2 {
		t.Errorf("expected at least 2 events (error result), got %d", len(mockBase.events))
	}
}

func TestMultipleWebSearchBlocks(t *testing.T) {
	mockBase := &mockBaseTransformer{}
	mockSvc := &mockService{
		results: []types.WebSearchResult{
			{Type: "web_search_result", Title: "Test", URL: "https://example.com", Content: "Content"},
		},
	}
	tr := NewTransformer(mockBase, mockSvc)

	// First web_search block
	startEvent1 := &sse.Event{
		Type: "content_block_start",
		Data: `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"web_search"}}`,
	}
	tr.Transform(startEvent1)

	deltaEvent1 := &sse.Event{
		Type: "content_block_delta",
		Data: `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"query\":\"query1\"}"}}`,
	}
	tr.Transform(deltaEvent1)

	// Second web_search block (interleaved)
	startEvent2 := &sse.Event{
		Type: "content_block_start",
		Data: `{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_2","name":"web_search"}}`,
	}
	tr.Transform(startEvent2)

	deltaEvent2 := &sse.Event{
		Type: "content_block_delta",
		Data: `{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"query\":\"query2\"}"}}`,
	}
	tr.Transform(deltaEvent2)

	// Stop first block
	stopEvent1 := &sse.Event{
		Type: "content_block_stop",
		Data: `{"type":"content_block_stop","index":0}`,
	}
	tr.Transform(stopEvent1)

	// Stop second block
	stopEvent2 := &sse.Event{
		Type: "content_block_stop",
		Data: `{"type":"content_block_stop","index":1}`,
	}
	tr.Transform(stopEvent2)

	// Should have 4 events (2 tool_results, each with start and stop)
	if len(mockBase.events) < 4 {
		t.Errorf("expected at least 4 events, got %d", len(mockBase.events))
	}
}

func TestCloseCleanup(t *testing.T) {
	mockBase := &mockBaseTransformer{}
	tr := NewTransformer(mockBase, nil)

	// Add a pending block
	tr.pendingBlocks["test"] = &pendingBlock{}
	tr.indexToID[0] = "test"

	// Close should clean up
	err := tr.Close()
	if err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	if len(tr.pendingBlocks) != 0 {
		t.Errorf("pendingBlocks not cleaned up")
	}
	if len(tr.indexToID) != 0 {
		t.Errorf("indexToID not cleaned up")
	}
}
