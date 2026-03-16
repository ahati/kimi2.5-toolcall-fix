// Package toolcall provides parsing and formatting for LLM tool call tokens.
// This file provides a shared tool call state structure used across transformers.
package toolcall

import (
	"strings"
)

// ToolCallState tracks the state of an in-progress tool call during streaming.
// This consolidates similar structs that were duplicated across multiple transformers.
type ToolCallState struct {
	// ID is the unique identifier for this tool call.
	// Format varies by provider: "call_xxx" (OpenAI), "toolu_xxx" (Anthropic)
	ID string

	// Name is the name of the function being called.
	Name string

	// Arguments accumulates the JSON arguments string during streaming.
	Arguments strings.Builder

	// Index is the position of this tool call in the output sequence.
	// Used for Anthropic content block indexing and OpenAI tool_calls array.
	Index int

	// BlockIndex is the content block index (Anthropic-specific).
	// For Anthropic, this differs from Index because text/thinking blocks
	// come before tool_use blocks.
	BlockIndex int
}

// NewToolCallState creates a new ToolCallState with the given parameters.
func NewToolCallState(id, name string, index int) *ToolCallState {
	return &ToolCallState{
		ID:    id,
		Name:  name,
		Index: index,
	}
}

// NewToolCallStateWithBlockIndex creates a new ToolCallState with both index types.
func NewToolCallStateWithBlockIndex(id, name string, index, blockIndex int) *ToolCallState {
	return &ToolCallState{
		ID:         id,
		Name:       name,
		Index:      index,
		BlockIndex: blockIndex,
	}
}

// AppendArguments appends a fragment of arguments to the accumulated state.
func (s *ToolCallState) AppendArguments(args string) {
	s.Arguments.WriteString(args)
}

// BuildArguments returns the complete accumulated arguments string.
func (s *ToolCallState) BuildArguments() string {
	return s.Arguments.String()
}

// Reset clears the state for reuse.
func (s *ToolCallState) Reset() {
	s.ID = ""
	s.Name = ""
	s.Arguments.Reset()
	s.Index = 0
	s.BlockIndex = 0
}

// ToolCallStateMap is a thread-safe map for tracking multiple tool calls by index.
// This is commonly needed during streaming where tool calls arrive incrementally.
type ToolCallStateMap struct {
	states map[int]*ToolCallState
}

// NewToolCallStateMap creates a new empty map.
func NewToolCallStateMap() *ToolCallStateMap {
	return &ToolCallStateMap{
		states: make(map[int]*ToolCallState),
	}
}

// Get retrieves a tool call state by index, or nil if not found.
func (m *ToolCallStateMap) Get(index int) *ToolCallState {
	return m.states[index]
}

// Set stores a tool call state at the given index.
func (m *ToolCallStateMap) Set(index int, state *ToolCallState) {
	m.states[index] = state
}

// GetOrCreate retrieves an existing state or creates a new one.
func (m *ToolCallStateMap) GetOrCreate(index int) *ToolCallState {
	if state, exists := m.states[index]; exists {
		return state
	}
	state := &ToolCallState{Index: index}
	m.states[index] = state
	return state
}

// Delete removes a tool call state.
func (m *ToolCallStateMap) Delete(index int) {
	delete(m.states, index)
}

// Len returns the number of tracked tool calls.
func (m *ToolCallStateMap) Len() int {
	return len(m.states)
}

// All returns all tool call states.
func (m *ToolCallStateMap) All() []*ToolCallState {
	result := make([]*ToolCallState, 0, len(m.states))
	for _, state := range m.states {
		result = append(result, state)
	}
	return result
}

// Clear removes all tool call states.
func (m *ToolCallStateMap) Clear() {
	m.states = make(map[int]*ToolCallState)
}
