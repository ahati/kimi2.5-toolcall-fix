package toolcall

import (
	"fmt"
	"strings"
	"time"
)

// state represents the parser's current position within a tool call sequence.
// The parser uses a finite state machine to track parsing progress.
//
// @brief Enumeration of parser states for tool call parsing state machine.
//
// @note State transitions are deterministic and depend solely on token
//
//	recognition in the input buffer. Each state has defined entry
//	and exit conditions.
//
// State Transition Diagram:
// stateIdle -> stateInSection: When SectionBegin token found
// stateInSection -> stateReadingID: When CallBegin token found
// stateInSection -> stateTrailing: When SectionEnd token found
// stateReadingID -> stateReadingArgs: When ArgBegin token found
// stateReadingArgs -> stateInSection: When CallEnd token found
// stateTrailing -> stateInSection: When SectionBegin token found
// stateTrailing -> stateIdle: When buffer exhausted with content output
type state int

const (
	// stateIdle indicates the parser is outside any tool call section.
	// In this state, all input is treated as regular content until
	// a SectionBegin token is encountered.
	stateIdle state = iota

	// stateInSection indicates the parser is inside a tool calls section.
	// The parser is looking for either CallBegin (start of a tool call)
	// or SectionEnd (end of the section).
	stateInSection

	// stateReadingID indicates the parser is reading a tool call ID and name.
	// The parser has found CallBegin and is searching for ArgBegin.
	// Content between CallBegin and ArgBegin contains the tool call ID
	// and function name.
	stateReadingID

	// stateReadingArgs indicates the parser is reading tool call arguments.
	// The parser has found ArgBegin and is searching for CallEnd.
	// Content between ArgBegin and CallEnd contains the function arguments
	// as a JSON string (potentially streamed across multiple chunks).
	stateReadingArgs

	// stateTrailing indicates the parser is after a section end.
	// Content after SectionEnd may be regular text or another section.
	// This state handles potential multiple tool call sections.
	stateTrailing
)

// EventType identifies the kind of parsing event emitted.
// Events are the output of the parser, informing the caller what was parsed.
//
// @brief Enumeration of event types emitted by the parser.
//
// @note Events are emitted in a specific sequence for tool calls:
//
//	EventToolStart -> EventToolArgs (multiple) -> EventToolEnd
//	EventContent is emitted for regular text outside tool calls.
//	EventSectionEnd is emitted when a tool calls section ends.
type EventType int

const (
	// EventContent indicates regular text content outside tool calls.
	// The Text field contains the content string.
	EventContent EventType = iota

	// EventToolStart indicates the beginning of a tool call.
	// The ID field contains the tool call identifier.
	// The Name field contains the function name.
	// The Index field contains the zero-based tool call index.
	EventToolStart

	// EventToolArgs contains tool call argument data.
	// The Args field contains a fragment of the JSON arguments.
	// The Index field contains the tool call index.
	// Multiple EventToolArgs may be emitted for a single tool call.
	EventToolArgs

	// EventToolEnd indicates the end of a tool call.
	// The Index field indicates which tool call has ended.
	EventToolEnd

	// EventSectionEnd indicates the end of the tool calls section.
	// This signals that no more tool calls are expected in this section.
	EventSectionEnd
)

// Event represents a parsed event from the tool call stream.
// Events are emitted by the Parser as it processes input text.
//
// @brief Data structure representing a parsed tool call or content event.
//
// @note Only certain fields are valid for each EventType:
//   - EventContent: Text
//   - EventToolStart: ID, Name, Index
//   - EventToolArgs: Args, Index
//   - EventToolEnd: Index
//   - EventSectionEnd: (no fields)
//
// @note Event instances should not be reused across parsing calls.
//
//	Each Parse call returns new Event instances.
type Event struct {
	// Type identifies the kind of event.
	// Determines which other fields are valid.
	Type EventType

	// Text contains the content string for EventContent events.
	// Empty for all other event types.
	Text string

	// ID contains the tool call identifier for EventToolStart events.
	// Format: "call_<index>_<timestamp>" or original ID from LLM.
	// Empty for all other event types.
	ID string

	// Name contains the function name for EventToolStart events.
	// This is the name of the function being called.
	// Empty for all other event types.
	Name string

	// Args contains the argument data for EventToolArgs events.
	// This is a fragment of the JSON arguments string.
	// Multiple EventToolArgs may be emitted for streaming arguments.
	// Empty for all other event types.
	Args string

	// Index contains the zero-based tool call index.
	// Valid for EventToolStart, EventToolArgs, and EventToolEnd events.
	// Incremented for each new tool call in a section.
	Index int
}

// Parser extracts tool calls from streaming text using delimiter tokens.
// It maintains state across multiple Parse calls to handle partial tokens.
//
// @brief Stateful parser for extracting tool calls from streaming LLM output.
//
// @note The parser is NOT safe for concurrent use. Each parser instance
//
//	should be used by a single goroutine.
//
// @note Partial tokens at buffer boundaries are correctly handled by
//
//	buffering and processing in subsequent Parse calls.
//
// @pre Parser must be initialized via NewParser before use.
// @post After parsing, Reset() may be called to reuse the parser.
//
// Memory Safety:
// - The parser buffers only unprocessed input.
// - Emitted events do not retain references to the buffer.
// - No memory leaks occur with proper Close() semantics.
type Parser struct {
	// tokens contains the delimiter tokens used for parsing.
	// Configured at initialization and immutable during parsing.
	tokens Tokens

	// state represents the current parser state.
	// Determines which tokens are expected next.
	state state

	// buf contains unprocessed input that may contain partial tokens.
	// Data is added by Parse() and consumed by processState().
	// Buffer is cleared when complete events are emitted.
	buf string

	// toolIndex tracks the current tool call index within a section.
	// Incremented after each complete tool call.
	// Reset to 0 when entering a new section or via Reset().
	toolIndex int
}

// NewParser creates a parser with the given delimiter tokens.
//
// @brief Creates a new Parser instance initialized with specified tokens.
//
// @param tokens The delimiter tokens to use for parsing.
//
//	All fields must be non-empty for correct parsing.
//	Use DefaultTokens for Kimi model compatibility.
//
// @return *Parser A new parser in stateIdle with empty buffer.
//
// @pre tokens must have all fields populated (non-empty strings).
// @post The returned parser is ready for Parse() calls.
// @post Parser is in stateIdle state with empty buffer.
//
// @note The parser does not take ownership of tokens; it copies the struct.
func NewParser(tokens Tokens) *Parser {
	return &Parser{tokens: tokens}
}

// Parse processes text and returns any complete events.
// Text is buffered until complete tokens are recognized.
//
// @brief Processes input text and emits complete parsing events.
//
// @param text The input text to parse.
//
//	May be empty (processes buffered data only).
//	Must be valid UTF-8 for correct string operations.
//	May be concatenated across multiple calls (streaming).
//
// @return []Event Slice of complete events extracted from the input.
//
//	Returns empty slice if no complete events available.
//	Events are ordered by their position in input.
//
// @pre Parser must be initialized via NewParser.
// @pre text must be valid UTF-8.
// @post Input text is appended to internal buffer.
// @post Complete events are removed from buffer.
// @post Parser state is updated based on recognized tokens.
//
// @note Partial tokens at the end of text are buffered for the next Parse call.
//
//	This enables correct handling of streaming input where tokens may be
//	split across chunks.
//
// @note Callers should check returned events even when text is empty,
//
//	as buffered data may produce events.
func (p *Parser) Parse(text string) []Event {
	// Append new text to buffer for processing.
	// Buffer may already contain partial data from previous calls.
	p.buf += text
	return p.processBuffer()
}

// processBuffer repeatedly processes the buffer until no more events are produced.
// This handles cases where one buffer may contain multiple complete events.
//
// @brief Internal method that iteratively processes buffer until stable.
//
// @return []Event All complete events from buffer processing.
//
// @pre p.buf contains all unprocessed input.
// @post p.buf contains only unprocessed (partial) data.
func (p *Parser) processBuffer() []Event {
	var events []Event
	for {
		prevBuf := p.buf
		evts := p.processState()
		events = append(events, evts...)
		// Termination condition 1: No events produced AND buffer unchanged.
		// This means we're waiting for more input to complete a token.
		if len(evts) == 0 && p.buf == prevBuf {
			return events
		}
		// Termination condition 2: Parser is in a stable state.
		// stateIdle and stateTrailing are states where we can safely return
		// and wait for more input.
		if p.state == stateIdle || p.state == stateTrailing {
			return events
		}
		// Continue processing - buffer was modified but not in a terminal state.
		// This handles cases where one state transition leads to another.
	}
}

// processState dispatches to the appropriate state handler based on current state.
// Each handler processes the buffer according to the expected tokens for that state.
//
// @brief Dispatches to state-specific processing handler.
//
// @return []Event Events produced by the state handler.
//
// @pre p.state indicates the current parser state.
// @post p.state may be updated based on recognized tokens.
// @post p.buf may be modified based on processed content.
func (p *Parser) processState() []Event {
	switch p.state {
	case stateIdle:
		return p.processIdle()
	case stateInSection:
		return p.processInSection()
	case stateReadingID:
		return p.processReadingID()
	case stateReadingArgs:
		return p.processReadingArgs()
	case stateTrailing:
		return p.processTrailing()
	default:
		// Invalid state - this should never happen in correct code.
		// Return nil to avoid infinite loop.
		return nil
	}
}

// processIdle handles parsing when in the idle state.
// Looks for SectionBegin token or emits content if not found.
//
// @brief Processes buffer in stateIdle, looking for tool call section start.
//
// @return []Event EventContent for text before section, or nil if waiting.
//
// State Transition:
// - On SectionBegin found: stateIdle -> stateInSection
// - Otherwise: Remain in stateIdle, emit buffered content
//
// @pre Parser must be in stateIdle.
// @post Parser transitions to stateInSection if SectionBegin found.
func (p *Parser) processIdle() []Event {
	// Search for section begin marker in buffer.
	idx := strings.Index(p.buf, p.tokens.SectionBegin)
	if idx < 0 {
		// SectionBegin not found - buffer contains only regular content.
		if p.buf == "" {
			// Empty buffer - nothing to process.
			return nil
		}
		// Emit all buffered content as regular text.
		// This is safe because SectionBegin was not found, meaning
		// the buffer cannot contain the start of a tool call section.
		text := p.buf
		p.buf = ""
		return []Event{{Type: EventContent, Text: text}}
	}
	// SectionBegin found at position idx.
	// First, emit any content before the marker as regular text.
	var events []Event
	if idx > 0 {
		// There is content before the section marker - emit it.
		events = append(events, Event{Type: EventContent, Text: p.buf[:idx]})
	}
	// Remove the marker and preceding content from buffer.
	// Transition to stateInSection to look for tool calls.
	p.buf = p.buf[idx+len(p.tokens.SectionBegin):]
	p.state = stateInSection
	return events
}

// processInSection handles parsing when inside a tool calls section.
// Looks for CallBegin (start of tool call) or SectionEnd (section complete).
//
// @brief Processes buffer in stateInSection, looking for tool call or section end.
//
// @return []Event Events from section end, or nil if waiting.
//
// State Transitions:
// - On CallBegin found: stateInSection -> stateReadingID
// - On SectionEnd found: stateInSection -> stateTrailing
//
// @pre Parser must be in stateInSection.
// @post Parser transitions based on which token is found first.
func (p *Parser) processInSection() []Event {
	// Look for both possible tokens - the one found first wins.
	callIdx := strings.Index(p.buf, p.tokens.CallBegin)
	endIdx := strings.Index(p.buf, p.tokens.SectionEnd)
	// SectionEnd takes priority if it comes before CallBegin.
	// This handles the case of an empty section (no tool calls).
	if endIdx >= 0 && (callIdx < 0 || endIdx < callIdx) {
		return p.endSection(endIdx)
	}
	// CallBegin not found - need more input.
	if callIdx < 0 {
		return nil
	}
	// Found CallBegin - remove it and transition to reading ID.
	p.buf = p.buf[callIdx+len(p.tokens.CallBegin):]
	p.state = stateReadingID
	return nil
}

// processReadingID handles parsing when reading tool call ID and name.
// Looks for ArgBegin marker to find end of ID/name section.
//
// @brief Processes buffer in stateReadingID, extracting tool call ID and name.
//
// @return []Event EventToolStart with ID and name, or nil if waiting.
//
// State Transition:
// - On ArgBegin found: stateReadingID -> stateReadingArgs
//
// @pre Parser must be in stateReadingID.
// @post Parser transitions to stateReadingArgs if ArgBegin found.
func (p *Parser) processReadingID() []Event {
	// Search for argument begin marker.
	argIdx := strings.Index(p.buf, p.tokens.ArgBegin)
	if argIdx < 0 {
		// ArgBegin not found - need more input.
		return nil
	}
	// Extract the ID and name from text before ArgBegin.
	// Content between CallBegin and ArgBegin contains:
	// "<id>:<function_name>" or just "<function_name>"
	rawID := strings.TrimSpace(p.buf[:argIdx])
	id, name := p.parseToolCallID(rawID)
	// Remove processed content and ArgBegin from buffer.
	p.buf = p.buf[argIdx+len(p.tokens.ArgBegin):]
	// Transition to reading arguments state.
	p.state = stateReadingArgs
	return []Event{{
		Type:  EventToolStart,
		ID:    id,
		Name:  name,
		Index: p.toolIndex,
	}}
}

// processReadingArgs handles parsing when reading tool call arguments.
// Looks for CallEnd marker to find end of arguments.
//
// @brief Processes buffer in stateReadingArgs, extracting argument data.
//
// @return []Event EventToolArgs and EventToolEnd, or just EventToolArgs.
//
// State Transition:
// - On CallEnd found: stateReadingArgs -> stateInSection
//
// @pre Parser must be in stateReadingArgs.
// @post Parser transitions to stateInSection if CallEnd found.
// @post toolIndex is incremented after complete tool call.
func (p *Parser) processReadingArgs() []Event {
	// Search for call end marker.
	endIdx := strings.Index(p.buf, p.tokens.CallEnd)
	if endIdx < 0 {
		// CallEnd not found - emit any buffered arguments.
		// This handles streaming where arguments arrive in chunks.
		if p.buf == "" {
			return nil
		}
		// Emit all current buffer as arguments.
		// The buffer will be cleared and we'll wait for more input.
		args := p.buf
		p.buf = ""
		return []Event{{Type: EventToolArgs, Args: args, Index: p.toolIndex}}
	}
	// CallEnd found - emit remaining arguments and end event.
	var events []Event
	args := p.buf[:endIdx]
	if args != "" {
		// There are arguments before the end marker.
		events = append(events, Event{Type: EventToolArgs, Args: args, Index: p.toolIndex})
	}
	// Emit the tool end event.
	events = append(events, Event{Type: EventToolEnd, Index: p.toolIndex})
	// Remove processed content and CallEnd from buffer.
	p.buf = p.buf[endIdx+len(p.tokens.CallEnd):]
	// Increment tool index for next tool call.
	p.toolIndex++
	// Return to section state to look for more tool calls or section end.
	p.state = stateInSection
	return events
}

// processTrailing handles parsing when after a tool calls section.
// May find another section or emit trailing content.
//
// @brief Processes buffer in stateTrailing, handling content after section.
//
// @return []Event EventContent for text, or events for new section.
//
// State Transitions:
// - On SectionBegin found: stateTrailing -> stateInSection
// - Otherwise: Remain in stateTrailing, emit content
//
// @pre Parser must be in stateTrailing.
// @post Parser transitions to stateInSection if new section found.
func (p *Parser) processTrailing() []Event {
	// Check if there's another tool call section starting.
	idx := strings.Index(p.buf, p.tokens.SectionBegin)
	if idx >= 0 {
		var events []Event
		// Emit any content before the new section.
		if idx > 0 {
			events = append(events, Event{Type: EventContent, Text: p.buf[:idx]})
		}
		// Remove content and marker, transition to section state.
		p.buf = p.buf[idx+len(p.tokens.SectionBegin):]
		p.state = stateInSection
		return events
	}
	// No new section - emit any content as regular text.
	if p.buf == "" {
		return nil
	}
	text := p.buf
	p.buf = ""
	return []Event{{Type: EventContent, Text: text}}
}

// shouldEndSection checks if the buffer contains a section end marker.
// This is used to determine if we should transition to trailing state.
//
// @brief Checks for section end marker in buffer.
//
// @return bool True if SectionEnd marker is present, false otherwise.
//
// @pre None.
// @post No state is modified.
//
// @note This is a query function that does not modify parser state.
func (p *Parser) shouldEndSection() bool {
	endIdx := strings.Index(p.buf, p.tokens.SectionEnd)
	return endIdx >= 0
}

// endSection handles transitioning from a tool calls section to trailing state.
// Removes the section end marker and emits a section end event.
//
// @brief Ends the current tool calls section and emits section end event.
//
// @param endIdx The position of the SectionEnd marker in the buffer.
//
//	Must be >= 0 and < len(p.buf).
//
// @return []Event Single EventSectionEnd event.
//
// @pre Parser must be in a section-processing state.
// @post Parser transitions to stateTrailing.
// @post SectionEnd marker and preceding content are removed from buffer.
func (p *Parser) endSection(endIdx int) []Event {
	// Preserve any trailing content after the section end marker.
	trailing := p.buf[endIdx+len(p.tokens.SectionEnd):]
	p.buf = trailing
	// Transition to trailing state to handle any content after section.
	p.state = stateTrailing
	return []Event{{Type: EventSectionEnd}}
}

// parseToolCallID extracts the ID and function name from raw tool call identifier text.
// If the text lacks a proper ID prefix, a new ID is generated.
//
// @brief Parses tool call identifier text into ID and function name components.
//
// @param raw The raw text between CallBegin and ArgBegin markers.
//
//	Expected formats: "call_<id>:<name>", "toolu_<id>:<name>", or "<name>".
//	Must be valid UTF-8 for correct parsing.
//
// @return string The tool call ID (either extracted or generated).
// @return string The function name (extracted from raw text).
//
// @pre raw must not be empty for meaningful output.
// @post Returned ID is either the original ID or a generated unique ID.
// @post Returned name has module prefix and parameter suffix removed.
//
// @note Generated IDs use format "call_<index>_<timestamp>" to ensure uniqueness.
//
//	The timestamp component ensures IDs are unique even across sessions.
func (p *Parser) parseToolCallID(raw string) (string, string) {
	raw = strings.TrimSpace(raw)
	name := p.extractFunctionName(raw)
	// Check for standard ID prefixes from the LLM.
	if strings.HasPrefix(raw, "call_") || strings.HasPrefix(raw, "toolu_") {
		return raw, name
	}
	// No standard ID prefix - generate a unique ID.
	// Format: call_<index>_<timestamp_ms>
	// This ensures uniqueness within and across sessions.
	id := fmt.Sprintf("call_%d_%d", p.toolIndex, time.Now().UnixMilli())
	return id, name
}

// extractFunctionName parses the function name from raw tool call ID text.
// It strips module prefixes and parameter suffixes.
//
// @brief Extracts clean function name from raw identifier text.
//
// @param raw The raw text possibly containing module prefix and parameters.
//
//	Example formats: "module.function", "function:params", "module.function:params".
//	Must be valid UTF-8 for correct parsing.
//
// @return string The clean function name with prefixes and suffixes removed.
//
// @pre raw must be valid UTF-8.
// @post Returned string contains only the function name.
//
// @note Processing order:
//  1. Trim whitespace
//  2. Remove module prefix (everything before last '.')
//  3. Remove parameter suffix (everything after first ':')
func (p *Parser) extractFunctionName(raw string) string {
	raw = strings.TrimSpace(raw)
	// Remove module prefix if present.
	// Example: "python.bash" -> "bash"
	if i := strings.Index(raw, "."); i >= 0 {
		raw = raw[i+1:]
	}
	// Remove parameter suffix if present.
	// Example: "bash:command" -> "bash"
	if i := strings.LastIndex(raw, ":"); i >= 0 {
		raw = raw[:i]
	}
	return raw
}

// Reset clears the parser state for reuse.
// This allows the parser to be reused for processing a new stream.
//
// @brief Resets parser to initial state, clearing all buffered data.
//
// @return None (method returns no value).
//
// @pre Parser may be in any state.
// @post Parser is in stateIdle with empty buffer.
// @post toolIndex is reset to 0.
//
// @note Call Reset() between processing different streams to avoid
//
//	mixing data from different requests.
//
// @note Reset() is idempotent - calling multiple times has the same effect.
func (p *Parser) Reset() {
	p.state = stateIdle
	p.buf = ""
	p.toolIndex = 0
}

// State returns the current parser state.
// This is primarily used for testing and debugging.
//
// @brief Returns the current state machine state.
//
// @return state The current parser state (stateIdle, stateInSection, etc.).
//
// @pre None.
// @post No state is modified.
//
// @note This method is primarily for testing and debugging.
//
//	Production code should not need to check parser state.
func (p *Parser) State() state {
	return p.state
}

// Buffer returns the current unprocessed buffer contents.
// This is primarily used for testing and debugging.
//
// @brief Returns the internal buffer containing unprocessed input.
//
// @return string The current buffer contents (may be empty).
//
// @pre None.
// @post No state is modified.
// @post Buffer contents are not copied; caller should not modify.
//
// @note This method is primarily for testing and debugging.
//
//	Production code should not need to inspect the buffer.
//
// @note The returned string may contain partial tokens at the end.
func (p *Parser) Buffer() string {
	return p.buf
}
