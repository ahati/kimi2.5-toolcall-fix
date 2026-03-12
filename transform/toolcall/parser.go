// Package toolcall provides tool call transformation functionality for streaming responses.
// It implements a state machine parser that extracts tool calls from special delimiter tokens
// and format-specific output formatters that emit properly formatted deltas.
package toolcall

import (
	"strings"

	"ai-proxy/types"
)

// State represents the current state of the tool call parser state machine.
// The parser transitions through states as it processes streaming text.
type State int

const (
	// StateIdle indicates the parser is not currently processing a tool call section.
	// In this state, text is passed through directly to the output handler.
	StateIdle State = iota

	// StateInSection indicates the parser has detected a tool calls section begin token
	// and is waiting for a tool call begin token or section end token.
	StateInSection

	// StateReadingID indicates the parser has detected a tool call begin token
	// and is accumulating the tool call ID/name until an argument begin token is found.
	StateReadingID

	// StateReadingArgs indicates the parser is accumulating tool call arguments
	// until a tool call end token is found.
	StateReadingArgs

	// StateTrailing indicates the parser has exited a tool calls section and may
	// encounter another section begin token for chained tool calls.
	StateTrailing
)

// TokenSet defines the delimiter tokens used to mark tool call boundaries in streaming text.
// These tokens are emitted by the upstream LLM to indicate the structure of tool calls.
type TokenSet struct {
	// SectionBegin marks the start of a tool calls section.
	SectionBegin string
	// CallBegin marks the start of an individual tool call within a section.
	CallBegin string
	// ArgBegin marks the start of tool call arguments.
	ArgBegin string
	// CallEnd marks the end of an individual tool call.
	CallEnd string
	// SectionEnd marks the end of the tool calls section.
	SectionEnd string
}

// DefaultTokenSet returns the default set of delimiter tokens used by the Kimi model.
//
// @brief    Returns the standard token set for tool call parsing.
// @return   TokenSet containing the default delimiter strings.
//
// @note     The tokens are specific to the Kimi model's output format.
func DefaultTokenSet() TokenSet {
	return TokenSet{
		SectionBegin: "<|tool_calls_section_begin|>",
		CallBegin:    "<|tool_call_begin|>",
		ArgBegin:     "<|tool_call_argument_begin|>",
		CallEnd:      "<|tool_call_end|>",
		SectionEnd:   "<|tool_calls_section_end|>",
	}
}

// EventHandler defines the interface for receiving parser events.
// Implementations handle different output formats (OpenAI, Anthropic, etc.).
type EventHandler interface {
	// OnText is called when regular text content is encountered outside tool calls.
	// The text should be emitted as-is to the output stream.
	OnText(text string)

	// OnToolCallStart is called when a new tool call is detected.
	// id is the normalized tool call identifier, name is the function name,
	// and index is the zero-based position in the tool call sequence.
	OnToolCallStart(id, name string, index int)

	// OnToolCallArgs is called when tool call arguments are encountered.
	// Multiple calls may occur for a single tool call as arguments are streamed.
	OnToolCallArgs(args string, index int)

	// OnToolCallEnd is called when a tool call is complete.
	// No more arguments will be received for this tool call index.
	OnToolCallEnd(index int)
}

// Parser implements a state machine for extracting tool calls from streaming text.
// It processes text incrementally, buffering partial tokens until complete delimiters
// are formed.
type Parser struct {
	// state is the current state machine state.
	state State
	// buf holds unprocessed text, including partial delimiter tokens.
	buf string
	// tokens defines the delimiter strings to search for.
	tokens TokenSet
	// handler receives events when parsing actions occur.
	handler EventHandler
	// toolIdx is the current tool call index, incremented after each completed call.
	toolIdx int
	// currentID holds the normalized ID of the tool call being parsed.
	currentID string
}

// NewParser creates a new parser instance with the specified token set and event handler.
//
// @brief    Initializes a new tool call parser.
// @param    tokens  The delimiter token set to use for parsing.
// @param    handler The event handler to receive parsing callbacks.
// @return   Pointer to a newly allocated Parser instance.
//
// @pre      tokens must contain valid delimiter strings.
// @pre      handler must not be nil.
// @post     Parser is in StateIdle with empty buffer.
func NewParser(tokens TokenSet, handler EventHandler) *Parser {
	return &Parser{
		state:   StateIdle,
		tokens:  tokens,
		handler: handler,
	}
}

// Feed processes incoming text through the parser state machine.
//
// @brief    Feeds text to the parser for processing.
// @param    text The text to process, may contain partial tokens.
//
// @note     Text may contain partial delimiter tokens that span multiple calls.
//
//	The parser buffers such partial tokens until complete.
//
// @note     This function may trigger output via the EventHandler callbacks.
//
// @pre      Parser must be initialized with NewParser().
// @post     Internal state is updated based on processed content.
func (p *Parser) Feed(text string) {
	p.buf += text
	p.process()
}

// Flush outputs any remaining buffered content.
// This should be called at the end of a stream to emit any trailing text or arguments.
//
// @brief    Emits remaining buffered content to the handler.
//
// @note     In StateIdle or StateTrailing, buffered text is emitted as regular text.
// @note     In StateReadingArgs, buffered content is emitted as tool arguments.
// @note     Other states do not produce output on flush.
//
// @post     Internal buffer is cleared.
func (p *Parser) Flush() {
	// Handle any remaining buffered content based on current state
	// This ensures no content is lost when the stream ends
	if p.buf != "" {
		switch p.state {
		case StateIdle, StateTrailing:
			// In these states, buffered content is regular text output
			p.handler.OnText(p.buf)
		case StateReadingArgs:
			// In this state, buffered content is tool call arguments
			// This handles cases where the stream ends before CallEnd token
			p.handler.OnToolCallArgs(p.buf, p.toolIdx)
		}
		// Clear buffer after emitting to prevent double-processing
		p.buf = ""
	}
}

// Reset clears the parser state and buffer, preparing for a new stream.
//
// @brief    Resets the parser to initial state for reuse.
//
// @post     Parser is in StateIdle with empty buffer.
// @post     Tool index and current ID are cleared.
func (p *Parser) Reset() {
	p.state = StateIdle
	p.buf = ""
	p.toolIdx = 0
	p.currentID = ""
}

// process runs the state machine until no more complete tokens are found.
// It is called internally by Feed and handles all state transitions.
//
// @brief    Executes the state machine processing loop.
//
// @note     This function handles all state transitions and event emission.
// @note     It returns when the buffer cannot be further processed with current state.
//
// @pre      Buffer contains text to process.
// @post     State machine is in a stable waiting state.
func (p *Parser) process() {
	for {
		switch p.state {
		case StateIdle:
			// Search for section begin token to detect start of tool calls
			idx := strings.Index(p.buf, p.tokens.SectionBegin)
			if idx < 0 {
				// No section begin found - emit all buffered text as regular content
				// and clear buffer since we've processed everything
				if p.buf != "" {
					p.handler.OnText(p.buf)
					p.buf = ""
				}
				return
			}

			// Emit any text before the section begin token as regular content
			// This preserves content that appears before tool calls in the stream
			if idx > 0 {
				p.handler.OnText(p.buf[:idx])
			}

			// Advance buffer past the section begin token and transition state
			// The buffer now contains content starting from inside the tool calls section
			p.buf = p.buf[idx+len(p.tokens.SectionBegin):]
			p.state = StateInSection

		case StateInSection:
			// Search for both call begin and section end tokens
			// We need both positions to determine the correct action
			idx := strings.Index(p.buf, p.tokens.CallBegin)
			endIdx := strings.Index(p.buf, p.tokens.SectionEnd)

			// Handle edge case: section end appears before call begin (or no call begin found)
			// This occurs when a tool calls section is empty or ends without any tool calls
			if endIdx >= 0 && (idx < 0 || endIdx < idx) {
				// Extract any content after the section end for trailing text output
				// This preserves content that follows the tool calls section
				trailing := p.buf[endIdx+len(p.tokens.SectionEnd):]
				p.buf = ""
				p.state = StateTrailing

				// Emit trailing content immediately if present
				// This handles cases where text follows directly after section end
				if trailing != "" {
					p.handler.OnText(trailing)
					p.buf = ""
				}
				return
			}

			// No call begin token found yet - wait for more data
			// The buffer may contain a partial token that will complete on next Feed()
			if idx < 0 {
				return
			}

			// Advance buffer past the call begin token and transition to reading ID
			// The buffer now contains the tool call ID/name followed by argument begin
			p.buf = p.buf[idx+len(p.tokens.CallBegin):]
			p.state = StateReadingID

		case StateReadingID:
			// Search for argument begin token which marks end of ID/name portion
			argIdx := strings.Index(p.buf, p.tokens.ArgBegin)
			if argIdx < 0 {
				// Token not found yet - buffer contains partial ID/name
				// Wait for more data to complete the token
				return
			}

			// Extract and normalize the tool call ID/name
			// The raw ID may be in format "name:id" or just "id"
			// NormalizeToolID ensures consistent ID format across different outputs
			rawID := strings.TrimSpace(p.buf[:argIdx])
			p.currentID = types.NormalizeToolID(rawID, p.toolIdx)
			name := types.ParseFunctionName(rawID)

			// Advance buffer past the argument begin token
			// Buffer now contains the tool call arguments
			p.buf = p.buf[argIdx+len(p.tokens.ArgBegin):]
			p.state = StateReadingArgs

			// Notify handler of new tool call with extracted ID and name
			p.handler.OnToolCallStart(p.currentID, name, p.toolIdx)

		case StateReadingArgs:
			// Search for call end token which marks completion of this tool call
			endIdx := strings.Index(p.buf, p.tokens.CallEnd)
			if endIdx < 0 {
				// No end token found - emit accumulated arguments as they stream in
				// This provides incremental updates for long argument strings
				if p.buf != "" {
					p.handler.OnToolCallArgs(p.buf, p.toolIdx)
					p.buf = ""
				}
				return
			}

			// Extract final argument portion before the call end token
			// This captures any arguments that came after the last emit
			args := p.buf[:endIdx]
			if args != "" {
				p.handler.OnToolCallArgs(args, p.toolIdx)
			}

			// Signal completion of this tool call to the handler
			p.handler.OnToolCallEnd(p.toolIdx)

			// Advance buffer past the call end token
			// Buffer may contain more tool calls or section end
			p.buf = p.buf[endIdx+len(p.tokens.CallEnd):]

			// Increment tool index for next call and return to section state
			// to look for additional tool calls or section end
			p.toolIdx++
			p.state = StateInSection

		case StateTrailing:
			// Check for another section begin token to handle chained tool call sections
			// Some models emit multiple tool call sections in sequence
			idx := strings.Index(p.buf, p.tokens.SectionBegin)
			if idx >= 0 {
				// Emit any text before the new section as regular content
				if idx > 0 {
					p.handler.OnText(p.buf[:idx])
				}

				// Advance buffer past section begin and re-enter section state
				// This allows processing of multiple tool call sections in one stream
				p.buf = p.buf[idx+len(p.tokens.SectionBegin):]
				p.state = StateInSection
				continue
			}

			// No new section found - emit buffered trailing text as regular content
			// This handles text that appears after all tool call sections
			if p.buf != "" {
				p.handler.OnText(p.buf)
				p.buf = ""
			}
			return
		}
	}
}
