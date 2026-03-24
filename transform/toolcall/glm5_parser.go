// Package toolcall provides parsing and formatting for LLM tool call tokens.
// This file contains the GLM-5 XML format parser.
package toolcall

import (
	"fmt"
	"strings"
	"time"
)

// glm5State represents the parser's current position within a GLM-5 tool call.
type glm5State int

const (
	// glm5StateIdle indicates the parser is outside any tool call.
	glm5StateIdle glm5State = iota
	// glm5StateInToolCall indicates the parser is inside a tool call, reading function name or between args.
	glm5StateInToolCall
	// glm5StateReadingArgKey indicates the parser is reading an argument key.
	glm5StateReadingArgKey
	// glm5StateReadingArgValue indicates the parser is reading an argument value.
	glm5StateReadingArgValue
)

// GLM5Parser extracts tool calls from GLM-5's XML format in reasoning_content.
// GLM-5 embeds tool calls like: <tool_call>func<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>
type GLM5Parser struct {
	state         glm5State
	buf           string
	toolName      string
	nameExtracted bool // track if we've already extracted the name
	currentKey    string
	currentValue  strings.Builder
	args          map[string]string
}

// NewGLM5Parser creates a new GLM-5 parser.
func NewGLM5Parser() *GLM5Parser {
	return &GLM5Parser{
		state: glm5StateIdle,
		args:  make(map[string]string),
	}
}

// Parse processes text and returns any complete tool call events.
// Text is buffered until complete tool calls are recognized.
func (p *GLM5Parser) Parse(text string) []Event {
	p.buf += text
	return p.processBuffer()
}

// processBuffer repeatedly processes the buffer until no more events are produced.
func (p *GLM5Parser) processBuffer() []Event {
	var events []Event
	for {
		prevBuf := p.buf
		evts := p.processState()
		events = append(events, evts...)
		// Stop if no events produced and buffer unchanged
		if len(evts) == 0 && p.buf == prevBuf {
			return events
		}
		// Continue processing in idle state to find more tool calls
	}
}

// processState dispatches to the appropriate state handler.
func (p *GLM5Parser) processState() []Event {
	switch p.state {
	case glm5StateIdle:
		return p.processIdle()
	case glm5StateInToolCall:
		return p.processInToolCall()
	case glm5StateReadingArgKey:
		return p.processReadingArgKey()
	case glm5StateReadingArgValue:
		return p.processReadingArgValue()
	default:
		return nil
	}
}

// processIdle looks for <tool_call> to start parsing.
func (p *GLM5Parser) processIdle() []Event {
	idx := strings.Index(p.buf, "<tool_call>")
	if idx < 0 {
		// No tool call found, emit any content as regular text
		if p.buf == "" {
			return nil
		}
		text := p.buf
		p.buf = ""
		return []Event{{Type: EventContent, Text: text}}
	}

	// Emit content before <tool_call> as regular text
	var events []Event
	if idx > 0 {
		events = append(events, Event{Type: EventContent, Text: p.buf[:idx]})
	}

	// Remove up to and including <tool_call>
	p.buf = p.buf[idx+len("<tool_call>"):]
	p.state = glm5StateInToolCall
	p.toolName = ""
	p.nameExtracted = false
	p.args = make(map[string]string)
	return events
}

// processInToolCall reads the function name until <arg_key> or </tool_call>.
func (p *GLM5Parser) processInToolCall() []Event {
	// Check for end of tool call
	endIdx := strings.Index(p.buf, "</tool_call>")
	argKeyIdx := strings.Index(p.buf, "<arg_key>")

	// If we haven't extracted the name yet, try to extract it
	if !p.nameExtracted {
		// Find where the name ends
		var nameEnd int
		if argKeyIdx >= 0 && (endIdx < 0 || argKeyIdx < endIdx) {
			nameEnd = argKeyIdx
		} else if endIdx >= 0 {
			nameEnd = endIdx
		} else {
			// Need more data
			return nil
		}
		p.toolName = strings.TrimSpace(p.buf[:nameEnd])
		p.nameExtracted = true
	}

	// Now determine what comes next
	if argKeyIdx >= 0 && (endIdx < 0 || argKeyIdx < endIdx) {
		// <arg_key> comes first - move to reading key
		p.buf = p.buf[argKeyIdx+len("<arg_key>"):]
		p.state = glm5StateReadingArgKey
		p.currentKey = ""
		return nil
	}

	if endIdx >= 0 {
		// </tool_call> comes first - emit the tool call
		p.buf = p.buf[endIdx+len("</tool_call>"):]
		p.state = glm5StateIdle
		return p.emitToolCallEvents()
	}

	// Need more data
	return nil
}

// processReadingArgKey reads the argument key until </arg_key>.
func (p *GLM5Parser) processReadingArgKey() []Event {
	idx := strings.Index(p.buf, "</arg_key>")
	if idx < 0 {
		// Need more data
		return nil
	}

	p.currentKey = p.buf[:idx]
	p.buf = p.buf[idx+len("</arg_key>"):]

	// Now skip past <arg_value> opening tag
	valIdx := strings.Index(p.buf, "<arg_value>")
	if valIdx < 0 {
		// Need more data - this shouldn't happen in valid XML
		return nil
	}
	p.buf = p.buf[valIdx+len("<arg_value>"):]

	p.state = glm5StateReadingArgValue
	p.currentValue.Reset()
	return nil
}

// processReadingArgValue reads the argument value until </arg_value>.
func (p *GLM5Parser) processReadingArgValue() []Event {
	idx := strings.Index(p.buf, "</arg_value>")
	if idx < 0 {
		// Need more data
		return nil
	}

	value := p.buf[:idx]
	p.currentValue.WriteString(value)
	p.args[p.currentKey] = p.currentValue.String()

	p.buf = p.buf[idx+len("</arg_value>"):]

	// Check what comes next: another <arg_key> or </tool_call>
	p.state = glm5StateInToolCall
	return nil
}

// emitToolCallEvents emits the tool call events.
func (p *GLM5Parser) emitToolCallEvents() []Event {
	if p.toolName == "" {
		return nil
	}

	// Build arguments as JSON object using strings.Builder
	var argsBuilder strings.Builder
	argsBuilder.WriteString("{")
	first := true
	for k, v := range p.args {
		if !first {
			argsBuilder.WriteString(",")
		}
		first = false
		// Escape the value for JSON
		argsBuilder.WriteString(`"`)
		argsBuilder.WriteString(k)
		argsBuilder.WriteString(`":`)
		argsBuilder.WriteString(encodeJSONString(v))
	}
	argsBuilder.WriteString("}")

	toolIndex := 0 // GLM-5 typically has one tool call at a time

	return []Event{
		{Type: EventToolStart, ID: generateToolCallID(toolIndex), Name: p.toolName, Index: toolIndex},
		{Type: EventToolArgs, Args: argsBuilder.String(), Index: toolIndex},
		{Type: EventToolEnd, Index: toolIndex},
	}
}

// encodeJSONString escapes a string for JSON encoding.
func encodeJSONString(s string) string {
	// Basic escaping for JSON
	s = strings.ReplaceAll(s, "\\", "\\\\")
	s = strings.ReplaceAll(s, `"`, `\"`)
	s = strings.ReplaceAll(s, "\n", "\\n")
	s = strings.ReplaceAll(s, "\r", "\\r")
	s = strings.ReplaceAll(s, "\t", "\\t")
	return `"` + s + `"`
}

// generateToolCallID creates a unique tool call ID.
func generateToolCallID(index int) string {
	return fmt.Sprintf("call_%d_%d", index, time.Now().UnixMilli())
}

// IsIdle returns true if the parser is not currently parsing a tool call.
func (p *GLM5Parser) IsIdle() bool {
	return p.state == glm5StateIdle
}

// Reset clears the parser state for reuse.
func (p *GLM5Parser) Reset() {
	p.state = glm5StateIdle
	p.buf = ""
	p.toolName = ""
	p.nameExtracted = false
	p.currentKey = ""
	p.currentValue.Reset()
	p.args = make(map[string]string)
}

// State returns the current parser state (for testing).
func (p *GLM5Parser) State() glm5State {
	return p.state
}
