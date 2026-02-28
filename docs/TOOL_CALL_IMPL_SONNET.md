# Design overview - "Sonnet"

**`Transformer` struct** holds a 5-state machine (`IDLE → IN_SECTION → READING_ID → READING_ARGS → TRAILING`) plus a `buf` string that accumulates raw reasoning text across chunk boundaries — this is the key to handling split tokens.

**`Transform(raw []byte)`** is the single entry point called per SSE line. It:
1. Short-circuits `[DONE]` and invalid JSON (passthrough)
2. Passes through chunks with no tool tokens if state is idle
3. Otherwise feeds into `processReasoning`

**`processReasoning`** runs a loop over the buffer, advancing the state machine:
- In `stateIdle`: scans for `<|tool_calls_section_begin|>`, flushes any preceding text as a `content` chunk
- In `stateInSection`: waits for `<|tool_call_begin|>` or `<|tool_calls_section_end|>`
- In `stateReadingID`: waits for `<|tool_call_argument_begin|>`, then parses function name and emits the header chunk
- In `stateReadingArgs`: flushes partial args each chunk; on `<|tool_call_end|>` finalizes and loops back
- In `stateTrailing`: emits trailing text as content

**11 unit tests** cover: passthrough (no tokens), `[DONE]`, single call all-in-one, split-across-chunks, multiple calls, mixed leading content, trailing content, empty arguments, `call_` ID format, usage-only chunks, invalid JSON, and token split at a chunk boundary.


```go
package downstream

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// Special tokens emitted by Kimi-K2.5
const (
	tokSectionBegin  = "<|tool_calls_section_begin|>"
	tokCallBegin     = "<|tool_call_begin|>"
	tokArgBegin      = "<|tool_call_argument_begin|>"
	tokCallEnd       = "<|tool_call_end|>"
	tokSectionEnd    = "<|tool_calls_section_end|>"
)

type parserState int

const (
	stateIdle parserState = iota
	stateInSection
	stateReadingID
	stateReadingArgs
	stateTrailing
)

// Chunk is a minimal OpenAI-compatible SSE chunk (subset used here).
type Chunk struct {
	ID      string   `json:"id,omitempty"`
	Object  string   `json:"object,omitempty"`
	Model   string   `json:"model,omitempty"`
	Created int64    `json:"created,omitempty"`
	Choices []Choice `json:"choices"`
	Usage   *Usage   `json:"usage,omitempty"`
}

type Choice struct {
	Index        int     `json:"index"`
	Delta        Delta   `json:"delta"`
	FinishReason *string `json:"finish_reason,omitempty"`
}

type Delta struct {
	Content          string      `json:"content,omitempty"`
	Reasoning        string      `json:"reasoning,omitempty"`
	ReasoningContent string      `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall  `json:"tool_calls,omitempty"`
}

type ToolCall struct {
	ID       string       `json:"id,omitempty"`
	Type     string       `json:"type,omitempty"`
	Index    int          `json:"index"`
	Function ToolFunction `json:"function"`
}

type ToolFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

// Transformer converts Kimi-K2.5 SSE stream to OpenAI tool_calls format.
type Transformer struct {
	state     parserState
	buf       string   // accumulates raw reasoning text across chunks
	toolIndex int
	currentID string
}

// Transform processes a raw SSE data line (without "data: " prefix) and
// returns zero or more SSE lines to emit downstream.
func (t *Transformer) Transform(raw []byte) ([][]byte, error) {
	s := strings.TrimSpace(string(raw))
	if s == "[DONE]" {
		return [][]byte{[]byte("data: [DONE]\n\n")}, nil
	}

	var chunk Chunk
	if err := json.Unmarshal([]byte(s), &chunk); err != nil {
		return [][]byte{[]byte("data: " + s + "\n\n")}, nil
	}

	// Pass-through if no choices or usage-only chunk
	if len(chunk.Choices) == 0 {
		return t.emit(chunk)
	}

	reasoning := chunk.Choices[0].Delta.Reasoning
	if reasoning == "" {
		reasoning = chunk.Choices[0].Delta.ReasoningContent
	}

	// No special tokens at all → pass through unchanged
	if !containsAnyToken(reasoning) && t.state == stateIdle {
		return t.emit(chunk)
	}

	return t.processReasoning(chunk, reasoning)
}

// processReasoning appends new reasoning text to the buffer and processes tokens.
func (t *Transformer) processReasoning(base Chunk, text string) ([][]byte, error) {
	t.buf += text
	var out [][]byte

	for {
		switch t.state {
		case stateIdle:
			idx := strings.Index(t.buf, tokSectionBegin)
			if idx < 0 {
				return out, nil // wait for more data
			}
			if idx > 0 {
				out = append(out, t.makeContentChunk(base, t.buf[:idx]))
			}
			t.buf = t.buf[idx+len(tokSectionBegin):]
			t.state = stateInSection

		case stateInSection:
			idx := strings.Index(t.buf, tokCallBegin)
			endIdx := strings.Index(t.buf, tokSectionEnd)

			if endIdx >= 0 && (idx < 0 || endIdx < idx) {
				// section end before next call
				trailing := t.buf[endIdx+len(tokSectionEnd):]
				t.buf = ""
				t.state = stateTrailing
				if trailing != "" {
					t.buf = trailing
					out = append(out, t.makeContentChunk(base, trailing))
					t.buf = ""
				}
				return out, nil
			}
			if idx < 0 {
				return out, nil // wait
			}
			t.buf = t.buf[idx+len(tokCallBegin):]
			t.state = stateReadingID

		case stateReadingID:
			argIdx := strings.Index(t.buf, tokArgBegin)
			if argIdx < 0 {
				return out, nil // wait for more
			}
			rawID := strings.TrimSpace(t.buf[:argIdx])
			// rawID is like "functions.bash:15" or "call_abc123"
			t.currentID, _ = parseToolCallID(rawID, t.toolIndex)
			name := parseFunctionName(rawID)
			t.buf = t.buf[argIdx+len(tokArgBegin):]
			t.state = stateReadingArgs
			out = append(out, t.makeToolCallHeader(base, name))

		case stateReadingArgs:
			endIdx := strings.Index(t.buf, tokCallEnd)
			if endIdx < 0 {
				// flush partial args
				if t.buf != "" {
					out = append(out, t.makeArgsDelta(base, t.buf))
					t.buf = ""
				}
				return out, nil
			}
			args := t.buf[:endIdx]
			if args != "" {
				out = append(out, t.makeArgsDelta(base, args))
			}
			t.buf = t.buf[endIdx+len(tokCallEnd):]
			t.toolIndex++
			t.state = stateInSection

		case stateTrailing:
			return out, nil
		}
	}
}

// --- helpers ---

func (t *Transformer) emit(c Chunk) ([][]byte, error) {
	b, err := json.Marshal(c)
	if err != nil {
		return nil, err
	}
	return [][]byte{[]byte("data: " + string(b) + "\n\n")}, nil
}

func (t *Transformer) makeContentChunk(base Chunk, content string) []byte {
	c := shallowCopy(base)
	c.Choices[0].Delta = Delta{Content: content}
	b, _ := json.Marshal(c)
	return []byte("data: " + string(b) + "\n\n")
}

func (t *Transformer) makeToolCallHeader(base Chunk, name string) []byte {
	c := shallowCopy(base)
	c.Choices[0].Delta = Delta{ToolCalls: []ToolCall{{
		ID:       t.currentID,
		Type:     "function",
		Index:    t.toolIndex,
		Function: ToolFunction{Name: name, Arguments: ""},
	}}}
	b, _ := json.Marshal(c)
	return []byte("data: " + string(b) + "\n\n")
}

func (t *Transformer) makeArgsDelta(base Chunk, args string) []byte {
	c := shallowCopy(base)
	c.Choices[0].Delta = Delta{ToolCalls: []ToolCall{{
		Index:    t.toolIndex,
		Function: ToolFunction{Arguments: args},
	}}}
	b, _ := json.Marshal(c)
	return []byte("data: " + string(b) + "\n\n")
}

func shallowCopy(c Chunk) Chunk {
	cp := c
	cp.Choices = []Choice{{Index: c.Choices[0].Index, FinishReason: c.Choices[0].FinishReason}}
	return cp
}

// parseToolCallID returns a stable call ID from the raw token between
// <|tool_call_begin|> and <|tool_call_argument_begin|>.
// Format possibilities: "call_abc123", "functions.bash:15", etc.
func parseToolCallID(raw string, index int) (string, string) {
	raw = strings.TrimSpace(raw)
	if strings.HasPrefix(raw, "call_") {
		return raw, ""
	}
	// fallback
	return fmt.Sprintf("call_%d_%d", index, time.Now().UnixMilli()), raw
}

// parseFunctionName extracts the function name from the ID token.
// "functions.bash:15" → "bash"
// "call_abc123" → "call_abc123" (used as-is for named calls)
func parseFunctionName(raw string) string {
	raw = strings.TrimSpace(raw)
	// strip "functions." prefix
	if i := strings.Index(raw, "."); i >= 0 {
		raw = raw[i+1:]
	}
	// strip ":index" suffix
	if i := strings.LastIndex(raw, ":"); i >= 0 {
		raw = raw[:i]
	}
	return raw
}

func containsAnyToken(s string) bool {
	return strings.Contains(s, "<|tool_call")
}
```