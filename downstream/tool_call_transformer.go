package downstream

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/tmaxmax/go-sse"
)

const (
	tokSectionBegin = "<|tool_calls_section_begin|>"
	tokCallBegin    = "<|tool_call_begin|>"
	tokArgBegin     = "<|tool_call_argument_begin|>"
	tokCallEnd      = "<|tool_call_end|>"
	tokSectionEnd   = "<|tool_calls_section_end|>"
)

type parserState int

const (
	stateIdle parserState = iota
	stateInSection
	stateReadingID
	stateReadingArgs
	stateTrailing
)

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
	Content          string     `json:"content,omitempty"`
	Reasoning        string     `json:"reasoning,omitempty"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
	FinishReason     *string    `json:"finish_reason,omitempty"`
}

type ToolCall struct {
	ID       string   `json:"id,omitempty"`
	Type     string   `json:"type,omitempty"`
	Index    int      `json:"index"`
	Function Function `json:"function"`
}

type Function struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

type ToolCallTransformer struct {
	output    io.Writer
	state     parserState
	buf       string
	toolIndex int
	currentID string
}

func NewToolCallTransformer(output io.Writer) *ToolCallTransformer {
	return &ToolCallTransformer{
		output: output,
		state:  stateIdle,
	}
}

func (t *ToolCallTransformer) Transform(event *sse.Event) {
	if event.Data == "" || event.Data == "[DONE]" {
		if event.Data == "[DONE]" {
			t.writeSSE([]byte("data: [DONE]\n\n"))
		}
		return
	}

	chunks, err := t.processEvent([]byte(event.Data))
	if err != nil {
		return
	}

	for _, chunk := range chunks {
		t.writeSSE(chunk)
	}
}

func (t *ToolCallTransformer) processEvent(raw []byte) ([][]byte, error) {
	s := strings.TrimSpace(string(raw))
	if s == "[DONE]" {
		return [][]byte{[]byte("data: [DONE]\n\n")}, nil
	}

	var chunk Chunk
	if err := json.Unmarshal([]byte(s), &chunk); err != nil {
		return [][]byte{[]byte("data: " + s + "\n\n")}, nil
	}

	// Promote finish_reason from delta up to choice (upstream puts it in the wrong place)
	if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.FinishReason != nil {
		chunk.Choices[0].FinishReason = chunk.Choices[0].Delta.FinishReason
		chunk.Choices[0].Delta.FinishReason = nil
	}

	if len(chunk.Choices) == 0 {
		return t.emit(chunk)
	}

	reasoning := chunk.Choices[0].Delta.Reasoning
	if reasoning == "" {
		reasoning = chunk.Choices[0].Delta.ReasoningContent
	}

	if !containsAnyToken(reasoning) && t.state == stateIdle {
		// if reasoning != "" {
		// 	chunk.Choices[0].Delta.Content = reasoning
		// 	chunk.Choices[0].Delta.Reasoning = ""
		// 	chunk.Choices[0].Delta.ReasoningContent = ""
		// }
		return t.emit(chunk)
	}

	return t.processReasoning(chunk, reasoning)
}

func (t *ToolCallTransformer) processReasoning(base Chunk, text string) ([][]byte, error) {
	t.buf += text
	var out [][]byte

	for {
		switch t.state {
		case stateIdle:
			idx := strings.Index(t.buf, tokSectionBegin)
			if idx < 0 {
				return out, nil
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
				return out, nil
			}
			t.buf = t.buf[idx+len(tokCallBegin):]
			t.state = stateReadingID

		case stateReadingID:
			argIdx := strings.Index(t.buf, tokArgBegin)
			if argIdx < 0 {
				return out, nil
			}
			rawID := strings.TrimSpace(t.buf[:argIdx])
			t.currentID, _ = parseToolCallID(rawID, t.toolIndex)
			name := parseFunctionName(rawID)
			t.buf = t.buf[argIdx+len(tokArgBegin):]
			t.state = stateReadingArgs
			out = append(out, t.makeToolCallHeader(base, name))

		case stateReadingArgs:
			endIdx := strings.Index(t.buf, tokCallEnd)
			if endIdx < 0 {
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

func (t *ToolCallTransformer) emit(c Chunk) ([][]byte, error) {
	b, err := json.Marshal(c)
	if err != nil {
		return nil, err
	}
	return [][]byte{[]byte("data: " + string(b) + "\n\n")}, nil
}

func (t *ToolCallTransformer) makeContentChunk(base Chunk, content string) []byte {
	c := shallowCopy(base)
	c.Choices[0].Delta = Delta{Content: content}
	b, _ := json.Marshal(c)
	return []byte("data: " + string(b) + "\n\n")
}

func (t *ToolCallTransformer) makeToolCallHeader(base Chunk, name string) []byte {
	c := shallowCopy(base)
	c.Choices[0].Delta = Delta{ToolCalls: []ToolCall{{
		ID:       t.currentID,
		Type:     "function",
		Index:    t.toolIndex,
		Function: Function{Name: name, Arguments: ""},
	}}}
	b, _ := json.Marshal(c)
	return []byte("data: " + string(b) + "\n\n")
}

func (t *ToolCallTransformer) makeArgsDelta(base Chunk, args string) []byte {
	c := shallowCopy(base)
	c.Choices[0].Delta = Delta{ToolCalls: []ToolCall{{
		Index:    t.toolIndex,
		Function: Function{Arguments: args},
	}}}
	b, _ := json.Marshal(c)
	return []byte("data: " + string(b) + "\n\n")
}

func shallowCopy(c Chunk) Chunk {
	cp := c
	cp.Choices = []Choice{{Index: c.Choices[0].Index, FinishReason: c.Choices[0].FinishReason}}
	return cp
}

func parseToolCallID(raw string, index int) (string, string) {
	raw = strings.TrimSpace(raw)
	if strings.HasPrefix(raw, "call_") {
		return raw, ""
	}
	return fmt.Sprintf("call_%d_%d", index, time.Now().UnixMilli()), raw
}

func parseFunctionName(raw string) string {
	raw = strings.TrimSpace(raw)
	if i := strings.Index(raw, "."); i >= 0 {
		raw = raw[i+1:]
	}
	if i := strings.LastIndex(raw, ":"); i >= 0 {
		raw = raw[:i]
	}
	return raw
}

func containsAnyToken(s string) bool {
	return strings.Contains(s, "<|tool_call")
}

func (t *ToolCallTransformer) writeSSE(data []byte) {
	if len(data) == 0 {
		return
	}
	t.output.Write(data)
}

func (t *ToolCallTransformer) Close() {
	if t.buf != "" && t.state == stateTrailing {
		return
	}
}

func (t *ToolCallTransformer) Flush() {
}
