## Implemenataio Approach "Gemini"

To solve the fragmented token issue, this converter uses a state machine and a buffer. Because tokens can arrive split across multiple SSE chunks (e.g., `<|tool`, `_call`, `_begin|>`), the parser buffers text and safely flushes content only when it confirms no special token marker (`<`) is forming.

### Core Transformation Logic

```go
package transformer

import (
	"encoding/json"
	"strings"
)

// Standard OpenAI format structures
type Chunk struct {
	Id      string   `json:"id"`
	Object  string   `json:"object"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   any      `json:"usage,omitempty"`
}
type Choice struct {
	Index        int    `json:"index"`
	Delta        Delta  `json:"delta"`
	FinishReason string `json:"finish_reason,omitempty"`
}
type Delta struct {
	Content   string     `json:"content,omitempty"`
	Reasoning string     `json:"reasoning,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}
type ToolCall struct {
	Index    int       `json:"index"`
	Id       string    `json:"id,omitempty"`
	Type     string    `json:"type,omitempty"`
	Function *Function `json:"function,omitempty"`
}
type Function struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

```

### State Machine & Converter

```go
const (
	StateIdle = iota; StateSection; StateInID; StateInArgs
	SecBeg  = "<|tool_calls_section_begin|>"
	SecEnd  = "<|tool_calls_section_end|>"
	CallBeg = "<|tool_call_begin|>"
	ArgBeg  = "<|tool_call_argument_begin|>"
	CallEnd = "<|tool_call_end|>"
)

type Converter struct {
	state  int
	buf    string
	out    []Chunk
	tcIdx  int
	currID string
}

func (c *Converter) Transform(data []byte) [][]byte {
	var chunk Chunk
	if json.Unmarshal(data, &chunk) != nil || len(chunk.Choices) == 0 {
		return [][]byte{data} // Passthrough usage or invalid chunks
	}
	
	c.out, c.buf = nil, c.buf+extractText(&chunk)
	for c.processStep(&chunk) {}
	c.handleFinish(&chunk)

	var res [][]byte
	for _, ch := range c.out {
		b, _ := json.Marshal(ch)
		res = append(res, b)
	}
	return res
}

```

### Handlers (Strictly under 20 lines)

```go
func (c *Converter) processStep(b *Chunk) bool {
	if len(c.buf) == 0 { return false }
	switch c.state {
	case StateIdle: return c.handleIdle(b)
	case StateSection: return c.handleSection(b)
	case StateInID: return c.handleInID(b)
	case StateInArgs: return c.handleInArgs(b)
	}
	return false
}

func (c *Converter) handleIdle(b *Chunk) bool {
	if idx := strings.Index(c.buf, SecBeg); idx >= 0 {
		c.emitContent(b, c.buf[:idx])
		c.buf, c.state = c.buf[idx+len(SecBeg):], StateSection
		return true
	}
	return c.flushSafe(b, true)
}

func (c *Converter) handleSection(b *Chunk) bool {
	c.buf = strings.TrimLeft(c.buf, " \n\r\t")
	if strings.HasPrefix(c.buf, CallBeg) {
		c.buf, c.state, c.currID = c.buf[len(CallBeg):], StateInID, ""
		return true
	}
	if strings.HasPrefix(c.buf, SecEnd) {
		c.buf, c.state = c.buf[len(SecEnd):], StateIdle
		return true
	}
	return c.trimToToken()
}

func (c *Converter) handleInID(b *Chunk) bool {
	if idx := strings.Index(c.buf, ArgBeg); idx >= 0 {
		c.currID += c.buf[:idx]
		c.buf = c.buf[idx+len(ArgBeg):]
		c.emitToolStart(b)
		c.state = StateInArgs
		return true
	}
	return c.accumulateID()
}

func (c *Converter) handleInArgs(b *Chunk) bool {
	if idx := strings.Index(c.buf, CallEnd); idx >= 0 {
		c.emitToolArg(b, c.buf[:idx])
		c.buf = c.buf[idx+len(CallEnd):]
		c.state, c.tcIdx = StateSection, c.tcIdx+1
		return true
	}
	return c.flushSafe(b, false)
}

```

### Safe Emission & Buffering Helpers

```go
func (c *Converter) flushSafe(b *Chunk, isContent bool) bool {
	idx, text := strings.Index(c.buf, "<"), c.buf
	if idx >= 0 {
		text, c.buf = c.buf[:idx], c.buf[idx:]
	} else {
		c.buf = ""
	}
	if isContent { c.emitContent(b, text) } else { c.emitToolArg(b, text) }
	return false
}

func (c *Converter) trimToToken() bool {
	if idx := strings.Index(c.buf, "<"); idx > 0 {
		c.buf = c.buf[idx:]
		return true
	} else if idx == -1 {
		c.buf = ""
	}
	return false
}

func (c *Converter) accumulateID() bool {
	if idx := strings.Index(c.buf, "<"); idx >= 0 {
		c.currID += c.buf[:idx]; c.buf = c.buf[idx:]
	} else {
		c.currID += c.buf; c.buf = ""
	}
	return false
}

func (c *Converter) handleFinish(b *Chunk) {
	if fr := b.Choices[0].FinishReason; fr != "" {
		if c.buf != "" { c.emitContent(b, c.buf); c.buf = "" }
		end := clone(b)
		end.Choices[0].FinishReason = fr
		c.out = append(c.out, end)
	}
}

```

### JSON Construct Generators

```go
func clone(b *Chunk) Chunk {
	return Chunk{Id: b.Id, Object: b.Object, Model: b.Model, Usage: b.Usage,
		Choices: []Choice{{Index: b.Choices[0].Index}}}
}

func extractText(c *Chunk) string {
	if r := c.Choices[0].Delta.Reasoning; r != "" { return r }
	return c.Choices[0].Delta.Content
}

func parseName(id string) string {
	parts := strings.Split(id, ".")
	return strings.Split(parts[len(parts)-1], ":")[0]
}

func (c *Converter) emitContent(b *Chunk, text string) {
	if text == "" { return }
	ch := clone(b)
	ch.Choices[0].Delta = Delta{Content: text}
	c.out = append(c.out, ch)
}

func (c *Converter) emitToolStart(b *Chunk) {
	ch, name := clone(b), parseName(c.currID)
	ch.Choices[0].Delta = Delta{ToolCalls: []ToolCall{{
		Index: c.tcIdx, Id: c.currID, Type: "function",
		Function: &Function{Name: name, Arguments: ""},
	}}}
	c.out = append(c.out, ch)
}

func (c *Converter) emitToolArg(b *Chunk, arg string) {
	if arg == "" { return }
	ch := clone(b)
	ch.Choices[0].Delta = Delta{ToolCalls: []ToolCall{{
		Index: c.tcIdx, Function: &Function{Arguments: arg},
	}}}
	c.out = append(c.out, ch)
}

```

---

### Example go-sse Integration

To use this with `tmaxmax/go-sse` on the downstream connection:

```go
// Inside your reverse proxy / request handler
converter := &transformer.Converter{}

// Assuming `reader` is an sse.Reader scanning the upstream Kimi response
for {
    ev, err := reader.Read()
    if err != nil { break }
    
    // Transform upstream Kimi payload to multiple valid OpenAI payloads
    transformedPayloads := converter.Transform([]byte(ev.Data))
    
    for _, payload := range transformedPayloads {
        // Write the transformed payloads downstream
        downstreamWriter.WriteEvent(&sse.Event{
            Type: ev.Type,
            ID:   ev.ID,
            Data: string(payload),
        })
    }
}

```

These tests focus on the trickiest part of SSE transformation: **token fragmentation**. Since a marker like `<|tool_call_begin|>` can be split into `"<|too"`, `"l_ca"`, and `"ll_begin|>"`, the converter must buffer and wait.

### Test Helper: Stream Simulator

This helper simulates a stream of chunks to the converter and collects the results.

```go
func runTestStream(c *Converter, inputs []string) []Chunk {
	var finalOutput []Chunk
	for _, input := range inputs {
		// Mocking the outer JSON structure for the test
		raw := []byte(`{"choices":[{"delta":{"reasoning":"` + input + `"}}],"id":"test"}`)
		results := c.Transform(raw)
		for _, res := range results {
			var ch Chunk
			json.Unmarshal(res, &ch)
			finalOutput = append(finalOutput, ch)
		}
	}
	return finalOutput
}

```

---

### 1. Fragmented Token Test

Tests if the state machine correctly holds data when a special marker is partially received.

```go
func TestFragmentedTokens(t *testing.T) {
	c := &Converter{}
	// Splitting "<|tool_calls_section_begin|>" and "<|tool_call_begin|>"
	stream := []string{"Pre-text <|tool_calls_", "section_begin|>", "<|tool_call", "_begin|>my.func:1<|tool_call_argument_begin|>{}", "<|tool_call_end|>", "<|tool_calls_section_end|>"}
	
	out := runTestStream(c, stream)

	if out[0].Choices[0].Delta.Content != "Pre-text " {
		t.Errorf("Expected pre-text, got %v", out[0].Choices[0].Delta.Content)
	}
	if out[1].Choices[0].Delta.ToolCalls[0].Function.Name != "func" {
		t.Errorf("Expected func name 'func', got %v", out[1].Choices[0].Delta.ToolCalls[0].Function.Name)
	}
}

```

### 2. Mixed Content & Arguments Test

Tests the transition from reasoning text to tool calls and back, including argument streaming.

```go
func TestMixedContentAndArgs(t *testing.T) {
	c := &Converter{}
	stream := []string{
		"Thinking... <|tool_calls_section_begin|><|tool_call_begin|>bash:1<|tool_call_argument_begin|>",
		`{"cmd"`, `: "ls"}`,
		"<|tool_call_end|><|tool_calls_section_end|> Done!"
	}
	
	out := runTestStream(c, stream)
	
	// Verify "Thinking..."
	if out[0].Choices[0].Delta.Content != "Thinking... " { t.Fail() }
	// Verify Tool Call Start
	if out[1].Choices[0].Delta.ToolCalls[0].Id != "bash:1" { t.Fail() }
	// Verify Arguments were streamed
	if out[2].Choices[0].Delta.ToolCalls[0].Function.Arguments != `{"cmd"` { t.Fail() }
	if out[3].Choices[0].Delta.ToolCalls[0].Function.Arguments != `: "ls"}` { t.Fail() }
	// Verify " Done!" trailing text
	if out[4].Choices[0].Delta.Content != " Done!" { t.Fail() }
}

```

### 3. Multiple Sequential Tool Calls

Tests that the converter resets state correctly for the second tool call in a single section.

```go
func TestMultipleToolCalls(t *testing.T) {
	c := &Converter{}
	raw := `<|tool_calls_section_begin|><|tool_call_begin|>a:1<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_call_begin|>b:2<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>`
	
	out := runTestStream(c, []string{raw})

	if len(out) != 4 { // 2 starts + 2 empty args (per emit logic)
		t.Fatalf("Expected 4 chunks for 2 tools, got %d", len(out))
	}
	if out[0].Choices[0].Delta.ToolCalls[0].Id != "a:1" { t.Error("Tool 1 ID mismatch") }
	if out[2].Choices[0].Delta.ToolCalls[0].Id != "b:2" { t.Error("Tool 2 ID mismatch") }
}

```

### 4. Buffer Safety (The "<" Edge Case)

Tests that a literal `<` that isn't a token is eventually flushed.

```go
func TestLiteralLessThan(t *testing.T) {
	c := &Converter{}
	// A math expression split: "if x <", " 5 then"
	out := runTestStream(c, []string{"if x <", " 5 then"})
	
	if len(out) < 2 { t.Fatal("Expected at least 2 chunks") }
	
	combined := out[0].Choices[0].Delta.Content + out[1].Choices[0].Delta.Content
	if combined != "if x < 5 then" {
		t.Errorf("Buffer safety failed, got: %s", combined)
	}
}

```

---

### Implementation Details Check

* **State Machine Consistency**: The `Converter` uses the `buf` to keep a "peek" at upcoming characters.
* **Performance**: Since it processes byte-by-byte only when tokens are detected and otherwise uses `strings.Index`, it remains O(n).
* **Memory**: The buffer only grows to the size of the longest possible token (~30 chars) unless the LLM generates a very long `tool_call_id` without hitting the next marker.

