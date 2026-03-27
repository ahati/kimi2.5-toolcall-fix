package toolcall

import (
	"encoding/json"
	"io"

	"ai-proxy/logging"
	"ai-proxy/transform"
	"ai-proxy/types"

	"github.com/tmaxmax/go-sse"
)

type OpenAITransformer struct {
	sseWriter             *transform.SSEWriter
	formatter             *OpenAIFormatter
	parser                *Parser
	glm5Parser            *GLM5Parser
	messageID             string
	model                 string
	inReasoning           bool
	toolCallTransform     bool
	glm5ToolCallTransform bool
}

func NewOpenAITransformer(output io.Writer) *OpenAITransformer {
	return &OpenAITransformer{
		sseWriter:  transform.NewSSEWriter(output),
		formatter:  NewOpenAIFormatter("", ""),
		parser:     NewParser(DefaultTokens),
		glm5Parser: NewGLM5Parser(),
	}
}

// SetKimiToolCallTransform enables or disables tool call extraction from reasoning content.
func (t *OpenAITransformer) SetKimiToolCallTransform(enabled bool) {
	t.toolCallTransform = enabled
}

// SetGLM5ToolCallTransform enables or disables GLM-5 XML tool call extraction.
func (t *OpenAITransformer) SetGLM5ToolCallTransform(enabled bool) {
	t.glm5ToolCallTransform = enabled
}

func (t *OpenAITransformer) Transform(event *sse.Event) error {
	if event.Data == "" {
		return nil
	}

	if event.Data == "[DONE]" {
		return t.writeData([]byte("[DONE]"))
	}

	var chunk types.Chunk
	if err := json.Unmarshal([]byte(event.Data), &chunk); err != nil {
		return t.writeData([]byte(event.Data))
	}

	return t.handleChunk(chunk, event.Data)
}

func (t *OpenAITransformer) handleChunk(chunk types.Chunk, rawData string) error {
	if t.messageID == "" && chunk.ID != "" {
		t.messageID = chunk.ID
		t.model = chunk.Model
		t.formatter.SetMessageID(chunk.ID)
		t.formatter.SetModel(chunk.Model)
	}

	if len(chunk.Choices) == 0 {
		if chunk.Usage != nil {
			return t.writeData([]byte(rawData))
		}
		return t.writeData([]byte(rawData))
	}

	choice := chunk.Choices[0]
	delta := choice.Delta

	if delta.Content != "" {
		t.inReasoning = false
		return t.write(t.formatter.FormatContent(delta.Content))
	}

	if len(delta.ToolCalls) > 0 {
		t.inReasoning = false
		for _, tc := range delta.ToolCalls {
			if tc.ID != "" && tc.Function.Name != "" {
				if err := t.write(t.formatter.FormatToolStart(tc.ID, tc.Function.Name, tc.Index)); err != nil {
					return err
				}
			}
			if tc.Function.Arguments != "" {
				if err := t.write(t.formatter.FormatToolArgs(tc.Function.Arguments, tc.Index)); err != nil {
					return err
				}
			}
		}
		return nil
	}

	text := delta.Reasoning
	if text == "" {
		text = delta.ReasoningContent
	}

	if text != "" {
		t.inReasoning = true
		return t.processText(text)
	}

	if choice.FinishReason != nil && *choice.FinishReason != "" {
		return t.writeData([]byte(rawData))
	}

	return nil
}

func (t *OpenAITransformer) processText(text string) error {
	// Always try GLM-5 parsing when enabled - let the parser's state machine handle detection
	if t.glm5ToolCallTransform {
		events := t.glm5Parser.Parse(text)
		if len(events) > 0 {
			// Check if any events are actual tool call events (not just content)
			hasToolCallEvents := false
			for _, e := range events {
				if e.Type == EventToolStart || e.Type == EventToolArgs || e.Type == EventToolEnd {
					hasToolCallEvents = true
					break
				}
			}
			// Only log "markup detected" when actual tool calls are found
			if hasToolCallEvents {
				logging.InfoMsg("[%s] GLM-5 tool call markup detected in reasoning content, extracting tool calls", t.messageID)
			}
			for _, e := range events {
				if e.Type == EventToolStart {
					logging.InfoMsg("[%s] GLM-5 tool call extracted: name=%s, id=%s, index=%d", t.messageID, e.Name, e.ID, e.Index)
				}
				if err := t.writeEvent(e); err != nil {
					return err
				}
			}
			return nil
		}
		// If parser might be parsing (buffering partial tag), don't emit as reasoning
		if t.glm5Parser.IsPotentiallyParsing() {
			return nil
		}
	}

	// Check for Kimi-style tool call markup (always try if not idle or contains markup)
	if !t.parser.IsIdle() || t.parser.tokens.ContainsAny(text) {
		logging.InfoMsg("[%s] Tool call markup detected in reasoning content, transforming to tool_calls format", t.messageID)
		events := t.parser.Parse(text)
		for _, e := range events {
			if err := t.writeEvent(e); err != nil {
				return err
			}
		}
		return nil
	}

	if t.inReasoning {
		return t.write(t.formatter.FormatReasoning(text))
	}
	return t.write(t.formatter.FormatContent(text))
}

func (t *OpenAITransformer) writeEvent(e Event) error {
	switch e.Type {
	case EventContent:
		return t.write(t.formatter.FormatContent(e.Text))
	case EventToolStart:
		logging.InfoMsg("[%s] Tool call parsed: name=%s, id=%s, index=%d", t.messageID, e.Name, e.ID, e.Index)
		return t.write(t.formatter.FormatToolStart(e.ID, e.Name, e.Index))
	case EventToolArgs:
		return t.write(t.formatter.FormatToolArgs(e.Args, e.Index))
	case EventToolEnd:
		return t.write(t.formatter.FormatToolEnd(e.Index))
	case EventSectionEnd:
		return t.write(t.formatter.FormatSectionEnd())
	}
	return nil
}

func (t *OpenAITransformer) write(data []byte) error {
	if len(data) == 0 {
		return nil
	}
	_, err := t.sseWriter.WriteRaw(data)
	return err
}

func (t *OpenAITransformer) writeData(data []byte) error {
	return t.sseWriter.WriteData(data)
}

func (t *OpenAITransformer) Flush() error {
	// Flush Kimi parser
	for {
		events := t.parser.Parse("")
		if len(events) == 0 {
			break
		}
		for _, e := range events {
			if err := t.writeEvent(e); err != nil {
				return err
			}
		}
	}
	// Flush GLM-5 parser
	for {
		events := t.glm5Parser.Parse("")
		if len(events) == 0 {
			return nil
		}
		for _, e := range events {
			if err := t.writeEvent(e); err != nil {
				return err
			}
		}
	}
}

func (t *OpenAITransformer) Close() error {
	return t.Flush()
}

// Initialize is called before the upstream request to perform any setup.
// For OpenAITransformer, this is a no-op as initialization happens on first event.
func (t *OpenAITransformer) Initialize() error {
	return nil
}

// HandleCancel handles cancellation requests.
// For OpenAITransformer, this is a no-op.
func (t *OpenAITransformer) HandleCancel() error {
	return nil
}
