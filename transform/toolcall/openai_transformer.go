package toolcall

import (
	"encoding/json"
	"io"
	"strings"

	"ai-proxy/logging"
	"ai-proxy/types"

	"github.com/tmaxmax/go-sse"
)

type OpenAITransformer struct {
	output       io.Writer
	formatter    *OpenAIFormatter
	parser       *Parser
	messageID    string
	model        string
	isKimiModel  bool
}

func NewOpenAITransformer(output io.Writer, model string) *OpenAITransformer {
	return &OpenAITransformer{
		output:      output,
		formatter:   NewOpenAIFormatter("", ""),
		parser:      NewParser(DefaultTokens),
		model:       model,
		isKimiModel: isKimiModel(model),
	}
}

// isKimiModel checks if the model name indicates it's a Kimi K2.5 model
func isKimiModel(model string) bool {
	lower := strings.ToLower(model)
	return strings.Contains(lower, "kimi") && strings.Contains(lower, "k2.5")
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
		return t.writeData([]byte(rawData))
	}

	choice := chunk.Choices[0]
	delta := choice.Delta

	if delta.Content != "" {
		return t.write(t.formatter.FormatContent(delta.Content))
	}

	text := delta.Reasoning
	if text == "" {
		text = delta.ReasoningContent
	}

	if text == "" {
		if choice.FinishReason != nil && *choice.FinishReason != "" {
			return t.writeData([]byte(rawData))
		}
		return nil
	}

	return t.processText(text)
}

func (t *OpenAITransformer) processText(text string) error {
	// Only parse tool calls for Kimi K2.5 models
	if !t.isKimiModel {
		return t.write(t.formatter.FormatContent(text))
	}

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
	_, err := t.output.Write(data)
	return err
}

func (t *OpenAITransformer) writeData(data []byte) error {
	if len(data) == 0 {
		return nil
	}
	_, err := t.output.Write([]byte("data: "))
	if err != nil {
		return err
	}
	_, err = t.output.Write(data)
	if err != nil {
		return err
	}
	_, err = t.output.Write([]byte("\n\n"))
	return err
}

func (t *OpenAITransformer) Flush() error {
	for {
		events := t.parser.Parse("")
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
