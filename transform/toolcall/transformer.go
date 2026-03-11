package toolcall

import (
	"encoding/json"
	"io"

	"ai-proxy/types"

	"github.com/tmaxmax/go-sse"
)

// Transformer combines a Parser and OutputFormatter to transform tool call markup
// into API-specific streaming output. It implements SSETransformer.
//
// @brief SSE transformer that converts proprietary tool call markup to API-specific formats.
//
// @note The transformer processes SSE events from upstream LLM APIs and extracts
//
//	tool call information from the "reasoning" or "reasoning_content" fields.
//	This is specific to models that embed tool calls in reasoning content.
//
// @note The transformer is NOT thread-safe. Create a new instance for each request.
// @note The transformer maintains internal state and must be properly closed to
//
//	release resources and flush remaining content.
//
// @pre Must be initialized via NewOpenAITransformer or NewAnthropicTransformer.
// @post After Close(), the transformer must not be used for further transformations.
//
// Data Flow:
// 1. SSE event received from upstream
// 2. Event data parsed as JSON chunk
// 3. Reasoning content extracted from chunk
// 4. Parser extracts tool call events from reasoning content
// 5. Formatter converts events to API-specific format
// 6. Formatted output written to destination
type Transformer struct {
	// parser extracts tool call events from reasoning content.
	// Initialized with DefaultTokens for Kimi model compatibility.
	parser *Parser

	// formatter converts parsed events to API-specific format.
	// Either OpenAIFormatter or AnthropicFormatter depending on constructor used.
	formatter OutputFormatter

	// output is the destination for formatted SSE data.
	// Must be a valid io.Writer that accepts concurrent writes.
	output io.Writer

	// messageID is the unique identifier for the response.
	// Extracted from the first upstream chunk and propagated to output.
	messageID string

	// model is the model name for the response.
	// Extracted from the first upstream chunk and propagated to output.
	model string

	// buf is an internal buffer for accumulating output.
	// Currently unused but reserved for future optimization.
	buf []byte
}

// NewOpenAITransformer creates a transformer that outputs OpenAI streaming format.
//
// @brief Creates a new Transformer configured for OpenAI streaming output.
//
// @param output The destination writer for formatted output.
//
//	Must be non-nil and writable.
//	Should support flushing for streaming responses.
//
// @param messageID The unique identifier for the chat completion.
//
//	May be empty (will be extracted from upstream response).
//	Format: typically "chatcmpl-xxx" or similar.
//
// @param model The model name for the response.
//
//	May be empty (will be extracted from upstream response).
//	Examples: "gpt-4", "moonshotai/Kimi-K2.5-TEE".
//
// @return *Transformer A new transformer ready for Transform calls.
//
// @pre output must be non-nil and writable.
// @post Transformer is ready to process SSE events.
// @post Parser is initialized with DefaultTokens.
// @post Formatter is OpenAIFormatter.
//
// @note Use this constructor for OpenAI-compatible clients.
func NewOpenAITransformer(output io.Writer, messageID, model string) *Transformer {
	return &Transformer{
		parser:    NewParser(DefaultTokens),
		formatter: NewOpenAIFormatter(messageID, model),
		output:    output,
		messageID: messageID,
		model:     model,
	}
}

// NewAnthropicTransformer creates a transformer that outputs Anthropic streaming format.
//
// @brief Creates a new Transformer configured for Anthropic streaming output.
//
// @param output The destination writer for formatted output.
//
//	Must be non-nil and writable.
//	Should support flushing for streaming responses.
//
// @param messageID The unique identifier for the message.
//
//	May be empty (Anthropic format may not require it).
//	Format: typically "msg_xxx" or similar.
//
// @param model The model name for the response.
//
//	May be empty (will be extracted from upstream response).
//	Examples: "claude-3-opus-20240229", "kimi-k2.5".
//
// @return *Transformer A new transformer ready for Transform calls.
//
// @pre output must be non-nil and writable.
// @post Transformer is ready to process SSE events.
// @post Parser is initialized with DefaultTokens.
// @post Formatter is AnthropicFormatter.
//
// @note Use this constructor for Anthropic-compatible clients.
func NewAnthropicTransformer(output io.Writer, messageID, model string) *Transformer {
	return &Transformer{
		parser:    NewParser(DefaultTokens),
		formatter: NewAnthropicFormatter(messageID, model),
		output:    output,
		messageID: messageID,
		model:     model,
	}
}

// Transform processes an SSE event and writes formatted output.
// It extracts tool calls from the reasoning content and reformats them.
//
// @brief Processes a single SSE event and writes transformed output.
//
// @param event The SSE event to process.
//
//	Must not be nil.
//	Data field should contain JSON chunk or "[DONE]" marker.
//
// @return error Returns nil on success.
//
//	Returns error if:
//	- Event data cannot be parsed as JSON (treated as raw content)
//	- Output write fails
//	- Parser encounters an error
//
// @pre event must not be nil.
// @pre output writer must be writable.
// @post Event is processed and output written (if applicable).
// @post Parser state is updated for subsequent events.
// @post messageID and model are extracted from first chunk (if empty).
//
// Processing Logic:
// 1. Skip empty events and "[DONE]" markers
// 2. Parse event data as JSON chunk
// 3. If parsing fails, treat as raw content
// 4. Extract messageID and model from first chunk
// 5. Check for content in delta
// 6. Extract reasoning content (reasoning or reasoning_content field)
// 7. Parse reasoning content for tool call markers
// 8. Format and write parsed events
//
// @note The transformer handles both regular content and tool call markup.
//
//	Regular content is passed through with formatting.
//	Tool call markup is extracted and reformatted.
func (t *Transformer) Transform(event *sse.Event) error {
	// Skip empty events and stream termination markers.
	// "[DONE]" is the standard SSE termination signal.
	if event.Data == "" || event.Data == "[DONE]" {
		return nil
	}

	// Attempt to parse event data as JSON chunk.
	// If parsing fails, treat the entire data as raw content.
	var chunk types.Chunk
	if err := json.Unmarshal([]byte(event.Data), &chunk); err != nil {
		// JSON parsing failed - treat as raw content string.
		// This handles malformed or non-JSON upstream responses.
		return t.write(t.formatter.FormatContent(string(event.Data)))
	}

	// Extract messageID and model from first chunk if not already set.
	// These are propagated to the output formatter for consistency.
	if t.messageID == "" && chunk.ID != "" {
		t.messageID = chunk.ID
		t.model = chunk.Model
		t.setMessageID(chunk.ID, chunk.Model)
	}

	// Skip chunks without choices (should not happen in normal operation).
	if len(chunk.Choices) == 0 {
		return nil
	}

	// Extract the delta from the first choice.
	// OpenAI format supports multiple choices but we use only the first.
	delta := chunk.Choices[0].Delta

	// Handle regular content in the delta.
	// Content is passed through directly with formatting.
	if delta.Content != "" {
		return t.write(t.formatter.FormatContent(delta.Content))
	}

	// Extract reasoning content where tool calls are embedded.
	// Try both field names for compatibility with different models.
	text := delta.Reasoning
	if text == "" {
		text = delta.ReasoningContent
	}

	// Skip if no reasoning content to process.
	if text == "" {
		return nil
	}

	// Parse the reasoning content for tool call markers.
	// The parser extracts tool call events from the markup.
	events := t.parser.Parse(text)

	// Write each parsed event to the output.
	for _, e := range events {
		if err := t.writeEvent(e); err != nil {
			return err
		}
	}

	return nil
}

// writeEvent formats and writes a single parsed event to the output.
//
// @brief Internal method to format and write a parser event.
//
// @param e The event to write.
//
//	Must have a valid Type field.
//
// @return error Returns nil on success.
//
//	Returns error if output write fails.
//
// @pre e.Type must be a valid EventType.
// @post Event is formatted and written to output.
func (t *Transformer) writeEvent(e Event) error {
	switch e.Type {
	case EventContent:
		return t.write(t.formatter.FormatContent(e.Text))
	case EventToolStart:
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

// write writes data to the output writer.
//
// @brief Internal method to write formatted data to output.
//
// @param data The data to write.
//
//	May be nil or empty (no-op).
//
// @return error Returns nil on success or if data is empty.
//
//	Returns error if output write fails.
//
// @pre output writer must be writable.
// @post Data is written to output (if non-empty).
func (t *Transformer) write(data []byte) error {
	// Skip nil or empty data to avoid unnecessary writes.
	if len(data) == 0 {
		return nil
	}
	_, err := t.output.Write(data)
	return err
}

// setMessageID updates the message ID in the formatter.
// This method uses type assertion to call the appropriate setter.
//
// @brief Internal method to propagate message ID to the formatter.
//
// @param id The message ID to set.
//
//	Must be valid UTF-8 for JSON encoding.
//
// @param model The model name to set.
//
//	Must be valid UTF-8 for JSON encoding.
//
// @return None (method returns no value).
//
// @pre Formatter must be OpenAIFormatter or AnthropicFormatter.
// @post Formatter's message ID and model are updated.
//
// @note Uses type assertion to handle different formatter types.
//
//	This is a design trade-off for flexibility.
func (t *Transformer) setMessageID(id, model string) {
	switch f := t.formatter.(type) {
	case *OpenAIFormatter:
		f.SetMessageID(id)
		f.SetModel(model)
	case *AnthropicFormatter:
		f.SetMessageID(id)
		f.SetModel(model)
	}
}

// Flush processes any remaining buffered content in the parser.
// This ensures all pending data is written to the output.
//
// @brief Flushes all buffered content from the parser to output.
//
// @return error Returns nil on success.
//
//	Returns error if:
//	- Parser produces events that fail to write
//	- Output write fails
//
// @pre Transform() should have been called for all events.
// @post All buffered content is written to output.
// @post Parser buffer is empty.
//
// @note Must be called after the last Transform() call to ensure
//
//	all pending content is written. Not calling Flush() may
//	result in truncated output at the client.
//
// @note Flush() is idempotent - calling multiple times has no effect
//
//	after the first call (parser buffer is empty).
func (t *Transformer) Flush() error {
	for {
		// Parse with empty string to flush remaining buffer.
		// The parser will emit any complete events from buffered data.
		events := t.parser.Parse("")
		if len(events) == 0 {
			// No more events - buffer is empty.
			return nil
		}
		// Write all flushed events to output.
		for _, e := range events {
			if err := t.writeEvent(e); err != nil {
				return err
			}
		}
	}
}

// Close flushes remaining content and releases resources.
// After Close(), the transformer must not be used.
//
// @brief Releases all resources held by the transformer.
//
// @return error Returns nil on success.
//
//	Returns error if Flush() fails.
//
// @pre None (safe to call even if never used).
// @post Transformer is in a closed state and must not be used.
// @post All buffered content is flushed.
//
// @note Close() calls Flush() internally before releasing resources.
//
//	After Close(), subsequent calls to Transform(), Flush(), or Close()
//	may panic or return errors.
func (t *Transformer) Close() error {
	if err := t.Flush(); err != nil {
		return err
	}
	return nil
}

// Parser returns the internal parser for testing and inspection.
//
// @brief Returns the internal Parser instance.
//
// @return *Parser The parser used by this transformer.
//
// @pre None.
// @post No state is modified.
//
// @note This method is primarily for testing and debugging.
//
//	Production code should not need to access the parser directly.
func (t *Transformer) Parser() *Parser {
	return t.parser
}

// Formatter returns the output formatter for testing and inspection.
//
// @brief Returns the internal OutputFormatter instance.
//
// @return OutputFormatter The formatter used by this transformer.
//
// @pre None.
// @post No state is modified.
//
// @note This method is primarily for testing and debugging.
//
//	Production code should not need to access the formatter directly.
func (t *Transformer) Formatter() OutputFormatter {
	return t.formatter
}
