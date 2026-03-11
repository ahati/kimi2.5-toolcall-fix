// Package protocols provides protocol adapters for different API formats.
// Each adapter implements the ProtocolAdapter interface to handle request/response
// transformation for a specific API format (OpenAI, Anthropic, or Bridge).
package protocols

import (
	"encoding/json"
	"io"
	"net/http"

	"ai-proxy/config"
	"ai-proxy/transform/toolcall"
	"ai-proxy/types"

	"github.com/gin-gonic/gin"
	"github.com/tmaxmax/go-sse"
)

// SSETransformer defines the interface for SSE event transformation.
// Implementations transform tool call tokens in SSE streams.
//
// @brief Interface for SSE stream transformation.
type SSETransformer interface {
	Transform(event *sse.Event)
	Flush()
	Close()
}

// ProtocolAdapter defines the interface for handling different API formats.
// Implementations transform requests and responses between client-facing and
// upstream API formats.
//
// @brief Interface for protocol-specific request/response handling.
//
// Implementations:
//   - OpenAIAdapter: Pass-through for OpenAI format
//   - AnthropicAdapter: Pass-through for Anthropic format
//   - BridgeAdapter: Transforms Anthropic requests to OpenAI upstream
type ProtocolAdapter interface {
	TransformRequest(body []byte) ([]byte, error)
	ValidateRequest(body []byte) error
	CreateTransformer(w io.Writer, base types.StreamChunk) SSETransformer
	UpstreamURL(cfg *config.Config) string
	UpstreamAPIKey(cfg *config.Config) string
	ForwardHeaders(src, dst http.Header)
	SendError(c *gin.Context, status int, msg string)
	IsStreamingRequest(body []byte) bool
}

// ToolCallTransformer transforms tool call tokens in SSE streams.
// It uses a parser to extract tool calls from text content and an output
// handler to emit them in the appropriate format.
//
// @brief Transformer for tool call content in SSE streams.
//
// @invariant parser != nil when transformer is initialized
// @invariant output != nil when transformer is initialized
// @invariant writer != nil when transformer is initialized
type ToolCallTransformer struct {
	parser  *toolcall.Parser
	output  toolcall.EventHandler
	writer  io.Writer
	base    types.StreamChunk
	flusher interface{ Flush() }
}

// NewToolCallTransformer creates a new tool call transformer.
//
// @brief    Creates a new ToolCallTransformer instance.
// @param    writer Destination for transformed SSE events.
// @param    base Base stream chunk for context.
// @param    output Handler for tool call events.
// @return   Pointer to newly created ToolCallTransformer.
//
// @note     Uses DefaultTokenSet for parsing tool call markers.
// @note     If output implements Flush(), it will be called on Flush().
func NewToolCallTransformer(writer io.Writer, base types.StreamChunk, output toolcall.EventHandler) *ToolCallTransformer {
	// DefaultTokenSet provides the standard markers for tool call detection
	tokens := toolcall.DefaultTokenSet()
	t := &ToolCallTransformer{
		output: output,
		writer: writer,
		base:   base,
	}
	// Parser feeds extracted tool calls to the output handler
	t.parser = toolcall.NewParser(tokens, output)
	// Store flusher for later use if output handler supports flushing
	if f, ok := output.(interface{ Flush() }); ok {
		t.flusher = f
	}
	return t
}

// Transform processes an SSE event and extracts tool calls.
//
// @brief    Processes SSE event for tool call extraction.
// @param    event The SSE event to transform.
//
// @note     Skips empty data or "[DONE]" markers (passes [DONE] through).
// @note     Extracts text from Content, Reasoning, or ReasoningContent fields.
// @note     Forwards tool_calls directly from delta to output handler.
func (t *ToolCallTransformer) Transform(event *sse.Event) {
	// Skip empty data lines or the final [DONE] marker
	// The [DONE] marker signals end of stream and needs to be forwarded to client
	if event.Data == "" || event.Data == "[DONE]" {
		if event.Data == "[DONE]" {
			// Forward the [DONE] marker to signal stream completion to client
			t.writer.Write([]byte("data: [DONE]\n\n"))
		}
		return
	}

	var chunk types.StreamChunk
	// Silently skip malformed JSON chunks - the stream may have partial/corrupted data
	if err := json.Unmarshal([]byte(event.Data), &chunk); err != nil {
		return
	}

	// Update base chunk for context (used by output formatters for ID, model, etc.)
	t.base = chunk

	if len(chunk.Choices) > 0 {
		delta := chunk.Choices[0].Delta
		// Extract text content, preferring Content over reasoning fields
		// Reasoning and ReasoningContent are alternate fields used by some LLM providers
		text := delta.Content
		if text == "" {
			// Fallback to Reasoning field (used by some providers for chain-of-thought)
			text = delta.Reasoning
		}
		if text == "" {
			// Second fallback to ReasoningContent (another reasoning field variant)
			text = delta.ReasoningContent
		}
		if text != "" {
			// Feed text to parser which extracts tool calls from special markers
			t.parser.Feed(text)
		}
		// Handle tool calls that are already structured in the delta (not in text)
		// Some upstream APIs send tool calls directly instead of as text markers
		for _, tc := range delta.ToolCalls {
			// Emit tool call start event with ID, name, and index for tracking
			t.output.OnToolCallStart(tc.ID, tc.Function.Name, tc.Index)
			if tc.Function.Arguments != "" {
				// Emit arguments incrementally as they arrive in the stream
				t.output.OnToolCallArgs(tc.Function.Arguments, tc.Index)
			}
			// Signal end of this tool call delta chunk (not necessarily the full call)
			t.output.OnToolCallEnd(tc.Index)
		}
	}
}

// Flush flushes any pending content in the parser and output.
//
// @brief    Flushes parser and output handler buffers.
//
// @note     Safe to call even if output does not implement Flush().
func (t *ToolCallTransformer) Flush() {
	// Flush parser to process any remaining buffered text (e.g., incomplete tool call markers)
	t.parser.Flush()
	// Flush output handler if it implements Flush() (writes pending SSE events)
	if t.flusher != nil {
		t.flusher.Flush()
	}
}

// Close closes the transformer and flushes any remaining content.
//
// @brief    Closes the transformer and flushes buffers.
//
// @note     Equivalent to calling Flush().
func (t *ToolCallTransformer) Close() {
	t.Flush()
}
