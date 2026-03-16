// Package handlers provides HTTP request handlers for the AI proxy endpoints.
// This file provides support for non-streaming (synchronous) responses.
package handlers

import (
	"encoding/json"
	"io"
	"strings"

	"ai-proxy/transform"

	"github.com/gin-gonic/gin"
	"github.com/tmaxmax/go-sse"
)

// ResponseAccumulator collects streaming events and builds a complete response.
// This enables non-streaming mode by accumulating SSE chunks into a single response.
type ResponseAccumulator interface {
	// Accumulate processes a single SSE event.
	Accumulate(event *sse.Event) error

	// Build returns the accumulated response.
	Build() interface{}

	// Reset clears the accumulator for reuse.
	Reset()
}

// ChunkAccumulator accumulates OpenAI Chat Completion chunks.
type ChunkAccumulator struct {
	chunks []*sse.Event
	sse    *transform.SSEWriter
}

// NewChunkAccumulator creates a new accumulator for OpenAI chunks.
func NewChunkAccumulator() *ChunkAccumulator {
	return &ChunkAccumulator{
		chunks: make([]*sse.Event, 0),
	}
}

// Accumulate stores an SSE event for later processing.
func (a *ChunkAccumulator) Accumulate(event *sse.Event) error {
	if event.Data == "" || event.Data == "[DONE]" {
		return nil
	}
	a.chunks = append(a.chunks, event)
	return nil
}

// Build assembles accumulated chunks into a complete ChatCompletionResponse.
func (a *ChunkAccumulator) Build() interface{} {
	if len(a.chunks) == 0 {
		return nil
	}

	response := &ChatCompletionResponse{
		Object:  "chat.completion",
		Choices: []ChatCompletionChoice{},
	}

	var content string
	var toolCalls []ToolCallData
	var finishReason string

	for _, event := range a.chunks {
		var chunk map[string]interface{}
		if err := json.Unmarshal([]byte(event.Data), &chunk); err != nil {
			continue
		}

		// Extract ID, model, created from first chunk
		if response.ID == "" {
			if id, ok := chunk["id"].(string); ok {
				response.ID = id
			}
			if model, ok := chunk["model"].(string); ok {
				response.Model = model
			}
			if created, ok := chunk["created"].(float64); ok {
				response.Created = int64(created)
			}
		}

		// Extract usage
		if usage, ok := chunk["usage"].(map[string]interface{}); ok {
			response.Usage = usage
		}

		// Extract content from choices
		if choices, ok := chunk["choices"].([]interface{}); ok && len(choices) > 0 {
			if choice, ok := choices[0].(map[string]interface{}); ok {
				// Extract finish_reason
				if fr, ok := choice["finish_reason"].(string); ok && fr != "" {
					finishReason = fr
				}

				// Extract delta content
				if delta, ok := choice["delta"].(map[string]interface{}); ok {
					if c, ok := delta["content"].(string); ok {
						content += c
					}
					if tc, ok := delta["tool_calls"].([]interface{}); ok {
						accumulateToolCalls(&toolCalls, tc)
					}
				}
			}
		}
	}

	// Build final choice
	choice := ChatCompletionChoice{
		Index: 0,
		Message: ChatCompletionMessage{
			Role:    "assistant",
			Content: content,
		},
		FinishReason: finishReason,
	}

	if len(toolCalls) > 0 {
		choice.Message.ToolCalls = toolCalls
	}

	response.Choices = []ChatCompletionChoice{choice}

	return response
}

// Reset clears all accumulated data.
func (a *ChunkAccumulator) Reset() {
	a.chunks = a.chunks[:0]
}

// ToolCallData represents accumulated tool call data.
type ToolCallData struct {
	ID        string `json:"id"`
	Type      string `json:"type"`
	Function  struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// accumulateToolCalls merges tool call deltas into accumulated data.
func accumulateToolCalls(accumulated *[]ToolCallData, deltas []interface{}) {
	for _, delta := range deltas {
		tc, ok := delta.(map[string]interface{})
		if !ok {
			continue
		}

		// Get index
		index := 0
		if idx, ok := tc["index"].(float64); ok {
			index = int(idx)
		}

		// Ensure slice is large enough
		for len(*accumulated) <= index {
			*accumulated = append(*accumulated, ToolCallData{Type: "function"})
		}

		// Merge data
		if id, ok := tc["id"].(string); ok && id != "" {
			(*accumulated)[index].ID = id
		}
		if t, ok := tc["type"].(string); ok && t != "" {
			(*accumulated)[index].Type = t
		}
		if fn, ok := tc["function"].(map[string]interface{}); ok {
			if name, ok := fn["name"].(string); ok && name != "" {
				(*accumulated)[index].Function.Name = name
			}
			if args, ok := fn["arguments"].(string); ok {
				(*accumulated)[index].Function.Arguments += args
			}
		}
	}
}

// ChatCompletionResponse represents a complete (non-streaming) chat completion.
type ChatCompletionResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []ChatCompletionChoice `json:"choices"`
	Usage   interface{}            `json:"usage,omitempty"`
}

// ChatCompletionChoice represents a choice in a complete response.
type ChatCompletionChoice struct {
	Index        int                  `json:"index"`
	Message      ChatCompletionMessage `json:"message"`
	FinishReason string               `json:"finish_reason,omitempty"`
}

// ChatCompletionMessage represents a message in a complete response.
type ChatCompletionMessage struct {
	Role      string        `json:"role"`
	Content   string        `json:"content"`
	ToolCalls []ToolCallData `json:"tool_calls,omitempty"`
}

// AnthropicResponseAccumulator accumulates Anthropic streaming events.
type AnthropicResponseAccumulator struct {
	events []*sse.Event
}

// NewAnthropicResponseAccumulator creates a new accumulator for Anthropic events.
func NewAnthropicResponseAccumulator() *AnthropicResponseAccumulator {
	return &AnthropicResponseAccumulator{
		events: make([]*sse.Event, 0),
	}
}

// Accumulate stores an SSE event.
func (a *AnthropicResponseAccumulator) Accumulate(event *sse.Event) error {
	if event.Data == "" {
		return nil
	}
	a.events = append(a.events, event)
	return nil
}

// Build assembles accumulated events into a complete Anthropic Message response.
func (a *AnthropicResponseAccumulator) Build() interface{} {
	if len(a.events) == 0 {
		return nil
	}

	response := &AnthropicMessageResponse{
		Type:    "message",
		Role:    "assistant",
		Content: []map[string]interface{}{},
	}

	var currentContent strings.Builder
	var currentToolCalls []map[string]interface{}
	var currentToolCall *map[string]interface{}

	for _, event := range a.events {
		var data map[string]interface{}
		if err := json.Unmarshal([]byte(event.Data), &data); err != nil {
			continue
		}

		eventType, _ := data["type"].(string)

		switch eventType {
		case "message_start":
			if msg, ok := data["message"].(map[string]interface{}); ok {
				if id, ok := msg["id"].(string); ok {
					response.ID = id
				}
				if model, ok := msg["model"].(string); ok {
					response.Model = model
				}
				if usage, ok := msg["usage"].(map[string]interface{}); ok {
					response.Usage = usage
				}
			}

		case "content_block_start":
			if cb, ok := data["content_block"].(map[string]interface{}); ok {
				cbType, _ := cb["type"].(string)
				if cbType == "tool_use" {
					tc := map[string]interface{}{
						"type":       "tool_use",
						"id":         cb["id"],
						"name":       cb["name"],
						"input":      "",
						"_accumulating": true,
					}
					currentToolCall = &tc
				}
			}

		case "content_block_delta":
			if delta, ok := data["delta"].(map[string]interface{}); ok {
				deltaType, _ := delta["type"].(string)
				switch deltaType {
				case "text_delta":
					if text, ok := delta["text"].(string); ok {
						currentContent.WriteString(text)
					}
				case "thinking_delta":
					if thinking, ok := delta["thinking"].(string); ok {
						currentContent.WriteString(thinking)
					}
				case "input_json_delta":
					if currentToolCall != nil {
						if partial, ok := delta["partial_json"].(string); ok {
							if args, ok := (*currentToolCall)["input"].(string); ok {
								(*currentToolCall)["input"] = args + partial
							} else {
								(*currentToolCall)["input"] = partial
							}
						}
					}
				}
			}

		case "content_block_stop":
			if currentToolCall != nil {
				delete(*currentToolCall, "_accumulating")
				currentToolCalls = append(currentToolCalls, *currentToolCall)
				currentToolCall = nil
			}

		case "message_delta":
			if delta, ok := data["delta"].(map[string]interface{}); ok {
				if sr, ok := delta["stop_reason"].(string); ok {
					response.StopReason = sr
				}
			}
			if usage, ok := data["usage"].(map[string]interface{}); ok {
				response.Usage = usage
			}
		}
	}

	// Build content blocks
	if currentContent.Len() > 0 {
		response.Content = append(response.Content, map[string]interface{}{
			"type": "text",
			"text": currentContent.String(),
		})
	}

	for _, tc := range currentToolCalls {
		response.Content = append(response.Content, tc)
	}

	return response
}

// Reset clears all accumulated data.
func (a *AnthropicResponseAccumulator) Reset() {
	a.events = a.events[:0]
}

// AnthropicMessageResponse represents a complete Anthropic message.
type AnthropicMessageResponse struct {
	ID         string                   `json:"id"`
	Type       string                   `json:"type"`
	Role       string                   `json:"role"`
	Model      string                   `json:"model"`
	Content    []map[string]interface{} `json:"content"`
	StopReason string                   `json:"stop_reason,omitempty"`
	Usage      interface{}              `json:"usage,omitempty"`
}

// StreamingStrategy defines how responses are delivered to clients.
type StreamingStrategy interface {
	// IsStreaming returns true if streaming mode is enabled.
	IsStreaming() bool
	// WriteResponse writes the response to the client.
	WriteResponse(c *gin.Context, body io.Reader, h Handler) error
}

// SyncMode handles non-streaming responses by accumulating events.
type SyncMode struct {
	accumulatorFactory func() ResponseAccumulator
}

// NewSyncMode creates a sync mode handler.
func NewSyncMode(factory func() ResponseAccumulator) *SyncMode {
	return &SyncMode{accumulatorFactory: factory}
}

// IsStreaming returns false for sync mode.
func (s *SyncMode) IsStreaming() bool { return false }

// WriteResponse accumulates streaming events and returns a complete response.
func (s *SyncMode) WriteResponse(c *gin.Context, body io.Reader, h Handler) error {
	accumulator := s.accumulatorFactory()

	// Read all SSE events
	for ev, err := range sse.Read(body, nil) {
		if err != nil {
			break
		}
		if err := accumulator.Accumulate(&ev); err != nil {
			return err
		}
	}

	// Build and return complete response
	response := accumulator.Build()
	c.JSON(200, response)
	return nil
}

// StreamingMode handles streaming responses.
type StreamingMode struct{}

// NewStreamingMode creates a streaming mode handler.
func NewStreamingMode() *StreamingMode {
	return &StreamingMode{}
}

// IsStreaming returns true for streaming mode.
func (s *StreamingMode) IsStreaming() bool { return true }

// WriteResponse streams events directly to the client.
func (s *StreamingMode) WriteResponse(c *gin.Context, body io.Reader, h Handler) error {
	// This is handled by the existing streaming logic
	return nil
}

// DetectStreamingMode determines if streaming is requested.
func DetectStreamingMode(body []byte) bool {
	var req struct {
		Stream bool `json:"stream"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return true // Default to streaming for backward compatibility
	}
	return req.Stream
}