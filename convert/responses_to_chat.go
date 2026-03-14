// Package convert provides converters between different API formats.
// This file implements OpenAI Responses API to OpenAI Chat Completions conversion.
package convert

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"

	"ai-proxy/types"

	"github.com/tmaxmax/go-sse"
)

// ResponsesToChatConverter converts OpenAI ResponsesRequest to ChatCompletionRequest.
type ResponsesToChatConverter struct{}

// NewResponsesToChatConverter creates a new converter for Responses to Chat format.
func NewResponsesToChatConverter() *ResponsesToChatConverter {
	return &ResponsesToChatConverter{}
}

// Convert transforms a ResponsesRequest body to ChatCompletionRequest format.
func (c *ResponsesToChatConverter) Convert(body []byte) ([]byte, error) {
	var req types.ResponsesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, fmt.Errorf("failed to parse ResponsesRequest: %w", err)
	}

	chatReq := c.convertRequest(&req)
	return json.Marshal(chatReq)
}

// convertRequest transforms a ResponsesRequest to ChatCompletionRequest.
func (c *ResponsesToChatConverter) convertRequest(req *types.ResponsesRequest) *types.ChatCompletionRequest {
	chatReq := &types.ChatCompletionRequest{
		Model:       req.Model,
		MaxTokens:   req.MaxOutputTokens,
		Stream:      req.Stream,
		Temperature: req.Temperature,
		TopP:        req.TopP,
	}

	// Convert instructions to system message
	if req.Instructions != "" {
		chatReq.System = req.Instructions
	}

	// Convert input to messages
	chatReq.Messages = c.convertInput(req.Input)

	// Convert tools
	chatReq.Tools = c.convertTools(req.Tools)

	return chatReq
}

// convertInput transforms Responses API input to Chat Completions messages.
// Input can be:
// - string: a simple user message
// - []InputItem: an array of input items
func (c *ResponsesToChatConverter) convertInput(input interface{}) []types.Message {
	if input == nil {
		return []types.Message{}
	}

	switch v := input.(type) {
	case string:
		if v == "" {
			return []types.Message{}
		}
		return []types.Message{
			{Role: "user", Content: v},
		}

	case []interface{}:
		return c.convertInputItems(v)

	default:
		// Try to marshal and unmarshal as InputItem array
		data, err := json.Marshal(input)
		if err != nil {
			return []types.Message{}
		}
		var items []types.InputItem
		if err := json.Unmarshal(data, &items); err != nil {
			return []types.Message{}
		}
		return c.convertInputItemsFromTyped(items)
	}
}

// convertInputItems converts an array of raw input items to messages.
func (c *ResponsesToChatConverter) convertInputItems(items []interface{}) []types.Message {
	messages := make([]types.Message, 0, len(items))

	for _, item := range items {
		itemMap, ok := item.(map[string]interface{})
		if !ok {
			continue
		}

		itemType, _ := itemMap["type"].(string)
		switch itemType {
		case "message":
			msg := c.convertInputItemToMessage(itemMap)
			if msg != nil {
				messages = append(messages, *msg)
			}
		}
	}

	return messages
}

// convertInputItemsFromTyped converts typed InputItem array to messages.
func (c *ResponsesToChatConverter) convertInputItemsFromTyped(items []types.InputItem) []types.Message {
	messages := make([]types.Message, 0, len(items))

	for _, item := range items {
		if item.Type == "message" {
			msg := types.Message{
				Role: item.Role,
			}

			// Convert content
			switch content := item.Content.(type) {
			case string:
				msg.Content = content
			case []interface{}:
				msg.Content = c.convertContentParts(content)
			default:
				msg.Content = item.Content
			}

			messages = append(messages, msg)
		}
	}

	return messages
}

// convertInputItemToMessage converts a single input item map to a Message.
func (c *ResponsesToChatConverter) convertInputItemToMessage(itemMap map[string]interface{}) *types.Message {
	role, _ := itemMap["role"].(string)
	if role == "" {
		role = "user"
	}

	msg := &types.Message{
		Role: role,
	}

	// Handle content
	if content, ok := itemMap["content"]; ok {
		switch v := content.(type) {
		case string:
			msg.Content = v
		case []interface{}:
			msg.Content = c.convertContentParts(v)
		default:
			msg.Content = content
		}
	}

	return msg
}

// convertContentParts converts Responses API content parts to Chat format.
func (c *ResponsesToChatConverter) convertContentParts(parts []interface{}) []interface{} {
	result := make([]interface{}, 0, len(parts))

	for _, part := range parts {
		partMap, ok := part.(map[string]interface{})
		if !ok {
			continue
		}

		partType, _ := partMap["type"].(string)
		switch partType {
		case "input_text":
			// Convert input_text to text
			if text, ok := partMap["text"].(string); ok {
				result = append(result, map[string]interface{}{
					"type": "text",
					"text": text,
				})
			}
		case "input_image":
			// Convert input_image to image_url
			if imageURL, ok := partMap["image_url"].(string); ok {
				result = append(result, map[string]interface{}{
					"type": "image_url",
					"image_url": map[string]interface{}{
						"url": imageURL,
					},
				})
			}
		default:
			// Pass through other content types
			result = append(result, part)
		}
	}

	return result
}

// convertTools transforms Responses API tools to Chat Completions tools.
func (c *ResponsesToChatConverter) convertTools(tools []types.ResponsesTool) []types.Tool {
	if len(tools) == 0 {
		return nil
	}

	result := make([]types.Tool, 0, len(tools))
	for _, tool := range tools {
		chatTool := c.convertTool(&tool)
		if chatTool != nil {
			result = append(result, *chatTool)
		}
	}

	if len(result) == 0 {
		return nil
	}
	return result
}

// convertTool transforms a single ResponsesTool to Tool.
func (c *ResponsesToChatConverter) convertTool(tool *types.ResponsesTool) *types.Tool {
	if tool.Type != "function" {
		// Only function tools are supported in Chat Completions
		return nil
	}

	// Handle both flat and nested tool formats
	name := tool.Name
	description := tool.Description
	parameters := tool.Parameters

	// If nested function format, use those values
	if tool.Function != nil {
		name = tool.Function.Name
		description = tool.Function.Description
		parameters = tool.Function.Parameters
	}

	if name == "" {
		return nil
	}

	return &types.Tool{
		Type: "function",
		Function: types.ToolFunction{
			Name:        name,
			Description: description,
			Parameters:  parameters,
		},
	}
}

// ResponsesToChatTransformer converts OpenAI Responses SSE to Chat Completions format.
type ResponsesToChatTransformer struct {
	w io.Writer

	// Response state
	responseID string
	model      string
	created    int64

	// Content tracking
	contentIndex    int
	toolCallIndex   int
	currentToolCall *responsesToolCallState

	// Content builders
	contentBuilder strings.Builder

	// Finish reason
	finishReason string
}

type responsesToolCallState struct {
	id        string
	name      string
	arguments strings.Builder
}

// NewResponsesToChatTransformer creates a new transformer for Responses to Chat format.
func NewResponsesToChatTransformer(w io.Writer) *ResponsesToChatTransformer {
	return &ResponsesToChatTransformer{
		w:       w,
		created: time.Now().Unix(),
	}
}

// Transform processes a Responses API SSE event and converts it to Chat Completions format.
func (t *ResponsesToChatTransformer) Transform(event *sse.Event) error {
	if event.Data == "" {
		return nil
	}

	if event.Data == "[DONE]" {
		return t.writeDone()
	}

	var respEvent types.ResponsesStreamEvent
	if err := json.Unmarshal([]byte(event.Data), &respEvent); err != nil {
		// Pass through unparseable events
		return t.writeData([]byte(event.Data))
	}

	return t.handleEvent(&respEvent)
}

// handleEvent processes a single Responses API event.
func (t *ResponsesToChatTransformer) handleEvent(event *types.ResponsesStreamEvent) error {
	switch event.Type {
	case "response.created":
		return t.handleResponseCreated(event)
	case "response.in_progress":
		return nil // No action needed
	case "response.output_item.added":
		return t.handleOutputItemAdded(event)
	case "response.content_part.added":
		return nil // No action needed
	case "response.output_text.delta":
		return t.handleOutputTextDelta(event)
	case "response.function_call_arguments.delta":
		return t.handleFunctionCallArgsDelta(event)
	case "response.content_part.done":
		return nil // No action needed
	case "response.output_item.done":
		return t.handleOutputItemDone(event)
	case "response.completed":
		return t.handleResponseCompleted(event)
	case "error":
		return t.handleError(event)
	default:
		// Pass through unknown events
		return nil
	}
}

// handleResponseCreated handles response.created event.
func (t *ResponsesToChatTransformer) handleResponseCreated(event *types.ResponsesStreamEvent) error {
	if event.Response != nil {
		t.responseID = event.Response.ID
		t.model = event.Response.Model
	}
	return nil
}

// handleOutputItemAdded handles response.output_item.added event.
func (t *ResponsesToChatTransformer) handleOutputItemAdded(event *types.ResponsesStreamEvent) error {
	if event.OutputItem == nil {
		return nil
	}

	// Handle function_call output items
	if event.OutputItem.Type == "function_call" {
		t.currentToolCall = &responsesToolCallState{
			id:   event.OutputItem.ID,
			name: event.OutputItem.CallID,
		}
		t.toolCallIndex++
	}

	return nil
}

// handleOutputTextDelta handles response.output_text.delta event.
func (t *ResponsesToChatTransformer) handleOutputTextDelta(event *types.ResponsesStreamEvent) error {
	if event.Delta == "" {
		return nil
	}

	t.contentBuilder.WriteString(event.Delta)

	chunk := t.createChunk()
	chunk.Choices = []types.Choice{
		{
			Index: t.contentIndex,
			Delta: types.Delta{
				Content: event.Delta,
			},
		},
	}

	return t.writeChunk(chunk)
}

// handleFunctionCallArgsDelta handles response.function_call_arguments.delta event.
func (t *ResponsesToChatTransformer) handleFunctionCallArgsDelta(event *types.ResponsesStreamEvent) error {
	if event.Delta == "" {
		return nil
	}

	if t.currentToolCall == nil {
		// Create tool call state if not exists
		t.currentToolCall = &responsesToolCallState{
			id: event.ItemID,
		}
		t.toolCallIndex++
	}

	t.currentToolCall.arguments.WriteString(event.Delta)

	// Send incremental tool call chunk
	chunk := t.createChunk()
	chunk.Choices = []types.Choice{
		{
			Index: 0,
			Delta: types.Delta{
				ToolCalls: []types.ToolCall{
					{
						ID:    t.currentToolCall.id,
						Type:  "function",
						Index: t.toolCallIndex - 1,
						Function: types.Function{
							Arguments: event.Delta,
						},
					},
				},
			},
		},
	}

	return t.writeChunk(chunk)
}

// handleOutputItemDone handles response.output_item.done event.
func (t *ResponsesToChatTransformer) handleOutputItemDone(event *types.ResponsesStreamEvent) error {
	if event.OutputItem == nil {
		return nil
	}

	// For function_call items, send the complete tool call
	if event.OutputItem.Type == "function_call" && t.currentToolCall != nil {
		// The tool call has been streamed incrementally, just clear state
		t.currentToolCall = nil
	}

	return nil
}

// handleResponseCompleted handles response.completed event.
func (t *ResponsesToChatTransformer) handleResponseCompleted(event *types.ResponsesStreamEvent) error {
	// Determine finish reason
	if event.Response != nil {
		if len(event.Response.Output) > 0 {
			for _, item := range event.Response.Output {
				if item.Type == "function_call" {
					t.finishReason = "tool_calls"
					break
				}
			}
		}
	}
	if t.finishReason == "" {
		t.finishReason = "stop"
	}

	// Send final chunk with finish_reason
	chunk := t.createChunk()
	chunk.Choices = []types.Choice{
		{
			Index:         t.contentIndex,
			Delta:         types.Delta{},
			FinishReason:  &t.finishReason,
		},
	}

	// Add usage if available
	if event.Response != nil && event.Response.Usage != nil {
		chunk.Usage = &types.Usage{
			PromptTokens:     event.Response.Usage.InputTokens,
			CompletionTokens: event.Response.Usage.OutputTokens,
			TotalTokens:      event.Response.Usage.TotalTokens,
		}
	}

	return t.writeChunk(chunk)
}

// handleError handles error events.
func (t *ResponsesToChatTransformer) handleError(event *types.ResponsesStreamEvent) error {
	if event.Error == nil {
		return nil
	}

	errResp := types.ErrorResponse{
		Error: types.ErrorDetail{
			Type:    "api_error",
			Message: event.Error.Message,
		},
	}
	if event.Error.Code != "" {
		errResp.Error.Code = event.Error.Code
	}

	data, err := json.Marshal(errResp)
	if err != nil {
		return err
	}

	return t.writeData(data)
}

// createChunk creates a base chunk with common fields.
func (t *ResponsesToChatTransformer) createChunk() *types.Chunk {
	return &types.Chunk{
		ID:      t.responseID,
		Object:  "chat.completion.chunk",
		Created: t.created,
		Model:   t.model,
	}
}

// writeChunk writes a chunk as SSE data.
func (t *ResponsesToChatTransformer) writeChunk(chunk *types.Chunk) error {
	data, err := json.Marshal(chunk)
	if err != nil {
		return err
	}
	return t.writeData(data)
}

// writeData writes raw data as SSE event.
func (t *ResponsesToChatTransformer) writeData(data []byte) error {
	if len(data) == 0 {
		return nil
	}
	_, err := t.w.Write([]byte("data: "))
	if err != nil {
		return err
	}
	_, err = t.w.Write(data)
	if err != nil {
		return err
	}
	_, err = t.w.Write([]byte("\n\n"))
	return err
}

// writeDone writes the [DONE] marker.
func (t *ResponsesToChatTransformer) writeDone() error {
	_, err := t.w.Write([]byte("data: [DONE]\n\n"))
	return err
}

// Flush writes any buffered data.
func (t *ResponsesToChatTransformer) Flush() error {
	return nil
}

// Close flushes and releases resources.
func (t *ResponsesToChatTransformer) Close() error {
	return t.Flush()
}