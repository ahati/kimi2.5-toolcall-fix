// Package convert provides converters between different API formats.
// This file implements OpenAI Responses API to OpenAI Chat Completions conversion.
package convert

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"

	"ai-proxy/conversation"
	"ai-proxy/logging"
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
// When previous_response_id is provided, it fetches the conversation history
// from the store and prepends it to the current input.
func (c *ResponsesToChatConverter) convertRequest(req *types.ResponsesRequest) *types.ChatCompletionRequest {
	// Fetch conversation history if previous_response_id is provided
	if req.PreviousResponseID != "" {
		if hist := conversation.GetFromDefault(req.PreviousResponseID); hist != nil {
			req.Input = prependHistoryToInput(hist, req.Input)
		} else {
			logging.InfoMsg("Warning: Previous response ID not found in conversation store: %s", req.PreviousResponseID)
		}
	}

	maxTokens := req.MaxOutputTokens
	if maxTokens == 0 {
		maxTokens = 65536 // Default max tokens (64k) for OpenAI-compatible APIs
	}

	chatReq := &types.ChatCompletionRequest{
		Model:       req.Model,
		MaxTokens:   maxTokens,
		Stream:      true,
		Temperature: req.Temperature,
		TopP:        req.TopP,
	}

	// Request usage statistics in the final streaming chunk.
	chatReq.StreamOptions = &types.StreamOptions{
		IncludeUsage: true,
	}

	// Convert parallel_tool_calls if set
	if req.ParallelToolCalls {
		chatReq.ParallelToolCalls = &req.ParallelToolCalls
	}

	// Convert input to messages
	chatReq.Messages = c.convertInput(req.Input)

	// Convert instructions to a prepended system message.
	if req.Instructions != "" {
		chatReq.Messages = append([]types.Message{{
			Role:    "system",
			Content: req.Instructions,
		}}, chatReq.Messages...)
	}

	// Convert tools
	chatReq.Tools = c.convertTools(req.Tools)

	// Convert tool_choice from Responses object form to Chat object form.
	chatReq.ToolChoice = c.convertToolChoice(req.ToolChoice)

	// Convert response_format
	chatReq.ResponseFormat = req.ResponseFormat

	// Forward reasoning effort to Chat Completions
	if req.Reasoning != nil && req.Reasoning.Effort != "" {
		chatReq.ReasoningEffort = req.Reasoning.Effort
	}

	// Convert metadata.user_id to user field
	if req.Metadata != nil {
		if userID, ok := req.Metadata["user_id"]; ok {
			if s, ok := userID.(string); ok && s != "" {
				chatReq.User = s
			}
		}
	}

	return chatReq
}

func (c *ResponsesToChatConverter) convertToolChoice(toolChoice interface{}) interface{} {
	if toolChoice == nil {
		return nil
	}

	raw, err := json.Marshal(toolChoice)
	if err != nil {
		return toolChoice
	}

	converted := ResponsesToolChoiceToChat(raw)
	if len(converted) == 0 {
		return toolChoice
	}

	var out interface{}
	if err := json.Unmarshal(converted, &out); err != nil {
		return toolChoice
	}

	return out
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

// inputItemGroup represents a logical grouping of input items
// that should become a single Chat Completions message.
// This handles the case where Codex sends function_call and message separately.
type inputItemGroup struct {
	itemType   string          // "message", "merged_assistant", "assistant_tool_calls", or "function_call_output"
	message    *types.Message  // Parsed message content (for message items)
	toolCalls  []types.ToolCall // Tool calls to merge (for function_call items)
	toolOutput *types.Message  // Tool output message (for function_call_output items)
}

// groupProcessor holds state during input item processing.
type groupProcessor struct {
	groups           []inputItemGroup
	pendingToolCalls []types.ToolCall
	converter        *ResponsesToChatConverter
}

// convertInputItems converts an array of raw input items to messages.
// It uses a two-phase approach: grouping then conversion.
func (c *ResponsesToChatConverter) convertInputItems(items []interface{}) []types.Message {
	groups := c.groupInputItems(items)
	return c.convertGroupsToMessages(groups)
}

// groupInputItems iterates over input items and delegates to handlers.
// Phase 1: Group related items that should become a single message.
func (c *ResponsesToChatConverter) groupInputItems(items []interface{}) []inputItemGroup {
	p := &groupProcessor{
		groups:    make([]inputItemGroup, 0, len(items)),
		converter: c,
	}

	for _, item := range items {
		itemMap, ok := item.(map[string]interface{})
		if !ok {
			continue
		}

		switch itemType, _ := itemMap["type"].(string); itemType {
		case "function_call":
			p.handleFunctionCallItem(itemMap)
		case "message":
			p.handleMessageItem(itemMap)
		case "function_call_output":
			p.handleFunctionCallOutputItem(itemMap)
		}
	}

	p.flushPendingToolCalls()
	return p.groups
}

// handleFunctionCallItem accumulates function_call items into pendingToolCalls.
func (p *groupProcessor) handleFunctionCallItem(itemMap map[string]interface{}) {
	if tc := p.converter.parseFunctionCallItem(itemMap); tc != nil {
		p.pendingToolCalls = append(p.pendingToolCalls, *tc)
	}
}

// handleMessageItem handles message items, merging with pending tool calls if assistant.
func (p *groupProcessor) handleMessageItem(itemMap map[string]interface{}) {
	role, _ := itemMap["role"].(string)

	// Check if this message has embedded tool_calls (combined format from prependHistoryToInput)
	if toolCallsRaw, ok := itemMap["tool_calls"]; ok && role == "assistant" {
		// Flush any pending tool calls first (they're orphaned)
		if len(p.pendingToolCalls) > 0 {
			p.flushPendingToolCalls()
		}
		// This is already a combined message with tool_calls
		msg := &types.Message{
			Role:      "assistant",
			ToolCalls: p.converter.extractToolCallsFromArray(toolCallsRaw),
		}
		if content, ok := itemMap["content"]; ok {
			msg.Content = p.converter.convertContentToValue(content)
		}
		p.groups = append(p.groups, inputItemGroup{
			itemType: "message",
			message:  msg,
		})
		return
	}

	msg := p.converter.parseMessageItem(itemMap)

	// If this is an assistant message and we have pending tool calls, MERGE them
	if role == "assistant" && len(p.pendingToolCalls) > 0 {
		p.groups = append(p.groups, inputItemGroup{
			itemType:  "merged_assistant",
			message:   msg,
			toolCalls: p.pendingToolCalls,
		})
		p.pendingToolCalls = nil
		return
	}

	// Flush any pending tool calls first (no assistant message to merge with)
	if len(p.pendingToolCalls) > 0 {
		p.flushPendingToolCalls()
	}

	// Add message as its own group
	if msg != nil {
		p.groups = append(p.groups, inputItemGroup{
			itemType: "message",
			message:  msg,
		})
	}
}

// handleFunctionCallOutputItem handles function_call_output items.
func (p *groupProcessor) handleFunctionCallOutputItem(itemMap map[string]interface{}) {
	// Flush any pending tool calls first
	if len(p.pendingToolCalls) > 0 {
		p.flushPendingToolCalls()
	}

	// Add tool output as its own group
	if toolMsg := p.converter.parseFunctionCallOutputItem(itemMap); toolMsg != nil {
		p.groups = append(p.groups, inputItemGroup{
			itemType:   "function_call_output",
			toolOutput: toolMsg,
		})
	}
}

// flushPendingToolCalls adds accumulated tool calls as a standalone group.
func (p *groupProcessor) flushPendingToolCalls() {
	if len(p.pendingToolCalls) == 0 {
		return
	}
	p.groups = append(p.groups, inputItemGroup{
		itemType:  "assistant_tool_calls",
		toolCalls: p.pendingToolCalls,
	})
	p.pendingToolCalls = nil
}

// convertGroupsToMessages converts grouped items to Chat Completions messages.
// Phase 2: Convert each group to its final message representation.
func (c *ResponsesToChatConverter) convertGroupsToMessages(groups []inputItemGroup) []types.Message {
	messages := make([]types.Message, 0, len(groups))

	for _, group := range groups {
		switch group.itemType {
		case "merged_assistant":
			// Combined assistant message with both content and tool_calls
			msg := *group.message
			msg.ToolCalls = group.toolCalls
			messages = append(messages, msg)

		case "assistant_tool_calls":
			// Assistant message with only tool_calls (no content)
			messages = append(messages, types.Message{
				Role:      "assistant",
				ToolCalls: group.toolCalls,
			})

		case "message":
			messages = append(messages, *group.message)

		case "function_call_output":
			messages = append(messages, *group.toolOutput)
		}
	}

	return messages
}

// parseMessageItem extracts a Message from a message input item.
func (c *ResponsesToChatConverter) parseMessageItem(itemMap map[string]interface{}) *types.Message {
	role, _ := itemMap["role"].(string)
	if role == "" {
		role = "user"
	}
	if role == "developer" {
		role = "system"
	}

	msg := &types.Message{Role: role}

	if content, ok := itemMap["content"]; ok {
		msg.Content = c.convertContentToValue(content)
	}

	return msg
}

// parseFunctionCallItem extracts a ToolCall from a function_call input item.
func (c *ResponsesToChatConverter) parseFunctionCallItem(itemMap map[string]interface{}) *types.ToolCall {
	name, _ := itemMap["name"].(string)
	arguments, _ := itemMap["arguments"].(string)
	callID, _ := itemMap["call_id"].(string)
	if callID == "" {
		callID, _ = itemMap["id"].(string)
	}

	if name == "" {
		return nil
	}

	return &types.ToolCall{
		ID:   callID,
		Type: "function",
		Function: types.Function{
			Name:      name,
			Arguments: arguments,
		},
	}
}

// parseFunctionCallOutputItem extracts a tool Message from a function_call_output input item.
func (c *ResponsesToChatConverter) parseFunctionCallOutputItem(itemMap map[string]interface{}) *types.Message {
	callID, _ := itemMap["call_id"].(string)
	if callID == "" {
		callID, _ = itemMap["tool_call_id"].(string)
	}
	output, _ := itemMap["output"].(string)

	if callID == "" {
		return nil
	}

	return &types.Message{
		Role:       "tool",
		ToolCallID: callID,
		Content:    output,
	}
}

// extractToolCallsFromArray extracts ToolCall slice from a tool_calls array.
func (c *ResponsesToChatConverter) extractToolCallsFromArray(toolCallsRaw interface{}) []types.ToolCall {
	if toolCallsRaw == nil {
		return nil
	}

	var toolCalls []types.ToolCall

	switch v := toolCallsRaw.(type) {
	case []interface{}:
		for _, tc := range v {
			if tcMap, ok := tc.(map[string]interface{}); ok {
				if extracted := c.parseFunctionCallItem(tcMap); extracted != nil {
					toolCalls = append(toolCalls, *extracted)
				}
			}
		}
	}

	return toolCalls
}

// convertContentToValue converts content to an appropriate value for Message.Content.
func (c *ResponsesToChatConverter) convertContentToValue(content interface{}) interface{} {
	if content == nil {
		return nil
	}

	switch v := content.(type) {
	case string:
		return v
	case []interface{}:
		return c.convertContent(v)
	default:
		return content
	}
}

// convertInputItemsFromTyped converts typed InputItem array to messages.
// It uses the same merge logic as convertInputItems for consistency.
func (c *ResponsesToChatConverter) convertInputItemsFromTyped(items []types.InputItem) []types.Message {
	// Convert typed items to generic format for grouping
	genericItems := make([]interface{}, len(items))
	for i, item := range items {
		itemMap := map[string]interface{}{"type": item.Type}
		if item.Role != "" {
			itemMap["role"] = item.Role
		}
		if item.Content != nil {
			itemMap["content"] = item.Content
		}
		if item.CallID != "" {
			itemMap["call_id"] = item.CallID
		}
		if item.ID != "" {
			itemMap["id"] = item.ID
		}
		if item.Name != "" {
			itemMap["name"] = item.Name
		}
		if item.Arguments != "" {
			itemMap["arguments"] = item.Arguments
		}
		if item.Output != "" {
			itemMap["output"] = item.Output
		}
		genericItems[i] = itemMap
	}

	// Reuse the same grouping logic
	return c.convertInputItems(genericItems)
}

// convertContentParts converts Responses API content parts to Chat format.
func (c *ResponsesToChatConverter) extractTextFromParts(parts []interface{}) string {
	var result strings.Builder
	for _, part := range parts {
		partMap, ok := part.(map[string]interface{})
		if !ok {
			continue
		}
		partType, _ := partMap["type"].(string)
		switch partType {
		case "input_text", "text", "output_text", "refusal":
			if text, ok := partMap["text"].(string); ok {
				if result.Len() > 0 {
					result.WriteString("\n")
				}
				result.WriteString(text)
			}
		case "input_file":
			// File attachments - extract filename as placeholder
			if fileData, ok := partMap["file_data"].(map[string]interface{}); ok {
				if filename, ok := fileData["filename"].(string); ok {
					if result.Len() > 0 {
						result.WriteString("\n")
					}
					result.WriteString("[File attached: " + filename + "]")
				}
			}
		}
	}
	return result.String()
}

func (c *ResponsesToChatConverter) hasNonTextParts(parts []interface{}) bool {
	for _, part := range parts {
		partMap, ok := part.(map[string]interface{})
		if !ok {
			continue
		}
		partType, _ := partMap["type"].(string)
		if partType != "input_text" && partType != "text" && partType != "output_text" && partType != "input_file" && partType != "refusal" {
			return true
		}
	}
	return false
}

func (c *ResponsesToChatConverter) convertContent(parts []interface{}) interface{} {
	if c.hasNonTextParts(parts) {
		return c.convertContentParts(parts)
	}
	return c.extractTextFromParts(parts)
}

func (c *ResponsesToChatConverter) convertContentParts(parts []interface{}) []interface{} {
	result := make([]interface{}, 0, len(parts))

	for _, part := range parts {
		partMap, ok := part.(map[string]interface{})
		if !ok {
			continue
		}

		partType, _ := partMap["type"].(string)
		switch partType {
		case "input_text", "output_text", "refusal":
			// Convert input_text, output_text, and refusal to text
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
		case "input_file":
			// Convert input_file to text placeholder
			if fileData, ok := partMap["file_data"].(map[string]interface{}); ok {
				if filename, ok := fileData["filename"].(string); ok {
					result = append(result, map[string]interface{}{
						"type": "text",
						"text": "[File attached: " + filename + "]",
					})
				}
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

	responseID string
	model      string
	created    int64

	contentIndex    int
	toolCallIndex   int
	currentToolCall *responsesToolCallState

	contentBuilder strings.Builder

	finishReason string

	toolNames map[string]string
}

type responsesToolCallState struct {
	id        string
	name      string
	arguments strings.Builder
	firstSent bool
}

// NewResponsesToChatTransformer creates a new transformer for Responses to Chat format.
func NewResponsesToChatTransformer(w io.Writer) *ResponsesToChatTransformer {
	return &ResponsesToChatTransformer{
		w:         w,
		created:   time.Now().Unix(),
		toolNames: make(map[string]string),
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

	if event.OutputItem.Type == "function_call" {
		t.currentToolCall = &responsesToolCallState{
			id:   event.OutputItem.ID,
			name: event.OutputItem.Name,
		}
		t.toolNames[event.OutputItem.ID] = event.OutputItem.Name
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
		t.toolCallIndex++
		name := t.toolNames[event.ItemID]
		t.currentToolCall = &responsesToolCallState{
			id:   event.ItemID,
			name: name,
		}
	}

	t.currentToolCall.arguments.WriteString(event.Delta)

	tc := types.ToolCall{
		Index: t.toolCallIndex - 1,
		Function: types.Function{
			Arguments: event.Delta,
		},
	}
	if !t.currentToolCall.firstSent {
		tc.ID = t.currentToolCall.id
		tc.Type = "function"
		tc.Function.Name = t.currentToolCall.name
		t.currentToolCall.firstSent = true
	}

	chunk := t.createChunk()
	chunk.Choices = []types.Choice{
		{
			Index: 0,
			Delta: types.Delta{
				ToolCalls: []types.ToolCall{tc},
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
			Index:        t.contentIndex,
			Delta:        types.Delta{},
			FinishReason: &t.finishReason,
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
