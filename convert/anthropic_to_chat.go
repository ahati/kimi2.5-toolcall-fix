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

const defaultMaxTokens = 4096

// ─────────────────────────────────────────────────────────────────────────────
// Anthropic → Chat Completions — Request
// ─────────────────────────────────────────────────────────────────────────────

// AnthropicToChatRequest converts an Anthropic MessageRequest into a Chat Completions Request.
//
// Dropped fields (no Chat Completions equivalent): top_k, thinking.
// stop_sequences → stop (OAI accepts array).
// metadata.user_id → user.
// stream_options.include_usage is automatically added when stream=true.
func AnthropicToChatRequest(req *types.MessageRequest) (*types.ChatCompletionRequest, error) {
	out := &types.ChatCompletionRequest{
		Model:      req.Model,
		Stream:     req.Stream,
		ToolChoice: AnthropicToolChoiceToChat(marshalToolChoice(req.ToolChoice)),
	}

	// max_tokens
	if req.MaxTokens > 0 {
		out.MaxTokens = req.MaxTokens
	}

	// temperature (Anthropic uses 0-1, Chat uses 0-2, so direct copy is fine)
	if req.Temperature > 0 {
		out.Temperature = req.Temperature
	}

	// top_p
	if req.TopP > 0 {
		out.TopP = req.TopP
	}

	// stop_sequences → stop
	if len(req.StopSequences) > 0 {
		out.Stop = req.StopSequences
	}

	// metadata.user_id → user
	if req.Metadata != nil && req.Metadata.UserID != "" {
		out.User = req.Metadata.UserID
	}

	// Always include usage when streaming so callers can track consumption.
	if req.Stream {
		out.StreamOptions = &types.StreamOptions{IncludeUsage: true}
	}

	// tools
	if len(req.Tools) > 0 {
		tools, err := anthropicToolsToChatTools(req.Tools)
		if err != nil {
			return nil, fmt.Errorf("converting tools: %w", err)
		}
		out.Tools = tools
	}

	// system → prepend system message
	system := extractSystemFromRequest(req.System)
	var messages []types.Message
	if system != "" {
		messages = append(messages, types.Message{
			Role:    "system",
			Content: system,
		})
	}

	converted, err := anthropicMessagesToChatMessages(req.Messages)
	if err != nil {
		return nil, fmt.Errorf("converting messages: %w", err)
	}
	out.Messages = append(messages, converted...)
	return out, nil
}

func marshalToolChoice(tc *types.ToolChoice) json.RawMessage {
	if tc == nil {
		return nil
	}
	b, _ := json.Marshal(tc)
	return b
}

func extractSystemFromRequest(system interface{}) string {
	if system == nil {
		return ""
	}
	switch s := system.(type) {
	case string:
		return s
	case json.RawMessage:
		return ExtractSystemText(s)
	default:
		b, _ := json.Marshal(system)
		return ExtractSystemText(b)
	}
}

func anthropicToolsToChatTools(tools []types.ToolDef) ([]types.Tool, error) {
	out := make([]types.Tool, len(tools))
	for i, t := range tools {
		out[i] = types.Tool{
			Type: "function",
			Function: types.ToolFunction{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.InputSchema,
			},
		}
	}
	return out, nil
}

func anthropicMessagesToChatMessages(msgs []types.MessageInput) ([]types.Message, error) {
	var out []types.Message
	for _, msg := range msgs {
		converted, err := anthropicMessageToChatMessages(msg)
		if err != nil {
			return nil, err
		}
		out = append(out, converted...)
	}
	return out, nil
}

// anthropicMessageToChatMessages converts one Anthropic message into one or
// more Chat Completions messages. Tool results in a user message become
// separate "tool" role messages.
func anthropicMessageToChatMessages(msg types.MessageInput) ([]types.Message, error) {
	switch content := msg.Content.(type) {
	case string:
		return []types.Message{{Role: msg.Role, Content: content}}, nil
	case []interface{}:
		return anthropicContentBlocksToChatMessages(msg.Role, content)
	case json.RawMessage:
		// Try to unmarshal as string first
		var s string
		if err := json.Unmarshal(content, &s); err == nil {
			return []types.Message{{Role: msg.Role, Content: s}}, nil
		}
		// Try as array of blocks
		var blocks []interface{}
		if err := json.Unmarshal(content, &blocks); err == nil {
			return anthropicContentBlocksToChatMessages(msg.Role, blocks)
		}
		return nil, fmt.Errorf("unknown content format")
	default:
		return nil, fmt.Errorf("unknown Anthropic content type: %T", content)
	}
}

func anthropicContentBlocksToChatMessages(role string, blocks []interface{}) ([]types.Message, error) {
	switch role {
	case "user":
		return anthropicUserBlocksToChatMessages(blocks)
	case "assistant":
		return anthropicAssistantBlocksToChatMessages(blocks)
	default:
		return nil, fmt.Errorf("unknown Anthropic role: %q", role)
	}
}

func anthropicUserBlocksToChatMessages(blocks []interface{}) ([]types.Message, error) {
	var toolResults []types.Message
	var contentParts []map[string]interface{}

	for _, item := range blocks {
		b, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		blockType, _ := b["type"].(string)
		switch blockType {
		case "tool_result":
			toolUseID, _ := b["tool_use_id"].(string)
			content := extractToolResultContent(b["content"])
			toolResults = append(toolResults, types.Message{
				Role:       "tool",
				ToolCallID: toolUseID,
				Content:    content,
			})
		case "text":
			text, _ := b["text"].(string)
			contentParts = append(contentParts, map[string]interface{}{
				"type": "text",
				"text": text,
			})
		case "image":
			src, ok := b["source"].(map[string]interface{})
			if !ok {
				continue
			}
			part, err := anthropicImageSourceToChatPart(src)
			if err != nil {
				return nil, err
			}
			contentParts = append(contentParts, part)
		}
	}

	var out []types.Message
	switch {
	case len(contentParts) == 1 && contentParts[0]["type"] == "text":
		// Simple text — use string content form.
		out = append(out, types.Message{
			Role:    "user",
			Content: contentParts[0]["text"].(string),
		})
	case len(contentParts) > 0:
		out = append(out, types.Message{
			Role:    "user",
			Content: contentParts,
		})
	}

	out = append(out, toolResults...)
	return out, nil
}

func extractToolResultContent(content interface{}) string {
	if content == nil {
		return ""
	}
	switch c := content.(type) {
	case string:
		return c
	case []interface{}:
		var parts []string
		for _, item := range c {
			if b, ok := item.(map[string]interface{}); ok {
				if b["type"] == "text" {
					if text, ok := b["text"].(string); ok {
						parts = append(parts, text)
					}
				}
			}
		}
		return strings.Join(parts, "\n")
	default:
		b, _ := json.Marshal(content)
		return string(b)
	}
}

func anthropicAssistantBlocksToChatMessages(blocks []interface{}) ([]types.Message, error) {
	var textParts []string
	var toolCalls []types.ToolCall

	for _, item := range blocks {
		b, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		blockType, _ := b["type"].(string)
		switch blockType {
		case "text":
			text, _ := b["text"].(string)
			textParts = append(textParts, text)
		case "tool_use":
			id, _ := b["id"].(string)
			name, _ := b["name"].(string)
			input := b["input"]
			var args string
			switch v := input.(type) {
			case string:
				args = v
			default:
				b, _ := json.Marshal(v)
				args = string(b)
			}
			if !json.Valid([]byte(args)) {
				args = "{}"
			}
			tc := types.ToolCall{ID: id, Type: "function"}
			tc.Function.Name = name
			tc.Function.Arguments = args
			toolCalls = append(toolCalls, tc)
		case "thinking":
			// Drop thinking blocks — no Chat Completions equivalent.
		}
	}

	msg := types.Message{Role: "assistant"}
	textContent := strings.Join(textParts, "")
	if textContent != "" {
		msg.Content = textContent
	}
	if len(toolCalls) > 0 {
		msg.ToolCalls = toolCalls
		if textContent == "" {
			// OAI convention: content is null when only tool calls are present.
			msg.Content = nil
		}
	}
	return []types.Message{msg}, nil
}

func anthropicImageSourceToChatPart(src map[string]interface{}) (map[string]interface{}, error) {
	srcType, _ := src["type"].(string)
	var url string
	switch srcType {
	case "base64":
		mediaType, _ := src["media_type"].(string)
		data, _ := src["data"].(string)
		url = BuildDataURI(mediaType, data)
	case "url":
		url, _ = src["url"].(string)
	default:
		return nil, fmt.Errorf("unknown image source type: %q", srcType)
	}
	return map[string]interface{}{
		"type": "image_url",
		"image_url": map[string]interface{}{
			"url": url,
		},
	}, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Anthropic → Chat Completions — Response
// ─────────────────────────────────────────────────────────────────────────────

// AnthropicResponseToChat converts an Anthropic message info into a Chat Completion Response.
func AnthropicResponseToChat(msg *types.MessageInfo) (*types.ChatCompletionRequest, error) {
	// Note: ChatCompletionRequest is used for responses too in this codebase
	// Actually we need a proper response type. Let me check what the codebase uses.
	// For now, we'll just return the information needed
	return nil, fmt.Errorf("use AnthropicMessageInfoToChatResponse for response conversion")
}

// AnthropicMessageInfoToChatResponse creates a Chat Completions response from Anthropic MessageInfo.
// Returns a map that can be serialized to JSON for the response.
func AnthropicMessageInfoToChatResponse(msg *types.MessageInfo) map[string]interface{} {
	response := map[string]interface{}{
		"id":      msg.ID,
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   msg.Model,
	}

	if msg.Usage != nil {
		response["usage"] = map[string]interface{}{
			"prompt_tokens":     msg.Usage.InputTokens,
			"completion_tokens": msg.Usage.OutputTokens,
			"total_tokens":      msg.Usage.InputTokens + msg.Usage.OutputTokens,
		}
	}

	content := ""
	var toolCalls []map[string]interface{}

	for _, b := range msg.Content {
		switch b.Type {
		case "text":
			content += b.Text
		case "tool_use":
			args := string(b.Input)
			if !json.Valid([]byte(args)) {
				args = "{}"
			}
			toolCalls = append(toolCalls, map[string]interface{}{
				"id":   b.ID,
				"type": "function",
				"function": map[string]interface{}{
					"name":      b.Name,
					"arguments": args,
				},
			})
		}
	}

	message := map[string]interface{}{
		"role":    "assistant",
		"content": content,
	}
	if len(toolCalls) > 0 {
		message["tool_calls"] = toolCalls
		if content == "" {
			message["content"] = nil
		}
	}

	finishReason := "stop"
	if len(toolCalls) > 0 {
		finishReason = "tool_calls"
	}
	response["choices"] = []map[string]interface{}{
		{
			"index":         0,
			"message":       message,
			"finish_reason": finishReason,
		},
	}
	return response
}

// ─────────────────────────────────────────────────────────────────────────────
// Anthropic → Chat Completions — Streaming
// ─────────────────────────────────────────────────────────────────────────────

// AnthropicToChatTransformer transforms Anthropic SSE events to Chat Completions SSE.
type AnthropicToChatTransformer struct {
	w io.Writer

	msgID       string
	model       string
	inputTokens int

	created          int64
	sequenceNumber   int
	contentIndex     int
	toolCallIndex    int
	toolCallIDs      map[int]string // block index -> tool call ID
	finishReason     string
	usage            *types.Usage
	messageStarted   bool
	toolCalls        []types.ToolCall
	currentToolCall  *anthropicToChatToolCallState
	contentBuilder   strings.Builder
	completed        bool
}

type anthropicToChatToolCallState struct {
	id        string
	name      string
	arguments strings.Builder
}

// NewAnthropicToChatTransformer creates a new transformer.
func NewAnthropicToChatTransformer(w io.Writer) *AnthropicToChatTransformer {
	return &AnthropicToChatTransformer{
		w:              w,
		created:        time.Now().Unix(),
		toolCallIDs:    make(map[int]string),
		sequenceNumber: 0,
	}
}

func (t *AnthropicToChatTransformer) nextSeq() int {
	t.sequenceNumber++
	return t.sequenceNumber
}

// Transform transforms an SSE event.
func (t *AnthropicToChatTransformer) Transform(event *sse.Event) error {
	if event.Data == "" || event.Data == "[DONE]" {
		return nil
	}

	// Parse the event type
	var base struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal([]byte(event.Data), &base); err != nil {
		return nil
	}

	switch base.Type {
	case "message_start":
		return t.handleMessageStart(event.Data)
	case "content_block_start":
		return t.handleContentBlockStart(event.Data)
	case "content_block_delta":
		return t.handleContentBlockDelta(event.Data)
	case "content_block_stop":
		return t.handleContentBlockStop(event.Data)
	case "message_delta":
		return t.handleMessageDelta(event.Data)
	case "message_stop":
		return t.handleMessageStop()
	}
	return nil
}

func (t *AnthropicToChatTransformer) handleMessageStart(data string) error {
	var e types.AnthropicMessageStartEvent
	if err := json.Unmarshal([]byte(data), &e); err != nil {
		return err
	}
	t.msgID = e.Message.ID
	t.model = e.Message.Model
	t.inputTokens = e.Message.Usage.InputTokens

	// Emit first chunk with role
	chunk := t.makeChunk(&types.Delta{Role: "assistant", Content: ""}, nil, nil)
	return t.writeChunk(chunk)
}

func (t *AnthropicToChatTransformer) handleContentBlockStart(data string) error {
	var e types.AnthropicContentBlockStartEvent
	if err := json.Unmarshal([]byte(data), &e); err != nil {
		return err
	}

	if e.ContentBlock.Type == "tool_use" {
		t.toolCallIDs[e.Index] = e.ContentBlock.ID
		t.currentToolCall = &anthropicToChatToolCallState{
			id:   e.ContentBlock.ID,
			name: e.ContentBlock.Name,
		}
		delta := &types.Delta{
			ToolCalls: []types.ToolCall{{
				Index: t.toolCallIndex,
				ID:    e.ContentBlock.ID,
				Type:  "function",
				Function: types.Function{
					Name: e.ContentBlock.Name,
				},
			}},
		}
		t.toolCallIndex++
		chunk := t.makeChunk(delta, nil, nil)
		return t.writeChunk(chunk)
	}
	return nil
}

func (t *AnthropicToChatTransformer) handleContentBlockDelta(data string) error {
	var e types.AnthropicContentBlockDeltaEvent
	if err := json.Unmarshal([]byte(data), &e); err != nil {
		return err
	}

	switch e.Delta.Type {
	case "text_delta":
		t.contentBuilder.WriteString(e.Delta.Text)
		chunk := t.makeChunk(&types.Delta{Content: e.Delta.Text}, nil, nil)
		return t.writeChunk(chunk)
	case "input_json_delta":
		if t.currentToolCall != nil {
			t.currentToolCall.arguments.WriteString(e.Delta.PartialJSON)
		}
		delta := &types.Delta{
			ToolCalls: []types.ToolCall{{
				Index: t.toolCallIndex - 1,
				Function: types.Function{
					Arguments: e.Delta.PartialJSON,
				},
			}},
		}
		chunk := t.makeChunk(delta, nil, nil)
		return t.writeChunk(chunk)
	}
	return nil
}

func (t *AnthropicToChatTransformer) handleContentBlockStop(data string) error {
	// Content block stop - no explicit action needed for Chat Completions
	return nil
}

func (t *AnthropicToChatTransformer) handleMessageDelta(data string) error {
	var e types.AnthropicMessageDeltaEvent
	if err := json.Unmarshal([]byte(data), &e); err != nil {
		return err
	}
	t.finishReason = MapStopReason(e.Delta.StopReason)
	t.usage = &types.Usage{
		PromptTokens:     t.inputTokens,
		CompletionTokens: e.Usage.OutputTokens,
		TotalTokens:      t.inputTokens + e.Usage.OutputTokens,
	}
	return nil
}

func (t *AnthropicToChatTransformer) handleMessageStop() error {
	if t.completed {
		return nil
	}

	fr := t.finishReason
	if fr == "" {
		fr = "stop"
	}
	usage := t.usage

	chunk := t.makeChunk(&types.Delta{}, &fr, usage)
	if err := t.writeChunk(chunk); err != nil {
		return err
	}
	t.completed = true
	return WriteSSEDone(t.w)
}

func (t *AnthropicToChatTransformer) makeChunk(delta *types.Delta, finishReason *string, usage *types.Usage) types.Chunk {
	return types.Chunk{
		ID:      t.msgID,
		Object:  "chat.completion.chunk",
		Created: t.created,
		Model:   t.model,
		Choices: []types.Choice{{
			Index:        0,
			Delta:        *delta,
			FinishReason: finishReason,
		}},
		Usage: usage,
	}
}

func (t *AnthropicToChatTransformer) writeChunk(chunk types.Chunk) error {
	data, err := json.Marshal(chunk)
	if err != nil {
		return err
	}
	return WriteSSEEvent(t.w, "", string(data))
}

// Flush flushes any pending data.
func (t *AnthropicToChatTransformer) Flush() error {
	return nil
}

// Close closes the transformer.
func (t *AnthropicToChatTransformer) Close() error {
	if !t.completed {
		return t.handleMessageStop()
	}
	return nil
}