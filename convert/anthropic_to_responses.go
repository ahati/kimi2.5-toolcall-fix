package convert

import (
	"encoding/json"
	"fmt"
	"io"
	"time"

	"ai-proxy/types"

	"github.com/tmaxmax/go-sse"
)

// ─────────────────────────────────────────────────────────────────────────────
// Anthropic → Responses — Request
// ─────────────────────────────────────────────────────────────────────────────

// AnthropicToResponsesRequest converts an Anthropic MessageRequest into a Responses Request.
//
// Dropped: top_k, stop_sequences (Responses API has no stop field).
func AnthropicToResponsesRequest(req *types.MessageRequest) (*types.ResponsesRequest, error) {
	out := &types.ResponsesRequest{
		Model:        req.Model,
		Stream:       req.Stream,
		Instructions: extractSystemFromRequest(req.System),
	}

	if req.MaxTokens > 0 {
		out.MaxOutputTokens = req.MaxTokens
	}

	if req.Temperature > 0 {
		out.Temperature = req.Temperature
	}

	if req.TopP > 0 {
		out.TopP = req.TopP
	}

	if req.Metadata != nil && req.Metadata.UserID != "" {
		out.Metadata = map[string]interface{}{"user_id": req.Metadata.UserID}
	}

	// thinking → reasoning
	if req.Thinking != nil && req.Thinking.Type == "enabled" {
		out.Reasoning = &types.ReasoningConfig{
			Effort: BudgetToReasoningEffort(req.Thinking.BudgetTokens),
		}
	}

	// tools
	for _, t := range req.Tools {
		out.Tools = append(out.Tools, types.ResponsesTool{
			Type:        "function",
			Name:        t.Name,
			Description: t.Description,
			Parameters:  t.InputSchema,
		})
	}
	if len(out.Tools) > 0 {
		out.ToolChoice = AnthropicToolChoiceToResponses(marshalToolChoice(req.ToolChoice))
	}

	input, err := anthropicMessagesToResponsesInput(req.Messages)
	if err != nil {
		return nil, fmt.Errorf("converting messages: %w", err)
	}
	out.Input = input
	return out, nil
}

func anthropicMessagesToResponsesInput(msgs []types.MessageInput) (interface{}, error) {
	if len(msgs) == 0 {
		return "", nil
	}

	var items []types.InputItem
	for _, msg := range msgs {
		msgItems, err := anthropicMessageToResponsesItems(msg)
		if err != nil {
			return nil, err
		}
		items = append(items, msgItems...)
	}

	if len(items) == 1 && items[0].Type == "message" && items[0].Role == "user" {
		// Return as simple string if single user message
		if str, ok := items[0].Content.(string); ok {
			return str, nil
		}
	}
	return items, nil
}

func anthropicMessageToResponsesItems(msg types.MessageInput) ([]types.InputItem, error) {
	switch content := msg.Content.(type) {
	case string:
		if msg.Role == "assistant" {
			return []types.InputItem{{
				Type: "message",
				Role: "assistant",
				Content: []types.ContentPart{{
					Type: "output_text",
					Text: content,
				}},
			}}, nil
		}
		return []types.InputItem{{
			Type:    "message",
			Role:    msg.Role,
			Content: content,
		}}, nil
	case []interface{}:
		return anthropicContentBlocksToResponsesItems(msg.Role, content)
	case json.RawMessage:
		// Try string first
		var s string
		if err := json.Unmarshal(content, &s); err == nil {
			if msg.Role == "assistant" {
				return []types.InputItem{{
					Type: "message",
					Role: "assistant",
					Content: []types.ContentPart{{
						Type: "output_text",
						Text: s,
					}},
				}}, nil
			}
			return []types.InputItem{{
				Type:    "message",
				Role:    msg.Role,
				Content: s,
			}}, nil
		}
		// Try as array of blocks
		var blocks []interface{}
		if err := json.Unmarshal(content, &blocks); err == nil {
			return anthropicContentBlocksToResponsesItems(msg.Role, blocks)
		}
		return nil, fmt.Errorf("unknown content format")
	default:
		return nil, fmt.Errorf("unknown content type: %T", content)
	}
}

func anthropicContentBlocksToResponsesItems(role string, blocks []interface{}) ([]types.InputItem, error) {
	switch role {
	case "user":
		return anthropicUserBlocksToResponsesItems(blocks)
	case "assistant":
		return anthropicAssistantBlocksToResponsesItems(blocks)
	default:
		return nil, fmt.Errorf("unknown role: %q", role)
	}
}

func anthropicUserBlocksToResponsesItems(blocks []interface{}) ([]types.InputItem, error) {
	var contentParts []types.ContentPart
	var functionOutputItems []types.InputItem

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
			functionOutputItems = append(functionOutputItems, types.InputItem{
				Type:   "function_call_output",
				CallID: toolUseID,
				Output: content,
			})
		case "text":
			text, _ := b["text"].(string)
			contentParts = append(contentParts, types.ContentPart{
				Type: "input_text",
				Text: text,
			})
		case "image":
			src, ok := b["source"].(map[string]interface{})
			if !ok {
				continue
			}
			imageURL, err := anthropicImageSourceToURL(src)
			if err != nil {
				return nil, err
			}
			contentParts = append(contentParts, types.ContentPart{
				Type:     "input_image",
				ImageURL: imageURL,
			})
		}
	}

	var out []types.InputItem
	if len(contentParts) > 0 {
		out = append(out, types.InputItem{
			Type:    "message",
			Role:    "user",
			Content: contentParts,
		})
	}
	out = append(out, functionOutputItems...)
	return out, nil
}

func anthropicAssistantBlocksToResponsesItems(blocks []interface{}) ([]types.InputItem, error) {
	var contentParts []types.ContentPart
	var functionCallItems []types.InputItem

	for _, item := range blocks {
		b, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		blockType, _ := b["type"].(string)
		switch blockType {
		case "text":
			text, _ := b["text"].(string)
			contentParts = append(contentParts, types.ContentPart{
				Type: "output_text",
				Text: text,
			})
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
			functionCallItems = append(functionCallItems, types.InputItem{
				Type:      "function_call",
				CallID:    id,
				Name:      name,
				Arguments: args,
			})
			// thinking — dropped (no Responses API equivalent in input)
		}
	}

	var out []types.InputItem
	if len(contentParts) > 0 {
		out = append(out, types.InputItem{
			Type:    "message",
			Role:    "assistant",
			Content: contentParts,
		})
	}
	out = append(out, functionCallItems...)
	return out, nil
}

func anthropicImageSourceToURL(src map[string]interface{}) (string, error) {
	srcType, _ := src["type"].(string)
	switch srcType {
	case "base64":
		mediaType, _ := src["media_type"].(string)
		data, _ := src["data"].(string)
		return BuildDataURI(mediaType, data), nil
	case "url":
		url, _ := src["url"].(string)
		return url, nil
	default:
		return "", fmt.Errorf("unknown image source type: %q", srcType)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Anthropic → Responses — Response
// ─────────────────────────────────────────────────────────────────────────────

// AnthropicMessageInfoToResponses converts an Anthropic MessageInfo into a Responses Response.
func AnthropicMessageInfoToResponses(msg *types.MessageInfo) (*types.ResponsesResponse, error) {
	out := &types.ResponsesResponse{
		ID:        msg.ID,
		Object:    "response",
		CreatedAt: time.Now().Unix(),
		Model:     msg.Model,
		Status:    "completed",
	}
	if msg.Usage != nil {
		out.Usage = &types.ResponsesUsage{
			InputTokens:  msg.Usage.InputTokens,
			OutputTokens: msg.Usage.OutputTokens,
			TotalTokens:  msg.Usage.InputTokens + msg.Usage.OutputTokens,
		}
	}

	var msgItem types.OutputItem
	msgItem.Type = "message"
	msgItem.Role = "assistant"

	for _, b := range msg.Content {
		switch b.Type {
		case "text":
			msgItem.Content = append(msgItem.Content, types.OutputContent{
				Type: "output_text",
				Text: b.Text,
			})
		case "tool_use":
			out.Output = append(out.Output, types.OutputItem{
				Type:      "function_call",
				CallID:    b.ID,
				Name:      b.Name,
				Arguments: string(b.Input),
			})
		}
	}
	if len(msgItem.Content) > 0 {
		// Prepend message item before any function_call items.
		out.Output = append([]types.OutputItem{msgItem}, out.Output...)
	}
	return out, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Anthropic → Responses — Streaming
// ─────────────────────────────────────────────────────────────────────────────

// AnthropicToResponsesTransformer transforms Anthropic SSE events to Responses SSE.
type AnthropicToResponsesTransformer struct {
	w io.Writer

	responseID string
	model      string
	created    int64

	outputIndex   int
	blockTypeMap  map[int]string // block index -> type
	toolCallItems map[int]*types.OutputItem
	toolCallArgs  map[int]string // block index -> accumulated arguments

	sequenceNumber int
	completed      bool

	// Token usage tracking from message_start
	inputTokens          int
	cacheReadInputTokens int
}

// NewAnthropicToResponsesTransformer creates a new transformer.
func NewAnthropicToResponsesTransformer(w io.Writer) *AnthropicToResponsesTransformer {
	return &AnthropicToResponsesTransformer{
		w:             w,
		created:       time.Now().Unix(),
		blockTypeMap:  make(map[int]string),
		toolCallItems: make(map[int]*types.OutputItem),
		toolCallArgs:  make(map[int]string),
	}
}

func (t *AnthropicToResponsesTransformer) nextSeq() int {
	t.sequenceNumber++
	return t.sequenceNumber
}

// Transform transforms an SSE event.
func (t *AnthropicToResponsesTransformer) Transform(event *sse.Event) error {
	if event.Data == "" || event.Data == "[DONE]" {
		return nil
	}

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
	}
	return nil
}

func (t *AnthropicToResponsesTransformer) handleMessageStart(data string) error {
	var e types.AnthropicMessageStartEvent
	if err := json.Unmarshal([]byte(data), &e); err != nil {
		return err
	}
	t.responseID = e.Message.ID
	t.model = e.Message.Model

	// Capture input token usage from message_start
	t.inputTokens = e.Message.Usage.InputTokens
	t.cacheReadInputTokens = e.Message.Usage.CacheReadInputTokens

	// Emit response.created
	resp := &types.ResponsesResponse{
		ID:     t.responseID,
		Object: "response",
		Model:  t.model,
		Status: "in_progress",
	}
	if err := t.emitEvent(string(types.EventResponseCreated), map[string]interface{}{
		"type":            string(types.EventResponseCreated),
		"sequence_number": t.nextSeq(),
		"response":        resp,
	}); err != nil {
		return err
	}

	// Emit response.in_progress
	return t.emitEvent(string(types.EventResponseInProgress), map[string]interface{}{
		"type":            string(types.EventResponseInProgress),
		"sequence_number": t.nextSeq(),
		"response":        resp,
	})
}

func (t *AnthropicToResponsesTransformer) handleContentBlockStart(data string) error {
	var e types.AnthropicContentBlockStartEvent
	if err := json.Unmarshal([]byte(data), &e); err != nil {
		return err
	}
	t.blockTypeMap[e.Index] = e.ContentBlock.Type

	switch e.ContentBlock.Type {
	case "text":
		// Emit response.output_item.added for message
		msgItem := &types.OutputItem{
			Type: "message",
			ID:   fmt.Sprintf("item_%d", e.Index),
			Role: "assistant",
		}
		if err := t.emitEvent(string(types.EventResponseOutputItemAdded), map[string]interface{}{
			"type":            string(types.EventResponseOutputItemAdded),
			"sequence_number": t.nextSeq(),
			"output_index":    t.outputIndex,
			"item":            msgItem,
		}); err != nil {
			return err
		}

		// Emit response.content_part.added
		part := &types.ContentPart{Type: "output_text"}
		if err := t.emitEvent(string(types.EventResponseContentPartAdded), map[string]interface{}{
			"type":            string(types.EventResponseContentPartAdded),
			"sequence_number": t.nextSeq(),
			"output_index":    t.outputIndex,
			"content_index":   0,
			"part":            part,
		}); err != nil {
			return err
		}
		t.outputIndex++

	case "tool_use":
		item := &types.OutputItem{
			Type:   "function_call",
			ID:     e.ContentBlock.ID,
			CallID: e.ContentBlock.ID,
			Name:   e.ContentBlock.Name,
		}
		t.toolCallItems[e.Index] = item
		t.toolCallArgs[e.Index] = ""
		if err := t.emitEvent(string(types.EventResponseOutputItemAdded), map[string]interface{}{
			"type":            string(types.EventResponseOutputItemAdded),
			"sequence_number": t.nextSeq(),
			"output_index":    t.outputIndex,
			"item":            item,
		}); err != nil {
			return err
		}
		t.outputIndex++
	}
	return nil
}

func (t *AnthropicToResponsesTransformer) handleContentBlockDelta(data string) error {
	var e types.AnthropicContentBlockDeltaEvent
	if err := json.Unmarshal([]byte(data), &e); err != nil {
		return err
	}

	switch e.Delta.Type {
	case "text_delta":
		return t.emitEvent(string(types.EventResponseOutputTextDelta), map[string]interface{}{
			"type":            string(types.EventResponseOutputTextDelta),
			"sequence_number": t.nextSeq(),
			"output_index":    e.Index,
			"delta":           e.Delta.Text,
		})
	case "input_json_delta":
		// Accumulate arguments
		t.toolCallArgs[e.Index] += e.Delta.PartialJSON
		return t.emitEvent(string(types.EventResponseFunctionCallArgumentsDelta), map[string]interface{}{
			"type":            string(types.EventResponseFunctionCallArgumentsDelta),
			"sequence_number": t.nextSeq(),
			"output_index":    e.Index,
			"delta":           e.Delta.PartialJSON,
		})
	}
	return nil
}

func (t *AnthropicToResponsesTransformer) handleContentBlockStop(data string) error {
	var e types.AnthropicContentBlockStopEvent
	if err := json.Unmarshal([]byte(data), &e); err != nil {
		return err
	}

	blockType := t.blockTypeMap[e.Index]
	switch blockType {
	case "text":
		if err := t.emitEvent(string(types.EventResponseOutputTextDone), map[string]interface{}{
			"type":            string(types.EventResponseOutputTextDone),
			"sequence_number": t.nextSeq(),
			"output_index":    e.Index,
		}); err != nil {
			return err
		}
	case "tool_use":
		args := t.toolCallArgs[e.Index]
		if err := t.emitEvent(string(types.EventResponseFunctionCallArgumentsDone), map[string]interface{}{
			"type":            string(types.EventResponseFunctionCallArgumentsDone),
			"sequence_number": t.nextSeq(),
			"output_index":    e.Index,
			"arguments":       args,
		}); err != nil {
			return err
		}
	}

	// Emit output_item.done
	return t.emitEvent(string(types.EventResponseOutputItemDone), map[string]interface{}{
		"type":            string(types.EventResponseOutputItemDone),
		"sequence_number": t.nextSeq(),
		"output_index":    e.Index,
	})
}

func (t *AnthropicToResponsesTransformer) handleMessageDelta(data string) error {
	var e types.AnthropicMessageDeltaEvent
	if err := json.Unmarshal([]byte(data), &e); err != nil {
		return err
	}

	status := "completed"
	eventType := types.EventResponseCompleted
	if e.Delta.StopReason == "max_tokens" {
		status = "incomplete"
		eventType = types.EventResponseIncomplete
	}

	// Calculate total tokens (include cache in input for OpenAI format)
	inputTokens := t.inputTokens + t.cacheReadInputTokens
	totalTokens := inputTokens + e.Usage.OutputTokens

	finalResp := &types.ResponsesResponse{
		ID:     t.responseID,
		Object: "response",
		Model:  t.model,
		Status: status,
		Usage: &types.ResponsesUsage{
			InputTokens:  inputTokens,
			OutputTokens: e.Usage.OutputTokens,
			TotalTokens:  totalTokens,
			InputTokensDetails: &types.InputTokensDetails{
				CachedTokens: t.cacheReadInputTokens,
			},
		},
	}

	if err := t.emitEvent(string(eventType), map[string]interface{}{
		"type":            string(eventType),
		"sequence_number": t.nextSeq(),
		"response":        finalResp,
	}); err != nil {
		return err
	}
	t.completed = true
	return nil
}

func (t *AnthropicToResponsesTransformer) emitEvent(eventType string, event map[string]interface{}) error {
	data, err := json.Marshal(event)
	if err != nil {
		return err
	}
	return WriteSSEEvent(t.w, eventType, string(data))
}

// Flush flushes any pending data.
func (t *AnthropicToResponsesTransformer) Flush() error {
	return nil
}

// Close closes the transformer.
func (t *AnthropicToResponsesTransformer) Close() error {
	if !t.completed {
		// Emit response.completed if not already done
		finalResp := &types.ResponsesResponse{
			ID:     t.responseID,
			Object: "response",
			Model:  t.model,
			Status: "completed",
		}
		return t.emitEvent(string(types.EventResponseCompleted), map[string]interface{}{
			"type":            string(types.EventResponseCompleted),
			"sequence_number": t.nextSeq(),
			"response":        finalResp,
		})
	}
	return nil
}
