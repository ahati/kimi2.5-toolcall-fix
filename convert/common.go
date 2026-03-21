// Package convert provides shared helper functions for converting between
// OpenAI and Anthropic API formats.
package convert

import (
	"ai-proxy/logging"
	"ai-proxy/types"
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"strings"
)

// ConvertAnthropicMessagesToOpenAI converts Anthropic messages to OpenAI format.
// Each message's content is converted from Anthropic's content blocks to OpenAI's format.
// Handles content as either a string or array of content blocks.
//
// For assistant messages with tool_use blocks, ToolCalls are populated.
// For user messages with tool_result blocks, ToolCallID is populated.
func ConvertAnthropicMessagesToOpenAI(anthMsgs []types.MessageInput) []types.Message {
	if len(anthMsgs) == 0 {
		return []types.Message{}
	}

	openMsgs := make([]types.Message, 0, len(anthMsgs))
	for _, anthMsg := range anthMsgs {
		openMsgs = append(openMsgs, convertAnthropicMessage(anthMsg))
	}
	return openMsgs
}

// convertAnthropicMessage transforms a single Anthropic message to OpenAI format.
func convertAnthropicMessage(anthMsg types.MessageInput) types.Message {
	openMsg := types.Message{Role: anthMsg.Role}

	switch content := anthMsg.Content.(type) {
	case string:
		openMsg.Content = content
	case []interface{}:
		openMsg.Content, openMsg.ToolCalls, openMsg.ToolCallID = ConvertContentBlocks(content)
	}

	return openMsg
}

// ConvertOpenAIMessagesToAnthropic converts OpenAI messages to Anthropic format.
// Each message's content is converted to Anthropic's content blocks format.
// Handles tool calls and tool results appropriately.
//
// Consecutive tool messages are batched into a single user message with multiple
// tool_result blocks, per Anthropic's message format requirements.
func ConvertOpenAIMessagesToAnthropic(openMsgs []types.Message) []types.MessageInput {
	if len(openMsgs) == 0 {
		return []types.MessageInput{}
	}

	anthMsgs := make([]types.MessageInput, 0, len(openMsgs))
	for _, openMsg := range openMsgs {
		// Handle tool messages with batching
		if openMsg.Role == "tool" && openMsg.ToolCallID != "" {
			// Check if previous message is a batched tool result message
			if len(anthMsgs) > 0 && isToolResultMessage(anthMsgs[len(anthMsgs)-1]) {
				// Append to the existing tool result message
				appendToolResultToMessage(&anthMsgs[len(anthMsgs)-1], openMsg)
			} else {
				// Create a new user message with tool_result
				anthMsgs = append(anthMsgs, createToolResultMessage(openMsg))
			}
			continue
		}

		// Non-tool messages are converted normally
		anthMsgs = append(anthMsgs, convertOpenAIMessage(openMsg))
	}
	return anthMsgs
}

// isToolResultMessage checks if an Anthropic message is a batched tool result message.
// A tool result message is a user message with content containing tool_result blocks.
func isToolResultMessage(anthMsg types.MessageInput) bool {
	if anthMsg.Role != "user" {
		return false
	}

	blocks, ok := anthMsg.Content.([]interface{})
	if !ok || len(blocks) == 0 {
		return false
	}

	// Check if first block is a tool_result
	if firstBlock, ok := blocks[0].(map[string]interface{}); ok {
		if blockType, ok := firstBlock["type"].(string); ok {
			return blockType == "tool_result"
		}
	}
	return false
}

// appendToolResultToMessage appends a tool result to an existing batched tool result message.
func appendToolResultToMessage(anthMsg *types.MessageInput, openMsg types.Message) {
	blocks, ok := anthMsg.Content.([]interface{})
	if !ok {
		blocks = []interface{}{}
	}

	newBlock := map[string]interface{}{
		"type":        "tool_result",
		"tool_use_id": openMsg.ToolCallID,
		"content":     FlattenToolResultContent(openMsg.Content),
	}

	anthMsg.Content = append(blocks, newBlock)
}

// createToolResultMessage creates a new user message with a single tool_result block.
func createToolResultMessage(openMsg types.Message) types.MessageInput {
	return types.MessageInput{
		Role: "user",
		Content: []interface{}{
			map[string]interface{}{
				"type":        "tool_result",
				"tool_use_id": openMsg.ToolCallID,
				"content":     FlattenToolResultContent(openMsg.Content),
			},
		},
	}
}

// convertOpenAIMessage transforms a single OpenAI message to Anthropic format.
func convertOpenAIMessage(openMsg types.Message) types.MessageInput {
	anthMsg := types.MessageInput{Role: openMsg.Role}

	// Handle tool response messages
	if openMsg.Role == "tool" && openMsg.ToolCallID != "" {
		anthMsg.Role = "user"
		anthMsg.Content = []map[string]interface{}{
			{
				"type":        "tool_result",
				"tool_use_id": openMsg.ToolCallID,
				"content":     FlattenToolResultContent(openMsg.Content),
			},
		}
		return anthMsg
	}

	// Handle assistant messages with tool calls
	if len(openMsg.ToolCalls) > 0 {
		blocks := []interface{}{}

		// Add text content first if present
		if text := ExtractTextFromContent(openMsg.Content); text != "" {
			blocks = append(blocks, map[string]interface{}{
				"type": "text",
				"text": text,
			})
		}

		// Add tool_use blocks
		for _, tc := range openMsg.ToolCalls {
			blocks = append(blocks, map[string]interface{}{
				"type":  "tool_use",
				"id":    tc.ID,
				"name":  tc.Function.Name,
				"input": unmarshalArgs(tc.Function.Arguments),
			})
		}
		anthMsg.Content = blocks
		return anthMsg
	}

	// Handle regular content
	switch c := openMsg.Content.(type) {
	case string:
		anthMsg.Content = c
	case []interface{}:
		// Convert OpenAI content parts to Anthropic format
		blocks := []interface{}{}
		for _, part := range c {
			if partMap, ok := part.(map[string]interface{}); ok {
				partType, _ := partMap["type"].(string)
				switch partType {
				case "text":
					if text, ok := partMap["text"].(string); ok {
						blocks = append(blocks, map[string]interface{}{
							"type": "text",
							"text": text,
						})
					}
				case "image_url":
					// Convert OpenAI image_url to Anthropic image format
					if imageURL, ok := partMap["image_url"].(map[string]interface{}); ok {
						if url, ok := imageURL["url"].(string); ok {
							if imageBlock, err := BuildAnthropicImageBlockFromURL(url); err == nil {
								blocks = append(blocks, imageBlock)
							} else {
								logging.DebugMsg("Skipping unsupported image_url value: %v", err)
							}
						}
					}
				}
			}
		}
		if len(blocks) > 0 {
			anthMsg.Content = blocks
		} else {
			anthMsg.Content = ""
		}
	default:
		anthMsg.Content = ""
	}

	return anthMsg
}

// unmarshalArgs parses a JSON arguments string into a map.
func unmarshalArgs(args string) map[string]interface{} {
	var result map[string]interface{}
	if args == "" {
		return result
	}
	if err := json.Unmarshal([]byte(args), &result); err != nil {
		return map[string]interface{}{}
	}
	return result
}

// FlattenToolResultContent converts a tool_result content payload into a string.
//
// It preserves plain text and array text blocks, discards non-text content, and
// returns an empty string for nil or unsupported values.
func FlattenToolResultContent(content interface{}) string {
	return extractTextFromContentValue(content, false, "\n")
}

// ConvertAnthropicToolsToOpenAI converts Anthropic tool definitions to OpenAI format.
// Anthropic uses "input_schema" while OpenAI uses "parameters".
func ConvertAnthropicToolsToOpenAI(anthTools []types.ToolDef) []types.Tool {
	if len(anthTools) == 0 {
		return []types.Tool{}
	}

	openTools := make([]types.Tool, 0, len(anthTools))
	for _, anthTool := range anthTools {
		openTools = append(openTools, types.Tool{
			Type: "function",
			Function: types.ToolFunction{
				Name:        anthTool.Name,
				Description: anthTool.Description,
				Parameters:  anthTool.InputSchema,
			},
		})
	}
	return openTools
}

// ConvertOpenAIToolsToAnthropic converts OpenAI tool definitions to Anthropic format.
// OpenAI uses "parameters" while Anthropic uses "input_schema".
func ConvertOpenAIToolsToAnthropic(openTools []types.Tool) []types.ToolDef {
	if len(openTools) == 0 {
		return []types.ToolDef{}
	}

	anthTools := make([]types.ToolDef, 0, len(openTools))
	for _, openTool := range openTools {
		if openTool.Type == "function" {
			anthTools = append(anthTools, types.ToolDef{
				Name:        openTool.Function.Name,
				Description: openTool.Function.Description,
				InputSchema: openTool.Function.Parameters,
			})
		}
	}
	return anthTools
}

// ExtractTextFromContent extracts text from various content formats.
// This is the SINGLE source of truth for text extraction across the codebase.
// All other implementations should call this function.
//
// Content can be:
//   - string: returned directly
//   - []interface{}: array of content blocks (Anthropic/OpenAI format)
//   - []map[string]interface{}: typed content blocks
//   - nil: returns empty string
//
// Supported content block types:
//   - "text", "input_text", "output_text", "refusal": extracts "text" field
//   - "thinking": extracts "thinking" field
func ExtractTextFromContent(content interface{}) string {
	return extractTextFromContentValue(content, true, "\n")
}

// ExtractSystemText extracts system content from Anthropic system payloads.
//
// It accepts the string and block-array forms used by Anthropic and returns a
// concatenated string, ignoring non-text blocks.
func ExtractSystemText(system interface{}) string {
	return extractTextFromContentValue(system, false, "")
}

func extractTextFromContentValue(content interface{}, includeThinking bool, separator string) string {
	if content == nil {
		return ""
	}

	switch c := content.(type) {
	case string:
		return c
	case json.RawMessage:
		return extractTextFromRawMessage(c, includeThinking, separator)
	case []interface{}:
		return extractTextFromInterfaceSlice(c, includeThinking, separator)
	case []map[string]interface{}:
		return extractTextFromMapSlice(c, includeThinking, separator)
	case []types.ContentBlock:
		return extractTextFromContentBlocks(c, includeThinking, separator)
	case types.ContentBlock:
		return extractTextFromContentBlock(c, includeThinking)
	case []types.SystemBlock:
		return extractTextFromSystemBlocks(c, separator)
	case types.SystemBlock:
		return c.Text
	default:
		return ""
	}
}

func extractTextFromRawMessage(data json.RawMessage, includeThinking bool, separator string) string {
	if len(data) == 0 {
		return ""
	}

	var stringValue string
	if err := json.Unmarshal(data, &stringValue); err == nil {
		return stringValue
	}

	var interfaceSlice []interface{}
	if err := json.Unmarshal(data, &interfaceSlice); err == nil {
		return extractTextFromInterfaceSlice(interfaceSlice, includeThinking, separator)
	}

	var mapSlice []map[string]interface{}
	if err := json.Unmarshal(data, &mapSlice); err == nil {
		return extractTextFromMapSlice(mapSlice, includeThinking, separator)
	}

	return ""
}

// extractTextFromInterfaceSlice extracts text from a slice of interface{} (untyped content blocks).
func extractTextFromInterfaceSlice(blocks []interface{}, includeThinking bool, separator string) string {
	var result strings.Builder
	for _, part := range blocks {
		if partMap, ok := part.(map[string]interface{}); ok {
			text := extractTextFromBlock(partMap, includeThinking)
			if text != "" {
				appendExtractedText(&result, text, separator)
			}
			continue
		}

		if contentBlock, ok := part.(types.ContentBlock); ok {
			text := extractTextFromContentBlock(contentBlock, includeThinking)
			if text != "" {
				appendExtractedText(&result, text, separator)
			}
		}
	}
	return result.String()
}

// extractTextFromMapSlice extracts text from a slice of map[string]interface{} (typed content blocks).
func extractTextFromMapSlice(blocks []map[string]interface{}, includeThinking bool, separator string) string {
	var result strings.Builder
	for _, part := range blocks {
		text := extractTextFromBlock(part, includeThinking)
		if text != "" {
			appendExtractedText(&result, text, separator)
		}
	}
	return result.String()
}

func extractTextFromContentBlocks(blocks []types.ContentBlock, includeThinking bool, separator string) string {
	var result strings.Builder
	for _, part := range blocks {
		text := extractTextFromContentBlock(part, includeThinking)
		if text != "" {
			appendExtractedText(&result, text, separator)
		}
	}
	return result.String()
}

func extractTextFromSystemBlocks(blocks []types.SystemBlock, separator string) string {
	var result strings.Builder
	for _, part := range blocks {
		if part.Text != "" {
			appendExtractedText(&result, part.Text, separator)
		}
	}
	return result.String()
}

func extractTextFromContentBlock(block types.ContentBlock, includeThinking bool) string {
	switch block.Type {
	case "text", "input_text", "output_text", "refusal":
		return block.Text
	case "thinking":
		if includeThinking {
			return block.Thinking
		}
	}

	return ""
}

func appendExtractedText(builder *strings.Builder, text, separator string) {
	if text == "" {
		return
	}
	if builder.Len() > 0 {
		builder.WriteString(separator)
	}
	builder.WriteString(text)
}

// extractTextFromBlock extracts text from a single content block based on its type.
func extractTextFromBlock(block map[string]interface{}, includeThinking bool) string {
	partType, _ := block["type"].(string)
	switch partType {
	case "text", "input_text", "output_text":
		if text, ok := block["text"].(string); ok {
			return text
		}
	case "refusal":
		if text, ok := block["text"].(string); ok && text != "" {
			return text
		}
		if refusal, ok := block["refusal"].(string); ok {
			return refusal
		}
	case "thinking":
		if includeThinking {
			if thinking, ok := block["thinking"].(string); ok {
				return thinking
			}
		}
	default:
		if partType != "" {
			logging.DebugMsg("Unknown content block type: %s", partType)
		}
	}
	return ""
}

// BuildAnthropicImageBlockFromURL converts an OpenAI/Responses image URL into
// an Anthropic image content block. Data URIs become base64 image sources;
// regular URLs are preserved as URL sources.
func BuildAnthropicImageBlockFromURL(url string) (map[string]interface{}, error) {
	source, err := OpenAIImageURLToAnthropicSource(url)
	if err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"type":   "image",
		"source": anthropicImageSourceToMap(source),
	}, nil
}

// OpenAIImageURLToAnthropicSource converts an OpenAI/Responses image URL into an
// Anthropic image source, supporting both data URIs and regular URLs.
func OpenAIImageURLToAnthropicSource(url string) (*types.ImageSource, error) {
	if url == "" {
		return nil, fmt.Errorf("image url is empty")
	}

	if strings.HasPrefix(url, "data:") {
		mediaType, data, err := ParseDataURI(url)
		if err != nil {
			return nil, err
		}
		return &types.ImageSource{
			Type:      "base64",
			MediaType: mediaType,
			Data:      data,
		}, nil
	}

	return &types.ImageSource{
		Type: "url",
		URL:  url,
	}, nil
}

// AnthropicImageSourceToURL converts an Anthropic image source into a regular
// image URL or data URI suitable for OpenAI/Responses image parts.
func AnthropicImageSourceToURL(source *types.ImageSource) (string, error) {
	if source == nil {
		return "", fmt.Errorf("image source is nil")
	}

	switch source.Type {
	case "base64":
		return BuildDataURI(source.MediaType, source.Data), nil
	case "url":
		return source.URL, nil
	default:
		return "", fmt.Errorf("unsupported image source type: %q", source.Type)
	}
}

// BuildChatImagePartFromAnthropicSource builds an OpenAI Chat image_url part
// from an Anthropic image source.
func BuildChatImagePartFromAnthropicSource(source *types.ImageSource) (map[string]interface{}, error) {
	url, err := AnthropicImageSourceToURL(source)
	if err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"type": "image_url",
		"image_url": map[string]interface{}{
			"url": url,
		},
	}, nil
}

// BuildResponsesImagePartFromAnthropicSource builds an OpenAI Responses image
// content part from an Anthropic image source.
func BuildResponsesImagePartFromAnthropicSource(source *types.ImageSource) (types.ContentPart, error) {
	url, err := AnthropicImageSourceToURL(source)
	if err != nil {
		return types.ContentPart{}, err
	}

	return types.ContentPart{
		Type:     "input_image",
		ImageURL: url,
	}, nil
}

func anthropicImageSourceToMap(source *types.ImageSource) map[string]interface{} {
	if source == nil {
		return nil
	}

	result := map[string]interface{}{
		"type": source.Type,
	}

	switch source.Type {
	case "base64":
		result["media_type"] = source.MediaType
		result["data"] = source.Data
	case "url":
		result["url"] = source.URL
	}

	return result
}

// ConvertContentBlocks converts Anthropic content blocks to OpenAI format.
// Returns: text content, tool calls, and tool_call_id.
//
// Anthropic content blocks can be:
//   - text: Simple text content
//   - tool_use: Tool call request (converted to OpenAI tool_calls)
//   - tool_result: Tool execution result (converted to OpenAI tool_call_id)
func ConvertContentBlocks(blocks []interface{}) (string, []types.ToolCall, string) {
	var textContent strings.Builder
	var toolCalls []types.ToolCall
	var toolCallID string

	for _, item := range blocks {
		m, ok := item.(map[string]interface{})
		if !ok {
			continue
		}

		switch m["type"] {
		case "text":
			if text, ok := m["text"].(string); ok {
				if textContent.Len() > 0 {
					textContent.WriteString("\n")
				}
				textContent.WriteString(text)
			}
		case "thinking":
			// Drop thinking blocks when converting away from Anthropic.
		case "tool_use":
			if id, ok := m["id"].(string); ok {
				if name, ok := m["name"].(string); ok {
					input, _ := json.Marshal(m["input"])
					toolCalls = append(toolCalls, types.ToolCall{
						ID:   id,
						Type: "function",
						Function: types.Function{
							Name:      name,
							Arguments: string(input),
						},
					})
				}
			}
		case "tool_result":
			if id, ok := m["tool_use_id"].(string); ok {
				toolCallID = id
			}
			flattened := FlattenToolResultContent(m["content"])
			if flattened != "" {
				if textContent.Len() > 0 {
					textContent.WriteString("\n")
				}
				textContent.WriteString(flattened)
			}
		}
	}

	return textContent.String(), toolCalls, toolCallID
}

// ExtractSystemMessage extracts a system message string from various formats.
// System can be: string, []interface{} (content blocks), or nil.
// Returns an empty string if no system content is found.
func ExtractSystemMessage(system interface{}) string {
	return ExtractSystemText(system)
}

// extractMediaType extracts the media type from a data URL.
// e.g., "data:image/png;base64,abc" -> "image/png"
func extractMediaType(dataURL string) string {
	if !strings.HasPrefix(dataURL, "data:") {
		return ""
	}
	// Remove "data:" prefix
	rest := dataURL[5:]
	// Find the semicolon or comma that ends the media type
	idx := strings.Index(rest, ";")
	if idx == -1 {
		idx = strings.Index(rest, ",")
	}
	if idx == -1 {
		return ""
	}
	return rest[:idx]
}

// extractBase64Data extracts the base64 data from a data URL.
// e.g., "data:image/png;base64,abc" -> "abc"
func extractBase64Data(dataURL string) string {
	idx := strings.Index(dataURL, ",")
	if idx == -1 {
		return ""
	}
	return dataURL[idx+1:]
}

// ─────────────────────────────────────────────────────────────────────────────
// Stop Reason / Finish Reason Mapping
// ─────────────────────────────────────────────────────────────────────────────

// StopReasonToFinishReason maps Anthropic stop_reason to Chat Completions finish_reason.
var StopReasonToFinishReason = map[string]string{
	"end_turn":      "stop",
	"max_tokens":    "length",
	"tool_use":      "tool_calls",
	"stop_sequence": "stop",
}

// FinishReasonToStopReason maps Chat Completions finish_reason to Anthropic stop_reason.
var FinishReasonToStopReason = map[string]string{
	"stop":           "end_turn",
	"length":         "max_tokens",
	"tool_calls":     "tool_use",
	"content_filter": "end_turn", // best approximation
}

// MapStopReason converts an Anthropic stop_reason to a Chat Completions finish_reason.
func MapStopReason(reason string) string {
	if v, ok := StopReasonToFinishReason[reason]; ok {
		return v
	}
	return "stop"
}

// MapFinishReason converts a Chat Completions finish_reason to an Anthropic stop_reason.
func MapFinishReason(reason string) string {
	if v, ok := FinishReasonToStopReason[reason]; ok {
		return v
	}
	return "end_turn"
}

// ─────────────────────────────────────────────────────────────────────────────
// Temperature Clamping
// ─────────────────────────────────────────────────────────────────────────────

// ClampTemperatureToAnthropic clamps temperature from OpenAI range (0-2) to Anthropic range (0-1).
func ClampTemperatureToAnthropic(t float64) float64 {
	return math.Min(math.Max(t, 0), 1.0)
}

// NormalizeAnthropicTemperature normalizes a temperature value for Anthropic-bound requests.
func NormalizeAnthropicTemperature(t float64) float64 {
	return ClampTemperatureToAnthropic(t)
}

// ─────────────────────────────────────────────────────────────────────────────
// Data URI Helpers
// ─────────────────────────────────────────────────────────────────────────────

// ParseDataURI parses a data URI into media type and base64 data.
// Expected format: "data:<mediaType>;base64,<data>"
// Returns mediaType, data, error.
func ParseDataURI(uri string) (string, string, error) {
	if !strings.HasPrefix(uri, "data:") {
		return "", "", fmt.Errorf("not a data URI: %q", uri)
	}
	rest := strings.TrimPrefix(uri, "data:")
	semi := strings.Index(rest, ";")
	if semi < 0 {
		return "", "", fmt.Errorf("malformed data URI (missing semicolon): %q", uri)
	}
	mediaType := rest[:semi]
	rest = rest[semi+1:]
	if !strings.HasPrefix(rest, "base64,") {
		return "", "", fmt.Errorf("only base64 data URIs are supported: %q", uri)
	}
	data := strings.TrimPrefix(rest, "base64,")
	return mediaType, data, nil
}

// BuildDataURI constructs a data URI from media type and base64 data.
func BuildDataURI(mediaType, data string) string {
	return fmt.Sprintf("data:%s;base64,%s", mediaType, data)
}

// ─────────────────────────────────────────────────────────────────────────────
// Role Alternation (Anthropic Requirement)
// ─────────────────────────────────────────────────────────────────────────────

// EnsureAlternatingRoles ensures messages alternate between user and assistant roles.
// Anthropic requires strict alternation. If consecutive messages have the same role,
// a filler message is inserted between them.
func EnsureAlternatingRoles(messages []types.MessageInput) []types.MessageInput {
	return NormalizeAnthropicMessages(messages)
}

// NormalizeAnthropicMessages ensures Anthropic message ordering starts with a
// user message and strictly alternates between user and assistant.
func NormalizeAnthropicMessages(messages []types.MessageInput) []types.MessageInput {
	if len(messages) == 0 {
		return messages
	}

	result := make([]types.MessageInput, 0, len(messages)+2)
	if messages[0].Role == "assistant" {
		result = append(result, emptyAnthropicUserMessage())
	}

	for _, msg := range messages {
		if len(result) > 0 && result[len(result)-1].Role == msg.Role {
			result = append(result, fillerAnthropicMessage(msg.Role))
		}
		result = append(result, msg)
	}

	return result
}

func emptyAnthropicUserMessage() types.MessageInput {
	return types.MessageInput{
		Role:    "user",
		Content: []interface{}{},
	}
}

func fillerAnthropicMessage(role string) types.MessageInput {
	if role == "assistant" {
		return types.MessageInput{
			Role:    "user",
			Content: []interface{}{},
		}
	}

	return types.MessageInput{
		Role:    "assistant",
		Content: []interface{}{},
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// SSE Utilities
// ─────────────────────────────────────────────────────────────────────────────

// SSEEvent represents a parsed Server-Sent Event.
type SSEEvent struct {
	Name string // event: line value (empty for Chat Completions chunks)
	Data string // data: line value (with "data: " prefix stripped)
}

// ReadSSEEvents reads SSE events from r, emitting them on the returned channel.
// The error channel receives at most one value and is always closed after the
// reader is exhausted or an error occurs.
//
// Each SSEEvent.Data already has the "data: " prefix stripped.
// The Chat Completions sentinel "data: [DONE]" is returned as-is; callers
// should check for it explicitly.
func ReadSSEEvents(r io.Reader) (<-chan SSEEvent, <-chan error) {
	events := make(chan SSEEvent, 16)
	errs := make(chan error, 1)

	go func() {
		defer close(events)
		defer close(errs)

		scanner := bufio.NewScanner(r)
		scanner.Buffer(make([]byte, 64*1024), 64*1024)

		var eventName string
		var dataLines []string

		flush := func() {
			if len(dataLines) == 0 {
				return
			}
			events <- SSEEvent{
				Name: eventName,
				Data: strings.Join(dataLines, "\n"),
			}
			eventName = ""
			dataLines = dataLines[:0]
		}

		for scanner.Scan() {
			line := scanner.Text()
			switch {
			case line == "":
				flush()
			case strings.HasPrefix(line, "event:"):
				eventName = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
			case strings.HasPrefix(line, "data:"):
				dataLines = append(dataLines, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
			case strings.HasPrefix(line, ":"):
				// SSE comment — ignore (e.g. Anthropic ping keep-alives).
			}
		}
		flush() // emit any trailing event

		if err := scanner.Err(); err != nil {
			errs <- err
		}
	}()

	return events, errs
}

// WriteSSEEvent formats and writes a named SSE event to w.
// For Chat Completions-style (unnamed) events, pass name as "".
func WriteSSEEvent(w io.Writer, name, data string) error {
	if name != "" {
		if _, err := fmt.Fprintf(w, "event: %s\n", name); err != nil {
			return err
		}
	}
	_, err := fmt.Fprintf(w, "data: %s\n\n", data)
	return err
}

// WriteSSEDone writes the Chat Completions sentinel "data: [DONE]\n\n".
func WriteSSEDone(w io.Writer) error {
	_, err := fmt.Fprint(w, "data: [DONE]\n\n")
	return err
}

// MarshalSSEData serializes v to JSON and returns the string form.
func MarshalSSEData(v interface{}) (string, error) {
	b, err := json.Marshal(v)
	return string(b), err
}

// ─────────────────────────────────────────────────────────────────────────────
// JSON Helpers
// ─────────────────────────────────────────────────────────────────────────────

// MustMarshal marshals v to JSON and panics on error (for use in tests/examples).
func MustMarshal(v interface{}) json.RawMessage {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return b
}

// IsJSONString reports whether raw starts with a JSON string ('"').
func IsJSONString(raw json.RawMessage) bool {
	return len(raw) > 0 && raw[0] == '"'
}

// IsJSONArray reports whether raw starts with a JSON array ('[').
func IsJSONArray(raw json.RawMessage) bool {
	return len(raw) > 0 && raw[0] == '['
}

// ─────────────────────────────────────────────────────────────────────────────
// Tool-Choice Converters
// ─────────────────────────────────────────────────────────────────────────────

// AnthropicToolChoiceObject is the object form of Anthropic tool_choice.
type AnthropicToolChoiceObject struct {
	Type string `json:"type"` // "auto" | "any" | "tool"
	Name string `json:"name,omitempty"`
}

// ChatToolChoiceObject is the object form of Chat Completions tool_choice.
type ChatToolChoiceObject struct {
	Type     string `json:"type"` // "function"
	Function struct {
		Name string `json:"name"`
	} `json:"function"`
}

// ResponsesToolChoiceObject is the object form of Responses API tool_choice.
type ResponsesToolChoiceObject struct {
	Type string `json:"type"` // "function"
	Name string `json:"name"`
}

// AnthropicToolChoiceToChat converts an Anthropic tool_choice into Chat Completions form.
func AnthropicToolChoiceToChat(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return nil
	}
	if IsJSONString(raw) {
		var s string
		_ = json.Unmarshal(raw, &s)
		switch s {
		case "auto":
			return MustMarshal("auto")
		case "any":
			return MustMarshal("required")
		}
		return MustMarshal("auto")
	}
	var obj AnthropicToolChoiceObject
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil
	}
	out := ChatToolChoiceObject{Type: "function"}
	out.Function.Name = obj.Name
	return MustMarshal(out)
}

// ChatToolChoiceToAnthropic converts a Chat Completions tool_choice into Anthropic form.
// Returns nil for "none" — caller must also remove tools from the request.
func ChatToolChoiceToAnthropic(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return nil
	}
	if IsJSONString(raw) {
		var s string
		_ = json.Unmarshal(raw, &s)
		switch s {
		case "auto":
			return MustMarshal("auto")
		case "required":
			return MustMarshal("any")
		case "none":
			return nil // caller must drop tools
		}
		return MustMarshal("auto")
	}
	var obj ChatToolChoiceObject
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil
	}
	return MustMarshal(AnthropicToolChoiceObject{Type: "tool", Name: obj.Function.Name})
}

// AnthropicToolChoiceToResponses converts an Anthropic tool_choice into Responses API form.
func AnthropicToolChoiceToResponses(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return nil
	}
	if IsJSONString(raw) {
		var s string
		_ = json.Unmarshal(raw, &s)
		switch s {
		case "auto":
			return MustMarshal("auto")
		case "any":
			return MustMarshal("required")
		}
		return MustMarshal("auto")
	}
	var obj AnthropicToolChoiceObject
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil
	}
	return MustMarshal(ResponsesToolChoiceObject{Type: "function", Name: obj.Name})
}

// ResponsesToolChoiceToChat converts a Responses API tool_choice into Chat Completions form.
func ResponsesToolChoiceToChat(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return nil
	}
	if IsJSONString(raw) {
		// "auto", "none", "required" all map 1-to-1.
		return raw
	}
	var obj ResponsesToolChoiceObject
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil
	}
	out := ChatToolChoiceObject{Type: "function"}
	out.Function.Name = obj.Name
	return MustMarshal(out)
}

// ChatToolChoiceToResponses converts a Chat Completions tool_choice to Responses API form.
func ChatToolChoiceToResponses(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return nil
	}
	if IsJSONString(raw) {
		// "auto", "none", "required" all map directly.
		return raw
	}
	var obj ChatToolChoiceObject
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil
	}
	return MustMarshal(ResponsesToolChoiceObject{Type: "function", Name: obj.Function.Name})
}

// ─────────────────────────────────────────────────────────────────────────────
// Reasoning / Thinking Budget Mapping
// ─────────────────────────────────────────────────────────────────────────────

// ReasoningEffortToBudget maps a Responses API reasoning effort string to an
// Anthropic thinking budget_tokens value.
func ReasoningEffortToBudget(effort string) int {
	switch effort {
	case "low":
		return 4000
	case "medium":
		return 8000
	case "high":
		return 16000
	default:
		return 8000
	}
}

// BudgetToReasoningEffort maps an Anthropic thinking budget_tokens value to a
// Responses API reasoning effort string.
func BudgetToReasoningEffort(budget int) string {
	switch {
	case budget <= 4000:
		return "low"
	case budget <= 10000:
		return "medium"
	default:
		return "high"
	}
}
