package downstream

import "encoding/json"

// Anthropic event types for streaming
type AnthropicEvent struct {
	Type         string            `json:"type"`
	Index        *int              `json:"index,omitempty"`
	Delta        json.RawMessage   `json:"delta,omitempty"`
	ContentBlock json.RawMessage   `json:"content_block,omitempty"`
	Message      *AnthropicMessage `json:"message,omitempty"`
	MessageUsage *AnthropicUsage   `json:"usage,omitempty"`
	StopReason   string            `json:"stop_reason,omitempty"`
	StopSequence *string           `json:"stop_sequence,omitempty"`
}

type AnthropicMessage struct {
	ID      string                  `json:"id"`
	Type    string                  `json:"type"`
	Role    string                  `json:"role"`
	Content []AnthropicContentBlock `json:"content"`
	Model   string                  `json:"model"`
	Usage   *AnthropicUsage         `json:"usage,omitempty"`
}

type AnthropicContentBlock struct {
	Type     string          `json:"type"`
	Text     string          `json:"text,omitempty"`
	ID       string          `json:"id,omitempty"`
	Name     string          `json:"name,omitempty"`
	Input    json.RawMessage `json:"input,omitempty"`
	Thinking string          `json:"thinking,omitempty"`
}

type AnthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// Delta types for content_block_delta events
type TextDelta struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type ThinkingDelta struct {
	Type     string `json:"type"`
	Thinking string `json:"thinking"`
}

type InputJSONDelta struct {
	Type        string `json:"type"`
	PartialJSON string `json:"partial_json"`
}

// Response wrapper for error handling
type AnthropicError struct {
	Type  string      `json:"type"`
	Error ErrorDetail `json:"error"`
}

type ErrorDetail struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}
