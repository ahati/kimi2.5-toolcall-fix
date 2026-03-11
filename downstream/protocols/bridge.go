// Package protocols provides protocol adapters for different API formats.
// Each adapter implements the ProtocolAdapter interface to handle request/response
// transformation for a specific API format (OpenAI, Anthropic, or Bridge).
package protocols

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"ai-proxy/config"
	"ai-proxy/logging"
	"ai-proxy/transform/toolcall"
	"ai-proxy/types"

	"github.com/gin-gonic/gin"
)

// BridgeAdapter implements ProtocolAdapter for Anthropic-to-OpenAI transformation.
// This adapter transforms Anthropic-format requests to OpenAI format for upstream,
// while preserving Anthropic-format responses for the client.
//
// @brief Adapter for Anthropic-to-OpenAI request transformation.
//
// @note Transforms request format from Anthropic to OpenAI.
// @note Response format remains Anthropic for client compatibility.
type BridgeAdapter struct{}

// NewBridgeAdapter creates a new Bridge adapter instance.
//
// @brief    Creates a new BridgeAdapter instance.
// @return   Pointer to newly created BridgeAdapter.
func NewBridgeAdapter() *BridgeAdapter {
	return &BridgeAdapter{}
}

// AnthropicToOpenAIRequest represents an Anthropic-format request.
// This is the input format received from Anthropic-compatible clients.
//
// @brief Input request structure in Anthropic format.
type AnthropicToOpenAIRequest struct {
	Model       string                    `json:"model"`
	Messages    []AnthropicMessageInput   `json:"messages"`
	MaxTokens   int                       `json:"max_tokens,omitempty"`
	Stream      bool                      `json:"stream,omitempty"`
	Tools       []AnthropicToolDefinition `json:"tools,omitempty"`
	System      interface{}               `json:"system,omitempty"`
	Temperature float64                   `json:"temperature,omitempty"`
	TopP        float64                   `json:"top_p,omitempty"`
}

// AnthropicMessageInput represents a single message in Anthropic format.
// Content can be a string or an array of content blocks.
//
// @brief Message structure in Anthropic format.
type AnthropicMessageInput struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}

// AnthropicToolDefinition represents a tool definition in Anthropic format.
// Tools have a name, description, and JSON schema for input parameters.
//
// @brief Tool definition structure in Anthropic format.
type AnthropicToolDefinition struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
}

// OpenAIRequest represents an OpenAI-format request.
// This is the output format sent to OpenAI-compatible upstream APIs.
//
// @brief Output request structure in OpenAI format.
type OpenAIRequest struct {
	Model       string          `json:"model"`
	Messages    []OpenAIMessage `json:"messages"`
	MaxTokens   int             `json:"max_tokens,omitempty"`
	Stream      bool            `json:"stream,omitempty"`
	Tools       []OpenAITool    `json:"tools,omitempty"`
	System      string          `json:"system,omitempty"`
	Temperature float64         `json:"temperature,omitempty"`
	TopP        float64         `json:"top_p,omitempty"`
}

// OpenAIMessage represents a single message in OpenAI format.
// May contain content, tool calls, or tool results.
//
// @brief Message structure in OpenAI format.
type OpenAIMessage struct {
	Role       string           `json:"role"`
	Content    interface{}      `json:"content,omitempty"`
	ToolCalls  []OpenAIToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
}

// OpenAITool represents a tool definition in OpenAI format.
// Contains the tool type and function definition.
//
// @brief Tool definition structure in OpenAI format.
type OpenAITool struct {
	Type     string         `json:"type"`
	Function OpenAIFunction `json:"function"`
}

// OpenAIFunction represents a function definition within a tool.
// Contains name, description, and JSON schema for parameters.
//
// @brief Function definition within an OpenAI tool.
type OpenAIFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// OpenAIToolCall represents a tool call in OpenAI format.
// Contains the tool call ID, type, index, and function details.
//
// @brief Tool call structure in OpenAI format.
type OpenAIToolCall struct {
	ID       string                 `json:"id"`
	Type     string                 `json:"type"`
	Index    int                    `json:"index"`
	Function OpenAIToolCallFunction `json:"function"`
}

// OpenAIToolCallFunction represents function details within a tool call.
// Contains the function name and arguments as a JSON string.
//
// @brief Function details within an OpenAI tool call.
type OpenAIToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// TransformRequest converts an Anthropic-format request to OpenAI format.
//
// @brief    Transforms Anthropic request to OpenAI format.
// @param    body The Anthropic-format request body.
// @return   Transformed OpenAI-format request body.
// @return   Error if JSON parsing fails.
//
// @note     Converts system messages, chat messages, and tool definitions.
// @note     Preserves model, max_tokens, stream, temperature, and top_p fields.
func (a *BridgeAdapter) TransformRequest(body []byte) ([]byte, error) {
	var anthReq AnthropicToOpenAIRequest
	if err := json.Unmarshal(body, &anthReq); err != nil {
		return nil, err
	}

	// Copy over fields that have identical semantics between Anthropic and OpenAI
	// These can be passed through without transformation
	openReq := OpenAIRequest{
		Model:       anthReq.Model,
		MaxTokens:   anthReq.MaxTokens,
		Stream:      anthReq.Stream,
		Temperature: anthReq.Temperature,
		TopP:        anthReq.TopP,
	}

	// Extract and set system prompt
	// Anthropic allows system as a top-level field, OpenAI may handle it differently
	if system := a.extractSystemMessage(anthReq.System); system != "" {
		openReq.System = system
	}

	// Convert each message from Anthropic's content block format to OpenAI's format
	// This handles text, tool_use, and tool_result content types
	for _, msg := range anthReq.Messages {
		openMsg := a.convertMessage(msg)
		openReq.Messages = append(openReq.Messages, openMsg)
	}

	// Convert tool definitions from Anthropic's input_schema to OpenAI's parameters
	// The JSON schema structure is the same, just the field name differs
	for _, tool := range anthReq.Tools {
		openTool := a.convertTool(tool)
		openReq.Tools = append(openReq.Tools, openTool)
	}

	return json.Marshal(openReq)
}

// extractSystemMessage extracts a string from the system field.
// The system field can be a string or an array of content blocks.
//
// @brief    Extracts system message string from Anthropic format.
// @param    system The system field value (string, array, or nil).
// @return   Extracted system message string, or empty string if nil.
//
// @note     Handles string and array of content block formats.
func (a *BridgeAdapter) extractSystemMessage(system interface{}) string {
	if system == nil {
		return ""
	}
	// Anthropic allows system as a simple string field
	if s, ok := system.(string); ok {
		return s
	}
	// Anthropic also supports system as an array of content blocks (e.g., with cache control)
	// OpenAI only accepts a single string, so we concatenate all text blocks
	if arr, ok := system.([]interface{}); ok {
		var content strings.Builder
		for _, item := range arr {
			if m, ok := item.(map[string]interface{}); ok {
				// Extract text from each content block, ignoring non-text types like cache_control
				if text, ok := m["text"].(string); ok {
					content.WriteString(text)
				}
			}
		}
		return content.String()
	}
	return ""
}

// convertMessage converts an Anthropic message to OpenAI format.
// Handles text content, tool use, and tool result content blocks.
//
// @brief    Converts Anthropic message to OpenAI format.
// @param    anthMsg The Anthropic-format message.
// @return   The converted OpenAI-format message.
//
// @note     Concatenates multiple text blocks with newlines.
// @note     Converts tool_use blocks to OpenAI tool_calls.
// @note     Extracts tool_use_id for tool_result blocks.
func (a *BridgeAdapter) convertMessage(anthMsg AnthropicMessageInput) OpenAIMessage {
	openMsg := OpenAIMessage{
		Role: anthMsg.Role,
	}

	switch content := anthMsg.Content.(type) {
	case string:
		// Simple string content - pass through directly
		openMsg.Content = content
	case []interface{}:
		// Anthropic content is an array of content blocks with different types
		// We need to aggregate text blocks and convert tool-related blocks
		var textContent strings.Builder
		var toolCalls []OpenAIToolCall
		var toolCallID string

		for _, item := range content {
			if m, ok := item.(map[string]interface{}); ok {
				switch m["type"] {
				case "text":
					// Concatenate multiple text blocks with newlines
					// Anthropic supports multiple text blocks in a single message
					if text, ok := m["text"].(string); ok {
						if textContent.Len() > 0 {
							textContent.WriteString("\n")
						}
						textContent.WriteString(text)
					}
				case "tool_use":
					// Convert Anthropic tool_use to OpenAI tool_calls format
					// Anthropic: {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
					// OpenAI: tool_calls array with nested function object
					if id, ok := m["id"].(string); ok {
						if name, ok := m["name"].(string); ok {
							// input is already an object, marshal to JSON string for OpenAI
							input, _ := json.Marshal(m["input"])
							toolCalls = append(toolCalls, OpenAIToolCall{
								ID:   id,
								Type: "function",
								Function: OpenAIToolCallFunction{
									Name:      name,
									Arguments: string(input),
								},
							})
						}
					}
				case "tool_result":
					// Extract tool_use_id for the tool response message
					// In OpenAI format, tool messages reference the original tool call via tool_call_id
					if id, ok := m["tool_use_id"].(string); ok {
						toolCallID = id
					}
				}
			}
		}

		// Only set fields that have values - omit empty fields
		if textContent.Len() > 0 {
			openMsg.Content = textContent.String()
		}
		if len(toolCalls) > 0 {
			openMsg.ToolCalls = toolCalls
		}
		if toolCallID != "" {
			openMsg.ToolCallID = toolCallID
		}
	}

	return openMsg
}

// convertTool converts an Anthropic tool definition to OpenAI format.
// Maps input_schema to parameters field.
//
// @brief    Converts Anthropic tool definition to OpenAI format.
// @param    anthTool The Anthropic-format tool definition.
// @return   The converted OpenAI-format tool definition.
func (a *BridgeAdapter) convertTool(anthTool AnthropicToolDefinition) OpenAITool {
	// Anthropic uses "input_schema" while OpenAI uses "parameters"
	// Both are JSON schemas, so we can pass through the raw JSON directly
	return OpenAITool{
		Type: "function",
		Function: OpenAIFunction{
			Name:        anthTool.Name,
			Description: anthTool.Description,
			Parameters:  anthTool.InputSchema,
		},
	}
}

// ValidateRequest checks if the request is a streaming request.
//
// @brief    Validates that the request is a streaming request.
// @param    body The request body to validate.
// @return   Error if request is non-streaming, nil otherwise.
//
// @note     Non-streaming requests are not supported.
func (a *BridgeAdapter) ValidateRequest(body []byte) error {
	if !a.IsStreamingRequest(body) {
		return ErrNonStreamingNotSupported
	}
	return nil
}

// CreateTransformer creates a tool call transformer for Anthropic output format.
// Uses Anthropic output because client expects Anthropic format responses.
//
// @brief    Creates transformer for Anthropic-format output.
// @param    w Writer for transformed output.
// @param    base Base stream chunk for context.
// @return   SSETransformer configured for Anthropic output.
func (a *BridgeAdapter) CreateTransformer(w io.Writer, base types.StreamChunk) SSETransformer {
	output := toolcall.NewAnthropicOutput(w, toolcall.ContextText, 0)
	return NewToolCallTransformer(w, base, output)
}

// UpstreamURL returns the configured OpenAI upstream URL.
// Bridge sends requests to OpenAI-compatible upstream.
//
// @brief    Gets the OpenAI upstream URL from configuration.
// @param    cfg Application configuration.
// @return   OpenAI upstream URL string.
func (a *BridgeAdapter) UpstreamURL(cfg *config.Config) string {
	return cfg.OpenAIUpstreamURL
}

// UpstreamAPIKey returns the configured OpenAI API key.
// Bridge uses OpenAI API key for upstream authentication.
//
// @brief    Gets the OpenAI API key from configuration.
// @param    cfg Application configuration.
// @return   OpenAI API key string.
func (a *BridgeAdapter) UpstreamAPIKey(cfg *config.Config) string {
	return cfg.OpenAIUpstreamAPIKey
}

// ForwardHeaders copies relevant headers from source to destination.
// Uses OpenAI-style header forwarding.
//
// @brief    Forwards protocol-specific headers to upstream.
// @param    src Source HTTP headers from client request.
// @param    dst Destination HTTP headers for upstream request.
//
// @note     Forwards headers with "X-" prefix and "Extra" header.
// @note     Also forwards Connection, Keep-Alive, Upgrade, and TE headers.
func (a *BridgeAdapter) ForwardHeaders(src, dst http.Header) {
	// Forward custom headers that may contain client-specific metadata or configuration
	// X- headers are commonly used for custom protocol extensions
	for k, v := range src {
		if strings.HasPrefix(k, "X-") || k == "Extra" {
			dst[k] = v
		}
	}
	// Forward connection-related headers for proper proxy behavior
	// These headers control HTTP connection handling and upgrade semantics
	for _, h := range []string{"Connection", "Keep-Alive", "Upgrade", "TE"} {
		if v := src.Get(h); v != "" {
			dst.Set(h, v)
		}
	}
}

// SendError sends an Anthropic-formatted error response.
// Uses Anthropic format because client expects Anthropic responses.
//
// @brief    Sends error response in Anthropic format.
// @param    c Gin context for the HTTP response.
// @param    status HTTP status code.
// @param    msg Error message.
//
// @note     Response format matches Anthropic error structure.
func (a *BridgeAdapter) SendError(c *gin.Context, status int, msg string) {
	logging.ErrorMsg("Bridge handler error: %s", msg)
	c.JSON(status, types.Error{
		Type: "error",
		Error: types.ErrorDetail{
			Type:    "invalid_request_error",
			Message: msg,
		},
	})
}

// IsStreamingRequest checks if the request body indicates streaming.
//
// @brief    Determines if request is a streaming request.
// @param    body The request body to check.
// @return   True if stream field is true, false otherwise.
//
// @note     Silently returns false if body cannot be parsed.
func (a *BridgeAdapter) IsStreamingRequest(body []byte) bool {
	var req struct {
		Stream bool `json:"stream"`
	}
	json.Unmarshal(body, &req)
	return req.Stream
}
