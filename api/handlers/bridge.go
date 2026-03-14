package handlers

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"ai-proxy/config"
	"ai-proxy/transform"
	"ai-proxy/transform/toolcall"
	"ai-proxy/types"

	"github.com/gin-gonic/gin"
)

// BridgeHandler converts Anthropic-format requests to OpenAI format
// and proxies to an OpenAI-compatible upstream.
// It implements the Handler interface for the /v1/openai-to-anthropic/messages endpoint.
//
// This handler:
//   - Accepts requests in Anthropic Messages format
//   - Transforms requests to OpenAI ChatCompletion format
//   - Forwards to OpenAI-compatible upstream API
//   - Transforms responses back to Anthropic format for the client
//
// @note This enables clients using Anthropic SDK to call OpenAI-compatible APIs.
type BridgeHandler struct {
	// cfg contains the application configuration including upstream URL and API key.
	// Must not be nil after construction.
	cfg *config.Config
	// model stores the model name from the request for use in CreateTransformer.
	model string
}

// NewBridgeHandler creates a Gin handler for the /v1/openai-to-anthropic/messages endpoint.
//
// @param cfg - Application configuration. Must not be nil.
// @return Gin handler function that processes bridge requests.
//
// @pre cfg != nil
// @pre cfg.OpenAIUpstreamURL != ""
// @pre cfg.OpenAIUpstreamAPIKey != "" (or valid Authorization header expected)
func NewBridgeHandler(cfg *config.Config) gin.HandlerFunc {
	return Handle(&BridgeHandler{cfg: cfg})
}

// ValidateRequest performs no additional validation for bridge requests.
// The Anthropic format request will be validated during transformation.
//
// @param body - Raw request body bytes (unused).
// @return Always returns nil (no validation performed).
//
// @note Validation happens during TransformRequest to avoid double parsing.
func (h *BridgeHandler) ValidateRequest(body []byte) error {
	// No validation - transformation will validate during JSON parsing
	return nil
}

// TransformRequest converts an Anthropic-format request body to OpenAI format.
// This is the core bridging logic that translates between API formats.
//
// @param body - Raw request body in Anthropic Messages format.
// @return Transformed body in OpenAI ChatCompletion format.
// @return Error if JSON parsing or transformation fails.
//
// @pre body is valid JSON.
// @post Returned body is valid OpenAI ChatCompletion format.
func (h *BridgeHandler) TransformRequest(body []byte) ([]byte, error) {
	// Extract model from request body for later use in CreateTransformer
	var req struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &req); err == nil {
		h.model = req.Model
	}
	return transformRequest(body)
}

// UpstreamURL returns the OpenAI-compatible upstream API URL.
// The bridge sends requests to an OpenAI-compatible API after transformation.
//
// @return URL string from configuration for the OpenAI-compatible upstream.
//
// @pre h.cfg != nil
func (h *BridgeHandler) UpstreamURL() string {
	return h.cfg.OpenAIUpstreamURL
}

// ResolveAPIKey returns the configured OpenAI upstream API key.
// The bridge uses OpenAI authentication after transforming the request.
//
// @param c - Gin context (unused in this implementation).
// @return API key string from configuration.
//
// @pre h.cfg != nil
func (h *BridgeHandler) ResolveAPIKey(c *gin.Context) string {
	return h.cfg.OpenAIUpstreamAPIKey
}

// ForwardHeaders copies X-* and Extra headers to the upstream request.
// Custom headers are forwarded but Anthropic-specific headers are not.
//
// @param c - Gin context containing the original request headers.
// @param req - Upstream request to receive forwarded headers.
//
// @pre c != nil, req != nil
// @post X-* headers are copied to upstream request.
// @post Extra header is copied if present.
// @note Anthropic-Version and Anthropic-Beta headers are NOT forwarded
// as they are specific to Anthropic's API.
func (h *BridgeHandler) ForwardHeaders(c *gin.Context, req *http.Request) {
	// Forward custom headers that may be used by the OpenAI-compatible upstream
	for k, v := range c.Request.Header {
		// Only forward X-* headers and Extra header
		// Do NOT forward Anthropic-specific headers as they may confuse OpenAI API
		if strings.HasPrefix(k, "X-") || k == "Extra" {
			req.Header[k] = v
		}
	}
}

// CreateTransformer builds an Anthropic SSE transformer to convert
// OpenAI responses back to Anthropic format.
// This ensures the client receives Anthropic-format responses.
//
// @param w - Writer to receive transformed output.
// @return Transformer for converting OpenAI SSE to Anthropic format.
//
// @pre w != nil and ready to receive writes.
// @post Caller must call Close() on returned transformer.
func (h *BridgeHandler) CreateTransformer(w io.Writer, model string) transform.SSETransformer {
	// Use the model from the request body (set during TransformRequest)
	return toolcall.NewAnthropicTransformer(w, h.model)
}

// WriteError sends an error response in Anthropic format.
// Maintains consistency with the expected Anthropic API response format.
//
// @param c - Gin context for writing the response.
// @param status - HTTP status code for the error.
// @param msg - Human-readable error message.
//
// @pre c != nil and response not yet written.
// @post Anthropic-format error response is written.
func (h *BridgeHandler) WriteError(c *gin.Context, status int, msg string) {
	sendAnthropicError(c, status, msg)
}

// transformRequest converts an Anthropic MessageRequest to an OpenAI ChatCompletionRequest.
// This handles the translation of all request fields including messages, tools, and parameters.
//
// @param body - Raw JSON body in Anthropic MessageRequest format.
// @return Transformed JSON in OpenAI ChatCompletionRequest format.
// @return Error if JSON parsing or marshaling fails.
//
// @pre body is valid JSON representing an Anthropic MessageRequest.
// @post Returned JSON is valid OpenAI ChatCompletionRequest format.
func transformRequest(body []byte) ([]byte, error) {
	// Parse the Anthropic format request
	var anthReq types.MessageRequest
	if err := json.Unmarshal(body, &anthReq); err != nil {
		return nil, err
	}

	// Build the OpenAI format request with corresponding fields
	openReq := types.ChatCompletionRequest{
		Model:       anthReq.Model,
		MaxTokens:   anthReq.MaxTokens,
		Stream:      anthReq.Stream,
		Temperature: anthReq.Temperature,
		TopP:        anthReq.TopP,
	}

	// Convert system message (may be string or array of content blocks)
	openReq.System = extractSystemMessage(anthReq.System)
	// Convert messages array (handles content blocks with tool use/results)
	openReq.Messages = convertMessages(anthReq.Messages)
	// Convert tool definitions (Anthropic input_schema -> OpenAI parameters)
	openReq.Tools = convertTools(anthReq.Tools)

	return json.Marshal(openReq)
}

// extractSystemMessage extracts a string system message from various
// Anthropic system field formats.
// Anthropic accepts system as a string or array of content blocks.
//
// @param system - System field value (may be nil, string, or []interface{}).
// @return String representation of the system message.
//
// @post Returns empty string if system is nil.
// @post Returns the string directly if system is a string.
// @post Concatenates text from content blocks if system is an array.
func extractSystemMessage(system interface{}) string {
	// Handle nil case - no system message
	if system == nil {
		return ""
	}

	// Handle string case - direct string system message
	if s, ok := system.(string); ok {
		return s
	}

	// Handle array of content blocks - concatenate text content
	// Anthropic allows system as array of blocks with type "text"
	if arr, ok := system.([]interface{}); ok {
		var content strings.Builder
		for _, item := range arr {
			// Each item should be a map with "type" and "text" fields
			if m, ok := item.(map[string]interface{}); ok {
				// Extract text field from content block
				if text, ok := m["text"].(string); ok {
					content.WriteString(text)
				}
			}
		}
		return content.String()
	}

	// Unknown format - return empty string
	return ""
}

// convertMessages transforms a slice of Anthropic messages to OpenAI format.
// Each message's content is converted from Anthropic's content blocks to OpenAI's format.
//
// @param anthMsgs - Slice of Anthropic MessageInput structures.
// @return Slice of OpenAI Message structures.
//
// @pre anthMsgs is a valid slice (may be empty).
// @post Each message is converted preserving role and content.
func convertMessages(anthMsgs []types.MessageInput) []types.Message {
	// Pre-allocate slice for efficiency
	openMsgs := make([]types.Message, 0, len(anthMsgs))
	// Convert each message individually
	for _, anthMsg := range anthMsgs {
		openMsgs = append(openMsgs, convertMessage(anthMsg))
	}
	return openMsgs
}

// convertMessage transforms a single Anthropic message to OpenAI format.
// Handles content as either a string or array of content blocks.
//
// @param anthMsg - Anthropic MessageInput to convert.
// @return OpenAI Message with converted content.
//
// @post Role is preserved unchanged.
// @post String content is preserved directly.
// @post Content blocks are converted to text, tool_calls, and tool_call_id.
func convertMessage(anthMsg types.MessageInput) types.Message {
	openMsg := types.Message{Role: anthMsg.Role}

	// Content can be a string or array of content blocks
	// Handle each format appropriately
	switch content := anthMsg.Content.(type) {
	case string:
		// Simple string content - use directly
		openMsg.Content = content
	case []interface{}:
		// Content blocks - extract text, tool calls, and tool results
		openMsg.Content, openMsg.ToolCalls, openMsg.ToolCallID = convertContentBlocks(content)
	}

	return openMsg
}

// convertContentBlocks extracts text content, tool calls, and tool result IDs
// from Anthropic content blocks.
//
// Anthropic content blocks can be:
//   - text: Simple text content
//   - tool_use: Tool call request (converted to OpenAI tool_calls)
//   - tool_result: Tool execution result (converted to OpenAI tool_call_id)
//
// @param blocks - Array of Anthropic content block objects.
// @return textContent - Concatenated text from text blocks.
// @return toolCalls - Slice of OpenAI ToolCall from tool_use blocks.
// @return toolCallID - ID from tool_result block for assistant messages.
//
// @pre blocks is a valid array (may be empty).
// @post Text from multiple text blocks is concatenated with newlines.
// @post Tool calls preserve ID, name, and arguments.
func convertContentBlocks(blocks []interface{}) (interface{}, []types.ToolCall, string) {
	var textContent strings.Builder
	var toolCalls []types.ToolCall
	var toolCallID string

	// Process each content block based on its type
	for _, item := range blocks {
		// Each block should be a map with a "type" field
		m, ok := item.(map[string]interface{})
		if !ok {
			// Skip non-map items (shouldn't happen with valid input)
			continue
		}

		// Handle each block type differently
		switch m["type"] {
		case "text":
			// Text block - extract and concatenate text
			if text, ok := m["text"].(string); ok {
				// Add newline separator between multiple text blocks
				if textContent.Len() > 0 {
					textContent.WriteString("\n")
				}
				textContent.WriteString(text)
			}
		case "tool_use":
			// Tool use block - convert to OpenAI tool call
			// Extract required fields
			if id, ok := m["id"].(string); ok {
				if name, ok := m["name"].(string); ok {
					// Marshal input to JSON string for OpenAI format
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
			// Tool result block - extract tool use ID
			// This is used in assistant messages to reference tool results
			if id, ok := m["tool_use_id"].(string); ok {
				toolCallID = id
			}
		}
	}

	return textContent.String(), toolCalls, toolCallID
}

// convertTools transforms Anthropic tool definitions to OpenAI format.
// Anthropic uses "input_schema" while OpenAI uses "parameters".
//
// @param anthTools - Slice of Anthropic ToolDef structures.
// @return Slice of OpenAI Tool structures.
//
// @pre anthTools is a valid slice (may be empty).
// @post Each tool is converted with name, description, and parameters preserved.
func convertTools(anthTools []types.ToolDef) []types.Tool {
	// Pre-allocate slice for efficiency
	openTools := make([]types.Tool, 0, len(anthTools))
	for _, anthTool := range anthTools {
		// Convert each tool definition to OpenAI format
		openTools = append(openTools, types.Tool{
			Type: "function",
			Function: types.ToolFunction{
				Name:        anthTool.Name,
				Description: anthTool.Description,
				// input_schema maps directly to parameters
				Parameters: anthTool.InputSchema,
			},
		})
	}
	return openTools
}
