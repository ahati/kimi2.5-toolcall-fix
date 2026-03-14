package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"ai-proxy/config"
	"ai-proxy/transform"
	"ai-proxy/transform/toolcall"
	"ai-proxy/types"

	"github.com/gin-gonic/gin"
)

// AnthropicToOpenAIHandler converts OpenAI Responses API format requests to Anthropic format
// and proxies to an Anthropic-compatible upstream.
// It implements the Handler interface for the /v1/anthropic-to-openai/responses endpoint.
//
// This handler:
//   - Accepts requests in OpenAI Responses API format
//   - Transforms requests to Anthropic Messages format
//   - Forwards to Anthropic-compatible upstream API
//   - Transforms responses back to OpenAI Responses API format for the client
//
// @note This enables clients using OpenAI SDK to call Anthropic-compatible APIs.
type AnthropicToOpenAIHandler struct {
	// cfg contains the application configuration including upstream URL and API key.
	// Must not be nil after construction.
	cfg *config.Config
}

// NewAnthropicToOpenAIHandler creates a Gin handler for the /v1/anthropic-to-openai/responses endpoint.
//
// @param cfg - Application configuration. Must not be nil.
// @return Gin handler function that processes bridge requests.
//
// @pre cfg != nil
// @pre cfg.AnthropicUpstreamURL != ""
// @pre cfg.AnthropicAPIKey != "" (or valid Authorization header expected)
func NewAnthropicToOpenAIHandler(cfg *config.Config) gin.HandlerFunc {
	return Handle(&AnthropicToOpenAIHandler{cfg: cfg})
}

// ValidateRequest performs validation for OpenAI Responses API requests.
// Checks that the request can be parsed as a valid ResponsesRequest.
//
// @param body - Raw request body bytes.
// @return Error if JSON parsing fails or required fields are missing.
func (h *AnthropicToOpenAIHandler) ValidateRequest(body []byte) error {
	// Parse to validate structure
	var req types.ResponsesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return err
	}
	if req.Model == "" {
		return fmt.Errorf("model is required")
	}
	return nil
}

// TransformRequest converts an OpenAI Responses API request body to Anthropic Messages format.
// This is the core bridging logic that translates between API formats.
//
// @param body - Raw request body in OpenAI Responses API format.
// @return Transformed body in Anthropic MessageRequest format.
// @return Error if JSON parsing or transformation fails.
func (h *AnthropicToOpenAIHandler) TransformRequest(body []byte) ([]byte, error) {
	return transformResponsesRequest(body)
}

// UpstreamURL returns the Anthropic-compatible upstream API URL.
// The bridge sends requests to an Anthropic-compatible API after transformation.
//
// @return URL string from configuration for the Anthropic upstream.
func (h *AnthropicToOpenAIHandler) UpstreamURL() string {
	return h.cfg.AnthropicUpstreamURL()
}

// ResolveAPIKey returns the configured Anthropic upstream API key.
// The bridge uses Anthropic authentication after transforming the request.
//
// @param c - Gin context (unused in this implementation).
// @return API key string from configuration.
func (h *AnthropicToOpenAIHandler) ResolveAPIKey(c *gin.Context) string {
	return h.cfg.AnthropicAPIKey()
}

// ForwardHeaders copies X-*, Anthropic-Version, and Anthropic-Beta headers to the upstream request.
// These headers are required for proper Anthropic API operation.
//
// @param c - Gin context containing the original request headers.
// @param req - Upstream request to receive forwarded headers.
func (h *AnthropicToOpenAIHandler) ForwardHeaders(c *gin.Context, req *http.Request) {
	// Forward headers that are important for Anthropic API
	for k, v := range c.Request.Header {
		// Forward custom headers with X- prefix
		// Forward Anthropic-Version which is required for API versioning
		// Forward Anthropic-Beta for beta feature access
		if strings.HasPrefix(k, "X-") || k == "Anthropic-Version" || k == "Anthropic-Beta" {
			req.Header[k] = v
		}
	}
}

// CreateTransformer builds an OpenAI Responses API transformer to convert
// Anthropic responses back to OpenAI Responses API format.
// This ensures the client receives OpenAI Responses API format responses.
//
// @param w - Writer to receive transformed output.
// @return Transformer for converting Anthropic SSE to OpenAI Responses API format.
func (h *AnthropicToOpenAIHandler) CreateTransformer(w io.Writer) transform.SSETransformer {
	return toolcall.NewResponsesTransformer(w)
}

// WriteError sends an error response in OpenAI Responses API format.
// Maintains consistency with the expected OpenAI Responses API error response format.
//
// @param c - Gin context for writing the response.
// @param status - HTTP status code for the error.
// @param msg - Human-readable error message.
func (h *AnthropicToOpenAIHandler) WriteError(c *gin.Context, status int, msg string) {
	sendOpenAIResponsesError(c, status, msg)
}

// sendOpenAIResponsesError sends an error response in OpenAI Responses API format.
func sendOpenAIResponsesError(c *gin.Context, status int, msg string) {
	event := map[string]interface{}{
		"type": "error",
		"error": map[string]interface{}{
			"code":    "invalid_request_error",
			"message": msg,
		},
	}
	c.Header("Content-Type", "text/event-stream")
	data, _ := json.Marshal(event)
	c.String(status, "data: "+string(data)+"\n\n")
}

// transformResponsesRequest converts an OpenAI ResponsesRequest to an Anthropic MessageRequest.
// This handles the translation of all request fields including input, tools, and parameters.
//
// @param body - Raw JSON body in OpenAI ResponsesRequest format.
// @return Transformed JSON in Anthropic MessageRequest format.
// @return Error if JSON parsing or marshaling fails.
func transformResponsesRequest(body []byte) ([]byte, error) {
	// Parse the OpenAI Responses API format request
	var openReq types.ResponsesRequest
	if err := json.Unmarshal(body, &openReq); err != nil {
		return nil, err
	}

	// Build the Anthropic format request using the client-provided model name
	// Anthropic requires max_tokens, so use a default if not provided
	maxTokens := openReq.MaxOutputTokens
	if maxTokens == 0 {
		maxTokens = 16384 // Default max tokens (16k) for Anthropic API
	}
	anthReq := types.MessageRequest{
		Model:       openReq.Model,
		MaxTokens:   maxTokens,
		Stream:      openReq.Stream,
		Temperature: openReq.Temperature,
		TopP:        openReq.TopP,
	}

	// Combine instructions with developer messages from input
	systemParts := []string{}
	if openReq.Instructions != "" {
		systemParts = append(systemParts, openReq.Instructions)
	}

	// Extract developer messages from input array
	if arr, ok := openReq.Input.([]interface{}); ok {
		for _, item := range arr {
			if msg, ok := item.(map[string]interface{}); ok {
				if role, ok := msg["role"].(string); ok && (role == "developer" || role == "system") {
					content := extractContent(msg["content"])
					if content != "" {
						systemParts = append(systemParts, content)
					}
				}
			}
		}
	}

	// Set system message if we have any parts
	if len(systemParts) > 0 {
		anthReq.System = strings.Join(systemParts, "\n\n")
	}

	// Convert input to messages (excluding developer/system)
	anthReq.Messages = convertInputToMessages(openReq.Input)

	// Convert tools
	anthReq.Tools = convertResponsesTools(openReq.Tools)

	// Convert tool_choice
	anthReq.ToolChoice = convertToolChoice(openReq.ToolChoice)

	return json.Marshal(anthReq)
}

// convertInputToMessages converts OpenAI Responses API input to Anthropic messages.
func convertInputToMessages(input interface{}) []types.MessageInput {
	if input == nil {
		return []types.MessageInput{}
	}

	if s, ok := input.(string); ok {
		return []types.MessageInput{
			{Role: "user", Content: s},
		}
	}

	if arr, ok := input.([]interface{}); ok {
		messages := make([]types.MessageInput, 0, len(arr))
		for _, item := range arr {
			if msg, ok := item.(map[string]interface{}); ok {
				itemType, _ := msg["type"].(string)

				switch itemType {
				case "message":
					role := "user"
					if r, ok := msg["role"].(string); ok {
						switch r {
						case "developer", "system":
							continue
						case "assistant":
							role = "assistant"
						default:
							role = "user"
						}
					}

					content := extractContent(msg["content"])
					if content == "" {
						continue
					}

					messages = append(messages, types.MessageInput{
						Role:    role,
						Content: content,
					})

				case "function_call":
					callID, _ := msg["call_id"].(string)
					name, _ := msg["name"].(string)
					args, _ := msg["arguments"].(string)

					var inputObj map[string]interface{}
					if args != "" {
						json.Unmarshal([]byte(args), &inputObj)
					}

					messages = append(messages, types.MessageInput{
						Role: "assistant",
						Content: []map[string]interface{}{
							{
								"type":  "tool_use",
								"id":    callID,
								"name":  name,
								"input": inputObj,
							},
						},
					})

				case "function_call_output":
					callID, _ := msg["call_id"].(string)
					output, _ := msg["output"].(string)

					messages = append(messages, types.MessageInput{
						Role: "user",
						Content: []map[string]interface{}{
							{
								"type":        "tool_result",
								"tool_use_id": callID,
								"content":     output,
							},
						},
					})

				case "reasoning":
					// Skip reasoning items - not needed for conversation history
				}
			}
		}
		return messages
	}

	return []types.MessageInput{}
}

// extractContent extracts text content from various formats.
// Handles string content and array of content parts.
func extractContent(content interface{}) string {
	if content == nil {
		return ""
	}

	// Handle string content directly
	if s, ok := content.(string); ok {
		return s
	}

	// Handle array of content parts
	if arr, ok := content.([]interface{}); ok {
		var result strings.Builder
		for _, part := range arr {
			if partMap, ok := part.(map[string]interface{}); ok {
				// Extract text from input_text or input_image parts
				partType, _ := partMap["type"].(string)
				switch partType {
				case "input_text":
					if text, ok := partMap["text"].(string); ok {
						if result.Len() > 0 {
							result.WriteString("\n")
						}
						result.WriteString(text)
					}
				case "input_image":
					// Images not supported by Anthropic text API - add placeholder
					if result.Len() > 0 {
						result.WriteString("\n")
					}
					result.WriteString("[Image attached]")
				}
			}
		}
		return result.String()
	}

	return ""
}

// convertResponsesTools converts OpenAI Responses API tools to Anthropic tools.
func convertResponsesTools(openTools []types.ResponsesTool) []types.ToolDef {
	if len(openTools) == 0 {
		return nil
	}

	anthTools := make([]types.ToolDef, 0, len(openTools))
	for _, openTool := range openTools {
		// Only handle function type tools
		// Skip "custom", "web_search", "file_search" etc.
		if openTool.Type == "function" {
			// Handle nested format (Function object)
			if openTool.Function != nil {
				anthTools = append(anthTools, types.ToolDef{
					Name:        openTool.Function.Name,
					Description: openTool.Function.Description,
					InputSchema: openTool.Function.Parameters,
				})
			} else if openTool.Name != "" {
				// Handle flat format (fields at top level)
				anthTools = append(anthTools, types.ToolDef{
					Name:        openTool.Name,
					Description: openTool.Description,
					InputSchema: openTool.Parameters,
				})
			}
		}
	}
	return anthTools
}

// convertToolChoice converts OpenAI tool_choice to Anthropic format.
// OpenAI: "none" | "auto" | "required" | {"type": "function", "function": {"name": "..."}}
// Anthropic: {"type": "auto"} | {"type": "any"} | {"type": "tool", "name": "..."}
func convertToolChoice(toolChoice interface{}) *types.ToolChoice {
	if toolChoice == nil {
		return nil
	}

	// Handle string values
	if s, ok := toolChoice.(string); ok {
		switch s {
		case "none":
			// Anthropic doesn't have "none" - return nil to omit field
			// The absence of tools will prevent tool use
			return nil
		case "auto":
			return &types.ToolChoice{Type: "auto"}
		case "required":
			return &types.ToolChoice{Type: "any"}
		default:
			return &types.ToolChoice{Type: "auto"}
		}
	}

	// Handle object format: {"type": "function", "function": {"name": "xyz"}}
	if obj, ok := toolChoice.(map[string]interface{}); ok {
		objType, _ := obj["type"].(string)
		if objType == "function" {
			// Extract function name
			if fn, ok := obj["function"].(map[string]interface{}); ok {
				if name, ok := fn["name"].(string); ok {
					return &types.ToolChoice{
						Type: "tool",
						Name: name,
					}
				}
			}
		}
	}

	// Default to auto for unknown formats
	return &types.ToolChoice{Type: "auto"}
}
