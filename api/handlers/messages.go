package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"ai-proxy/config"
	"ai-proxy/router"
	"ai-proxy/transform"
	"ai-proxy/transform/toolcall"
	"ai-proxy/types"

	"github.com/gin-gonic/gin"
)

// MessagesHandler handles Anthropic Messages API requests.
// It implements the Handler interface for the /v1/messages endpoint.
//
// This handler:
//   - Accepts requests in Anthropic Messages format
//   - Routes to the appropriate upstream based on model configuration
//   - For Anthropic providers: passes through requests without transformation
//   - For OpenAI providers: converts Anthropic→OpenAI Chat, transforms responses back
//   - Supports streaming responses with tool use handling
//
// @note This enables clients using Anthropic SDK to call any provider.
type MessagesHandler struct {
	// cfg contains the application configuration including providers and models.
	// Must not be nil after construction.
	cfg *config.Config
	// router resolves model names to providers. May be nil for legacy behavior.
	modelRouter router.Router
	// route is the resolved route for the current request.
	// Set during ValidateRequest for use in subsequent methods.
	route *router.ResolvedRoute
	// originalModel is the model name from the original request.
	// Preserved for response transformation.
	originalModel string
}

// NewMessagesHandler creates a Gin handler for the /v1/messages endpoint.
//
// @param cfg - Application configuration. Must not be nil.
// @param r - Router for model resolution. May be nil for legacy behavior.
// @return Gin handler function that processes message requests.
//
// @pre cfg != nil
func NewMessagesHandler(cfg *config.Config, r router.Router) gin.HandlerFunc {
	return Handle(&MessagesHandler{
		cfg:         cfg,
		modelRouter: r,
	})
}

// ValidateRequest validates the request and resolves the model route.
// It parses the request to extract the model name and resolves it to a provider.
//
// @param body - Raw request body bytes.
// @return Error if JSON parsing fails or model cannot be resolved.
func (h *MessagesHandler) ValidateRequest(body []byte) error {
	// If no router, use legacy behavior
	if h.modelRouter == nil {
		return nil
	}

	// Parse to get model name
	var req struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return nil // Let upstream handle invalid JSON
	}

	if req.Model == "" {
		return nil // Let upstream handle missing model
	}

	// Resolve the model to a route
	route, err := h.modelRouter.Resolve(req.Model)
	if err != nil {
		return nil // Use fallback behavior
	}

	h.route = route
	h.originalModel = req.Model
	return nil
}

// TransformRequest converts the request body based on the upstream provider type.
// For Anthropic providers: passes through without transformation.
// For OpenAI providers: converts Anthropic Messages to OpenAI Chat Completions.
//
// @param body - Raw request body in Anthropic Messages format.
// @return Transformed body in the appropriate upstream format.
// @return Error if transformation fails.
func (h *MessagesHandler) TransformRequest(body []byte) ([]byte, error) {
	// If no route resolved, pass through (legacy behavior)
	if h.route == nil {
		return body, nil
	}

	// Update model in request to the resolved model
	var req map[string]interface{}
	if err := json.Unmarshal(body, &req); err != nil {
		return body, nil
	}
	req["model"] = h.route.Model
	updatedBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal updated request: %w", err)
	}

	switch h.route.Provider.Type {
	case "openai":
		// Convert Anthropic MessageRequest to OpenAI ChatCompletionRequest
		return transformAnthropicToChat(updatedBody)
	case "anthropic":
		// Pass through without transformation
		return updatedBody, nil
	default:
		// Unknown provider type - pass through as-is
		return updatedBody, nil
	}
}

// UpstreamURL returns the upstream API URL based on the resolved provider.
//
// @return URL string for the upstream API endpoint.
func (h *MessagesHandler) UpstreamURL() string {
	if h.route != nil {
		url := h.route.Provider.BaseURL
		// For OpenAI providers, append the chat completions path if needed
		if h.route.Provider.Type == "openai" && !strings.HasSuffix(url, "/chat/completions") {
			url = strings.TrimSuffix(url, "/") + "/chat/completions"
		}
		return url
	}
	// Legacy behavior - use Anthropic upstream
	return h.cfg.GetAnthropicUpstreamURL()
}

// ResolveAPIKey returns the API key for the resolved provider.
//
// @param c - Gin context (unused in this implementation).
// @return API key string from the provider configuration.
func (h *MessagesHandler) ResolveAPIKey(c *gin.Context) string {
	if h.route != nil {
		return h.route.Provider.GetAPIKey()
	}
	// Legacy behavior
	return h.cfg.GetAnthropicAPIKey()
}

// ForwardHeaders copies headers to the upstream request based on provider type.
// For OpenAI providers: forwards X-* headers only.
// For Anthropic providers: forwards X-*, Anthropic-Version, and Anthropic-Beta headers.
//
// @param c - Gin context containing the original request headers.
// @param req - Upstream request to receive forwarded headers.
func (h *MessagesHandler) ForwardHeaders(c *gin.Context, req *http.Request) {
	providerType := "anthropic" // default
	if h.route != nil {
		providerType = h.route.Provider.Type
	}

	switch providerType {
	case "openai":
		// Forward custom headers that may be used by OpenAI-compatible upstream
		for k, v := range c.Request.Header {
			// Only forward X-* headers and Extra header
			// Do NOT forward Anthropic-specific headers as they may confuse OpenAI API
			if strings.HasPrefix(k, "X-") || k == "Extra" {
				req.Header[k] = v
			}
		}
	case "anthropic":
		// Forward headers that are important for Anthropic API
		for k, v := range c.Request.Header {
			// Forward custom headers with X- prefix
			// Forward Anthropic-Version which is required for API versioning
			// Forward Anthropic-Beta for beta feature access
			if strings.HasPrefix(k, "X-") || k == "Anthropic-Version" || k == "Anthropic-Beta" {
				req.Header[k] = v
			}
		}
	default:
		// Forward X-* headers by default
		forwardCustomHeaders(c, req, "X-")
	}
}

// CreateTransformer builds an SSE transformer based on the provider type.
// For Anthropic providers: uses AnthropicTransformer for tool call handling.
// For OpenAI providers: uses AnthropicToChatTransformer to convert responses back to Anthropic format.
//
// @param w - Writer to receive transformed output.
// @return Transformer for processing SSE events.
func (h *MessagesHandler) CreateTransformer(w io.Writer) transform.SSETransformer {
	if h.route == nil {
		// Legacy behavior - use Anthropic transformer
		return toolcall.NewAnthropicTransformer(w)
	}

	switch h.route.Provider.Type {
	case "openai":
		// Convert OpenAI Chat responses back to Anthropic format
		return toolcall.NewAnthropicTransformer(w)
	case "anthropic":
		// Use Anthropic transformer for tool call handling
		if h.route.ToolCallTransform {
			return toolcall.NewAnthropicTransformer(w)
		}
		return transform.NewPassthroughTransformer(w)
	default:
		return transform.NewPassthroughTransformer(w)
	}
}

// WriteError sends an error response in Anthropic format.
// Maintains consistency with Anthropic API error responses.
//
// @param c - Gin context for writing the response.
// @param status - HTTP status code for the error.
// @param msg - Human-readable error message.
func (h *MessagesHandler) WriteError(c *gin.Context, status int, msg string) {
	sendAnthropicError(c, status, msg)
}

// transformAnthropicToChat converts an Anthropic MessageRequest to OpenAI ChatCompletionRequest.
// This reuses the transformation logic from the bridge handler.
func transformAnthropicToChat(body []byte) ([]byte, error) {
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
	openReq.Messages = convertAnthropicMessages(anthReq.Messages)
	// Convert tool definitions (Anthropic input_schema -> OpenAI parameters)
	openReq.Tools = convertAnthropicTools(anthReq.Tools)

	return json.Marshal(openReq)
}

// extractSystemMessage extracts a string system message from various
// Anthropic system field formats.
func extractSystemMessage(system interface{}) string {
	if system == nil {
		return ""
	}

	if s, ok := system.(string); ok {
		return s
	}

	if arr, ok := system.([]interface{}); ok {
		var content strings.Builder
		for _, item := range arr {
			if m, ok := item.(map[string]interface{}); ok {
				if text, ok := m["text"].(string); ok {
					content.WriteString(text)
				}
			}
		}
		return content.String()
	}

	return ""
}

// convertAnthropicMessages transforms a slice of Anthropic messages to OpenAI format.
func convertAnthropicMessages(anthMsgs []types.MessageInput) []types.Message {
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
		openMsg.Content, openMsg.ToolCalls, openMsg.ToolCallID = convertAnthropicContentBlocks(content)
	}

	return openMsg
}

// convertAnthropicContentBlocks extracts text content, tool calls, and tool result IDs
// from Anthropic content blocks.
func convertAnthropicContentBlocks(blocks []interface{}) (interface{}, []types.ToolCall, string) {
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
		}
	}

	return textContent.String(), toolCalls, toolCallID
}

// convertAnthropicTools transforms Anthropic tool definitions to OpenAI format.
func convertAnthropicTools(anthTools []types.ToolDef) []types.Tool {
	if len(anthTools) == 0 {
		return nil
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