package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"ai-proxy/config"
	"ai-proxy/convert"
	"ai-proxy/logging"
	"ai-proxy/router"
	"ai-proxy/transform"
	"ai-proxy/transform/toolcall"
	wstransform "ai-proxy/transform/websearch"
	"ai-proxy/types"
	"ai-proxy/websearch"

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
//   - Supports web search tool interception when web search service is enabled
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
	return func(c *gin.Context) {
		h := &MessagesHandler{
			cfg:         cfg,
			modelRouter: r,
		}
		Handle(h)(c)
	}
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

	// Resolve the model to a route with incoming protocol context
	// The messages endpoint receives requests in Anthropic format
	route, err := h.modelRouter.ResolveWithProtocol(req.Model, "anthropic")
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
// @param ctx - Context for the request (unused in this handler).
// @param body - Raw request body in Anthropic Messages format.
// @return Transformed body in the appropriate upstream format.
// @return Error if transformation fails.
func (h *MessagesHandler) TransformRequest(ctx context.Context, body []byte) ([]byte, error) {
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

	// Passthrough optimization - no transformation needed
	if h.route.IsPassthrough {
		return updatedBody, nil
	}

	switch h.route.OutputProtocol {
	case "openai":
		// Convert Anthropic MessageRequest to OpenAI ChatCompletionRequest
		transformed, err := transformAnthropicToChat(updatedBody)
		if err != nil {
			return nil, err
		}

		// Inject reasoning_split if configured
		if h.route.ReasoningSplit {
			var req map[string]interface{}
			if err := json.Unmarshal(transformed, &req); err == nil {
				req["reasoning_split"] = true
				transformed, _ = json.Marshal(req)
			}
		}
		return transformed, nil
	case "anthropic":
		// Normalize web_search_tool_result to tool_result for upstream compatibility
		// Many Anthropic-compatible providers don't support web_search_tool_result natively
		return convert.NormalizeWebSearchToolResultsInMessages(updatedBody), nil
	case "responses":
		// Convert Anthropic Messages to Responses API format
		return convert.TransformAnthropicToResponses(updatedBody)
	default:
		// Unknown protocol - pass through as-is
		return updatedBody, nil
	}
}

// UpstreamURL returns the upstream API URL based on the resolved provider.
//
// @return URL string for the upstream API endpoint.
func (h *MessagesHandler) UpstreamURL() string {
	if h.route != nil {
		return h.route.Provider.GetEndpoint(h.route.OutputProtocol)
	}
	return ""
}

// ResolveAPIKey returns the API key for the resolved provider.
//
// @param c - Gin context (unused in this implementation).
// @return API key string from the provider configuration.
func (h *MessagesHandler) ResolveAPIKey(c *gin.Context) string {
	if h.route != nil {
		return h.route.Provider.GetAPIKey()
	}
	return ""
}

// ForwardHeaders copies headers to the upstream request based on provider type.
// For OpenAI providers: forwards X-* headers only.
// For Anthropic providers: forwards X-*, Anthropic-Version, and Anthropic-Beta headers.
//
// @param c - Gin context containing the original request headers.
// @param req - Upstream request to receive forwarded headers.
func (h *MessagesHandler) ForwardHeaders(c *gin.Context, req *http.Request) {
	outputProtocol := "anthropic" // default
	if h.route != nil {
		outputProtocol = h.route.OutputProtocol
	}

	switch outputProtocol {
	case "openai":
		// Forward custom headers only
		forwardCustomHeaders(c, req, "X-")
	case "anthropic":
		// Forward Anthropic-specific headers
		for k, v := range c.Request.Header {
			if strings.HasPrefix(k, "X-") || k == "Anthropic-Version" || k == "Anthropic-Beta" {
				req.Header[k] = v
			}
		}
	default:
		// Forward X-* headers by default
		forwardCustomHeaders(c, req, "X-")
	}
}

// CreateTransformer builds an SSE transformer for converting upstream responses.
// For OpenAI providers: converts Chat Completions to Anthropic format.
// For Anthropic providers: passes through SSE events.
// If web search service is enabled, wraps the transformer to intercept web_search tool calls.
//
// @param w - Writer to receive transformed output.
// @return Transformer for processing SSE events.
func (h *MessagesHandler) CreateTransformer(w io.Writer) transform.SSETransformer {
	// No route resolved: pass through
	if h.route == nil {
		return h.wrapWithWebSearch(transform.NewPassthroughTransformer(w))
	}

	// Passthrough optimization
	if h.route.IsPassthrough {
		return h.wrapWithWebSearch(transform.NewPassthroughTransformer(w))
	}

	var baseTransformer transform.SSETransformer

	switch h.route.OutputProtocol {
	case "openai":
		// OpenAI to Anthropic transformer
		transformer := toolcall.NewAnthropicTransformer(w)
		transformer.SetGLM5ToolCallTransform(h.route.GLM5ToolCallTransform)
		transformer.SetKimiToolCallTransform(h.route.KimiToolCallTransform)
		baseTransformer = transformer
	case "anthropic":
		// Passthrough for native Anthropic
		baseTransformer = transform.NewPassthroughTransformer(w)
	default:
		return transform.NewPassthroughTransformer(w)
	}

	// Wrap with web search transformer if enabled
	return h.wrapWithWebSearch(baseTransformer)
}

// wrapWithWebSearch wraps the base transformer with web search interception if enabled.
//
// @param base - The base transformer to wrap.
// @return The wrapped transformer, or the base transformer if web search is not enabled.
func (h *MessagesHandler) wrapWithWebSearch(base transform.SSETransformer) transform.SSETransformer {
	// Get the web search adapter (returns nil if service is not enabled)
	adapter := websearch.GetDefaultAdapter()
	if adapter == nil || !adapter.IsEnabled() {
		return base
	}

	// Wrap the base transformer with web search interception
	return wstransform.NewTransformer(base, adapter)
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
	// Force streaming mode - this proxy only supports SSE streaming
	stream := anthReq.Stream
	if !stream {
		logging.InfoMsg("Forcing stream=true for upstream request (client did not specify)")
		stream = true
	}
	openReq := types.ChatCompletionRequest{
		Model:       anthReq.Model,
		MaxTokens:   anthReq.MaxTokens,
		Stream:      stream,
		Temperature: anthReq.Temperature,
		TopP:        anthReq.TopP,
		// Request usage statistics from upstream (required for Anthropic SDK)
		StreamOptions: &types.StreamOptions{IncludeUsage: true},
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
// Uses the shared ExtractTextFromContent from convert package.
func extractSystemMessage(system interface{}) string {
	return convert.ExtractTextFromContent(system)
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
		// Only pure tool_result turns can be represented as OpenAI tool messages.
		if openMsg.ToolCallID != "" && isPureAnthropicToolResultTurn(content) {
			openMsg.Role = "tool"
		}
	}

	return openMsg
}

func isPureAnthropicToolResultTurn(blocks []interface{}) bool {
	if len(blocks) == 0 {
		return false
	}

	for _, item := range blocks {
		block, ok := item.(map[string]interface{})
		if !ok {
			return false
		}
		blockType, _ := block["type"].(string)
		if blockType != "tool_result" {
			return false
		}
	}

	return true
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
			// Extract tool_use_id and content
			if id, ok := m["tool_use_id"].(string); ok {
				toolCallID = id
			}
			// Extract the tool result content
			if content, ok := m["content"]; ok {
				switch c := content.(type) {
				case string:
					if textContent.Len() > 0 {
						textContent.WriteString("\n")
					}
					textContent.WriteString(c)
				case []interface{}:
					// Handle array content blocks within tool_result
					for _, block := range c {
						if blockMap, ok := block.(map[string]interface{}); ok {
							if blockType, ok := blockMap["type"].(string); ok && blockType == "text" {
								if t, ok := blockMap["text"].(string); ok {
									if textContent.Len() > 0 {
										textContent.WriteString("\n")
									}
									textContent.WriteString(t)
								}
							}
						}
					}
				}
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

// ModelInfo returns the downstream and upstream model names for logging.
func (h *MessagesHandler) ModelInfo() (downstreamModel string, upstreamModel string) {
	downstreamModel = h.originalModel
	if h.route != nil {
		upstreamModel = h.route.Model
	}
	return
}
