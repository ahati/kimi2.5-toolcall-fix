package handlers

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"ai-proxy/router"
	"ai-proxy/transform"
	"ai-proxy/transform/toolcall"
	"ai-proxy/types"

	"github.com/gin-gonic/gin"
)

// BridgeHandler converts Anthropic-format requests to OpenAI format.
type BridgeHandler struct {
	router        router.Router
	resolvedRoute *router.ResolvedRoute
}

// NewBridgeHandler creates a Gin handler for the /v1/openai-to-anthropic/messages endpoint.
func NewBridgeHandler(r router.Router) gin.HandlerFunc {
	return Handle(&BridgeHandler{router: r})
}

// ValidateRequest performs no additional validation for bridge requests.
func (h *BridgeHandler) ValidateRequest(body []byte) error {
	return nil
}

// TransformRequest converts an Anthropic-format request body to OpenAI format.
func (h *BridgeHandler) TransformRequest(body []byte) ([]byte, error) {
	var req struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return body, nil
	}

	route, err := h.router.Resolve(req.Model)
	if err != nil {
		return body, nil
	}

	h.resolvedRoute = route

	return transformRequest(body)
}

// UpstreamURL returns the resolved provider's base URL.
func (h *BridgeHandler) UpstreamURL() string {
	if h.resolvedRoute != nil && h.resolvedRoute.Provider != nil {
		return h.resolvedRoute.Provider.BaseURL
	}
	return ""
}

// ResolveAPIKey returns the resolved provider's API key.
func (h *BridgeHandler) ResolveAPIKey(c *gin.Context) string {
	if h.resolvedRoute != nil && h.resolvedRoute.Provider != nil {
		return h.resolvedRoute.Provider.GetAPIKey()
	}
	return ""
}

// ForwardHeaders copies X-* and Extra headers to the upstream request.
func (h *BridgeHandler) ForwardHeaders(c *gin.Context, req *http.Request) {
	for k, v := range c.Request.Header {
		if strings.HasPrefix(k, "X-") || k == "Extra" {
			req.Header[k] = v
		}
	}
}

// CreateTransformer builds an Anthropic SSE transformer.
func (h *BridgeHandler) CreateTransformer(w io.Writer) transform.SSETransformer {
	return toolcall.NewAnthropicTransformer(w)
}

// WriteError sends an error response in Anthropic format.
func (h *BridgeHandler) WriteError(c *gin.Context, status int, msg string) {
	sendAnthropicError(c, status, msg)
}

func transformRequest(body []byte) ([]byte, error) {
	var anthReq types.MessageRequest
	if err := json.Unmarshal(body, &anthReq); err != nil {
		return nil, err
	}

	openReq := types.ChatCompletionRequest{
		Model:       anthReq.Model,
		MaxTokens:   anthReq.MaxTokens,
		Stream:      anthReq.Stream,
		Temperature: anthReq.Temperature,
		TopP:        anthReq.TopP,
	}

	openReq.System = extractSystemMessage(anthReq.System)
	openReq.Messages = convertMessages(anthReq.Messages)
	openReq.Tools = convertTools(anthReq.Tools)

	return json.Marshal(openReq)
}

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

func convertMessages(anthMsgs []types.MessageInput) []types.Message {
	openMsgs := make([]types.Message, 0, len(anthMsgs))
	for _, anthMsg := range anthMsgs {
		openMsgs = append(openMsgs, convertMessage(anthMsg))
	}
	return openMsgs
}

func convertMessage(anthMsg types.MessageInput) types.Message {
	openMsg := types.Message{Role: anthMsg.Role}
	switch content := anthMsg.Content.(type) {
	case string:
		openMsg.Content = content
	case []interface{}:
		openMsg.Content, openMsg.ToolCalls, openMsg.ToolCallID = convertContentBlocks(content)
	}
	return openMsg
}

func convertContentBlocks(blocks []interface{}) (interface{}, []types.ToolCall, string) {
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

func convertTools(anthTools []types.ToolDef) []types.Tool {
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
