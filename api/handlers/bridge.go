package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"ai-proxy/router"
	"ai-proxy/transform"
	"ai-proxy/transform/toolcall"
	"ai-proxy/types"

	"github.com/gin-gonic/gin"
)

type BridgeHandler struct{}

func NewBridgeHandler(r router.Router) gin.HandlerFunc {
	return HandleWithRouter(&BridgeHandler{}, r)
}

func (h *BridgeHandler) ValidateRequest(body []byte) error {
	return nil
}

func (h *BridgeHandler) ExtractModel(body []byte) (string, error) {
	var req struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return "", fmt.Errorf("invalid JSON: %w", err)
	}
	return req.Model, nil
}

func (h *BridgeHandler) TransformRequest(body []byte) ([]byte, error) {
	return body, nil
}

func (h *BridgeHandler) TransformRequestWithRoute(body []byte, route *router.ResolvedRoute) ([]byte, error) {
	var req map[string]interface{}
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, fmt.Errorf("failed to parse request: %w", err)
	}
	req["model"] = route.Model
	updatedBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal updated request: %w", err)
	}

	switch route.Provider.Type {
	case "openai":
		return transformAnthropicToOpenAI(updatedBody)
	case "anthropic":
		return updatedBody, nil
	default:
		return updatedBody, nil
	}
}

func (h *BridgeHandler) UpstreamURL() string {
	return ""
}

func (h *BridgeHandler) UpstreamURLWithRoute(route *router.ResolvedRoute) string {
	switch route.Provider.Type {
	case "openai":
		baseURL := route.Provider.BaseURL
		if !strings.HasSuffix(baseURL, "/chat/completions") {
			baseURL = strings.TrimSuffix(baseURL, "/") + "/chat/completions"
		}
		return baseURL
	case "anthropic":
		return route.Provider.BaseURL
	default:
		return route.Provider.BaseURL
	}
}

func (h *BridgeHandler) ResolveAPIKey(c *gin.Context) string {
	return ""
}

func (h *BridgeHandler) ResolveAPIKeyWithRoute(route *router.ResolvedRoute) string {
	return route.Provider.GetAPIKey()
}

func (h *BridgeHandler) ForwardHeaders(c *gin.Context, req *http.Request) {
}

func (h *BridgeHandler) ForwardHeadersWithRoute(c *gin.Context, req *http.Request, route *router.ResolvedRoute) {
	switch route.Provider.Type {
	case "openai":
		for k, v := range c.Request.Header {
			if strings.HasPrefix(k, "X-") || k == "Extra" {
				req.Header[k] = v
			}
		}
	case "anthropic":
		for k, v := range c.Request.Header {
			if strings.HasPrefix(k, "X-") || k == "Anthropic-Version" || k == "Anthropic-Beta" {
				req.Header[k] = v
			}
		}
	default:
		forwardCustomHeaders(c, req, "X-")
	}
}

func (h *BridgeHandler) CreateTransformer(w io.Writer) transform.SSETransformer {
	return transform.NewPassthroughTransformer(w)
}

func (h *BridgeHandler) CreateTransformerWithRoute(w io.Writer, route *router.ResolvedRoute) transform.SSETransformer {
	switch route.Provider.Type {
	case "openai":
		return toolcall.NewAnthropicTransformer(w)
	case "anthropic":
		if route.ToolCallTransform {
			return toolcall.NewAnthropicTransformer(w)
		}
		return transform.NewPassthroughTransformer(w)
	default:
		return transform.NewPassthroughTransformer(w)
	}
}

func (h *BridgeHandler) WriteError(c *gin.Context, status int, msg string) {
	sendAnthropicError(c, status, msg)
}

func transformAnthropicToOpenAI(body []byte) ([]byte, error) {
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
