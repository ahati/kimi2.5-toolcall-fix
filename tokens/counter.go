// Package tokens provides local token counting for Anthropic-compatible APIs.
// This package implements fallback token counting when the upstream API
// does not support the /v1/messages/count_tokens endpoint.
package tokens

import (
	"encoding/json"
	"fmt"

	"ai-proxy/types"

	"github.com/pkoukk/tiktoken-go"
)

// Counter provides token counting functionality.
// It uses tiktoken for BPE tokenization compatible with OpenAI/Claude models.
type Counter struct {
	encoder *tiktoken.Tiktoken
}

// NewCounter creates a new token counter with the specified encoding.
// Supported encodings: "cl100k_base" (GPT-4, Claude), "p50k_base", "r50k_base", "gpt2"
//
// @param encodingName - The tiktoken encoding to use.
// @return *Counter - A configured token counter.
// @return error - Error if encoding is not supported.
func NewCounter(encodingName string) (*Counter, error) {
	encoder, err := tiktoken.GetEncoding(encodingName)
	if err != nil {
		return nil, fmt.Errorf("failed to get encoding %q: %w", encodingName, err)
	}

	return &Counter{encoder: encoder}, nil
}

// CountTokensForModel creates a counter appropriate for the given model.
// Uses cl100k_base for Claude-compatible models.
//
// @param model - The model identifier (e.g., "kimi-k2.5", "claude-3").
// @return *Counter - A token counter configured for the model.
// @return error - Error if no suitable encoding is found.
func CountTokensForModel(model string) (*Counter, error) {
	// Most modern models (GPT-4, Claude 3, Kimi) use cl100k_base
	// This is a reasonable default for Anthropic-compatible APIs
	return NewCounter("cl100k_base")
}

// CountMessageTokens counts tokens for a complete message request.
// Includes tokens for messages, system prompt, tools, and Anthropic formatting overhead.
//
// @param req - The token counting request.
// @return int - Total input token count.
// @return error - Error if counting fails.
func (c *Counter) CountMessageTokens(req *types.MessageCountTokensRequest) (int, error) {
	total := 0

	// Count system prompt tokens
	if req.System != nil {
		systemTokens, err := c.countSystemContent(req.System)
		if err != nil {
			return 0, fmt.Errorf("count system content: %w", err)
		}
		total += systemTokens
	}

	// Count tokens for tools
	for _, tool := range req.Tools {
		toolTokens := c.countToolDefinition(tool)
		total += toolTokens
	}

	// Count tokens for each message
	for _, msg := range req.Messages {
		msgTokens, err := c.countMessage(msg)
		if err != nil {
			return 0, fmt.Errorf("count message: %w", err)
		}
		total += msgTokens
	}

	// Add formatting overhead for Anthropic message structure
	// This accounts for JSON structure, role markers, etc.
	total += c.countFormattingOverhead(len(req.Messages), len(req.Tools))

	return total, nil
}

// countSystemContent counts tokens for system content (string or structured).
//
// @param system - System content (can be string or []ContentBlock).
// @return int - Token count for system content.
// @return error - Error if content format is invalid.
func (c *Counter) countSystemContent(system interface{}) (int, error) {
	switch v := system.(type) {
	case string:
		return len(c.encoder.Encode(v, nil, nil)), nil
	case []interface{}:
		// Structured content blocks
		return c.countContentBlocks(v), nil
	default:
		// Try to marshal and count
		data, err := json.Marshal(system)
		if err != nil {
			return 0, fmt.Errorf("marshal system content: %w", err)
		}
		return len(c.encoder.Encode(string(data), nil, nil)), nil
	}
}

// countContentBlocks counts tokens for structured content blocks.
//
// @param blocks - Array of content blocks.
// @return int - Token count.
func (c *Counter) countContentBlocks(blocks []interface{}) int {
	total := 0
	for _, block := range blocks {
		blockJSON, _ := json.Marshal(block)
		total += len(c.encoder.Encode(string(blockJSON), nil, nil))
	}
	return total
}

// countToolDefinition counts tokens for a tool definition.
// Includes the name, description, and input schema.
//
// @param tool - Tool definition.
// @return int - Token count for tool.
func (c *Counter) countToolDefinition(tool types.ToolDef) int {
	total := 0

	// Tool name
	total += len(c.encoder.Encode(tool.Name, nil, nil))

	// Tool description
	total += len(c.encoder.Encode(tool.Description, nil, nil))

	// Input schema
	if tool.InputSchema != nil {
		total += len(c.encoder.Encode(string(tool.InputSchema), nil, nil))
	}

	// Overhead for tool structure
	total += 10

	return total
}

// countMessage counts tokens for a single message.
// Handles both string content and structured content blocks.
//
// @param msg - The message to count.
// @return int - Token count.
// @return error - Error if content format is invalid.
func (c *Counter) countMessage(msg types.MessageInput) (int, error) {
	total := 0

	// Role marker tokens
	total += len(c.encoder.Encode(msg.Role, nil, nil))

	// Count content based on type
	switch v := msg.Content.(type) {
	case string:
		total += len(c.encoder.Encode(v, nil, nil))
	case []interface{}:
		total += c.countContentBlocks(v)
	case []types.ContentBlock:
		for _, block := range v {
			total += c.countContentBlock(block)
		}
	default:
		// Try to marshal content
		contentJSON, err := json.Marshal(msg.Content)
		if err != nil {
			return 0, fmt.Errorf("marshal content: %w", err)
		}
		total += len(c.encoder.Encode(string(contentJSON), nil, nil))
	}

	// Message structure overhead
	total += 4

	return total, nil
}

// countContentBlock counts tokens for a single content block.
//
// @param block - Content block.
// @return int - Token count.
func (c *Counter) countContentBlock(block types.ContentBlock) int {
	total := len(c.encoder.Encode(block.Type, nil, nil))

	switch block.Type {
	case "text":
		total += len(c.encoder.Encode(block.Text, nil, nil))
	case "tool_use":
		total += len(c.encoder.Encode(block.Name, nil, nil))
		total += len(c.encoder.Encode(block.ID, nil, nil))
		if block.Input != nil {
			total += len(c.encoder.Encode(string(block.Input), nil, nil))
		}
	case "thinking":
		total += len(c.encoder.Encode(block.Thinking, nil, nil))
	}

	return total
}

// countFormattingOverhead counts Anthropic API formatting overhead.
// This accounts for JSON structure, message delimiters, etc.
//
// @param messageCount - Number of messages.
// @param toolCount - Number of tools.
// @return int - Overhead token count.
func (c *Counter) countFormattingOverhead(messageCount, toolCount int) int {
	// Base overhead for request structure
	overhead := 20

	// Additional overhead per message
	overhead += messageCount * 5

	// Additional overhead per tool
	overhead += toolCount * 3

	return overhead
}

// CountTokens is a convenience function that counts tokens for a request.
// It creates a counter for the specified model and returns the token count.
//
// @param req - The token counting request.
// @return int - Total input token count.
// @return error - Error if counting fails.
func CountTokens(req *types.MessageCountTokensRequest) (int, error) {
	counter, err := CountTokensForModel(req.Model)
	if err != nil {
		return 0, err
	}

	return counter.CountMessageTokens(req)
}
