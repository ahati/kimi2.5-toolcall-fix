package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type AnthropicEvent struct {
	Type         string          `json:"type"`
	Index        *int            `json:"index,omitempty"`
	Delta        json.RawMessage `json:"delta,omitempty"`
	ContentBlock json.RawMessage `json:"content_block,omitempty"`
	StopReason   string          `json:"stop_reason,omitempty"`
	Usage        json.RawMessage `json:"usage,omitempty"`
}

type MessageDelta struct {
	StopReason   string  `json:"stop_reason,omitempty"`
	StopSequence *string `json:"stop_sequence,omitempty"`
}

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

type ContentBlock struct {
	Type     string          `json:"type"`
	Text     string          `json:"text,omitempty"`
	Thinking string          `json:"thinking,omitempty"`
	Name     string          `json:"name,omitempty"`
	ID       string          `json:"id,omitempty"`
	Input    json.RawMessage `json:"input,omitempty"`
}

type BlockContent struct {
	Type      string
	Text      string
	Thinking  string
	ToolName  string
	ToolID    string
	ToolInput string
	Index     int
	Completed bool
}

func main() {
	logDir := "/workspaces/kimi-k2.5-fix-proxy/sse_logs"

	files, err := filepath.Glob(filepath.Join(logDir, "*.log"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error finding log files: %v\n", err)
		os.Exit(1)
	}

	if len(files) == 0 {
		fmt.Println("No log files found")
		return
	}

	sort.Strings(files)

	// Analyze last 20 files
	startIdx := 0
	if len(files) > 20 {
		startIdx = len(files) - 20
	}

	fmt.Printf("Analyzing %d log files (of %d total)\n\n", len(files)-startIdx, len(files))

	for i, file := range files[startIdx:] {
		fmt.Printf("=== File %d/%d: %s ===\n", i+1, len(files)-startIdx, filepath.Base(file))
		analyzeFile(file)
		fmt.Println()
	}
}

func analyzeFile(filename string) {
	file, err := os.Open(filename)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error opening file: %v\n", err)
		return
	}
	defer file.Close()

	blocks := make(map[int]*BlockContent)
	var stopReason string
	var hasMessageStart bool
	var hasMessageStop bool
	totalEvents := 0

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		jsonStr := strings.TrimPrefix(line, "data: ")
		if jsonStr == "" || jsonStr == "[DONE]" {
			continue
		}

		var event AnthropicEvent
		if err := json.Unmarshal([]byte(jsonStr), &event); err != nil {
			continue
		}

		totalEvents++

		switch event.Type {
		case "message_start":
			hasMessageStart = true

		case "message_stop":
			hasMessageStop = true

		case "message_delta":
			var msgDelta MessageDelta
			if err := json.Unmarshal(event.Delta, &msgDelta); err == nil {
				if msgDelta.StopReason != "" {
					stopReason = msgDelta.StopReason
				}
			}

		case "content_block_start":
			var block ContentBlock
			if err := json.Unmarshal(event.ContentBlock, &block); err == nil {
				idx := 0
				if event.Index != nil {
					idx = *event.Index
				}
				blocks[idx] = &BlockContent{
					Type:      block.Type,
					ToolName:  block.Name,
					ToolID:    block.ID,
					Index:     idx,
					Completed: false,
				}
			}

		case "content_block_delta":
			idx := 0
			if event.Index != nil {
				idx = *event.Index
			}

			if blocks[idx] == nil {
				blocks[idx] = &BlockContent{Index: idx}
			}

			// Try to parse as different delta types
			var textDelta TextDelta
			if json.Unmarshal(event.Delta, &textDelta) == nil {
				if textDelta.Type == "text_delta" {
					blocks[idx].Text += textDelta.Text
				}
			}

			var thinkingDelta ThinkingDelta
			if json.Unmarshal(event.Delta, &thinkingDelta) == nil {
				if thinkingDelta.Type == "thinking_delta" {
					blocks[idx].Thinking += thinkingDelta.Thinking
				}
			}

			var inputDelta InputJSONDelta
			if json.Unmarshal(event.Delta, &inputDelta) == nil {
				if inputDelta.Type == "input_json_delta" {
					blocks[idx].ToolInput += inputDelta.PartialJSON
				}
			}

		case "content_block_stop":
			idx := 0
			if event.Index != nil {
				idx = *event.Index
			}
			if blocks[idx] != nil {
				blocks[idx].Completed = true
			}
		}
	}

	// Print analysis
	fmt.Printf("Events: %d | Message: start=%v stop=%v | Stop reason: %q\n",
		totalEvents, hasMessageStart, hasMessageStop, stopReason)

	// Sort blocks by index
	var indices []int
	for idx := range blocks {
		indices = append(indices, idx)
	}
	sort.Ints(indices)

	for _, idx := range indices {
		block := blocks[idx]
		fmt.Printf("  Block %d [%s]: ", idx, block.Type)

		if block.Type == "text" {
			if block.Text == "" {
				fmt.Printf("⚠️  EMPTY TEXT\n")
			} else {
				text := block.Text
				if len(text) > 80 {
					text = text[:77] + "..."
				}
				fmt.Printf("%d chars: %q\n", len(block.Text), text)
			}
		} else if block.Type == "thinking" {
			if block.Thinking == "" {
				fmt.Printf("⚠️  EMPTY THINKING\n")
			} else {
				text := block.Thinking
				if len(text) > 80 {
					text = text[:77] + "..."
				}
				fmt.Printf("%d chars: %q\n", len(block.Thinking), text)
			}
		} else if block.Type == "tool_use" {
			fmt.Printf("tool=%q input=%d chars completed=%v\n",
				block.ToolName, len(block.ToolInput), block.Completed)
		} else {
			fmt.Printf("unknown type\n")
		}
	}

	// Check for issues
	if !hasMessageStart {
		fmt.Printf("  ⚠️  MISSING message_start event\n")
	}
	if !hasMessageStop {
		fmt.Printf("  ⚠️  MISSING message_stop event\n")
	}

	if len(blocks) == 0 {
		fmt.Printf("  ⚠️  NO CONTENT BLOCKS\n")
	} else {
		for _, idx := range indices {
			block := blocks[idx]
			if !block.Completed {
				fmt.Printf("  ⚠️  Block %d not completed\n", idx)
			}
			if block.Type == "text" && block.Text == "" && block.Thinking == "" && block.ToolName == "" {
				fmt.Printf("  ⚠️  Block %d has no content\n", idx)
			}
		}
	}
}
