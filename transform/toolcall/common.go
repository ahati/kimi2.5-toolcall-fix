package toolcall

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"ai-proxy/types"
)

func intPtr(i int) *int {
	return &i
}

func parseToolCallID(raw string, index int) string {
	raw = strings.TrimSpace(raw)
	if strings.HasPrefix(raw, "toolu_") || strings.HasPrefix(raw, "call_") {
		return raw
	}
	return fmt.Sprintf("toolu_%d_%d", index, time.Now().UnixMilli())
}

func parseFunctionName(raw string) string {
	raw = strings.TrimSpace(raw)
	if i := strings.Index(raw, "."); i >= 0 {
		raw = raw[i+1:]
	}
	if i := strings.LastIndex(raw, ":"); i >= 0 {
		raw = raw[:i]
	}
	return raw
}

func serializeAnthropicEvent(event types.Event) []byte {
	data, err := json.Marshal(event)
	if err != nil {
		// Return minimal valid event on marshal error
		return []byte(fmt.Sprintf("event: %s\ndata: {}\n\n", event.Type))
	}
	return []byte(fmt.Sprintf("event: %s\ndata: %s\n\n", event.Type, string(data)))
}

func marshalJSON(v interface{}) []byte {
	data, err := json.Marshal(v)
	if err != nil {
		return []byte("{}")
	}
	return data
}
