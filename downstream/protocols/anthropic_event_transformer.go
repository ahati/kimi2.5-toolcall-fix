/*******************************************************************************
 * @file    anthropic_event_transformer.go
 * @brief   Anthropic SSE event transformer for tool call extraction.
 *
 * @details This file implements the AnthropicEventTransformer which processes
 *          Server-Sent Events (SSE) from the Anthropic API and transforms
 *          tool calls embedded in thinking content blocks into proper
 *          Anthropic tool_use content blocks.
 *
 *          The transformer uses a state machine to parse special tokens
 *          that delineate tool calls within the thinking stream:
 *          - <|tool_calls_section_begin|> : Start of tool calls section
 *          - <|tool_call_begin|>          : Start of individual tool call
 *          - <|tool_call_argument_begin|> : Start of tool call arguments
 *          - <|tool_call_end|>           : End of individual tool call
 *          - <|tool_calls_section_end|>  : End of tool calls section
 *
 * @note    This transformer is designed to work with the Kimi K2.5 model
 *          which outputs tool calls in a non-standard format within
 *          thinking blocks, requiring transformation to proper Anthropic
 *          API format.
 *
 * @copyright Copyright (c) 2024
 ******************************************************************************/

package protocols

import (
	"encoding/json"
	"io"
	"strings"

	"ai-proxy/types"

	"github.com/tmaxmax/go-sse"
)

/*******************************************************************************
 * @struct   AnthropicEventTransformer
 * @brief    Transforms Anthropic SSE events with embedded tool calls.
 *
 * @details  This transformer processes SSE events from the Anthropic API
 *           and detects tool calls embedded within thinking content blocks.
 *           When tool calls are detected, they are extracted and emitted
 *           as proper tool_use content blocks in the Anthropic streaming
 *           format.
 *
 * @note     The transformer maintains state across multiple events to handle
 *           tool calls that may span multiple SSE chunks.
 ******************************************************************************/
type AnthropicEventTransformer struct {
	output     io.Writer      /*!< Output writer for transformed SSE events.        */
	messageID  string         /*!< Current message ID from message_start event.      */
	blockIndex int            /*!< Next available content block index.              */
	inThinking bool           /*!< Flag: currently processing a thinking block.      */
	inText     bool           /*!< Flag: currently processing a text block.           */
	state      anthropicState /*!< Current state machine state.                       */
	buf        string         /*!< Buffer for incomplete tool call token parsing.    */
	toolIdx    int            /*!< Counter for processed tool calls.                 */
	currentID  string         /*!< Current tool call ID being processed.             */
}

/*******************************************************************************
 * @enum     anthropicState
 * @brief    State machine states for tool call parsing.
 *
 * @details  The state machine progresses through these states as it
 *           encounters special tokens in the thinking stream:
 *           - Idle: Not currently parsing tool calls
 *           - InSection: Inside a tool_calls_section
 *           - ReadingID: Reading tool call ID/name
 *           - ReadingArgs: Reading tool call arguments
 *           - Trailing: Processing trailing content after section end
 ******************************************************************************/
type anthropicState int

/*******************************************************************************
 * @brief    State machine constant definitions.
 *
 * @details  These constants represent the valid states of the tool call
 *           parsing state machine. Each state corresponds to a specific
 *           phase in the tool call extraction process.
 ******************************************************************************/
const (
	anthropicStateIdle        anthropicState = iota /*!< Not in a tool calls section.           */
	anthropicStateInSection                         /*!< Inside tool_calls_section_begin/end.   */
	anthropicStateReadingID                         /*!< Reading tool ID after call_begin.      */
	anthropicStateReadingArgs                       /*!< Reading arguments after argument_begin. */
	anthropicStateTrailing                          /*!< Processing content after section end.  */
)

/*******************************************************************************
 * @brief    Special token constants for tool call detection.
 *
 * @details  These tokens are emitted by the upstream model to delineate
 *           tool calls within the thinking content stream. The transformer
 *           detects these tokens and uses them to structure proper tool_use
 *           content blocks.
 *
 * @note     Token format: <|prefix|> with descriptive suffixes.
 ******************************************************************************/
const (
	tokSectionBegin = "<|tool_calls_section_begin|>" /*!< Marks start of tool calls section.   */
	tokCallBegin    = "<|tool_call_begin|>"          /*!< Marks start of individual tool call. */
	tokArgBegin     = "<|tool_call_argument_begin|>" /*!< Marks start of tool arguments.       */
	tokCallEnd      = "<|tool_call_end|>"            /*!< Marks end of individual tool call.   */
	tokSectionEnd   = "<|tool_calls_section_end|>"   /*!< Marks end of tool calls section.     */
)

/*******************************************************************************
 * @brief    Creates a new AnthropicEventTransformer instance.
 *
 * @details  Initializes the transformer with the provided output writer
 *           and sets the initial state to idle. The transformer is ready
 *           to process SSE events immediately after creation.
 *
 * @param    output The io.Writer to receive transformed SSE events.
 *
 * @return   Pointer to a newly allocated AnthropicEventTransformer.
 *
 * @pre      output must not be nil.
 * @post     Transformer is initialized in idle state with empty buffer.
 *
 * @note     The caller is responsible for ensuring the output writer
 *           remains valid for the lifetime of the transformer.
 ******************************************************************************/
func NewAnthropicEventTransformer(output io.Writer) *AnthropicEventTransformer {
	return &AnthropicEventTransformer{
		output: output,
		state:  anthropicStateIdle,
	}
}

/*******************************************************************************
 * @brief    Transforms an incoming SSE event.
 *
 * @details  This is the main entry point for SSE event processing. It
 *           parses the event data as JSON, determines the event type,
 *           and delegates to the appropriate handler method. Special
 *           handling is applied to thinking deltas that may contain
 *           embedded tool calls.
 *
 * @param    event Pointer to the SSE event to transform. May be nil.
 *
 * @pre      The transformer must be properly initialized.
 * @post     Event is either written to output or buffered for further
 *           processing depending on the current state.
 *
 * @note     Empty or "[DONE]" events are passed through with minimal
 *           processing. Unparseable events result in an error event.
 *
 * @see      handleMessageStart, handleContentBlockStart, handleContentBlockDelta
 ******************************************************************************/
func (t *AnthropicEventTransformer) Transform(event *sse.Event) {
	/* Handle empty data or stream termination marker. */
	if event.Data == "" || event.Data == "[DONE]" {
		if event.Data == "[DONE]" {
			t.output.Write([]byte("data: [DONE]\n\n"))
		}
		return
	}

	/* Parse the event data as an Anthropic event structure. */
	var anthropicEvent types.Event
	if err := json.Unmarshal([]byte(event.Data), &anthropicEvent); err != nil {
		t.writeSSE([]byte("event: error\ndata: {\"type\": \"error\"}\n\n"))
		return
	}

	/* Dispatch to appropriate handler based on event type. */
	switch anthropicEvent.Type {
	case "message_start":
		t.handleMessageStart(&anthropicEvent)
	case "content_block_start":
		t.handleContentBlockStart(&anthropicEvent)
	case "content_block_delta":
		t.handleContentBlockDelta(&anthropicEvent)
	case "content_block_stop":
		t.handleContentBlockStop(&anthropicEvent)
	case "message_delta":
		t.handleMessageDelta(&anthropicEvent)
	case "message_stop", "ping":
		t.writeEvent(&anthropicEvent)
	default:
		t.writeEvent(&anthropicEvent)
	}
}

/*******************************************************************************
 * @brief    Handles message_start events.
 *
 * @details  Extracts the message ID from the event and resets the block
 *           index counter. The event is then passed through to the output.
 *
 * @param    event Pointer to the message_start event to process.
 *
 * @pre      event must not be nil.
 * @post     messageID is set if present in event; blockIndex is reset to 0.
 *
 * @note     The message ID is stored for potential use in constructing
 *           tool call IDs if not provided by the upstream model.
 ******************************************************************************/
func (t *AnthropicEventTransformer) handleMessageStart(event *types.Event) {
	if event.Message != nil {
		t.messageID = event.Message.ID
		t.blockIndex = 0
	}
	t.writeEvent(event)
}

/*******************************************************************************
 * @brief    Handles content_block_start events.
 *
 * @details  Determines the type of content block being started and sets
 *           the appropriate tracking flags (inThinking or inText). The
 *           block index is updated to track the next available index.
 *
 * @param    event Pointer to the content_block_start event to process.
 *
 * @pre      event must not be nil; event.ContentBlock must be valid JSON.
 * @post     inThinking or inText flags are set based on block type;
 *           blockIndex is updated if event.Index >= blockIndex.
 *
 * @note     Block types other than "thinking" or "text" do not affect
 *           tracking flags but are still passed through.
 ******************************************************************************/
func (t *AnthropicEventTransformer) handleContentBlockStart(event *types.Event) {
	var block types.ContentBlock
	if err := json.Unmarshal(event.ContentBlock, &block); err == nil {
		/* Set tracking flags based on block type for delta processing. */
		if block.Type == "thinking" {
			t.inThinking = true
		} else if block.Type == "text" {
			t.inText = true
		}
	}
	t.writeEvent(event)

	/* Update block index to next available slot. */
	if event.Index != nil && *event.Index >= t.blockIndex {
		t.blockIndex = *event.Index + 1
	}
}

/*******************************************************************************
 * @brief    Handles content_block_delta events.
 *
 * @details  Routes delta events to appropriate handlers based on current
 *           content block type. Thinking deltas may contain embedded tool
 *           calls and are processed through the state machine. Text deltas
 *           are passed through directly.
 *
 * @param    event Pointer to the content_block_delta event to process.
 *
 * @pre      event must not be nil; event.Delta must be valid JSON if processed.
 * @post     Event may be transformed or buffered depending on content.
 *
 * @note     For thinking blocks, tool calls embedded in the thinking text
 *           are extracted and emitted as separate tool_use blocks.
 ******************************************************************************/
func (t *AnthropicEventTransformer) handleContentBlockDelta(event *types.Event) {
	/* Process thinking deltas for potential embedded tool calls. */
	if t.inThinking {
		var delta types.ThinkingDelta
		if err := json.Unmarshal(event.Delta, &delta); err == nil && delta.Type == "thinking_delta" {
			t.processThinkingDelta(delta.Thinking, event.Index)
			return
		}
	}

	/* Pass through text deltas unchanged. */
	if t.inText {
		var delta types.TextDelta
		if err := json.Unmarshal(event.Delta, &delta); err == nil && delta.Type == "text_delta" {
			t.writeEvent(event)
			return
		}
	}

	t.writeEvent(event)
}

/*******************************************************************************
 * @brief    Processes a thinking delta for embedded tool calls.
 *
 * @details  Checks if the thinking text contains tool call section markers.
 *           If markers are present, the text is buffered and processed
 *           through the state machine. Otherwise, the delta is emitted
 *           directly as a thinking_delta event.
 *
 * @param    thinking The thinking text content to process.
 * @param    index    Pointer to the content block index (may be nil).
 *
 * @pre      thinking may be empty but must be a valid string.
 * @post     Either the delta is written directly or added to buffer.
 *
 * @note     The buffer accumulates content across multiple delta events
 *           until a complete tool call can be parsed.
 ******************************************************************************/
func (t *AnthropicEventTransformer) processThinkingDelta(thinking string, index *int) {
	idx := 0
	if index != nil {
		idx = *index
	}

	/* Check for tool call section marker to determine processing path. */
	if !strings.Contains(thinking, tokSectionBegin) {
		/* No tool calls detected - emit thinking delta directly. */
		t.writeContentBlockDelta(idx, types.ThinkingDelta{
			Type:     "thinking_delta",
			Thinking: thinking,
		})
		return
	}

	/* Append to buffer for state machine processing. */
	t.buf += thinking
	t.processBuffer(idx)
}

/*******************************************************************************
 * @brief    Processes the buffer through the state machine.
 *
 * @details  Implements a state machine that parses tool call tokens from
 *           the accumulated buffer. The state machine transitions between
 *           states as it encounters special tokens, emitting appropriate
 *           SSE events for each parsed component.
 *
 * @param    idx The content block index for emitted events.
 *
 * @pre      Buffer may contain partial or complete tool call data.
 * @post     Complete tool calls are emitted; buffer contains remaining
 *           unparsed content; state reflects current parsing position.
 *
 * @note     State machine transitions:
 *           - Idle -> InSection: When tokSectionBegin found
 *           - InSection -> ReadingID: When tokCallBegin found
 *           - ReadingID -> ReadingArgs: When tokArgBegin found
 *           - ReadingArgs -> InSection: When tokCallEnd found
 *           - InSection -> Idle: When tokSectionEnd found
 *
 * @warning  This method modifies t.buf and t.state as side effects.
 ******************************************************************************/
func (t *AnthropicEventTransformer) processBuffer(idx int) {
	for {
		switch t.state {
		case anthropicStateIdle:
			/*
			 * IDLE STATE: Looking for tool_calls_section_begin token.
			 * Any content before the token is emitted as thinking content.
			 */
			secIdx := strings.Index(t.buf, tokSectionBegin)
			if secIdx < 0 {
				/* No section begin found - emit buffered content as thinking. */
				if t.buf != "" {
					t.writeContentBlockDelta(idx, types.ThinkingDelta{
						Type:     "thinking_delta",
						Thinking: t.buf,
					})
					t.buf = ""
				}
				return
			}
			/* Emit any content before the section marker as thinking. */
			if secIdx > 0 {
				t.writeContentBlockDelta(idx, types.ThinkingDelta{
					Type:     "thinking_delta",
					Thinking: t.buf[:secIdx],
				})
			}
			/* Remove processed content and transition to in-section state. */
			t.buf = t.buf[secIdx+len(tokSectionBegin):]
			t.state = anthropicStateInSection

		case anthropicStateInSection:
			/*
			 * IN-SECTION STATE: Looking for call_begin or section_end tokens.
			 * Must check for section_end first to handle early termination.
			 */
			callIdx := strings.Index(t.buf, tokCallBegin)
			endIdx := strings.Index(t.buf, tokSectionEnd)

			/* Check if section ends before next call (or no more calls). */
			if endIdx >= 0 && (callIdx < 0 || endIdx < callIdx) {
				/* Extract any trailing content after section end marker. */
				trailing := t.buf[endIdx+len(tokSectionEnd):]
				t.buf = ""
				t.state = anthropicStateIdle
				/* Emit trailing content as thinking if present. */
				if trailing != "" {
					t.writeContentBlockDelta(idx, types.ThinkingDelta{
						Type:     "thinking_delta",
						Thinking: trailing,
					})
				}
				return
			}
			/* Wait for more data if call_begin not found yet. */
			if callIdx < 0 {
				return
			}
			/* Advance past call_begin marker to read tool ID. */
			t.buf = t.buf[callIdx+len(tokCallBegin):]
			t.state = anthropicStateReadingID

		case anthropicStateReadingID:
			/*
			 * READING-ID STATE: Extracting tool call ID and name.
			 * ID/name is everything before arg_begin token.
			 */
			argIdx := strings.Index(t.buf, tokArgBegin)
			if argIdx < 0 {
				/* Need more data to complete ID parsing. */
				return
			}
			/* Parse and normalize the tool call ID and name. */
			rawID := strings.TrimSpace(t.buf[:argIdx])
			t.currentID = t.normalizeID(rawID)
			name := t.parseName(rawID)
			t.buf = t.buf[argIdx+len(tokArgBegin):]
			t.state = anthropicStateReadingArgs

			/*
			 * Close the current thinking block and emit tool_use start.
			 * The thinking block index is closed, then a new tool_use
			 * block is started at the next available block index.
			 */
			t.writeContentBlockStop(idx)
			t.blockIndex++
			t.writeToolUseStart(t.blockIndex, t.currentID, name)

		case anthropicStateReadingArgs:
			/*
			 * READING-ARGS STATE: Streaming tool call arguments as JSON.
			 * Arguments continue until call_end token is encountered.
			 */
			endIdx := strings.Index(t.buf, tokCallEnd)
			if endIdx < 0 {
				/* Stream available argument content and wait for more. */
				if t.buf != "" {
					t.writeInputJSONDelta(t.blockIndex, t.buf)
					t.buf = ""
				}
				return
			}
			/* Emit final argument content before call_end marker. */
			args := t.buf[:endIdx]
			if args != "" {
				t.writeInputJSONDelta(t.blockIndex, args)
			}
			/* Close the tool_use block and prepare for next tool call. */
			t.writeContentBlockStop(t.blockIndex)
			t.blockIndex++
			t.buf = t.buf[endIdx+len(tokCallEnd):]
			t.toolIdx++
			t.state = anthropicStateInSection
		}
	}
}

/*******************************************************************************
 * @brief    Handles content_block_stop events.
 *
 * @details  Resets tracking flags and flushes any remaining buffered
 *           content. This ensures partial tool calls are properly
 *           emitted before the block ends.
 *
 * @param    event Pointer to the content_block_stop event to process.
 *
 * @pre      event must not be nil.
 * @post     inThinking and inText flags are reset to false; state is
 *           reset to idle; buffer is cleared.
 *
 * @note     Flushing is only performed when state indicates active tool
 *           call parsing (state != idle with buffered content).
 ******************************************************************************/
func (t *AnthropicEventTransformer) handleContentBlockStop(event *types.Event) {
	/* Reset content block type tracking flags. */
	t.inThinking = false
	t.inText = false

	/* Flush any remaining buffered content if mid-parse. */
	if t.buf != "" && t.state != anthropicStateIdle {
		idx := 0
		if event.Index != nil {
			idx = *event.Index
		}
		t.flushBuffer(idx)
	}

	/* Reset state machine and clear buffer. */
	t.state = anthropicStateIdle
	t.buf = ""
	t.writeEvent(event)
}

/*******************************************************************************
 * @brief    Flushes remaining buffered content.
 *
 * @details  Emits any content remaining in the buffer based on the
 *           current state machine state. Only ReadingArgs state
 *           requires special handling for partial JSON content.
 *
 * @param    idx The content block index for emitted events.
 *
 * @pre      Buffer may contain partial tool call data.
 * @post     Buffer content is emitted if applicable; buffer unchanged.
 *
 * @note     This method does not clear the buffer; the caller is
 *           responsible for buffer management.
 ******************************************************************************/
func (t *AnthropicEventTransformer) flushBuffer(idx int) {
	switch t.state {
	case anthropicStateReadingArgs:
		/* Emit remaining argument content as JSON delta. */
		if t.buf != "" {
			t.writeInputJSONDelta(t.blockIndex, t.buf)
		}
	}
}

/*******************************************************************************
 * @brief    Handles message_delta events.
 *
 * @details  Passes message_delta events through unchanged. These events
 *           typically contain usage statistics or other metadata.
 *
 * @param    event Pointer to the message_delta event to process.
 *
 * @pre      event must not be nil.
 * @post     Event is written to output unchanged.
 ******************************************************************************/
func (t *AnthropicEventTransformer) handleMessageDelta(event *types.Event) {
	t.writeEvent(event)
}

/*******************************************************************************
 * @brief    Normalizes a tool call ID to Anthropic format.
 *
 * @details  Ensures the ID follows Anthropic's tool call ID format.
 *           IDs starting with "call_" or "toolu_" are returned unchanged.
 *           Other IDs are prefixed with "toolu_" and colons are replaced
 *           with underscores.
 *
 * @param    raw The raw tool call ID to normalize.
 *
 * @return   The normalized ID string conforming to Anthropic format.
 *
 * @pre      raw may be empty but must be a valid string.
 * @post     Returned string is either unchanged or prefixed with "toolu_".
 *
 * @note     Anthropic tool call IDs typically start with "toolu_" prefix.
 ******************************************************************************/
func (t *AnthropicEventTransformer) normalizeID(raw string) string {
	raw = strings.TrimSpace(raw)
	/* Return unchanged if already properly prefixed. */
	if strings.HasPrefix(raw, "call_") || strings.HasPrefix(raw, "toolu_") {
		return raw
	}
	/* Prefix and sanitize for Anthropic format. */
	return "toolu_" + strings.ReplaceAll(raw, ":", "_")
}

/*******************************************************************************
 * @brief    Extracts the tool name from a raw ID string.
 *
 * @details  Parses the tool name from the raw ID format which may include
 *           namespace prefix and call identifier. The name is extracted
 *           by removing the call identifier (after last colon) and
 *           taking the final component (after last dot).
 *
 * @param    raw The raw tool call ID/name string to parse.
 *
 * @return   The extracted tool name string.
 *
 * @pre      raw may be empty but must be a valid string.
 * @post     Returned string contains only the tool name without
 *           namespace or call identifier components.
 *
 * @note     Input format examples:
 *           - "namespace.toolname:callid" -> "toolname"
 *           - "toolname:callid" -> "toolname"
 *           - "toolname" -> "toolname"
 ******************************************************************************/
func (t *AnthropicEventTransformer) parseName(raw string) string {
	raw = strings.TrimSpace(raw)
	/* Remove call identifier after last colon. */
	if i := strings.LastIndex(raw, ":"); i >= 0 {
		raw = raw[:i]
	}
	/* Extract final component after last dot. */
	if i := strings.LastIndex(raw, "."); i >= 0 {
		raw = raw[i+1:]
	}
	return raw
}

/*******************************************************************************
 * @brief    Writes an event to the output stream.
 *
 * @details  Serializes the event to JSON and formats it as an SSE event
 *           with the appropriate event type header.
 *
 * @param    event Pointer to the event to write.
 *
 * @pre      event must not be nil; event.Type must be set.
 * @post     Event is written to output in SSE format.
 *
 * @note     Output format: "event: <type>\ndata: <json>\n\n"
 ******************************************************************************/
func (t *AnthropicEventTransformer) writeEvent(event *types.Event) {
	data, _ := json.Marshal(event)
	line := "event: " + event.Type + "\ndata: " + string(data) + "\n\n"
	t.output.Write([]byte(line))
}

/*******************************************************************************
 * @brief    Writes raw bytes to the output stream.
 *
 * @details  Directly writes the provided bytes without any formatting
 *           or modification. Used for pre-formatted SSE data.
 *
 * @param    data The bytes to write to output.
 *
 * @pre      data may be empty but should be valid SSE format if non-empty.
 * @post     Data is written to output stream.
 ******************************************************************************/
func (t *AnthropicEventTransformer) writeSSE(data []byte) {
	t.output.Write(data)
}

/*******************************************************************************
 * @brief    Writes a content_block_start event.
 *
 * @details  Constructs and emits a content_block_start SSE event with
 *           the specified index and content block data.
 *
 * @param    index The content block index for this block.
 * @param    block The content block data to include.
 *
 * @pre      block must have Type field set.
 * @post     A content_block_start event is written to output.
 ******************************************************************************/
func (t *AnthropicEventTransformer) writeContentBlockStart(index int, block types.ContentBlock) {
	blockJSON, _ := json.Marshal(block)
	event := types.Event{
		Type:         "content_block_start",
		Index:        intPtr(index),
		ContentBlock: blockJSON,
	}
	t.writeEvent(&event)
}

/*******************************************************************************
 * @brief    Writes a content_block_delta event.
 *
 * @details  Constructs and emits a content_block_delta SSE event with
 *           the specified index and delta data.
 *
 * @param    index The content block index for this delta.
 * @param    delta The delta data (must be JSON-serializable).
 *
 * @pre      delta must be JSON-serializable.
 * @post     A content_block_delta event is written to output.
 ******************************************************************************/
func (t *AnthropicEventTransformer) writeContentBlockDelta(index int, delta interface{}) {
	deltaJSON, _ := json.Marshal(delta)
	event := types.Event{
		Type:  "content_block_delta",
		Index: intPtr(index),
		Delta: deltaJSON,
	}
	t.writeEvent(&event)
}

/*******************************************************************************
 * @brief    Writes a content_block_stop event.
 *
 * @details  Emits a content_block_stop SSE event to signal the end of
 *           a content block at the specified index.
 *
 * @param    index The content block index that is stopping.
 *
 * @pre      index must be a valid content block index.
 * @post     A content_block_stop event is written to output.
 ******************************************************************************/
func (t *AnthropicEventTransformer) writeContentBlockStop(index int) {
	event := types.Event{
		Type:  "content_block_stop",
		Index: intPtr(index),
	}
	t.writeEvent(&event)
}

/*******************************************************************************
 * @brief    Writes a tool_use content block start event.
 *
 * @details  Constructs and emits a content_block_start event for a
 *           tool_use block. The block is initialized with an empty
 *           JSON object as input, which will be populated by
 *           subsequent input_json_delta events.
 *
 * @param    index The content block index for this tool_use block.
 * @param    id    The unique identifier for this tool call.
 * @param    name  The name of the tool being called.
 *
 * @pre      id and name must not be empty.
 * @post     A content_block_start event for tool_use is written.
 *
 * @note     The input field is initialized as an empty JSON object "{}".
 ******************************************************************************/
func (t *AnthropicEventTransformer) writeToolUseStart(index int, id, name string) {
	block := types.ContentBlock{
		Type:  "tool_use",
		ID:    id,
		Name:  name,
		Input: json.RawMessage("{}"),
	}
	t.writeContentBlockStart(index, block)
}

/*******************************************************************************
 * @brief    Writes an input_json_delta event for tool arguments.
 *
 * @details  Emits a content_block_delta event containing partial JSON
 *           for the tool call arguments. These deltas accumulate to
 *           form the complete tool input JSON.
 *
 * @param    index The content block index for this tool_use block.
 * @param    args  The partial JSON string to emit.
 *
 * @pre      index must correspond to an active tool_use block.
 * @post     An input_json_delta event is written to output.
 *
 * @note     The args string may be partial JSON that will be
 *           concatenated with other deltas by the client.
 ******************************************************************************/
func (t *AnthropicEventTransformer) writeInputJSONDelta(index int, args string) {
	delta := types.InputJSONDelta{
		Type:        "input_json_delta",
		PartialJSON: args,
	}
	t.writeContentBlockDelta(index, delta)
}

/*******************************************************************************
 * @brief    Flushes any pending buffered content.
 *
 * @details  Called at stream end to ensure all buffered content is
 *           processed. Currently a no-op as the state machine
 *           handles flushing during normal operation.
 *
 * @pre      Transformer must be properly initialized.
 * @post     Any pending content should be written to output.
 ******************************************************************************/
func (t *AnthropicEventTransformer) Flush() {
}

/*******************************************************************************
 * @brief    Releases resources held by the transformer.
 *
 * @details  Cleans up any resources held by the transformer. Currently
 *           a no-op as no resources require explicit cleanup.
 *
 * @pre      Transformer must be properly initialized.
 * @post     Transformer resources are released.
 ******************************************************************************/
func (t *AnthropicEventTransformer) Close() {
}

/*******************************************************************************
 * @brief    Creates a pointer to an int value.
 *
 * @details  Helper function to create int pointers for use in event
 *           structures that require pointer types for JSON omitempty
 *           behavior.
 *
 * @param    i The integer value to point to.
 *
 * @return   Pointer to the provided integer value.
 *
 * @post     Returned pointer references the integer value.
 ******************************************************************************/
func intPtr(i int) *int {
	return &i
}
