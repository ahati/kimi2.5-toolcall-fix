# Logging Module Requirements and Design

## Overview

Structured per-request logging module capturing bidirectional traffic between client and upstream LLM API with temporal correlation. All four data streams are captured in a single structured JSON file per request.

## Requirements

### Functional Requirements

1. **Capture All Traffic Directions**
   - Downstream TX: Client request entering the proxy
   - Upstream TX: Request sent to upstream LLM API
   - Upstream RX: Response received from upstream LLM API
   - Downstream RX: Response sent back to client

2. **Structured Data Storage**
   - Request/response bodies parsed as JSON when valid
   - Headers captured (sanitized)
   - SSE chunks captured individually with timing
   - Single JSON file per request for temporal correlation

3. **Timing Information**
   - Absolute timestamps (RFC3339)
   - Relative offsets from request start (milliseconds)
   - Latency analysis between capture points

4. **Data Sanitization**
   - Authorization headers masked as `***`
   - Sensitive headers configurable for masking

5. **Non-Functional Requirements**
   - No capture size limits (full body capture)
   - No retention policies (files persist indefinitely)
   - Both streaming and non-streaming responses handled uniformly
   - Zero impact on streaming performance (async flush)

## Design

### Module Structure

```
logging/
  capture.go      # Core types and context
  recorder.go     # Per-request recording logic
  storage.go      # File writing and directory management
  middleware.go   # Gin framework integration
```

### Core Types

#### CaptureContext

Flows through the entire request lifecycle, attached to `context.Context`. Uses request-provided ID, never generates new IDs.

```go
type CaptureContext struct {
    RequestID string  // From X-Request-ID header or request body 'id' field
    StartTime time.Time
    Recorder  *RequestRecorder
}
```

#### RequestRecorder

Accumulates all data for a single request. Thread-safe via mutex.

```go
type RequestRecorder struct {
    mu sync.Mutex
    
    // Metadata
    RequestID   string
    StartedAt   time.Time
    Method      string
    Path        string
    ClientIP    string
    
    // Four capture points
    DownstreamRequest  *HTTPRequestCapture
    UpstreamRequest    *HTTPRequestCapture
    UpstreamResponse   *HTTPResponseCapture
    DownstreamResponse *HTTPResponseCapture
}
```

#### HTTPRequestCapture

Structured request data at capture point.

```go
type HTTPRequestCapture struct {
    At      time.Time
    Headers map[string]string
    Body    json.RawMessage
    RawBody []byte
}
```

#### HTTPResponseCapture

Streaming response with SSE chunks.

```go
type HTTPResponseCapture struct {
    StatusCode int
    Headers    map[string]string
    Chunks     []SSEChunk
}

type SSEChunk struct {
    OffsetMS int64
    Event    string
    Data     json.RawMessage
    Raw      string
}
```

### Storage Format

Single JSON file per request with temporal correlation.

```json
{
  "request_id": "req_7f8a9b2c",
  "started_at": "2025-03-03T12:00:00.123Z",
  "duration_ms": 2500,
  "method": "POST",
  "path": "/v1/chat/completions",
  "client_ip": "192.168.1.100",
  
  "downstream_request": {
    "at": "2025-03-03T12:00:00.123Z",
    "headers": {
      "content-type": "application/json",
      "authorization": "***"
    },
    "body": {
      "model": "kimi-k2.5",
      "messages": [{"role": "user", "content": "Hello"}],
      "stream": true
    }
  },
  
  "upstream_request": {
    "at": "2025-03-03T12:00:00.135Z",
    "headers": {
      "content-type": "application/json",
      "authorization": "***"
    },
    "body": {
      "model": "kimi-k2.5",
      "messages": [{"role": "user", "content": "Hello"}],
      "stream": true
    }
  },
  
  "upstream_response": {
    "status_code": 200,
    "headers": {
      "content-type": "text/event-stream"
    },
    "chunks": [
      {
        "offset_ms": 245,
        "event": "",
        "data": {"choices": [{"delta": {"content": "Hi"}}]}
      },
      {
        "offset_ms": 312,
        "event": "",
        "data": {"choices": [{"delta": {"content": " there"}}]}
      }
    ]
  },
  
  "downstream_response": {
    "chunks": [
      {
        "offset_ms": 248,
        "data": {"choices": [{"delta": {"content": "Hi"}}]}
      },
      {
        "offset_ms": 315,
        "data": {"choices": [{"delta": {"content": " there"}}]}
      }
    ]
  }
}
```

### Directory Layout

Uses request-provided ID for filename. If multiple requests have same ID, suffix with timestamp.

```
logs/
  2025-03-03/
    chatcmpl-abc123.json
    chatcmpl-abc123_20250303-120005.json  # duplicate ID, timestamp suffixed
    msg_01X8yJoxJWmMzN6xG1Y9sTPo.json
```

### Integration Points

| Component | Integration | Action |
|-----------|-------------|--------|
| Gin Router | Middleware | Create `CaptureContext`, attach to request context, record downstream request headers/body |
| upstream/client.go | Method parameter | Accept `CaptureContext`, record upstream request before sending |
| upstream/client.go | Response wrapping | Wrap `resp.Body` with `CapturingReader`, record SSE chunks as read |
| downstream/unified_handler.go | Response streaming | Record transformed chunks sent to client |
| downstream/unified_handler.go | Deferred close | Async flush `RequestRecorder` to disk |

### Capture Flow

```
1. Request Enters (Gin Middleware)
   ├─ Extract RequestID from X-Request-ID header or request body 'id' field
   ├─ Create CaptureContext with RequestID + StartTime
   ├─ Record downstream_request (headers + body)
   └─ Attach to c.Request.Context()
```

### Configuration

Uses existing `--sse-log-dir` flag (env: `SSELOG_DIR`). No new flags or environment variables.

```go
type CaptureConfig struct {
    Directory string // From --sse-log-dir / SSELOG_DIR
}
```

Capture is enabled when `SSELogDir` is non-empty. Disabled when empty.

### Implementation Notes

#### Thread Safety

- `RequestRecorder` uses `sync.Mutex` for all mutations
- SSE chunks append under lock, minimize lock time
- Final JSON marshal happens after all data collected

#### Sanitization

Headers masked by default:
- `Authorization`
- `X-Api-Key`
- `Cookie`
- `Set-Cookie`

Values replaced with `***`.

#### Error Handling

- JSON parse failures: Store raw string in `Raw` field, leave `Data` null
- I/O failures during capture: Log error, continue without capture data
- Storage failures: Log error, do not retry (best-effort logging)

#### Async Flush

To avoid blocking response streaming:

```go
// In handler, defer after streaming completes
defer func() {
    if ctx := GetCaptureContext(c.Request.Context()); ctx != nil {
        go func() {
            if err := storage.Write(ctx.Recorder); err != nil {
                logging.ErrorMsg("Failed to write capture: %v", err)
            }
        }()
    }
}()
```

### Migration from LoggingTransformer

The existing `LoggingTransformer` will be replaced. The `--sse-log-dir` flag and `SSELOG_DIR` env var remain unchanged.

Changes:
1. Remove `downstream/logging_transformer.go`
2. Remove transformer instantiation from handlers
3. New capture system uses same `SSELogDir` config field
4. Output format changes from raw SSE to structured JSON

### Testing Strategy

1. **Unit tests** for each component (recorder, storage, sanitization)
2. **Integration tests** with test HTTP server simulating upstream
3. **End-to-end tests** verifying complete capture file matches expected structure
4. **Performance tests** ensuring async flush doesn't impact throughput

### Request ID Extraction

The capture system extracts the request ID from the incoming request in priority order:

1. `X-Request-ID` header (if present)
2. Request body JSON `id` field (if present)
3. `x-request-id` header (lowercase variant)

If no ID is found, the request is not captured (capture skipped).

This design ensures:
- No synthetic IDs generated
- Full traceability to original client request
- Compatible with existing request tracing systems
- Filenames match the request ID exactly

### Open Questions

1. Body parsing: Strict JSON only, or attempt to parse SSE data lines?
2. Concurrent writes: Directory-level lock needed for duplicate ID handling?

## Implementation Status

### Completed Features (March 2026)

✅ **ResponseRecorder Implementation**
- Fixed `downstream_response` to capture transformed events (not upstream events)
- All 3 handlers use `ResponseRecorder` wrapper
- Proper SSE parsing to extract event type and data

✅ **Request ID Extraction**
- OpenAI: Extracted from first SSE chunk `id` field
- Anthropic: Extracted from `message_start` event `message.id` field
- Errors: Fallback hash-based ID (`err_<hash>_<timestamp>`)

✅ **Endpoint Coverage**
All 3 endpoints fully logged:
1. `POST /v1/chat/completions` (OpenAI format)
2. `POST /v1/messages` (Anthropic format)
3. `POST /v1/openai-to-anthropic/messages` (Bridge)

✅ **Error Response Logging**
- HTTP status codes captured
- Error response bodies logged
- Fallback request IDs for errors without upstream ID

## Log Organization

### Directory Structure

```
sse_logs/
└── YYYY-MM-DD/
    ├── YYYYMMDD-HHMMSS_<request_id>.json
    ├── YYYYMMDD-HHMMSS_<request_id>.json
    └── ...
```

### Filename Format

```
YYYYMMDD-HHMMSS_<request_id>.json
```

- `YYYYMMDD-HHMMSS`: Timestamp when request was received
- `<request_id>`: Request ID from upstream response
  - OpenAI format: `chatcmpl-<id>`
  - Anthropic format: `msg_<uuid>`
  - Error fallback: `err_<hash>_<timestamp>`

### Example Log Files

```
sse_logs/
└── 2026-03-03/
    ├── 20260303-213222_chatcmpl-90852f1db3124dc7.json
    ├── 20260303-213223_msg_acf67c1e-eb0e-475f-8f76-a5162bc7d9d1.json
    └── 20260303-213225_chatcmpl-b87d51b3ef1910c9.json
```

## Connection Details

### Request Lifecycle

1. **Client Connection**
   - TCP connection established
   - Request headers received
   - Request body read and logged

2. **Upstream Connection**
   - New HTTP connection to upstream
   - Request transformed and forwarded
   - Response streamed back

3. **Response Streaming**
   - SSE events received from upstream
   - Events transformed (if needed)
   - Events written to client
   - All events logged

4. **Connection Cleanup**
   - Upstream connection closed
   - Client connection closed
   - Log file written asynchronously

### Timing Information

Each log entry includes timing data:

- `started_at`: When proxy received request
- `duration_ms`: Total time from start to finish
- `offset_ms` (in chunks): Time offset from start for each SSE chunk

Example timing analysis:
```json
{
  "started_at": "2026-03-03T21:32:22.123456Z",
  "duration_ms": 3259,
  "downstream_response": {
    "chunks": [
      {"offset_ms": 1355, "event": "message_start"},
      {"offset_ms": 1591, "event": "content_block_delta"},
      {"offset_ms": 1591, "event": "message_delta"}
    ]
  }
}
```

## ResponseRecorder Implementation

### Background

Previously, `downstream_response` captured upstream events instead of transformed downstream events. This was fixed by implementing a `ResponseRecorder` wrapper.

### Implementation Details

**File:** `downstream/response_recorder.go`

```go
type ResponseRecorder struct {
    writer http.ResponseWriter
    chunks *[]logging.SSEChunk
    start  time.Time
}

func (r *ResponseRecorder) Write(data []byte) (int, error) {
    if len(data) > 0 {
        dataForLogging := extractDataPart(data)
        chunk := logging.NewSSEChunk(
            logging.OffsetMS(r.start),
            extractEventType(data),
            dataForLogging,
        )
        *r.chunks = append(*r.chunks, chunk)
    }
    return r.writer.Write(data)
}
```

The `ResponseRecorder`:
1. Wraps the HTTP response writer
2. Captures all bytes written to response
3. Parses SSE format to extract event type and data
4. Stores chunks for logging

### Handlers Using ResponseRecorder

All endpoints use `ResponseRecorder` via the unified handler:

**File:** `downstream/unified_handler.go`

```go
recorder := NewResponseRecorder(c.Writer, downstreamCapture)
transformer := adapter.CreateTransformer(recorder, types.StreamChunk{})
```

The `ResponseRecorder`:
1. Wraps the HTTP response writer
2. Captures all bytes written to response
3. Parses SSE format to extract event type and data
4. Stores chunks for logging

### Protocol Adapters

The unified handler works with all protocol adapters:

1. **OpenAIAdapter** - OpenAI format (`/v1/chat/completions`)
2. **AnthropicAdapter** - Anthropic format (`/v1/messages`)
3. **BridgeAdapter** - OpenAI→Anthropic bridge (`/v1/openai-to-anthropic/messages`)

## Error Handling

### Error Response Logging

Errors are logged with:
- HTTP status code
- Error response body
- Request context

Example error log:
```json
{
  "path": "/v1/openai-to-anthropic/messages",
  "request_id": "err_a1b2c3d4_1234567890",
  "upstream_response": {
    "status_code": 401
  },
  "downstream_response": {
    "status_code": 401,
    "chunks": [
      {
        "event": "error",
        "data": {"detail": "Invalid token."}
      }
    ]
  }
}
```

### Fallback Request IDs

When upstream doesn't provide a request ID (e.g., authentication errors):
- Generate fallback ID: `err_<hash>_<timestamp>`
- Hash based on error response body
- Ensures unique filenames even for errors

## Querying Logs

### Using jq

```bash
# Find all logs with tool calls
cat sse_logs/2026-03-03/*.json | jq 'select(.downstream_response.chunks[].data.content_block.type == "tool_use")'

# Find slow requests (>5 seconds)
cat sse_logs/2026-03-03/*.json | jq 'select(.duration_ms > 5000)'

# Count requests per endpoint
cat sse_logs/2026-03-03/*.json | jq -s 'group_by(.path) | map({path: .[0].path, count: length})'

# Extract all request IDs
cat sse_logs/2026-03-03/*.json | jq -r '.request_id'
```

### Finding Specific Logs

```bash
# By request ID
ls sse_logs/2026-03-03/*chatcmpl-90852f1db3124dc7.json

# By endpoint
ls sse_logs/2026-03-03/*_msg_*.json  # Anthropic
ls sse_logs/2026-03-03/*_chatcmpl*.json  # OpenAI

# By time range
ls -lt sse_logs/2026-03-03/ | grep "21:32"
```

## Performance Considerations

- Logging is asynchronous (doesn't block response)
- Log files written after response completes
- Storage uses exclusive file creation (prevents overwrites)
- Sensitive headers are sanitized (Authorization, API keys)

## Troubleshooting

### Missing Logs

1. Check `SSELOG_DIR` environment variable is set
2. Verify directory is writable
3. Check server logs for "Failed to write capture" errors

### Empty downstream_response

If `downstream_response.chunks` is empty:
- Check if ResponseRecorder is being used
- Verify transformer is writing to recorder, not directly to writer

### Request ID Extraction

Request IDs are extracted from:
- OpenAI: First SSE chunk `id` field
- Anthropic: `message_start` event `message.id` field
- Errors: Fallback hash-based ID

## Related Files

- `logging/middleware.go` - Capture middleware setup
- `logging/capture.go` - Capture context and types
- `logging/storage.go` - Log file storage
- `logging/recorder.go` - Response recording
- `downstream/response_recorder.go` - ResponseRecorder wrapper
- `downstream/handler.go` - OpenAI completions handler
- `downstream/anthropic_handler.go` - Anthropic messages handler
- `downstream/openai_to_anthropic_handler.go` - Bridge handler
