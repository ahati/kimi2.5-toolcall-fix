package convert

import (
	"io"

	"ai-proxy/transform"
)

// RequestConverter converts request bodies between formats.
type RequestConverter interface {
	// Convert transforms a request body from one format to another.
	Convert(body []byte) ([]byte, error)
}

// ResponseTransformer transforms SSE response streams between formats.
// It wraps the SSETransformer interface for use in conversion pipelines.
type ResponseTransformer interface {
	transform.SSETransformer
}

// ConverterPair holds matching request and response converters.
type ConverterPair struct {
	Request  RequestConverter
	Response func(io.Writer) ResponseTransformer
}
