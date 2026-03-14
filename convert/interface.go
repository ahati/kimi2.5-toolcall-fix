package convert

import (
	"io"

	"ai-proxy/transform"
	"github.com/tmaxmax/go-sse"
)

type RequestConverter interface {
	Convert(body []byte) ([]byte, error)
}

type ResponseTransformer interface {
	Transform(event *sse.Event) error
	Flush() error
	Close() error
}

type ConverterPair struct {
	Request  RequestConverter
	Response func(io.Writer) ResponseTransformer
}

var _ transform.SSETransformer = (ResponseTransformer)(nil)
