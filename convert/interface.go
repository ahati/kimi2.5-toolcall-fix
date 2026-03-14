package convert

import (
	"io"

	"ai-proxy/transform"
)

type RequestConverter interface {
	Convert(body []byte) ([]byte, error)
}

type ResponseTransformer interface {
	transform.SSETransformer
}

type ConverterPair struct {
	Request  RequestConverter
	Response func(w io.Writer) transform.SSETransformer
}
