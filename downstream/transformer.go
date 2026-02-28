package downstream

import "github.com/tmaxmax/go-sse"

type SSETransformer interface {
	Transform(event *sse.Event)
}
