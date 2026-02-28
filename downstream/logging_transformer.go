package downstream

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/tmaxmax/go-sse"
)

type LoggingTransformer struct {
	file *os.File
}

func NewLoggingTransformer(logDir string) (*LoggingTransformer, error) {
	if logDir == "" {
		return nil, nil
	}
	if err := os.MkdirAll(logDir, 0755); err != nil {
		return nil, fmt.Errorf("create log dir: %w", err)
	}

	timestamp := time.Now().Format("2006-01-02_15-04-05")
	filename := fmt.Sprintf("sse_%s.log", timestamp)
	filePath := filepath.Join(logDir, filename)

	file, err := os.Create(filePath)
	if err != nil {
		return nil, fmt.Errorf("create log file: %w", err)
	}

	return &LoggingTransformer{
		file: file,
	}, nil
}

func (l *LoggingTransformer) Transform(event *sse.Event) {
	if l == nil || l.file == nil || event.Data == "" {
		return
	}
	l.file.Write([]byte("data: " + event.Data + "\n\n"))
}

func (l *LoggingTransformer) Close() {
	if l == nil || l.file == nil {
		return
	}
	l.file.Close()
}
