package logging

import (
	"bytes"
	"io"
	"log"
	"os"
	"strings"
	"sync"
	"testing"
)

func resetLoggers() {
	Info = nil
	Error = nil
	once = sync.Once{}
}

func captureOutput(f func()) (stdout string, stderr string) {
	oldStdout := os.Stdout
	oldStderr := os.Stderr

	rOut, wOut, _ := os.Pipe()
	rErr, wErr, _ := os.Pipe()
	os.Stdout = wOut
	os.Stderr = wErr

	f()

	wOut.Close()
	wErr.Close()

	var bufOut, bufErr bytes.Buffer
	io.Copy(&bufOut, rOut)
	io.Copy(&bufErr, rErr)

	os.Stdout = oldStdout
	os.Stderr = oldStderr

	return bufOut.String(), bufErr.String()
}

func TestInit(t *testing.T) {
	resetLoggers()

	Init()

	if Info == nil {
		t.Error("Info logger should be initialized")
	}
	if Error == nil {
		t.Error("Error logger should be initialized")
	}
}

func TestInit_Idempotent(t *testing.T) {
	resetLoggers()

	Init()
	firstInfo := Info
	firstError := Error

	Init()

	if Info != firstInfo {
		t.Error("Info logger should not change on second Init call")
	}
	if Error != firstError {
		t.Error("Error logger should not change on second Init call")
	}
}

func TestInfoMsg(t *testing.T) {
	resetLoggers()

	stdout, _ := captureOutput(func() {
		InfoMsg("test message %d", 42)
	})

	if !strings.Contains(stdout, "[INFO]") {
		t.Errorf("Expected [INFO] prefix, got: %s", stdout)
	}
	if !strings.Contains(stdout, "test message 42") {
		t.Errorf("Expected 'test message 42', got: %s", stdout)
	}
}

func TestErrorMsg(t *testing.T) {
	resetLoggers()

	_, stderr := captureOutput(func() {
		ErrorMsg("error occurred: %s", "failed")
	})

	if !strings.Contains(stderr, "[ERROR]") {
		t.Errorf("Expected [ERROR] prefix, got: %s", stderr)
	}
	if !strings.Contains(stderr, "error occurred: failed") {
		t.Errorf("Expected 'error occurred: failed', got: %s", stderr)
	}
}

func TestInfoMsg_AutoInit(t *testing.T) {
	resetLoggers()

	if Info != nil {
		t.Error("Info should be nil before auto-init test")
	}

	stdout, _ := captureOutput(func() {
		InfoMsg("auto init test")
	})

	if Info == nil {
		t.Error("Info should be auto-initialized")
	}
	if !strings.Contains(stdout, "auto init test") {
		t.Errorf("Expected 'auto init test', got: %s", stdout)
	}
}

func TestErrorMsg_AutoInit(t *testing.T) {
	resetLoggers()

	if Error != nil {
		t.Error("Error should be nil before auto-init test")
	}

	_, stderr := captureOutput(func() {
		ErrorMsg("auto init error test")
	})

	if Error == nil {
		t.Error("Error should be auto-initialized")
	}
	if !strings.Contains(stderr, "auto init error test") {
		t.Errorf("Expected 'auto init error test', got: %s", stderr)
	}
}

func TestInfoMsg_NoFormat(t *testing.T) {
	resetLoggers()

	stdout, _ := captureOutput(func() {
		InfoMsg("plain message")
	})

	if !strings.Contains(stdout, "plain message") {
		t.Errorf("Expected 'plain message', got: %s", stdout)
	}
}

func TestErrorMsg_NoFormat(t *testing.T) {
	resetLoggers()

	_, stderr := captureOutput(func() {
		ErrorMsg("plain error")
	})

	if !strings.Contains(stderr, "plain error") {
		t.Errorf("Expected 'plain error', got: %s", stderr)
	}
}

func TestMultipleInits(t *testing.T) {
	resetLoggers()

	var loggers []*log.Logger
	for i := 0; i < 10; i++ {
		Init()
		loggers = append(loggers, Info)
	}

	for i := 1; i < len(loggers); i++ {
		if loggers[i] != loggers[0] {
			t.Errorf("Logger instance changed at iteration %d", i)
		}
	}
}

func TestConcurrentAccess(t *testing.T) {
	resetLoggers()

	var wg sync.WaitGroup
	numGoroutines := 100
	errCh := make(chan error, numGoroutines*2)

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			InfoMsg("goroutine %d", id)
		}(i)

		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			ErrorMsg("error goroutine %d", id)
		}(i)
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Error(err)
	}
}

func TestConcurrentInit(t *testing.T) {
	resetLoggers()

	var wg sync.WaitGroup
	numGoroutines := 100
	loggerRefs := make([]*log.Logger, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			Init()
			loggerRefs[idx] = Info
		}(i)
	}

	wg.Wait()

	for i := 1; i < numGoroutines; i++ {
		if loggerRefs[i] != loggerRefs[0] {
			t.Errorf("Logger instance differs at index %d", i)
		}
	}
}

func TestConcurrentInfoMsg(t *testing.T) {
	resetLoggers()

	var wg sync.WaitGroup
	numGoroutines := 50

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			InfoMsg("concurrent test %d", id)
		}(i)
	}

	wg.Wait()
}

func TestConcurrentErrorMsg(t *testing.T) {
	resetLoggers()

	var wg sync.WaitGroup
	numGoroutines := 50

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			ErrorMsg("concurrent error %d", id)
		}(i)
	}

	wg.Wait()
}

func TestLoggerPrefix(t *testing.T) {
	tests := []struct {
		name     string
		fn       func()
		expected string
		isStderr bool
	}{
		{
			name:     "InfoMsg has [INFO] prefix",
			fn:       func() { InfoMsg("test") },
			expected: "[INFO]",
			isStderr: false,
		},
		{
			name:     "ErrorMsg has [ERROR] prefix",
			fn:       func() { ErrorMsg("test") },
			expected: "[ERROR]",
			isStderr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resetLoggers()

			stdout, stderr := captureOutput(tt.fn)

			var output string
			if tt.isStderr {
				output = stderr
			} else {
				output = stdout
			}

			if !strings.Contains(output, tt.expected) {
				t.Errorf("Expected prefix %s in output, got: %s", tt.expected, output)
			}
		})
	}
}

func TestLogMessageFormatting(t *testing.T) {
	tests := []struct {
		name     string
		fn       func()
		expected string
		isStderr bool
	}{
		{
			name:     "InfoMsg with string arg",
			fn:       func() { InfoMsg("hello %s", "world") },
			expected: "hello world",
			isStderr: false,
		},
		{
			name:     "InfoMsg with int arg",
			fn:       func() { InfoMsg("count: %d", 123) },
			expected: "count: 123",
			isStderr: false,
		},
		{
			name:     "InfoMsg with multiple args",
			fn:       func() { InfoMsg("%s = %d", "value", 42) },
			expected: "value = 42",
			isStderr: false,
		},
		{
			name:     "ErrorMsg with string arg",
			fn:       func() { ErrorMsg("failed: %s", "timeout") },
			expected: "failed: timeout",
			isStderr: true,
		},
		{
			name:     "ErrorMsg with int arg",
			fn:       func() { ErrorMsg("code: %d", 500) },
			expected: "code: 500",
			isStderr: true,
		},
		{
			name:     "ErrorMsg with multiple args",
			fn:       func() { ErrorMsg("%s: %v", "error", "details") },
			expected: "error: details",
			isStderr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resetLoggers()

			stdout, stderr := captureOutput(tt.fn)

			var output string
			if tt.isStderr {
				output = stderr
			} else {
				output = stdout
			}

			if !strings.Contains(output, tt.expected) {
				t.Errorf("Expected %q in output, got: %s", tt.expected, output)
			}
		})
	}
}

func TestInitAfterReset(t *testing.T) {
	resetLoggers()
	Init()
	if Info == nil || Error == nil {
		t.Error("Init should initialize both loggers")
	}

	resetLoggers()
	if Info != nil || Error != nil {
		t.Error("Reset should clear both loggers")
	}

	Init()
	if Info == nil || Error == nil {
		t.Error("Init after reset should reinitialize both loggers")
	}
}

func TestLoggerFlags(t *testing.T) {
	resetLoggers()

	stdout, _ := captureOutput(func() {
		Init()
		InfoMsg("test")
	})

	if !strings.Contains(stdout, "[INFO]") {
		t.Errorf("Logger should include prefix, got: %s", stdout)
	}

	parts := strings.Split(strings.TrimSpace(stdout), " ")
	if len(parts) < 3 {
		t.Errorf("Logger should include timestamp, got parts: %v", parts)
	}
}
