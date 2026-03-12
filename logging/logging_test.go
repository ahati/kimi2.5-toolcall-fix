package logging

import (
	"bytes"
	"os"
	"strings"
	"sync"
	"testing"
)

func TestInit(t *testing.T) {
	once = sync.Once{}
	Info = nil
	Error = nil

	Init()

	if Info == nil {
		t.Error("Info logger should not be nil after Init")
	}
	if Error == nil {
		t.Error("Error logger should not be nil after Init")
	}
}

func TestInit_Idempotent(t *testing.T) {
	once = sync.Once{}
	Info = nil
	Error = nil

	Init()
	firstInfo := Info
	firstError := Error

	Init()

	if Info != firstInfo {
		t.Error("Info should be the same instance after multiple Init calls")
	}
	if Error != firstError {
		t.Error("Error should be the same instance after multiple Init calls")
	}
}

func TestInfoMsg(t *testing.T) {
	once = sync.Once{}
	Info = nil

	old := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	InfoMsg("test message: %s", "value")

	w.Close()
	os.Stdout = old

	var buf bytes.Buffer
	buf.ReadFrom(r)
	output := buf.String()

	if !strings.Contains(output, "[INFO]") {
		t.Errorf("Expected [INFO] in output, got: %s", output)
	}
	if !strings.Contains(output, "test message: value") {
		t.Errorf("Expected 'test message: value' in output, got: %s", output)
	}
}

func TestErrorMsg(t *testing.T) {
	once = sync.Once{}
	Error = nil

	old := os.Stderr
	r, w, _ := os.Pipe()
	os.Stderr = w

	ErrorMsg("error message: %s", "test")

	w.Close()
	os.Stderr = old

	var buf bytes.Buffer
	buf.ReadFrom(r)
	output := buf.String()

	if !strings.Contains(output, "[ERROR]") {
		t.Errorf("Expected [ERROR] in output, got: %s", output)
	}
	if !strings.Contains(output, "error message: test") {
		t.Errorf("Expected 'error message: test' in output, got: %s", output)
	}
}

func TestInfoMsg_NoFormat(t *testing.T) {
	once = sync.Once{}
	Info = nil

	old := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	InfoMsg("simple message")

	w.Close()
	os.Stdout = old

	var buf bytes.Buffer
	buf.ReadFrom(r)
	output := buf.String()

	if !strings.Contains(output, "simple message") {
		t.Errorf("Expected 'simple message' in output, got: %s", output)
	}
}

func TestErrorMsg_NoFormat(t *testing.T) {
	once = sync.Once{}
	Error = nil

	old := os.Stderr
	r, w, _ := os.Pipe()
	os.Stderr = w

	ErrorMsg("simple error")

	w.Close()
	os.Stderr = old

	var buf bytes.Buffer
	buf.ReadFrom(r)
	output := buf.String()

	if !strings.Contains(output, "simple error") {
		t.Errorf("Expected 'simple error' in output, got: %s", output)
	}
}

func TestInfoMsg_MultipleArgs(t *testing.T) {
	once = sync.Once{}
	Info = nil

	old := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	InfoMsg("test %s %d %f", "string", 42, 3.14)

	w.Close()
	os.Stdout = old

	var buf bytes.Buffer
	buf.ReadFrom(r)
	output := buf.String()

	if !strings.Contains(output, "test string 42") {
		t.Errorf("Expected formatted output, got: %s", output)
	}
}
