// Package logging provides structured logging and request/response capture functionality.
// It captures bidirectional traffic between client and upstream LLM API for debugging
// and analysis purposes.
package logging

import (
	"log"
	"os"
	"sync"
)

// Package-level logger instances for structured logging.
// Initialized once on first use via sync.Once for thread-safety.
var (
	// Info provides standard output logging with [INFO] prefix.
	Info *log.Logger
	// Error provides standard error logging with [ERROR] prefix.
	Error *log.Logger
	// once ensures logger initialization happens exactly once.
	once sync.Once
)

// Init initializes the global logger instances.
//
// @brief    Initializes Info and Error loggers with standard prefixes.
// @pre      None.
// @post     Info logger writes to stdout with [INFO] prefix.
// @post     Error logger writes to stderr with [ERROR] prefix.
// @note     Thread-safe via sync.Once; multiple calls are safe.
// @note     Called automatically by InfoMsg/ErrorMsg if not initialized.
func Init() {
	once.Do(func() {
		Info = log.New(os.Stdout, "[INFO] ", log.LstdFlags)
		Error = log.New(os.Stderr, "[ERROR] ", log.LstdFlags)
	})
}

// InfoMsg logs an informational message using the Info logger.
//
// @brief    Logs formatted message to stdout with [INFO] prefix.
// @param    format Printf-style format string.
// @param    v     Variable arguments for format string.
// @pre      None (auto-initializes if needed).
// @post     Message written to stdout with timestamp.
// @note     Automatically calls Init() if Info logger is nil.
func InfoMsg(format string, v ...interface{}) {
	if Info == nil {
		Init()
	}
	Info.Printf(format, v...)
}

// ErrorMsg logs an error message using the Error logger.
//
// @brief    Logs formatted message to stderr with [ERROR] prefix.
// @param    format Printf-style format string.
// @param    v     Variable arguments for format string.
// @pre      None (auto-initializes if needed).
// @post     Message written to stderr with timestamp.
// @note     Automatically calls Init() if Error logger is nil.
func ErrorMsg(format string, v ...interface{}) {
	if Error == nil {
		Init()
	}
	Error.Printf(format, v...)
}
