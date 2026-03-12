// Package logging provides simple logging utilities with info and error levels.
// Loggers are initialized lazily and are safe for concurrent use.
package logging

import (
	"log"
	"os"
	"sync"
)

var (
	// Info is the info-level logger for general operational messages.
	// Writes to stdout with "[INFO] " prefix.
	// Nil until Init() is called; use InfoMsg() for safe access.
	Info *log.Logger
	// Error is the error-level logger for error and warning messages.
	// Writes to stderr with "[ERROR] " prefix.
	// Nil until Init() is called; use ErrorMsg() for safe access.
	Error *log.Logger
	// once ensures Init() is only executed once.
	// Used for thread-safe lazy initialization.
	once sync.Once
)

// Init initializes the Info and Error loggers. It is safe to call multiple times.
// The actual initialization happens only once due to sync.Once.
//
// @post Info logger is initialized writing to stdout with "[INFO] " prefix
// @post Error logger is initialized writing to stderr with "[ERROR] " prefix
// @post Loggers include timestamp in output
// @note This function is called automatically by InfoMsg and ErrorMsg if needed
// @note Thread-safe; multiple goroutines can call Init concurrently
func Init() {
	once.Do(func() {
		// Initialize loggers with standard flags including timestamp
		Info = log.New(os.Stdout, "[INFO] ", log.LstdFlags)
		Error = log.New(os.Stderr, "[ERROR] ", log.LstdFlags)
	})
}

// InfoMsg logs an info-level message using the configured format.
// It lazily initializes the logger if needed.
// Format follows fmt.Printf conventions.
//
// @param format - the format string for the log message
// @param v - variadic arguments for format string substitution
// @pre format must be a valid fmt.Printf format string
// @post Message is written to stdout with "[INFO] " prefix and timestamp
// @note Safe to call before Init(); will auto-initialize
// @note Thread-safe for concurrent use
func InfoMsg(format string, v ...interface{}) {
	// Lazy initialization ensures logger is ready even if Init() wasn't called
	if Info == nil {
		Init()
	}
	Info.Printf(format, v...)
}

// ErrorMsg logs an error-level message using the configured format.
// It lazily initializes the logger if needed.
// Format follows fmt.Printf conventions.
//
// @param format - the format string for the log message
// @param v - variadic arguments for format string substitution
// @pre format must be a valid fmt.Printf format string
// @post Message is written to stderr with "[ERROR] " prefix and timestamp
// @note Safe to call before Init(); will auto-initialize
// @note Thread-safe for concurrent use
func ErrorMsg(format string, v ...interface{}) {
	// Lazy initialization ensures logger is ready even if Init() wasn't called
	if Error == nil {
		Init()
	}
	Error.Printf(format, v...)
}
