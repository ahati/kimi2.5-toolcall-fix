package logging

import (
	"log"
	"os"
)

var (
	Info  *log.Logger
	Error *log.Logger
)

func Init() {
	Info = log.New(os.Stdout, "[INFO] ", log.LstdFlags)
	Error = log.New(os.Stderr, "[ERROR] ", log.LstdFlags)
}

func InfoMsg(format string, v ...interface{}) {
	Info.Printf(format, v...)
}

func ErrorMsg(format string, v ...interface{}) {
	Error.Printf(format, v...)
}
