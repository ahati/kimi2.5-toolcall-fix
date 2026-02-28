.PHONY: build install

BINARY_NAME=ai-proxy
INSTALL_DIR=/home/hati/.local/bin
BUILD_DIR=.

build:
	go build -o $(BINARY_NAME) .

install:
	install -m 755 $(BINARY_NAME) $(INSTALL_DIR)/$(BINARY_NAME)
