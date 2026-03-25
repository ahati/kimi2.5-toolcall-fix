.PHONY: build install clean clean-cache

BINARY_NAME=ai-proxy
INSTALL_DIR=/home/hati/.local/bin
BUILD_DIR=.
CONFIG_DIR=$(HOME)/.config/ai-proxy
CONFIG_FILE=$(CONFIG_DIR)/config.json

# llama.cpp configuration - cache is tied to summarizer module version
LLAMA_VERSION?=b8508
LLAMA_REPO?=https://github.com/ggml-org/llama.cpp.git
SUMMARIZER_VERSION=$(shell grep 'github.com/ahati/reasoning-summarizer' go.mod | awk '{print $$2}')
LLAMA_CACHE_DIR=$(shell go env GOCACHE)/github.com-ahati-reasoning-summarizer-$(SUMMARIZER_VERSION)/llama-cpp

# Use ninja if available, otherwise make
CMAKE_GENERATOR=$(shell command -v ninja >/dev/null 2>&1 && echo "Ninja" || echo "Unix Makefiles")
CMAKE_BUILD_CMD=$(shell command -v ninja >/dev/null 2>&1 && echo "ninja" || echo "make -j$$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)")

# Check if llama cache is valid (has all required libraries)
LLAMA_LIBS=$(LLAMA_CACHE_DIR)/build/src/libllama.a $(LLAMA_CACHE_DIR)/build/ggml/src/libggml.a $(LLAMA_CACHE_DIR)/build/ggml/src/libggml-base.a $(LLAMA_CACHE_DIR)/build/ggml/src/libggml-cpu.a

build:
	@if [ -f "$(LLAMA_CACHE_DIR)/build/src/libllama.a" ]; then \
		echo "=== Using cached llama.cpp for summarizer $(SUMMARIZER_VERSION) ==="; \
	else \
		echo "=== Building llama.cpp for summarizer $(SUMMARIZER_VERSION) ==="; \
		echo "=== Cache: $(LLAMA_CACHE_DIR) ==="; \
		mkdir -p $(LLAMA_CACHE_DIR) && \
		git clone --depth 1 --branch $(LLAMA_VERSION) $(LLAMA_REPO) $(LLAMA_CACHE_DIR) && \
		cd $(LLAMA_CACHE_DIR) && mkdir -p build && cd build && \
		cmake .. -G "$(CMAKE_GENERATOR)" -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DCMAKE_BUILD_TYPE=Release && \
		$(CMAKE_BUILD_CMD) llama ggml ggml-base ggml-cpu; \
	fi
	@echo "=== Building ai-proxy ==="
	@CGO_ENABLED=1 \
	CGO_CFLAGS="-I$(LLAMA_CACHE_DIR)/include -I$(LLAMA_CACHE_DIR)/ggml/include" \
	CGO_CXXFLAGS="-I$(LLAMA_CACHE_DIR)/include -I$(LLAMA_CACHE_DIR)/ggml/include" \
	CGO_LDFLAGS="$(LLAMA_LIBS) -lstdc++ -lm -lpthread -lgomp" \
	go build -o $(BINARY_NAME) .
	@echo "=== Build complete ==="

install: $(CONFIG_DIR)
	install -m 755 $(BINARY_NAME) $(INSTALL_DIR)/$(BINARY_NAME)
	install -m 644 test-config.json $(CONFIG_FILE)

$(CONFIG_DIR):
	mkdir -p $(CONFIG_DIR)

clean:
	rm -f $(BINARY_NAME)

clean-cache:
	rm -rf $(shell go env GOCACHE)/github.com-ahati-reasoning-summarizer-*
