.PHONY: build install clean clean-cache generate build-cuda

BINARY_NAME=ai-proxy
INSTALL_DIR=/home/hati/.local/bin
BUILD_DIR=.
CONFIG_DIR=$(HOME)/.config/ai-proxy
CONFIG_FILE=$(CONFIG_DIR)/config.json

# llama.cpp configuration - cache in project .build directory (gitignored)
LLAMA_VERSION?=b8508
LLAMA_REPO?=https://github.com/ggml-org/llama.cpp.git

# CUDA configuration - set CUDA=true for GPU support
CUDA?=false
CUDA_ARCH?=$(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.' || echo "89")

ifeq ($(CUDA),true)
LLAMA_CACHE_DIR=$(CURDIR)/.build/llama-cpp-$(LLAMA_VERSION)-cuda
else
LLAMA_CACHE_DIR=$(CURDIR)/.build/llama-cpp-$(LLAMA_VERSION)
endif

# Use ninja if available, otherwise make
CMAKE_GENERATOR=$(shell command -v ninja >/dev/null 2>&1 && echo "Ninja" || echo "Unix Makefiles")
CMAKE_BUILD_CMD=$(shell command -v ninja >/dev/null 2>&1 && echo "ninja" || echo "make -j$$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)")

# Check if llama cache is valid (has all required libraries)
LLAMA_LIBS=$(LLAMA_CACHE_DIR)/build/src/libllama.a $(LLAMA_CACHE_DIR)/build/ggml/src/libggml.a $(LLAMA_CACHE_DIR)/build/ggml/src/libggml-base.a $(LLAMA_CACHE_DIR)/build/ggml/src/libggml-cpu.a

# Add CUDA libraries if enabled
ifeq ($(CUDA),true)
LLAMA_LIBS += $(LLAMA_CACHE_DIR)/build/ggml/src/libggml-cuda.a
CUDA_LDFLAGS=-lcudart -lcublas -lcublasLt
endif

# Generate fetches and builds llama.cpp using go generate
generate:
	@echo "=== Generating llama.cpp build ==="
	@CUDA=$(CUDA) CUDA_ARCH=$(CUDA_ARCH) go generate ./llama

build:
ifeq ($(CUDA),true)
	@echo "=== Building with CUDA support ==="
endif
	@if [ -f "$(LLAMA_CACHE_DIR)/build/src/libllama.a" ]; then \
		echo "=== Using cached llama.cpp ==="; \
	else \
		echo "=== Building llama.cpp ==="; \
		echo "=== Cache: $(LLAMA_CACHE_DIR) ==="; \
		mkdir -p $(LLAMA_CACHE_DIR) && \
		git clone --depth 1 --branch $(LLAMA_VERSION) $(LLAMA_REPO) $(LLAMA_CACHE_DIR) && \
		cd $(LLAMA_CACHE_DIR) && mkdir -p build && cd build && \
		cmake .. -G "$(CMAKE_GENERATOR)" -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DCMAKE_BUILD_TYPE=Release \
		$(if $(filter true,$(CUDA)),-DLLAMA_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH),) && \
		$(CMAKE_BUILD_CMD) llama ggml ggml-base ggml-cpu $(if $(filter true,$(CUDA)),ggml-cuda,); \
	fi
	@echo "=== Building ai-proxy ==="
	@CGO_ENABLED=1 \
	CGO_CFLAGS="-I$(LLAMA_CACHE_DIR)/include -I$(LLAMA_CACHE_DIR)/ggml/include" \
	CGO_CXXFLAGS="-I$(LLAMA_CACHE_DIR)/include -I$(LLAMA_CACHE_DIR)/ggml/include" \
	CGO_LDFLAGS="$(LLAMA_LIBS) $(CUDA_LDFLAGS) -lstdc++ -lm -lpthread -lgomp" \
	go build -tags llama -o $(BINARY_NAME) .
	@echo "=== Build complete ==="

# Convenience target for CUDA build
build-cuda:
	@$(MAKE) build CUDA=true

install: $(CONFIG_DIR)
	install -m 755 $(BINARY_NAME) $(INSTALL_DIR)/$(BINARY_NAME)
	install -m 644 test-config.json $(CONFIG_FILE)

$(CONFIG_DIR):
	mkdir -p $(CONFIG_DIR)

clean:
	rm -f $(BINARY_NAME)

clean-cache:
	rm -rf .build
