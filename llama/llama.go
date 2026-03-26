//go:build llama

// Package llama provides CGo bindings to llama.cpp.
//
// SETUP INSTRUCTIONS:
//
// Run go generate to fetch and build llama.cpp:
//
//	go generate ./llama
//
// This will download and compile llama.cpp, then print the required environment variables.
// Set those variables and rebuild:
//
//	CGO_ENABLED=1 go build
//
// Manual Setup (Alternative):
//
// 1. Clone and build llama.cpp:
//
//	git clone --depth 1 --branch b8508 https://github.com/ggml-org/llama.cpp.git
//	cd llama.cpp && mkdir build && cd build
//	cmake .. -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
//	make -j$(nproc) llama ggml ggml-base ggml-cpu
//
// 2. Set CGO flags to point to llama.cpp (replace /path/to/llama.cpp):
//
//	export CGO_CFLAGS="-I/path/to/llama.cpp/include -I/path/to/llama.cpp/ggml/include"
//	export CGO_CXXFLAGS="-I/path/to/llama.cpp/include -I/path/to/llama.cpp/ggml/include"
//	export CGO_LDFLAGS="/path/to/llama.cpp/build/src/libllama.a /path/to/llama.cpp/build/ggml/src/libggml.a /path/to/llama.cpp/build/ggml/src/libggml-base.a /path/to/llama.cpp/build/ggml/src/libggml-cpu.a -lstdc++ -lm -lpthread -lgomp"
//
// 3. Build your project:
//
//	CGO_ENABLED=1 go build
package llama

//go:generate go run generate.go

/*
#include "llama.h"
#include <stdlib.h>
*/
import "C"

import (
	"errors"
	"fmt"
	"strings"
	"unsafe"
)

// BackendInit initializes the llama backend. Must be called before any other operations.
func BackendInit() { C.llama_backend_init() }

// BackendFree frees resources used by the llama backend.
func BackendFree() { C.llama_backend_free() }

// LoadBackends loads all available backends (CPU, GPU, etc.).
func LoadBackends() { C.ggml_backend_load_all() }

// Model represents a loaded llama model.
type Model struct{ ptr *C.struct_llama_model }

// LoadModel loads a GGUF model from the given path.
// gpuLayers specifies the number of layers to offload to GPU (0 for CPU-only).
func LoadModel(path string, gpuLayers int) (*Model, error) {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	params := C.llama_model_default_params()
	params.n_gpu_layers = C.int32_t(gpuLayers)
	ptr := C.llama_model_load_from_file(cpath, params)
	if ptr == nil {
		return nil, fmt.Errorf("failed to load model: %s", path)
	}
	return &Model{ptr: ptr}, nil
}

// Close frees the model resources.
func (m *Model) Close() {
	if m.ptr != nil {
		C.llama_model_free(m.ptr)
		m.ptr = nil
	}
}

// Context represents an inference context for a model.
type Context struct {
	ptr   *C.struct_llama_context
	model *Model
}

// NewContext creates a new inference context for the given model.
func NewContext(model *Model, ctxSize, threads int) (*Context, error) {
	params := C.llama_context_default_params()
	params.n_ctx = C.uint32_t(ctxSize)
	params.n_threads = C.int32_t(threads)
	params.n_batch = C.uint32_t(ctxSize) // Set batch size to context size
	ptr := C.llama_init_from_model(model.ptr, params)
	if ptr == nil {
		return nil, errors.New("failed to create context")
	}
	return &Context{ptr: ptr, model: model}, nil
}

// Close frees the context resources.
func (c *Context) Close() {
	if c.ptr != nil {
		C.llama_free(c.ptr)
		c.ptr = nil
	}
}

// ClearMemory clears the context memory, allowing new sequences.
func (c *Context) ClearMemory() {
	mem := C.llama_get_memory(c.ptr)
	C.llama_memory_clear(mem, true)
}

// Tokenize converts text to token IDs.
func (c *Context) Tokenize(text string) ([]int32, error) {
	ctext := C.CString(text)
	defer C.free(unsafe.Pointer(ctext))
	vocab := C.llama_model_get_vocab(c.model.ptr)
	maxTokens := len(text) + 256
	tokens := make([]C.llama_token, maxTokens)
	n := C.llama_tokenize(vocab, ctext, C.int(len(text)), &tokens[0], C.int(maxTokens), true, false)
	if n < 0 {
		return nil, errors.New("tokenization failed")
	}
	result := make([]int32, n)
	for i := 0; i < int(n); i++ {
		result[i] = int32(tokens[i])
	}
	return result, nil
}

// Decode processes tokens through the model.
func (c *Context) Decode(tokens []int32) error {
	batch := C.llama_batch_get_one((*C.llama_token)(unsafe.Pointer(&tokens[0])), C.int(len(tokens)))
	if ret := C.llama_decode(c.ptr, batch); ret < 0 {
		return fmt.Errorf("decode failed: %d", ret)
	}
	return nil
}

// Sampler represents a token sampling strategy.
type Sampler struct{ ptr *C.struct_llama_sampler }

// NewGreedySampler creates a greedy sampler (always picks highest probability token).
func NewGreedySampler() *Sampler { return &Sampler{ptr: C.llama_sampler_init_greedy()} }

// Close frees the sampler resources.
func (s *Sampler) Close() {
	if s.ptr != nil {
		C.llama_sampler_free(s.ptr)
		s.ptr = nil
	}
}

// Sample returns the next token from the context using this sampler.
func (s *Sampler) Sample(ctx *Context) int32 {
	return int32(C.llama_sampler_sample(s.ptr, ctx.ptr, -1))
}

// TokenToPiece converts a token ID to its string representation.
func (m *Model) TokenToPiece(token int32) string {
	buf := make([]byte, 64)
	vocab := C.llama_model_get_vocab(m.ptr)
	n := C.llama_token_to_piece(vocab, C.llama_token(token), (*C.char)(unsafe.Pointer(&buf[0])), C.int32_t(len(buf)), 0, false)
	if n <= 0 {
		return ""
	}
	return strings.ReplaceAll(string(buf[:n]), "Ġ", " ")
}

// EOSToken returns the end-of-sequence token ID for this model.
func (m *Model) EOSToken() int32 {
	return int32(C.llama_vocab_eos(C.llama_model_get_vocab(m.ptr)))
}