//go:build ignore

// This file is run by go:generate to fetch and build llama.cpp.
// Run: go generate ./llama
package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
)

const (
	llamaVersion = "b8508"
	llamaRepo    = "https://github.com/ggml-org/llama.cpp.git"
)

// Build configuration - set CUDA=true to enable GPU support
var (
	cudaEnabled = os.Getenv("CUDA") == "true" || os.Getenv("CUDA") == "1"
	cudaArch    = os.Getenv("CUDA_ARCH") // e.g., "89" for RTX 4090, "86" for RTX 3090
)

func main() {
	// Determine cache directory - default to .build in current directory
	cacheDir := os.Getenv("LLAMA_CACHE_DIR")
	if cacheDir == "" {
		// Find module root
		modRoot := findModuleRoot()
		if cudaEnabled {
			cacheDir = filepath.Join(modRoot, ".build", "llama-cpp-"+llamaVersion+"-cuda")
		} else {
			cacheDir = filepath.Join(modRoot, ".build", "llama-cpp-"+llamaVersion)
		}
	}

	llamaDir := filepath.Join(cacheDir, "llama.cpp")
	buildDir := filepath.Join(llamaDir, "build")

	// Check if already built
	libPath := filepath.Join(buildDir, "src", "libllama.a")
	if _, err := os.Stat(libPath); err == nil {
		fmt.Printf("llama.cpp already built at %s\n", llamaDir)
		printEnvVars(cacheDir)
		return
	}

	// Clone llama.cpp
	fmt.Printf("Cloning llama.cpp %s to %s...\n", llamaVersion, llamaDir)
	if _, err := os.Stat(llamaDir); os.IsNotExist(err) {
		cmd := exec.Command("git", "clone", "--depth", "1", "--branch", llamaVersion, llamaRepo, llamaDir)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			fmt.Fprintf(os.Stderr, "Failed to clone llama.cpp: %v\n", err)
			os.Exit(1)
		}
	}

	// Create build directory
	if err := os.MkdirAll(buildDir, 0755); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create build directory: %v\n", err)
		os.Exit(1)
	}

	// Configure with CMake
	fmt.Println("Configuring with CMake...")
	if cudaEnabled {
		fmt.Println("CUDA support ENABLED")
	}
	cmakeGenerator := "Unix Makefiles"
	if _, err := exec.LookPath("ninja"); err == nil {
		cmakeGenerator = "Ninja"
	}

	cmakeCmd := []string{
		"..",
		"-G", cmakeGenerator,
		"-DBUILD_SHARED_LIBS=OFF",
		"-DLLAMA_BUILD_EXAMPLES=OFF",
		"-DLLAMA_BUILD_SERVER=OFF",
		"-DCMAKE_BUILD_TYPE=Release",
	}

	// Add CUDA support if enabled
	if cudaEnabled {
		cmakeCmd = append(cmakeCmd, "-DLLAMA_CUDA=ON")
		if cudaArch != "" {
			cmakeCmd = append(cmakeCmd, fmt.Sprintf("-DCMAKE_CUDA_ARCHITECTURES=%s", cudaArch))
		}
	}

	cmd := exec.Command("cmake", cmakeCmd...)
	cmd.Dir = buildDir
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to configure llama.cpp: %v\n", err)
		os.Exit(1)
	}

	// Build
	fmt.Println("Building llama.cpp...")
	buildCmd := "make"
	buildArgs := []string{"-j", fmt.Sprintf("%d", runtime.NumCPU()), "llama", "ggml", "ggml-base", "ggml-cpu"}
	if cudaEnabled {
		buildArgs = append(buildArgs, "ggml-cuda")
	}
	if cmakeGenerator == "Ninja" {
		buildCmd = "ninja"
		buildArgs = []string{"llama", "ggml", "ggml-base", "ggml-cpu"}
		if cudaEnabled {
			buildArgs = append(buildArgs, "ggml-cuda")
		}
	}

	cmd = exec.Command(buildCmd, buildArgs...)
	cmd.Dir = buildDir
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to build llama.cpp: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("llama.cpp built successfully!")
	printEnvVars(cacheDir)
}

func printEnvVars(cacheDir string) {
	llamaDir := filepath.Join(cacheDir, "llama.cpp")
	buildDir := filepath.Join(llamaDir, "build")

	fmt.Println("\n=== Set these environment variables before building ===")
	fmt.Printf("export CGO_CFLAGS=\"-I%s/include -I%s/ggml/include\"\n", llamaDir, llamaDir)
	fmt.Printf("export CGO_CXXFLAGS=\"-I%s/include -I%s/ggml/include\"\n", llamaDir, llamaDir)

	ldflags := fmt.Sprintf("%s/src/libllama.a %s/ggml/src/libggml.a %s/ggml/src/libggml-base.a %s/ggml/src/libggml-cpu.a",
		buildDir, buildDir, buildDir, buildDir)

	if cudaEnabled {
		ldflags += fmt.Sprintf(" %s/ggml/src/libggml-cuda.a", buildDir)
		ldflags += " -lcudart -lcublas -lcublasLt"
		fmt.Println("CUDA libraries included in linker flags")
	}

	ldflags += " -lstdc++ -lm -lpthread -lgomp"
	fmt.Printf("export CGO_LDFLAGS=\"%s\"\n", ldflags)
	fmt.Println("\nThen run: CGO_ENABLED=1 go build")
}

// findModuleRoot finds the Go module root directory.
func findModuleRoot() string {
	dir, err := os.Getwd()
	if err != nil {
		return "."
	}

	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "."
		}
		dir = parent
	}
}