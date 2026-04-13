//go:build llamacpp_native && cgo

package llamacppnative

/*
#cgo CXXFLAGS: -std=c++17 -DIMGSEARCH_LLAMA_NATIVE_ENABLED
#cgo CPPFLAGS: -I${SRCDIR}/../../../deps/llama.cpp/include
#cgo CPPFLAGS: -I${SRCDIR}/../../../deps/llama.cpp/ggml/include
#cgo CPPFLAGS: -I${SRCDIR}/../../../deps/llama.cpp/tools/mtmd
#cgo LDFLAGS: -L${SRCDIR}/../../../deps/llama.cpp/build/bin -lllama -lmtmd
#cgo LDFLAGS: -Wl,-rpath,${SRCDIR}/../../../deps/llama.cpp/build/bin
#include <stdlib.h>
#include "bridge.h"
*/
import "C"

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"unsafe"
)

const (
	defaultQueryInstruction   = "Retrieve images or text relevant to the user's query."
	defaultPassageInstruction = "Represent this image or text for retrieval."
	defaultImageMaxSide       = 512
)

type Config struct {
	ModelPath          string
	VisionModelPath    string
	Dimensions         int
	GPULayers          int
	UseGPU             bool
	ContextSize        int
	BatchSize          int
	Threads            int
	ImageMaxSide       int
	ImageMaxTokens     int
	QueryInstruction   string
	PassageInstruction string
}

type Embedder struct {
	mu                 sync.Mutex
	handle             *C.imgsearch_llama_handle
	dimensions         int
	queryInstruction   string
	passageInstruction string
}

func New(cfg Config) (*Embedder, error) {
	modelPath := strings.TrimSpace(cfg.ModelPath)
	if modelPath == "" {
		return nil, fmt.Errorf("llama-cpp-native model path is required")
	}
	visionPath := strings.TrimSpace(cfg.VisionModelPath)
	if visionPath == "" {
		return nil, fmt.Errorf("llama-cpp-native vision model path is required")
	}
	if cfg.Dimensions <= 0 {
		return nil, fmt.Errorf("llama-cpp-native dimensions must be positive")
	}

	imageMaxSide := cfg.ImageMaxSide
	if imageMaxSide == 0 {
		imageMaxSide = defaultImageMaxSide
	}
	if imageMaxSide < 0 {
		return nil, fmt.Errorf("llama-cpp-native image max side must be non-negative")
	}
	if cfg.ImageMaxTokens < 0 {
		return nil, fmt.Errorf("llama-cpp-native image max tokens must be non-negative")
	}

	queryInstruction := strings.TrimSpace(cfg.QueryInstruction)
	if queryInstruction == "" {
		queryInstruction = defaultQueryInstruction
	}
	passageInstruction := strings.TrimSpace(cfg.PassageInstruction)
	if passageInstruction == "" {
		passageInstruction = defaultPassageInstruction
	}

	cModelPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cModelPath))
	cVisionPath := C.CString(visionPath)
	defer C.free(unsafe.Pointer(cVisionPath))

	useGPU := 0
	if cfg.UseGPU {
		useGPU = 1
	}

	h := C.imgsearch_llama_new(
		cModelPath,
		cVisionPath,
		C.int32_t(cfg.GPULayers),
		C.int32_t(cfg.ContextSize),
		C.int32_t(cfg.BatchSize),
		C.int32_t(cfg.Threads),
		C.int32_t(useGPU),
		C.int32_t(imageMaxSide),
		C.int32_t(cfg.ImageMaxTokens),
	)
	if h == nil {
		msg := strings.TrimSpace(C.GoString(C.imgsearch_llama_global_error()))
		if msg == "" {
			msg = "failed to initialize llama.cpp-native embedder"
		}
		return nil, fmt.Errorf("%s", msg)
	}

	actualDims := int(C.imgsearch_llama_dims(h))
	if actualDims <= 0 {
		C.imgsearch_llama_free(h)
		return nil, fmt.Errorf("llama-cpp-native returned invalid embedding dimensions")
	}
	if actualDims != cfg.Dimensions {
		C.imgsearch_llama_free(h)
		return nil, fmt.Errorf("llama-cpp-native dimensions mismatch: got %d want %d", actualDims, cfg.Dimensions)
	}

	return &Embedder{
		handle:             h,
		dimensions:         actualDims,
		queryInstruction:   queryInstruction,
		passageInstruction: passageInstruction,
	}, nil
}

func (e *Embedder) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.handle != nil {
		C.imgsearch_llama_free(e.handle)
		e.handle = nil
	}
	return nil
}

func (e *Embedder) EmbedText(_ context.Context, text string) ([]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.handle == nil {
		return nil, fmt.Errorf("llama-cpp-native embedder is closed")
	}
	if strings.TrimSpace(text) == "" {
		return nil, fmt.Errorf("text is empty")
	}

	out := make([]float32, e.dimensions)
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))
	cInstruction := C.CString(e.queryInstruction)
	defer C.free(unsafe.Pointer(cInstruction))

	if C.imgsearch_llama_embed_text(
		e.handle,
		cText,
		cInstruction,
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int32_t(len(out)),
	) != 0 {
		return nil, e.lastErrorLocked()
	}

	return out, nil
}

func (e *Embedder) EmbedImage(_ context.Context, path string) ([]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.handle == nil {
		return nil, fmt.Errorf("llama-cpp-native embedder is closed")
	}
	if strings.TrimSpace(path) == "" {
		return nil, fmt.Errorf("image path is empty")
	}

	out := make([]float32, e.dimensions)
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	cInstruction := C.CString(e.passageInstruction)
	defer C.free(unsafe.Pointer(cInstruction))

	if C.imgsearch_llama_embed_image(
		e.handle,
		cPath,
		cInstruction,
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int32_t(len(out)),
	) != 0 {
		return nil, e.lastErrorLocked()
	}

	return out, nil
}

func (e *Embedder) lastErrorLocked() error {
	if e == nil {
		return fmt.Errorf("llama-cpp-native operation failed")
	}
	msg := ""
	if e.handle != nil {
		msg = strings.TrimSpace(C.GoString(C.imgsearch_llama_last_error(e.handle)))
	}
	if msg == "" {
		msg = strings.TrimSpace(C.GoString(C.imgsearch_llama_global_error()))
	}
	if msg == "" {
		msg = "llama-cpp-native operation failed"
	}
	return fmt.Errorf("%s", msg)
}
