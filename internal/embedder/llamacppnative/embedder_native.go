//go:build cgo

package llamacppnative

/*
#cgo CXXFLAGS: -std=c++17 -DIMGSEARCH_LLAMA_NATIVE_ENABLED
#cgo CPPFLAGS: -I${SRCDIR}/../../../deps/llama.cpp/common
#cgo CPPFLAGS: -I${SRCDIR}/../../../deps/llama.cpp/vendor
#cgo CPPFLAGS: -I${SRCDIR}/../../../deps/llama.cpp/include
#cgo CPPFLAGS: -I${SRCDIR}/../../../deps/llama.cpp/ggml/include
#cgo CPPFLAGS: -I${SRCDIR}/../../../deps/llama.cpp/tools/mtmd
#cgo LDFLAGS: -L${SRCDIR}/../../../deps/llama.cpp/build/bin -lllama -lmtmd -lggml -lggml-base
#cgo LDFLAGS: ${SRCDIR}/../../../deps/llama.cpp/build/common/libcommon.a
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
	defaultImageMaxSide       = 384
	defaultMaxSequences       = 1
	defaultFlashAttnType      = -1
	defaultCacheType          = -1
)

type Config struct {
	ModelPath          string
	VisionModelPath    string
	Dimensions         int
	GPULayers          int
	UseGPU             bool
	ContextSize        int
	BatchSize          int
	MaxSequences       int
	Threads            int
	ImageMaxSide       int
	ImageMaxTokens     int
	QueryInstruction   string
	PassageInstruction string
	FlashAttnType      int
	CacheTypeK         int
	CacheTypeV         int
}

type Embedder struct {
	mu                 sync.Mutex
	handle             *C.imgsearch_llama_handle
	dimensions         int
	maxSequences       int
	imageMaxSide       int
	queryInstruction   string
	passageInstruction string
}

type preparedEmbedPath struct {
	path    string
	cleanup func()
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

	maxSequences := cfg.MaxSequences
	if maxSequences <= 0 {
		maxSequences = defaultMaxSequences
	}

	flashAttnType := cfg.FlashAttnType
	if flashAttnType == 0 {
		flashAttnType = defaultFlashAttnType
	}
	cacheTypeK := cfg.CacheTypeK
	if cacheTypeK == 0 {
		cacheTypeK = defaultCacheType
	}
	cacheTypeV := cfg.CacheTypeV
	if cacheTypeV == 0 {
		cacheTypeV = defaultCacheType
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
		C.int32_t(flashAttnType),
		C.int32_t(cacheTypeK),
		C.int32_t(cacheTypeV),
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
		maxSequences:       maxSequences,
		imageMaxSide:       imageMaxSide,
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

func (e *Embedder) EmbedText(ctx context.Context, text string) ([]float32, error) {
	if err := ensureContextActive(ctx); err != nil {
		return nil, err
	}

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

func (e *Embedder) EmbedImage(ctx context.Context, path string) ([]float32, error) {
	vecs, err := e.EmbedImages(ctx, []string{path})
	if err != nil {
		return nil, err
	}
	if len(vecs) != 1 {
		return nil, fmt.Errorf("llama-cpp-native batch image embedding count mismatch: got %d want 1", len(vecs))
	}
	return vecs[0], nil
}

func (e *Embedder) EmbedImages(ctx context.Context, paths []string) ([][]float32, error) {
	if err := ensureContextActive(ctx); err != nil {
		return nil, err
	}
	if len(paths) == 0 {
		return nil, fmt.Errorf("image path batch is empty")
	}

	prepared := make([]preparedEmbedPath, 0, len(paths))
	for _, path := range paths {
		preprocessedPath, cleanup, err := preprocessImageForEmbeddingWithVipsgen(path, e.imageMaxSide)
		if err != nil {
			for _, item := range prepared {
				item.cleanup()
			}
			return nil, err
		}
		prepared = append(prepared, preparedEmbedPath{path: preprocessedPath, cleanup: cleanup})
	}
	defer func() {
		for _, item := range prepared {
			item.cleanup()
		}
	}()

	if err := ensureContextActive(ctx); err != nil {
		return nil, err
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	if e.handle == nil {
		return nil, fmt.Errorf("llama-cpp-native embedder is closed")
	}

	results := make([][]float32, 0, len(prepared))
	for start := 0; start < len(prepared); start += e.maxSequences {
		end := start + e.maxSequences
		if end > len(prepared) {
			end = len(prepared)
		}
		chunk := prepared[start:end]
		chunkOut, err := e.embedPreparedPathsLocked(ctx, chunk)
		if err != nil {
			return nil, err
		}
		results = append(results, chunkOut...)
	}

	return results, nil
}

func (e *Embedder) embedPreparedPathsLocked(ctx context.Context, paths []preparedEmbedPath) ([][]float32, error) {
	if len(paths) == 0 {
		return nil, nil
	}

	results := make([][]float32, len(paths))
	for i := range paths {
		vec := make([]float32, e.dimensions)
		if err := ensureContextActive(ctx); err != nil {
			return nil, err
		}
		if err := e.embedImagePathLocked(paths[i].path, vec); err != nil {
			return nil, err
		}
		results[i] = vec
	}
	return results, nil
}

func (e *Embedder) embedImagePathLocked(path string, out []float32) error {
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
		return e.lastErrorLocked()
	}

	return nil
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

func ensureContextActive(ctx context.Context) error {
	if ctx == nil {
		return nil
	}
	return ctx.Err()
}
