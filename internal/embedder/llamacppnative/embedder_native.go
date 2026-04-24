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
	"errors"
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
	ModelPath              string
	VisionModelPath        string
	Dimensions             int
	GPULayers              int
	UseGPU                 bool
	ContextSize            int
	BatchSize              int
	MaxSequences           int
	Threads                int
	ImageMaxSide           int
	AnnotationImageMaxSide int
	ImageMaxTokens         int
	AnnotationTemperature  float32
	AnnotationSeed         int64
	QueryInstruction       string
	PassageInstruction     string
	FlashAttnType          int
	CacheTypeK             int
	CacheTypeV             int
}

type Embedder struct {
	mu                     sync.Mutex
	handle                 *C.imgsearch_llama_handle
	dimensions             int
	maxSequences           int
	imageMaxSide           int
	annotationImageMaxSide int
	annotationTemperature  float32
	annotationSeed         int64
	queryInstruction       string
	passageInstruction     string
}

type EmbedInspect struct {
	TextChunks     int
	ImageChunks    int
	TextTokens     int
	ImageTokens    int
	TotalTokens    int
	TotalPositions int
	MaxImageTokens int
	MaxImageNX     int
	MaxImageNY     int
	NCtx           int
	NCtxSeq        int
	NSeqMax        int
	NBatch         int
}

type preparedEmbedPath struct {
	path    string
	cleanup func()
}

type pipelinePreprocessError struct {
	err error
}

func (e pipelinePreprocessError) Error() string {
	if e.err == nil {
		return "pipeline preprocess failed"
	}
	return fmt.Sprintf("pipeline preprocess failed: %v", e.err)
}

func (e pipelinePreprocessError) Unwrap() error {
	return e.err
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

	annotationImageMaxSide := cfg.AnnotationImageMaxSide
	if annotationImageMaxSide == 0 {
		annotationImageMaxSide = imageMaxSide
	}
	if annotationImageMaxSide < 0 {
		return nil, fmt.Errorf("llama-cpp-native annotation image max side must be non-negative")
	}

	queryInstruction := strings.TrimSpace(cfg.QueryInstruction)
	if queryInstruction == "" {
		queryInstruction = defaultQueryInstruction
	}
	passageInstruction := strings.TrimSpace(cfg.PassageInstruction)
	if passageInstruction == "" {
		passageInstruction = defaultPassageInstruction
	}

	annotationTemperature, annotationSeed, err := normalizeAnnotationSampling(cfg.AnnotationTemperature, cfg.AnnotationSeed)
	if err != nil {
		return nil, err
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
		C.int32_t(maxSequences),
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
		handle:                 h,
		dimensions:             actualDims,
		maxSequences:           maxSequences,
		imageMaxSide:           imageMaxSide,
		annotationImageMaxSide: annotationImageMaxSide,
		annotationTemperature:  annotationTemperature,
		annotationSeed:         annotationSeed,
		queryInstruction:       queryInstruction,
		passageInstruction:     passageInstruction,
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
	if err := ensureContextActive(ctx); err != nil {
		return nil, err
	}

	// Fast-fail before preprocessing when the embedder is already closed.
	// executePreparedImageLocked performs the authoritative handle check.
	e.mu.Lock()
	isClosed := e.handle == nil
	e.mu.Unlock()
	if isClosed {
		return nil, fmt.Errorf("llama-cpp-native embedder is closed")
	}

	prepared, err := e.prepareImageForEmbedding(ctx, path)
	if err != nil {
		return nil, err
	}
	if prepared.cleanup == nil {
		prepared.cleanup = func() {}
	}
	defer prepared.cleanup()

	return e.executePreparedImage(ctx, prepared)
}

func (e *Embedder) EmbedImages(ctx context.Context, paths []string) ([][]float32, error) {
	if err := ensureContextActive(ctx); err != nil {
		return nil, err
	}
	if len(paths) == 0 {
		return nil, fmt.Errorf("image path batch is empty")
	}

	chunkSize := e.maxSequences
	if chunkSize <= 0 {
		chunkSize = 1
	}

	results := make([][]float32, 0, len(paths))
	for start := 0; start < len(paths); start += chunkSize {
		end := start + chunkSize
		if end > len(paths) {
			end = len(paths)
		}
		e.mu.Lock()
		if e.handle == nil {
			e.mu.Unlock()
			return nil, fmt.Errorf("llama-cpp-native embedder is closed")
		}
		chunkOut, err := runEmbedChunk(
			ctx,
			paths[start:end],
			func(preprocessCtx context.Context, path string) (preparedEmbedPath, error) {
				return e.prepareImageForEmbedding(preprocessCtx, path)
			},
			e.executePreparedImageLocked,
		)
		e.mu.Unlock()
		if err != nil {
			return nil, err
		}
		results = append(results, chunkOut...)
	}

	return results, nil
}

func (e *Embedder) prepareImageForEmbedding(ctx context.Context, path string) (preparedEmbedPath, error) {
	if err := ensureContextActive(ctx); err != nil {
		return preparedEmbedPath{}, err
	}
	preparedPath, cleanup, err := preprocessImageForEmbeddingWithVipsgen(path, e.imageMaxSide)
	if err != nil {
		return preparedEmbedPath{}, err
	}
	return preparedEmbedPath{path: preparedPath, cleanup: cleanup}, nil
}

func (e *Embedder) executePreparedImage(ctx context.Context, prepared preparedEmbedPath) ([]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.executePreparedImageLocked(ctx, prepared)
}

func (e *Embedder) executePreparedImageLocked(ctx context.Context, prepared preparedEmbedPath) ([]float32, error) {
	if err := ensureContextActive(ctx); err != nil {
		return nil, err
	}
	if e.handle == nil {
		return nil, fmt.Errorf("llama-cpp-native embedder is closed")
	}
	vec := make([]float32, e.dimensions)
	if err := e.embedImagePathLocked(prepared.path, vec); err != nil {
		return nil, err
	}
	return vec, nil
}

func runEmbedChunk(
	ctx context.Context,
	paths []string,
	preprocess func(context.Context, string) (preparedEmbedPath, error),
	embed func(context.Context, preparedEmbedPath) ([]float32, error),
) ([][]float32, error) {
	chunkOut, err := runEmbedPipeline(ctx, paths, preprocess, embed)
	if err == nil {
		return chunkOut, nil
	}

	var preprocessErr pipelinePreprocessError
	if !errors.As(err, &preprocessErr) {
		return nil, err
	}

	return runEmbedSerial(ctx, paths, preprocess, embed)
}

func runEmbedSerial(
	ctx context.Context,
	paths []string,
	preprocess func(context.Context, string) (preparedEmbedPath, error),
	embed func(context.Context, preparedEmbedPath) ([]float32, error),
) ([][]float32, error) {
	if len(paths) == 0 {
		return [][]float32{}, nil
	}

	results := make([][]float32, 0, len(paths))
	for _, path := range paths {
		if err := ensureContextActive(ctx); err != nil {
			return nil, err
		}
		prepared, err := preprocess(ctx, path)
		if err != nil {
			return nil, err
		}
		if prepared.cleanup == nil {
			prepared.cleanup = func() {}
		}
		if err := ensureContextActive(ctx); err != nil {
			prepared.cleanup()
			return nil, err
		}
		vec, err := embed(ctx, prepared)
		prepared.cleanup()
		if err != nil {
			return nil, err
		}
		results = append(results, vec)
	}

	return results, nil
}

func runEmbedPipeline(
	ctx context.Context,
	paths []string,
	preprocess func(context.Context, string) (preparedEmbedPath, error),
	embed func(context.Context, preparedEmbedPath) ([]float32, error),
) ([][]float32, error) {
	if len(paths) == 0 {
		return [][]float32{}, nil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	preparedCh := make(chan preparedEmbedPath)
	producerErrCh := make(chan error, 1)
	go func() {
		defer close(preparedCh)
		for _, path := range paths {
			if err := ensureContextActive(runCtx); err != nil {
				producerErrCh <- err
				return
			}
			prepared, err := preprocess(runCtx, path)
			if err != nil {
				producerErrCh <- pipelinePreprocessError{err: err}
				return
			}
			if prepared.cleanup == nil {
				prepared.cleanup = func() {}
			}
			select {
			case preparedCh <- prepared:
			case <-runCtx.Done():
				prepared.cleanup()
				producerErrCh <- runCtx.Err()
				return
			}
		}
		producerErrCh <- nil
	}()

	producerDone := false
	waitProducer := func() error {
		if producerDone {
			return nil
		}
		producerDone = true
		return <-producerErrCh
	}

	results := make([][]float32, 0, len(paths))
	for prepared := range preparedCh {
		if err := ensureContextActive(runCtx); err != nil {
			prepared.cleanup()
			cancel()
			_ = waitProducer()
			return nil, err
		}
		vec, err := embed(runCtx, prepared)
		prepared.cleanup()
		if err != nil {
			cancel()
			_ = waitProducer()
			return nil, err
		}
		results = append(results, vec)
	}

	if err := waitProducer(); err != nil {
		if ctxErr := ensureContextActive(ctx); ctxErr != nil {
			return nil, ctxErr
		}
		return nil, err
	}

	return results, nil
}

func (e *Embedder) InspectImageEmbedding(ctx context.Context, path string) (EmbedInspect, error) {
	var out EmbedInspect
	if err := ensureContextActive(ctx); err != nil {
		return out, err
	}

	prepared, err := e.prepareImageForEmbedding(ctx, path)
	if err != nil {
		return out, err
	}
	defer prepared.cleanup()

	e.mu.Lock()
	defer e.mu.Unlock()

	if e.handle == nil {
		return out, fmt.Errorf("llama-cpp-native embedder is closed")
	}

	cPath := C.CString(prepared.path)
	defer C.free(unsafe.Pointer(cPath))
	cInstruction := C.CString(e.passageInstruction)
	defer C.free(unsafe.Pointer(cInstruction))

	var nativeOut C.imgsearch_llama_embed_inspect
	if C.imgsearch_llama_inspect_embed_image(e.handle, cPath, cInstruction, &nativeOut) != 0 {
		return out, e.lastErrorLocked()
	}

	out = EmbedInspect{
		TextChunks:     int(nativeOut.text_chunks),
		ImageChunks:    int(nativeOut.image_chunks),
		TextTokens:     int(nativeOut.text_tokens),
		ImageTokens:    int(nativeOut.image_tokens),
		TotalTokens:    int(nativeOut.total_tokens),
		TotalPositions: int(nativeOut.total_positions),
		MaxImageTokens: int(nativeOut.max_image_tokens),
		MaxImageNX:     int(nativeOut.max_image_nx),
		MaxImageNY:     int(nativeOut.max_image_ny),
		NCtx:           int(nativeOut.n_ctx),
		NCtxSeq:        int(nativeOut.n_ctx_seq),
		NSeqMax:        int(nativeOut.n_seq_max),
		NBatch:         int(nativeOut.n_batch),
	}

	return out, nil
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
