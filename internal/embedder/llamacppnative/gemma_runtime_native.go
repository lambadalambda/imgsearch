//go:build cgo

package llamacppnative

/*
#include <stdlib.h>
#include "bridge.h"
*/
import "C"

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"path/filepath"
	"strings"
	"sync"
	"time"
	"unsafe"

	"imgsearch/internal/annotation"
	coreembedder "imgsearch/internal/embedder"
)

const gemmaDescriptionJSONSchema = `{"type":"object","properties":{"short_description":{"type":"string"},"labels":{"type":"array","items":{"type":"string"},"minItems":1,"maxItems":8,"uniqueItems":true}},"required":["short_description","labels"],"additionalProperties":false}`
const gemmaTagsJSONSchema = `{"type":"object","properties":{"tags":{"type":"array","items":{"type":"string"},"minItems":3,"maxItems":10,"uniqueItems":true}},"required":["tags"],"additionalProperties":false}`
const gemmaDescriptionAndTagsJSONSchema = `{"type":"object","properties":{"description":{"type":"string"},"tags":{"type":"array","items":{"type":"string"},"minItems":3,"maxItems":10,"uniqueItems":true}},"required":["description","tags"],"additionalProperties":false}`

const gemmaGenerationOutputBufferSize = 32 * 1024
const gemmaAnnotationTopP = 0.95

type gemmaImageDescription struct {
	ShortDescription string   `json:"short_description"`
	Labels           []string `json:"labels"`
}

type gemmaImageTags struct {
	Tags []string `json:"tags"`
}

type gemmaImageDescriptionAndTags struct {
	Description string   `json:"description"`
	Tags        []string `json:"tags"`
}

type nativeGemmaRuntimeConfig struct {
	ModelPath                  string
	VisionModelPath            string
	GPULayers                  int
	UseGPU                     bool
	ContextSize                int
	BatchSize                  int
	Threads                    int
	ImageMaxSide               int
	ImageMaxTokens             int
	AnnotationTemperature      float32
	AnnotationSeed             int64
	AnnotationNGramSpeculation bool
	FlashAttnType              int
	CacheTypeK                 int
	CacheTypeV                 int
}

type AnnotatorConfig = nativeGemmaRuntimeConfig

type nativeGemmaRuntime struct {
	mu                    sync.Mutex
	handle                *C.imgsearch_llama_handle
	imageMaxSide          int
	annotationTemperature float32
	annotationSeed        int64
}

type generationTiming struct {
	ImagePreprocessMS         int64
	NativeDecodeMS            int64
	TokenizeMS                int64
	PrefillMS                 int64
	GenerateMS                int64
	PromptTokens              int
	GeneratedTokens           int
	SpeculativeDraftedTokens  int
	SpeculativeAcceptedTokens int
}

type Annotator = nativeGemmaRuntime

func (e *Embedder) AnnotateImage(ctx context.Context, imagePath string) (coreembedder.ImageAnnotation, error) {
	return e.AnnotateImageWithOptions(ctx, imagePath, coreembedder.ImageAnnotationOptions{})
}

func (e *Embedder) AnnotateImageWithOptions(ctx context.Context, imagePath string, opts coreembedder.ImageAnnotationOptions) (coreembedder.ImageAnnotation, error) {
	if err := ensureContextActive(ctx); err != nil {
		return coreembedder.ImageAnnotation{}, err
	}

	e.mu.Lock()
	defer e.mu.Unlock()
	if e.handle == nil {
		return coreembedder.ImageAnnotation{}, fmt.Errorf("llama-cpp-native embedder is closed")
	}
	annotationImageMaxSide := annotationImageMaxSideWithMultiplier(e.annotationImageMaxSide, opts.ImageMaxSideMultiplier)

	resp, _, err := describeAndTagImageWithHandle(
		ctx,
		e.handle,
		"annotate_image",
		annotationImageMaxSide,
		e.annotationTemperature,
		e.annotationSeed,
		imagePath,
		annotation.ImageSystemPrompt,
		annotation.ImageUserPrompt(opts.OriginalName),
		annotation.FullJSONSchema,
		annotation.ImageMaxTokens,
		annotation.ImageRetrySystemPrompt,
		annotation.ImageRetryUserPrompt,
		annotation.ImageRetryMaxTokens,
	)
	if err != nil {
		return coreembedder.ImageAnnotation{}, err
	}

	return resp.ImageAnnotation(), nil
}

func (e *Embedder) AnnotateVideo(ctx context.Context, input coreembedder.VideoAnnotationInput) (coreembedder.VideoAnnotation, error) {
	if err := ensureContextActive(ctx); err != nil {
		return coreembedder.VideoAnnotation{}, err
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.handle == nil {
		return coreembedder.VideoAnnotation{}, fmt.Errorf("llama-cpp-native embedder is closed")
	}
	representativeFramePath := strings.TrimSpace(input.RepresentativeFramePath)
	if representativeFramePath == "" {
		return coreembedder.VideoAnnotation{}, fmt.Errorf("representative frame path is required")
	}
	if len(input.Frames) == 0 {
		return coreembedder.VideoAnnotation{}, fmt.Errorf("at least one frame annotation is required")
	}
	userPrompt, err := annotation.VideoUserPrompt(input)
	if err != nil {
		return coreembedder.VideoAnnotation{}, err
	}
	annotationImageMaxSide := annotationImageMaxSideWithMultiplier(e.annotationImageMaxSide, input.ImageMaxSideMultiplier)

	resp, _, err := describeAndTagImageWithHandle(
		ctx,
		e.handle,
		"annotate_video",
		annotationImageMaxSide,
		e.annotationTemperature,
		e.annotationSeed,
		representativeFramePath,
		annotation.VideoSystemPrompt,
		userPrompt,
		annotation.FullJSONSchema,
		annotation.VideoMaxTokens,
		annotation.VideoRetrySystemPrompt,
		annotation.VideoRetryUserPrompt,
		annotation.VideoRetryMaxTokens,
	)
	if err != nil {
		return coreembedder.VideoAnnotation{}, err
	}

	return resp.VideoAnnotation(), nil
}

func (e *Embedder) AnnotateVideoFrame(ctx context.Context, imagePath string, opts coreembedder.ImageAnnotationOptions) (coreembedder.ImageAnnotation, error) {
	if err := ensureContextActive(ctx); err != nil {
		return coreembedder.ImageAnnotation{}, err
	}

	e.mu.Lock()
	defer e.mu.Unlock()
	if e.handle == nil {
		return coreembedder.ImageAnnotation{}, fmt.Errorf("llama-cpp-native embedder is closed")
	}
	annotationImageMaxSide := annotationImageMaxSideWithMultiplier(e.annotationImageMaxSide, opts.ImageMaxSideMultiplier)

	resp, _, err := describeAndTagImageWithHandle(
		ctx,
		e.handle,
		"annotate_video_frame",
		annotationImageMaxSide,
		e.annotationTemperature,
		e.annotationSeed,
		imagePath,
		annotation.VideoFrameSystemPrompt,
		annotation.VideoFrameUserPrompt(opts.OriginalName),
		annotation.CompactJSONSchema,
		annotation.VideoFrameMaxTokens,
		annotation.VideoFrameRetrySystemPrompt,
		annotation.VideoFrameRetryUserPrompt,
		annotation.VideoFrameRetryMaxTokens,
	)
	if err != nil {
		return coreembedder.ImageAnnotation{}, err
	}

	return resp.ImageAnnotation(), nil
}

func NewAnnotator(cfg AnnotatorConfig) (*Annotator, error) {
	return newGemmaNativeRuntime(nativeGemmaRuntimeConfig(cfg))
}

func newGemmaNativeRuntime(cfg nativeGemmaRuntimeConfig) (*nativeGemmaRuntime, error) {
	modelPath := strings.TrimSpace(cfg.ModelPath)
	if modelPath == "" {
		return nil, fmt.Errorf("Gemma model path is required")
	}
	visionPath := strings.TrimSpace(cfg.VisionModelPath)
	if visionPath == "" {
		return nil, fmt.Errorf("Gemma mmproj path is required")
	}

	imageMaxSide := cfg.ImageMaxSide
	if imageMaxSide <= 0 {
		imageMaxSide = defaultImageMaxSide
	}
	annotationTemperature, annotationSeed, err := normalizeAnnotationSampling(cfg.AnnotationTemperature, cfg.AnnotationSeed)
	if err != nil {
		return nil, err
	}

	cModelPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cModelPath))
	cVisionPath := C.CString(visionPath)
	defer C.free(unsafe.Pointer(cVisionPath))

	flashAttnType := cfg.FlashAttnType
	if flashAttnType == 0 {
		flashAttnType = -1
	}
	cacheTypeK := cfg.CacheTypeK
	if cacheTypeK == 0 {
		cacheTypeK = -1
	}
	cacheTypeV := cfg.CacheTypeV
	if cacheTypeV == 0 {
		cacheTypeV = -1
	}

	h := C.imgsearch_llama_new(
		cModelPath,
		cVisionPath,
		C.int32_t(cfg.GPULayers),
		C.int32_t(cfg.ContextSize),
		C.int32_t(cfg.BatchSize),
		// Annotator runs single-sequence generation on seq_id=0.
		C.int32_t(defaultMaxSequences),
		C.int32_t(cfg.Threads),
		boolToCInt32(cfg.UseGPU),
		C.int32_t(imageMaxSide),
		C.int32_t(cfg.ImageMaxTokens),
		C.int32_t(flashAttnType),
		C.int32_t(cacheTypeK),
		C.int32_t(cacheTypeV),
		boolToCInt32(cfg.AnnotationNGramSpeculation),
	)
	if h == nil {
		msg := strings.TrimSpace(C.GoString(C.imgsearch_llama_global_error()))
		if msg == "" {
			msg = "failed to initialize native Gemma runtime"
		}
		return nil, fmt.Errorf("%s", msg)
	}

	return &nativeGemmaRuntime{
		handle:                h,
		imageMaxSide:          imageMaxSide,
		annotationTemperature: annotationTemperature,
		annotationSeed:        annotationSeed,
	}, nil
}

func (r *nativeGemmaRuntime) Close() error {
	if r == nil {
		return nil
	}

	r.mu.Lock()
	handle := r.handle
	r.handle = nil
	r.mu.Unlock()

	if handle == nil {
		return nil
	}
	C.imgsearch_llama_free(handle)
	return nil
}

func (r *nativeGemmaRuntime) AnnotateImage(ctx context.Context, imagePath string) (coreembedder.ImageAnnotation, error) {
	return r.AnnotateImageWithOptions(ctx, imagePath, coreembedder.ImageAnnotationOptions{})
}

func (r *nativeGemmaRuntime) AnnotateImageWithOptions(ctx context.Context, imagePath string, opts coreembedder.ImageAnnotationOptions) (coreembedder.ImageAnnotation, error) {
	if err := ensureContextActive(ctx); err != nil {
		return coreembedder.ImageAnnotation{}, err
	}
	if r == nil {
		return coreembedder.ImageAnnotation{}, fmt.Errorf("native Gemma runtime is closed")
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	if r.handle == nil {
		return coreembedder.ImageAnnotation{}, fmt.Errorf("native Gemma runtime is closed")
	}
	annotationImageMaxSide := annotationImageMaxSideWithMultiplier(r.imageMaxSide, opts.ImageMaxSideMultiplier)

	resp, _, err := describeAndTagImageWithHandle(
		ctx,
		r.handle,
		"annotate_image",
		annotationImageMaxSide,
		r.annotationTemperature,
		r.annotationSeed,
		imagePath,
		annotation.ImageSystemPrompt,
		annotation.ImageUserPrompt(opts.OriginalName),
		annotation.FullJSONSchema,
		annotation.ImageMaxTokens,
		annotation.ImageRetrySystemPrompt,
		annotation.ImageRetryUserPrompt,
		annotation.ImageRetryMaxTokens,
	)
	if err != nil {
		return coreembedder.ImageAnnotation{}, err
	}

	return resp.ImageAnnotation(), nil
}

func (r *nativeGemmaRuntime) AnnotateVideo(ctx context.Context, input coreembedder.VideoAnnotationInput) (coreembedder.VideoAnnotation, error) {
	if err := ensureContextActive(ctx); err != nil {
		return coreembedder.VideoAnnotation{}, err
	}
	if r == nil {
		return coreembedder.VideoAnnotation{}, fmt.Errorf("native Gemma runtime is closed")
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	if r.handle == nil {
		return coreembedder.VideoAnnotation{}, fmt.Errorf("native Gemma runtime is closed")
	}
	representativeFramePath := strings.TrimSpace(input.RepresentativeFramePath)
	if representativeFramePath == "" {
		return coreembedder.VideoAnnotation{}, fmt.Errorf("representative frame path is required")
	}
	if len(input.Frames) == 0 {
		return coreembedder.VideoAnnotation{}, fmt.Errorf("at least one frame annotation is required")
	}
	userPrompt, err := annotation.VideoUserPrompt(input)
	if err != nil {
		return coreembedder.VideoAnnotation{}, err
	}
	annotationImageMaxSide := annotationImageMaxSideWithMultiplier(r.imageMaxSide, input.ImageMaxSideMultiplier)

	resp, _, err := describeAndTagImageWithHandle(
		ctx,
		r.handle,
		"annotate_video",
		annotationImageMaxSide,
		r.annotationTemperature,
		r.annotationSeed,
		representativeFramePath,
		annotation.VideoSystemPrompt,
		userPrompt,
		annotation.FullJSONSchema,
		annotation.VideoMaxTokens,
		annotation.VideoRetrySystemPrompt,
		annotation.VideoRetryUserPrompt,
		annotation.VideoRetryMaxTokens,
	)
	if err != nil {
		return coreembedder.VideoAnnotation{}, err
	}

	return resp.VideoAnnotation(), nil
}

func (r *nativeGemmaRuntime) AnnotateVideoFrame(ctx context.Context, imagePath string, opts coreembedder.ImageAnnotationOptions) (coreembedder.ImageAnnotation, error) {
	if err := ensureContextActive(ctx); err != nil {
		return coreembedder.ImageAnnotation{}, err
	}
	if r == nil {
		return coreembedder.ImageAnnotation{}, fmt.Errorf("native Gemma runtime is closed")
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	if r.handle == nil {
		return coreembedder.ImageAnnotation{}, fmt.Errorf("native Gemma runtime is closed")
	}
	annotationImageMaxSide := annotationImageMaxSideWithMultiplier(r.imageMaxSide, opts.ImageMaxSideMultiplier)

	resp, _, err := describeAndTagImageWithHandle(
		ctx,
		r.handle,
		"annotate_video_frame",
		annotationImageMaxSide,
		r.annotationTemperature,
		r.annotationSeed,
		imagePath,
		annotation.VideoFrameSystemPrompt,
		annotation.VideoFrameUserPrompt(opts.OriginalName),
		annotation.CompactJSONSchema,
		annotation.VideoFrameMaxTokens,
		annotation.VideoFrameRetrySystemPrompt,
		annotation.VideoFrameRetryUserPrompt,
		annotation.VideoFrameRetryMaxTokens,
	)
	if err != nil {
		return coreembedder.ImageAnnotation{}, err
	}

	return resp.ImageAnnotation(), nil
}

func (r *nativeGemmaRuntime) DescribeImage(ctx context.Context, imagePath string) (gemmaImageDescription, string, error) {
	systemPrompt := "You are a precise vision model. Return only valid JSON that describes visible image content."
	userPrompt := "Describe this image in concrete visual terms. Return JSON with short_description as one sentence and labels as 3-8 lowercase tags. Mention the main subject and setting when obvious. Avoid speculation."
	raw, err := r.generateImageJSON(ctx, imagePath, systemPrompt, userPrompt, gemmaDescriptionJSONSchema, 160)
	if err != nil {
		return gemmaImageDescription{}, "", err
	}

	var description gemmaImageDescription
	if err := json.Unmarshal([]byte(raw), &description); err != nil {
		jsonOnly, ok := annotation.ExtractJSONObject(raw)
		if !ok {
			return gemmaImageDescription{}, raw, fmt.Errorf("decode description JSON: %w; raw=%q", err, raw)
		}
		if err := json.Unmarshal([]byte(jsonOnly), &description); err != nil {
			return gemmaImageDescription{}, raw, fmt.Errorf("decode extracted description JSON: %w; raw=%q", err, raw)
		}
	}
	description.ShortDescription = strings.TrimSpace(description.ShortDescription)
	for i, label := range description.Labels {
		description.Labels[i] = strings.ToLower(strings.TrimSpace(label))
	}
	description.Labels = annotation.NormalizeUniqueTags(description.Labels)

	return description, raw, nil
}

func (r *nativeGemmaRuntime) AutoTagImage(ctx context.Context, imagePath string) (gemmaImageTags, string, error) {
	systemPrompt := "You are a precise image autotagger. Return only valid JSON with concise tags for visible content."
	userPrompt := "Generate the most relevant autotags for this image. Return JSON with tags as an array of 3 to 10 unique lowercase tags. Prioritize the main subject, obvious attributes, and the setting. Use short search-friendly tags only. Avoid speculation, full sentences, and duplicate or overly generic tags unless they are clearly useful."
	raw, err := r.generateImageJSON(ctx, imagePath, systemPrompt, userPrompt, gemmaTagsJSONSchema, 120)
	if err != nil {
		return gemmaImageTags{}, "", err
	}

	var tags gemmaImageTags
	if err := json.Unmarshal([]byte(raw), &tags); err != nil {
		jsonOnly, ok := annotation.ExtractJSONObject(raw)
		if !ok {
			return gemmaImageTags{}, raw, fmt.Errorf("decode tags JSON: %w; raw=%q", err, raw)
		}
		if err := json.Unmarshal([]byte(jsonOnly), &tags); err != nil {
			return gemmaImageTags{}, raw, fmt.Errorf("decode extracted tags JSON: %w; raw=%q", err, raw)
		}
	}
	for i, tag := range tags.Tags {
		tags.Tags[i] = strings.ToLower(strings.TrimSpace(tag))
	}
	tags.Tags = annotation.NormalizeUniqueTags(tags.Tags)

	return tags, raw, nil
}

func (r *nativeGemmaRuntime) DescribeAndTagImage(ctx context.Context, imagePath string) (gemmaImageDescriptionAndTags, string, error) {
	systemPrompt := "You are a precise vision model. Return only valid JSON with a useful medium-length description and concise tags for visible image content."
	userPrompt := "Describe this image in one medium-length paragraph of 2 to 4 sentences. Focus on the main subject, visible attributes, composition, and setting. Then return 3 to 10 unique lowercase tags that are most relevant for search. Avoid speculation, avoid repetition, and keep the description grounded in what is visible."
	raw, err := r.generateImageJSON(ctx, imagePath, systemPrompt, userPrompt, gemmaDescriptionAndTagsJSONSchema, 220)
	if err != nil {
		return gemmaImageDescriptionAndTags{}, "", err
	}

	var result gemmaImageDescriptionAndTags
	if err := json.Unmarshal([]byte(raw), &result); err != nil {
		jsonOnly, ok := annotation.ExtractJSONObject(raw)
		if !ok {
			return gemmaImageDescriptionAndTags{}, raw, fmt.Errorf("decode description+tags JSON: %w; raw=%q", err, raw)
		}
		if err := json.Unmarshal([]byte(jsonOnly), &result); err != nil {
			return gemmaImageDescriptionAndTags{}, raw, fmt.Errorf("decode extracted description+tags JSON: %w; raw=%q", err, raw)
		}
	}

	result.Description = strings.TrimSpace(result.Description)
	for i, tag := range result.Tags {
		result.Tags[i] = strings.ToLower(strings.TrimSpace(tag))
	}
	result.Tags = annotation.NormalizeUniqueTags(result.Tags)

	return result, raw, nil
}

func describeAndTagImageWithHandle(
	ctx context.Context,
	handle *C.imgsearch_llama_handle,
	kind string,
	imageMaxSide int,
	annotationTemperature float32,
	annotationSeed int64,
	imagePath string,
	systemPrompt string,
	userPrompt string,
	jsonSchema string,
	maxTokens int,
	retrySystemPrompt string,
	retryUserPrompt string,
	retryMaxTokens int,
) (annotation.Response, string, error) {
	raw, timing, err := generateImageJSONForHandleWithTiming(ctx, handle, imageMaxSide, annotationTemperature, annotationSeed, imagePath, systemPrompt, userPrompt, jsonSchema, maxTokens)
	if err != nil {
		return annotation.Response{}, "", err
	}
	log.Printf("%s", formatGenerationTimingLog(kind, "primary", imagePath, timing))

	var result annotation.Response
	if err := annotation.DecodeJSONObject(raw, &result); err != nil {
		retryRaw, retryTiming, retryErr := generateImageJSONForHandleWithTiming(ctx, handle, imageMaxSide, annotationTemperature, annotationSeed, imagePath, retrySystemPrompt, retryUserPrompt, jsonSchema, retryMaxTokens)
		if retryErr != nil {
			return annotation.Response{}, raw, fmt.Errorf("%v; retry failed: %w", err, retryErr)
		}
		log.Printf("%s", formatGenerationTimingLog(kind, "retry", imagePath, retryTiming))
		var retryResult annotation.Response
		if retryDecodeErr := annotation.DecodeJSONObject(retryRaw, &retryResult); retryDecodeErr != nil {
			return annotation.Response{}, retryRaw, fmt.Errorf("%v; retry decode failed: %w", err, retryDecodeErr)
		}
		result = retryResult
		raw = retryRaw
	}

	result = annotation.Normalize(result)

	return result, raw, nil
}

func (r *nativeGemmaRuntime) generateImageJSON(ctx context.Context, imagePath string, systemPrompt string, userPrompt string, jsonSchema string, maxTokens int) (string, error) {
	if err := ensureContextActive(ctx); err != nil {
		return "", err
	}
	if r == nil {
		return "", fmt.Errorf("native Gemma runtime is closed")
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	if r.handle == nil {
		return "", fmt.Errorf("native Gemma runtime is closed")
	}
	return generateImageJSONForHandle(ctx, r.handle, r.imageMaxSide, r.annotationTemperature, r.annotationSeed, imagePath, systemPrompt, userPrompt, jsonSchema, maxTokens)
}

func generateImageJSONForHandle(ctx context.Context, handle *C.imgsearch_llama_handle, imageMaxSide int, annotationTemperature float32, annotationSeed int64, imagePath string, systemPrompt string, userPrompt string, jsonSchema string, maxTokens int) (string, error) {
	raw, _, err := generateImageJSONForHandleWithTiming(ctx, handle, imageMaxSide, annotationTemperature, annotationSeed, imagePath, systemPrompt, userPrompt, jsonSchema, maxTokens)
	return raw, err
}

func generateImageJSONForHandleWithTiming(ctx context.Context, handle *C.imgsearch_llama_handle, imageMaxSide int, annotationTemperature float32, annotationSeed int64, imagePath string, systemPrompt string, userPrompt string, jsonSchema string, maxTokens int) (string, generationTiming, error) {
	if err := ensureContextActive(ctx); err != nil {
		return "", generationTiming{}, err
	}
	if handle == nil {
		return "", generationTiming{}, fmt.Errorf("native Gemma runtime is closed")
	}

	preprocessStartedAt := time.Now()
	preprocessedPath, cleanup, err := preprocessImageForEmbeddingWithVipsgen(imagePath, imageMaxSide)
	if err != nil {
		return "", generationTiming{}, err
	}
	defer cleanup()
	timing := generationTiming{ImagePreprocessMS: time.Since(preprocessStartedAt).Milliseconds()}
	if err := ensureContextActive(ctx); err != nil {
		return "", timing, err
	}

	cImagePath := C.CString(preprocessedPath)
	defer C.free(unsafe.Pointer(cImagePath))
	cSystemPrompt := C.CString(systemPrompt)
	defer C.free(unsafe.Pointer(cSystemPrompt))
	cUserPrompt := C.CString(userPrompt)
	defer C.free(unsafe.Pointer(cUserPrompt))
	cSchema := C.CString(jsonSchema)
	defer C.free(unsafe.Pointer(cSchema))

	buf := make([]byte, gemmaGenerationOutputBufferSize)
	nativeTiming := C.imgsearch_llama_generate_timings{}
	res := C.imgsearch_llama_generate_image_with_timings(
		handle,
		cImagePath,
		cSystemPrompt,
		cUserPrompt,
		cSchema,
		C.int32_t(maxTokens),
		C.float(annotationTemperature),
		C.float(gemmaAnnotationTopP),
		C.int64_t(annotationSeed),
		(*C.char)(unsafe.Pointer(&buf[0])),
		C.int32_t(len(buf)),
		&nativeTiming,
	)
	timing.NativeDecodeMS = int64(nativeTiming.native_decode_ms)
	timing.TokenizeMS = int64(nativeTiming.tokenize_ms)
	timing.PrefillMS = int64(nativeTiming.prefill_ms)
	timing.GenerateMS = int64(nativeTiming.generate_ms)
	timing.PromptTokens = int(nativeTiming.prompt_tokens)
	timing.GeneratedTokens = int(nativeTiming.generated_tokens)
	timing.SpeculativeDraftedTokens = int(nativeTiming.speculative_drafted_tokens)
	timing.SpeculativeAcceptedTokens = int(nativeTiming.speculative_accepted_tokens)
	if res != 0 {
		msg := strings.TrimSpace(C.GoString(C.imgsearch_llama_last_error(handle)))
		if msg == "" {
			msg = strings.TrimSpace(C.GoString(C.imgsearch_llama_global_error()))
		}
		if msg == "" {
			msg = "native Gemma generation failed"
		}
		return "", timing, fmt.Errorf("%s", msg)
	}

	raw := strings.TrimSpace(C.GoString((*C.char)(unsafe.Pointer(&buf[0]))))
	if raw == "" {
		return raw, timing, fmt.Errorf("native Gemma generation returned empty output")
	}

	return raw, timing, nil
}

func formatGenerationTimingLog(kind string, attempt string, imagePath string, timing generationTiming) string {
	return fmt.Sprintf(
		"native annotation timing kind=%s attempt=%s file=%q preprocess=%dms native_decode=%dms tokenize=%dms prefill=%dms generate=%dms generated_tokens=%d prompt_tokens=%d speculative_drafted_tokens=%d speculative_accepted_tokens=%d",
		strings.TrimSpace(kind),
		strings.TrimSpace(attempt),
		filepath.Base(strings.TrimSpace(imagePath)),
		timing.ImagePreprocessMS,
		timing.NativeDecodeMS,
		timing.TokenizeMS,
		timing.PrefillMS,
		timing.GenerateMS,
		timing.GeneratedTokens,
		timing.PromptTokens,
		timing.SpeculativeDraftedTokens,
		timing.SpeculativeAcceptedTokens,
	)
}

func boolToCInt32(v bool) C.int32_t {
	if v {
		return 1
	}
	return 0
}
