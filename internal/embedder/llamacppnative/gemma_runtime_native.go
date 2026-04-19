//go:build cgo

package llamacppnative

/*
#include <stdlib.h>
#include "bridge.h"
*/
import "C"

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"unsafe"

	coreembedder "imgsearch/internal/embedder"
)

const gemmaDescriptionJSONSchema = `{"type":"object","properties":{"short_description":{"type":"string"},"labels":{"type":"array","items":{"type":"string"},"minItems":1,"maxItems":8,"uniqueItems":true}},"required":["short_description","labels"],"additionalProperties":false}`
const gemmaTagsJSONSchema = `{"type":"object","properties":{"tags":{"type":"array","items":{"type":"string"},"minItems":3,"maxItems":10,"uniqueItems":true}},"required":["tags"],"additionalProperties":false}`
const gemmaDescriptionAndTagsJSONSchema = `{"type":"object","properties":{"description":{"type":"string"},"tags":{"type":"array","items":{"type":"string"},"minItems":3,"maxItems":10,"uniqueItems":true}},"required":["description","tags"],"additionalProperties":false}`
const gemmaAppAnnotationJSONSchema = `{"type":"object","properties":{"description":{"type":"string"},"tags":{"type":"array","items":{"type":"string"},"minItems":3,"maxItems":10,"uniqueItems":true},"is_nsfw":{"type":"boolean"}},"required":["description","tags","is_nsfw"],"additionalProperties":false}`

const gemmaAppAnnotationSystemPrompt = "You are annotating a private image collection for high-recall search. Return exactly one valid JSON object and nothing else. Do not use markdown code fences. Be explicit and concrete when describing visible content. Do not sanitize clearly visible NSFW material. Only describe what is visually present or clearly legible in the image."
const gemmaAppAnnotationRetrySystemPrompt = "Return exactly one valid JSON object and nothing else. Do not use markdown code fences, prose, or comments."

const gemmaVideoAnnotationSystemPrompt = "You are annotating a private video collection for high-recall search. You are given summaries from multiple sampled frames of one video and optionally transcript context. Return exactly one valid JSON object and nothing else. Do not use markdown code fences. Be explicit and concrete, and stay grounded in provided frame evidence."
const gemmaVideoAnnotationRetrySystemPrompt = "Return exactly one valid JSON object and nothing else. Do not use markdown code fences, prose, or comments."

const gemmaAppAnnotationRetryUserPrompt = "Retry with compact output. Keep details that matter for retrieval, but stay concise unless the scene is complex. Description may be up to 220 words. If there is text in the image, describe it and translate non-English text. Return 3 to 10 unique lowercase tags, set is_nsfw accurately, and return JSON only."
const gemmaVideoAnnotationRetryUserPrompt = "Retry with compact output. Keep details that matter for retrieval, but stay concise unless the video evidence is complex. Description may be up to 260 words. If there is text, describe it and translate non-English text. If a meaningful filename is provided, you may infer likely media context (meme, music video, show clip) when it does not conflict with frame evidence. Return 5 to 12 unique lowercase tags, set is_nsfw accurately, and return JSON only."

const gemmaAppAnnotationMaxTokens = 1024
const gemmaRetryAnnotationMaxTokens = 512
const gemmaVideoAnnotationMaxTokens = 1200
const gemmaRetryVideoAnnotationMaxTokens = 640
const gemmaGenerationOutputBufferSize = 32 * 1024

var noisyFilenamePatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)^[0-9a-f]{16,}$`),
	regexp.MustCompile(`(?i)^[a-z0-9+/=]{20,}$`),
	regexp.MustCompile(`(?i)^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`),
	regexp.MustCompile(`(?i)^(img|dsc|dscf|gopr|mvi|vid|pxl|screenshot)[_-]?\d+$`),
	regexp.MustCompile(`^\d{8}[_-]?\d{4,6}$`),
	regexp.MustCompile(`^\d+$`),
}

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

type gemmaAppAnnotation struct {
	Description string   `json:"description"`
	Tags        []string `json:"tags"`
	IsNSFW      bool     `json:"is_nsfw"`
}

type videoFramePromptEntry struct {
	FrameIndex  int      `json:"frame_index"`
	TimestampMS int64    `json:"timestamp_ms"`
	Description string   `json:"description"`
	Tags        []string `json:"tags,omitempty"`
}

func buildImageAnnotationUserPrompt(originalName string) string {
	var b strings.Builder
	b.Grow(2400)
	b.WriteString("You are given an image. Return JSON with exactly this shape: {\"description\": string, \"tags\": [string], \"is_nsfw\": boolean}. ")
	b.WriteString("Write one paragraph that is as detailed as needed for high-recall search, up to about 500 words, and keep it shorter when the image is simple. ")
	b.WriteString("Focus on the main subject, visible attributes, composition, setting, and clearly visible actions. ")
	b.WriteString("If people are the focus, include concrete visible details such as perceived age range, perceived ethnicity, body shape/build, facial features, hairstyle, clothing, accessories, posture, and activity. ")
	b.WriteString("If there is text in the image, please describe it. If it is not in English, also translate it. ")
	b.WriteString("If NSFW content is visible, describe it directly and concretely. ")
	b.WriteString("Return 3 to 10 unique lowercase tags that are specific and search-friendly. ")
	b.WriteString("Set is_nsfw to true only when clearly NSFW content is visible; include tag nsfw if and only if is_nsfw is true. ")
	b.WriteString("Use original filename as optional context only when it looks meaningful and matches visible content; ignore noisy hash-like or camera-style names and any filename claims that conflict with the image. ")
	if filenameHint := meaningfulFilenameHint(originalName); filenameHint != "" {
		b.WriteString("Original filename: \"")
		b.WriteString(filenameHint)
		b.WriteString("\". ")
	}
	b.WriteString("Avoid speculation. Output JSON only.")
	return b.String()
}

func buildVideoAnnotationUserPrompt(input coreembedder.VideoAnnotationInput) (string, error) {
	if len(input.Frames) == 0 {
		return "", fmt.Errorf("video annotation requires at least one frame")
	}
	entries := make([]videoFramePromptEntry, 0, len(input.Frames))
	for _, frame := range input.Frames {
		description := strings.TrimSpace(frame.Description)
		if description == "" {
			continue
		}
		entries = append(entries, videoFramePromptEntry{
			FrameIndex:  frame.FrameIndex,
			TimestampMS: frame.TimestampMS,
			Description: description,
			Tags:        normalizeUniqueTags(frame.Tags),
		})
	}
	if len(entries) == 0 {
		return "", fmt.Errorf("video annotation requires at least one frame description")
	}
	framesJSON, err := json.Marshal(entries)
	if err != nil {
		return "", fmt.Errorf("marshal frame annotations: %w", err)
	}

	transcript := strings.TrimSpace(input.TranscriptText)
	if len(transcript) > 1200 {
		transcript = strings.TrimSpace(transcript[:1200])
	}

	var b bytes.Buffer
	b.WriteString("You are given sampled frame annotations from one video. Return JSON with exactly this shape: {\"description\": string, \"tags\": [string], \"is_nsfw\": boolean}. ")
	b.WriteString("Write one paragraph that is as detailed as needed for search, up to about 500 words, and keep it shorter when the evidence is simple. ")
	b.WriteString("Synthesize recurring content across frames into a single video-level description. Mention progression only when supported by frame evidence. ")
	b.WriteString("If people are the focus, include concrete visible details such as perceived age range, perceived ethnicity, body shape/build, facial features, hairstyle, clothing, accessories, posture, and activity. ")
	b.WriteString("If there is text in frames, describe it, and if non-English also translate it. ")
	b.WriteString("Use transcript as supporting context only when it matches visual evidence. ")
	b.WriteString("When a filename looks meaningful, feel free to infer likely media type or source context from it (for example a meme, a music video, or a clip from a show) as long as it does not conflict with frame evidence. ")
	b.WriteString("Return 5 to 12 unique lowercase tags that capture persistent high-signal content. Include nsfw if and only if is_nsfw is true. ")
	b.WriteString("Avoid unsupported specifics and avoid one-off details that are not central. Output JSON only.\n")
	if filenameHint := meaningfulFilenameHint(input.OriginalName); filenameHint != "" {
		b.WriteString("Original filename: \"")
		b.WriteString(filenameHint)
		b.WriteString("\"\n")
	}
	b.WriteString(fmt.Sprintf("Duration (ms): %d\n", input.DurationMS))
	b.WriteString("Sampled frame annotations (chronological):\n")
	b.Write(framesJSON)
	b.WriteString("\n")
	if transcript != "" {
		b.WriteString("Optional transcript snippet:\n")
		b.WriteString(sanitizePromptSnippet(transcript, 1200))
		b.WriteString("\n")
	}
	return b.String(), nil
}

func meaningfulFilenameHint(originalName string) string {
	trimmed := strings.TrimSpace(originalName)
	if trimmed == "" {
		return ""
	}
	base := filepath.Base(trimmed)
	stem := strings.TrimSuffix(base, filepath.Ext(base))
	stem = strings.TrimSpace(stem)
	if stem == "" {
		return ""
	}
	normalized := strings.Join(strings.Fields(strings.NewReplacer("_", " ", "-", " ", ".", " ").Replace(stem)), " ")
	normalized = sanitizePromptSnippet(normalized, 80)
	if normalized == "" {
		return ""
	}
	lower := strings.ToLower(normalized)
	for _, re := range noisyFilenamePatterns {
		if re.MatchString(lower) {
			return ""
		}
	}
	alphaTokenCount := 0
	for _, token := range strings.Fields(lower) {
		letters := 0
		for _, r := range token {
			if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') {
				letters++
			}
		}
		if letters >= 3 {
			alphaTokenCount++
		}
	}
	if alphaTokenCount < 2 {
		return ""
	}
	return normalized
}

func sanitizePromptSnippet(input string, maxLen int) string {
	trimmed := strings.TrimSpace(input)
	if trimmed == "" {
		return ""
	}
	cleaned := strings.Map(func(r rune) rune {
		if r < 32 {
			return -1
		}
		return r
	}, trimmed)
	if maxLen > 0 && len(cleaned) > maxLen {
		cleaned = strings.TrimSpace(cleaned[:maxLen])
	}
	return cleaned
}

type nativeGemmaRuntimeConfig struct {
	ModelPath       string
	VisionModelPath string
	GPULayers       int
	UseGPU          bool
	ContextSize     int
	BatchSize       int
	Threads         int
	ImageMaxSide    int
	ImageMaxTokens  int
	FlashAttnType   int
	CacheTypeK      int
	CacheTypeV      int
}

type AnnotatorConfig = nativeGemmaRuntimeConfig

type nativeGemmaRuntime struct {
	mu           sync.Mutex
	handle       *C.imgsearch_llama_handle
	imageMaxSide int
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

	annotation, _, err := describeAndTagImageWithHandle(
		ctx,
		e.handle,
		e.imageMaxSide,
		imagePath,
		gemmaAppAnnotationSystemPrompt,
		buildImageAnnotationUserPrompt(opts.OriginalName),
		gemmaAppAnnotationJSONSchema,
		gemmaAppAnnotationMaxTokens,
		gemmaAppAnnotationRetrySystemPrompt,
		gemmaAppAnnotationRetryUserPrompt,
		gemmaRetryAnnotationMaxTokens,
	)
	if err != nil {
		return coreembedder.ImageAnnotation{}, err
	}

	return coreembedder.ImageAnnotation{Description: annotation.Description, Tags: annotation.Tags}, nil
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
	userPrompt, err := buildVideoAnnotationUserPrompt(input)
	if err != nil {
		return coreembedder.VideoAnnotation{}, err
	}

	annotation, _, err := describeAndTagImageWithHandle(
		ctx,
		e.handle,
		e.imageMaxSide,
		representativeFramePath,
		gemmaVideoAnnotationSystemPrompt,
		userPrompt,
		gemmaAppAnnotationJSONSchema,
		gemmaVideoAnnotationMaxTokens,
		gemmaVideoAnnotationRetrySystemPrompt,
		gemmaVideoAnnotationRetryUserPrompt,
		gemmaRetryVideoAnnotationMaxTokens,
	)
	if err != nil {
		return coreembedder.VideoAnnotation{}, err
	}

	return coreembedder.VideoAnnotation{Description: annotation.Description, Tags: annotation.Tags, IsNSFW: annotation.IsNSFW}, nil
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
		C.int32_t(cfg.Threads),
		boolToCInt32(cfg.UseGPU),
		C.int32_t(imageMaxSide),
		C.int32_t(cfg.ImageMaxTokens),
		C.int32_t(flashAttnType),
		C.int32_t(cacheTypeK),
		C.int32_t(cacheTypeV),
	)
	if h == nil {
		msg := strings.TrimSpace(C.GoString(C.imgsearch_llama_global_error()))
		if msg == "" {
			msg = "failed to initialize native Gemma runtime"
		}
		return nil, fmt.Errorf("%s", msg)
	}

	return &nativeGemmaRuntime{handle: h, imageMaxSide: imageMaxSide}, nil
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

	annotation, _, err := describeAndTagImageWithHandle(
		ctx,
		r.handle,
		r.imageMaxSide,
		imagePath,
		gemmaAppAnnotationSystemPrompt,
		buildImageAnnotationUserPrompt(opts.OriginalName),
		gemmaAppAnnotationJSONSchema,
		gemmaAppAnnotationMaxTokens,
		gemmaAppAnnotationRetrySystemPrompt,
		gemmaAppAnnotationRetryUserPrompt,
		gemmaRetryAnnotationMaxTokens,
	)
	if err != nil {
		return coreembedder.ImageAnnotation{}, err
	}

	return coreembedder.ImageAnnotation{Description: annotation.Description, Tags: annotation.Tags}, nil
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
	userPrompt, err := buildVideoAnnotationUserPrompt(input)
	if err != nil {
		return coreembedder.VideoAnnotation{}, err
	}

	annotation, _, err := describeAndTagImageWithHandle(
		ctx,
		r.handle,
		r.imageMaxSide,
		representativeFramePath,
		gemmaVideoAnnotationSystemPrompt,
		userPrompt,
		gemmaAppAnnotationJSONSchema,
		gemmaVideoAnnotationMaxTokens,
		gemmaVideoAnnotationRetrySystemPrompt,
		gemmaVideoAnnotationRetryUserPrompt,
		gemmaRetryVideoAnnotationMaxTokens,
	)
	if err != nil {
		return coreembedder.VideoAnnotation{}, err
	}

	return coreembedder.VideoAnnotation{Description: annotation.Description, Tags: annotation.Tags, IsNSFW: annotation.IsNSFW}, nil
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
		jsonOnly, ok := extractJSONObject(raw)
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
	description.Labels = normalizeUniqueTags(description.Labels)

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
		jsonOnly, ok := extractJSONObject(raw)
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
	tags.Tags = normalizeUniqueTags(tags.Tags)

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
		jsonOnly, ok := extractJSONObject(raw)
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
	result.Tags = normalizeUniqueTags(result.Tags)

	return result, raw, nil
}

func describeAndTagImageWithHandle(
	ctx context.Context,
	handle *C.imgsearch_llama_handle,
	imageMaxSide int,
	imagePath string,
	systemPrompt string,
	userPrompt string,
	jsonSchema string,
	maxTokens int,
	retrySystemPrompt string,
	retryUserPrompt string,
	retryMaxTokens int,
) (gemmaAppAnnotation, string, error) {
	raw, err := generateImageJSONForHandle(ctx, handle, imageMaxSide, imagePath, systemPrompt, userPrompt, jsonSchema, maxTokens)
	if err != nil {
		return gemmaAppAnnotation{}, "", err
	}

	var result gemmaAppAnnotation
	if err := decodeGemmaJSONObject(raw, &result, "description+tags"); err != nil {
		retryRaw, retryErr := generateImageJSONForHandle(ctx, handle, imageMaxSide, imagePath, retrySystemPrompt, retryUserPrompt, jsonSchema, retryMaxTokens)
		if retryErr != nil {
			return gemmaAppAnnotation{}, raw, fmt.Errorf("%v; retry failed: %w", err, retryErr)
		}
		var retryResult gemmaAppAnnotation
		if retryDecodeErr := decodeGemmaJSONObject(retryRaw, &retryResult, "description+tags retry"); retryDecodeErr != nil {
			return gemmaAppAnnotation{}, retryRaw, fmt.Errorf("%v; retry decode failed: %w", err, retryDecodeErr)
		}
		result = retryResult
		raw = retryRaw
	}

	result.Description = strings.TrimSpace(result.Description)
	result.Tags = normalizeUniqueTags(result.Tags)
	result.Tags = applyNSFWTag(result.Tags, result.IsNSFW)

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
	return generateImageJSONForHandle(ctx, r.handle, r.imageMaxSide, imagePath, systemPrompt, userPrompt, jsonSchema, maxTokens)
}

func generateImageJSONForHandle(ctx context.Context, handle *C.imgsearch_llama_handle, imageMaxSide int, imagePath string, systemPrompt string, userPrompt string, jsonSchema string, maxTokens int) (string, error) {
	if err := ensureContextActive(ctx); err != nil {
		return "", err
	}
	if handle == nil {
		return "", fmt.Errorf("native Gemma runtime is closed")
	}

	preprocessedPath, cleanup, err := preprocessImageForEmbeddingWithVipsgen(imagePath, imageMaxSide)
	if err != nil {
		return "", err
	}
	defer cleanup()
	if err := ensureContextActive(ctx); err != nil {
		return "", err
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
	res := C.imgsearch_llama_generate_image(
		handle,
		cImagePath,
		cSystemPrompt,
		cUserPrompt,
		cSchema,
		C.int32_t(maxTokens),
		C.float(0),
		C.float(1),
		(*C.char)(unsafe.Pointer(&buf[0])),
		C.int32_t(len(buf)),
	)
	if res != 0 {
		msg := strings.TrimSpace(C.GoString(C.imgsearch_llama_last_error(handle)))
		if msg == "" {
			msg = strings.TrimSpace(C.GoString(C.imgsearch_llama_global_error()))
		}
		if msg == "" {
			msg = "native Gemma generation failed"
		}
		return "", fmt.Errorf("%s", msg)
	}

	raw := strings.TrimSpace(C.GoString((*C.char)(unsafe.Pointer(&buf[0]))))
	if raw == "" {
		return raw, fmt.Errorf("native Gemma generation returned empty output")
	}

	return raw, nil
}

func boolToCInt32(v bool) C.int32_t {
	if v {
		return 1
	}
	return 0
}

func extractJSONObject(raw string) (string, bool) {
	start := strings.IndexByte(raw, '{')
	if start < 0 {
		return "", false
	}

	depth := 0
	inString := false
	escaped := false
	for i := start; i < len(raw); i++ {
		ch := raw[i]
		if inString {
			if escaped {
				escaped = false
				continue
			}
			if ch == '\\' {
				escaped = true
				continue
			}
			if ch == '"' {
				inString = false
			}
			continue
		}

		switch ch {
		case '"':
			inString = true
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return raw[start : i+1], true
			}
		}
	}

	return "", false
}

func decodeGemmaJSONObject(raw string, out any, label string) error {
	normalizedRaw := stripMarkdownCodeFences(strings.TrimSpace(raw))
	if err := json.Unmarshal([]byte(normalizedRaw), out); err != nil {
		jsonOnly, ok := extractJSONObject(normalizedRaw)
		if !ok {
			return fmt.Errorf("decode %s JSON: %w; raw=%q", label, err, raw)
		}
		if err := json.Unmarshal([]byte(jsonOnly), out); err != nil {
			return fmt.Errorf("decode extracted %s JSON: %w; raw=%q", label, err, raw)
		}
	}
	return nil
}

func stripMarkdownCodeFences(raw string) string {
	trimmed := strings.TrimSpace(raw)
	if !strings.HasPrefix(trimmed, "```") {
		return trimmed
	}
	lines := strings.Split(trimmed, "\n")
	if len(lines) == 0 {
		return trimmed
	}
	if strings.HasPrefix(strings.TrimSpace(lines[0]), "```") {
		lines = lines[1:]
	}
	if len(lines) > 0 && strings.TrimSpace(lines[len(lines)-1]) == "```" {
		lines = lines[:len(lines)-1]
	}
	return strings.TrimSpace(strings.Join(lines, "\n"))
}

func normalizeUniqueTags(tags []string) []string {
	seen := make(map[string]struct{}, len(tags))
	normalized := make([]string, 0, len(tags))
	for _, tag := range tags {
		trimmed := strings.ToLower(strings.TrimSpace(tag))
		if trimmed == "" {
			continue
		}
		if _, ok := seen[trimmed]; ok {
			continue
		}
		seen[trimmed] = struct{}{}
		normalized = append(normalized, trimmed)
	}
	return normalized
}

func applyNSFWTag(tags []string, isNSFW bool) []string {
	normalized := normalizeUniqueTags(tags)
	filtered := normalized[:0]
	for _, tag := range normalized {
		if tag == "nsfw" {
			continue
		}
		filtered = append(filtered, tag)
	}
	if isNSFW {
		if len(filtered) >= 10 {
			filtered = filtered[:9]
		}
		filtered = append(filtered, "nsfw")
	}
	return filtered
}
