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
	"strings"
	"sync"
	"unsafe"

	coreembedder "imgsearch/internal/embedder"
)

const gemmaDescriptionJSONSchema = `{"type":"object","properties":{"short_description":{"type":"string"},"labels":{"type":"array","items":{"type":"string"},"minItems":1,"maxItems":8,"uniqueItems":true}},"required":["short_description","labels"],"additionalProperties":false}`
const gemmaTagsJSONSchema = `{"type":"object","properties":{"tags":{"type":"array","items":{"type":"string"},"minItems":3,"maxItems":10,"uniqueItems":true}},"required":["tags"],"additionalProperties":false}`
const gemmaDescriptionAndTagsJSONSchema = `{"type":"object","properties":{"description":{"type":"string"},"tags":{"type":"array","items":{"type":"string"},"minItems":3,"maxItems":10,"uniqueItems":true}},"required":["description","tags"],"additionalProperties":false}`
const gemmaAppAnnotationJSONSchema = `{"type":"object","properties":{"description":{"type":"string"},"tags":{"type":"array","items":{"type":"string"},"minItems":3,"maxItems":10,"uniqueItems":true},"is_nsfw":{"type":"boolean"}},"required":["description","tags","is_nsfw"],"additionalProperties":false}`

const gemmaAppAnnotationSystemPrompt = "You are annotating a private image collection. Return exactly one valid JSON object and nothing else. Do not use markdown code fences. Descriptions may be explicit when the image contains sexual, erotic, nude, fetish, or graphic content; do not sanitize or euphemize clearly visible NSFW material. If the image contains legible text, captions, memes, signs, labels, or screenshots, include the important visible text in the description."

const gemmaAppAnnotationUserPrompt = "Describe this image in one concise paragraph of 1 to 3 sentences and keep it under about 110 words. Focus on the main subject, visible attributes, composition, setting, and any clearly visible actions. This is a private collection, so if the image contains explicit nudity, sexual activity, fetish content, or other not-safe-for-work material, describe that content directly and concretely instead of softening it. If the image contains legible text, transcribe the important visible text needed to understand or search the image, but do not exhaustively list every tiny text fragment. Mention when the image is a meme, screenshot, poster, sign, label, or other text-heavy image. Then return 3 to 10 unique lowercase tags that are most relevant for search. If the image contains explicit nudity, sexual content, fetish content, graphic gore, or other clearly not-safe-for-work material, set is_nsfw to true and include the tag nsfw. Otherwise set is_nsfw to false and do not include the tag nsfw. Avoid speculation, avoid repetition, and keep the description grounded in what is visible. Return JSON only."

const gemmaAppAnnotationRetrySystemPrompt = "Return exactly one valid JSON object and nothing else. Do not use markdown code fences, prose, or comments."

const gemmaAppAnnotationRetryUserPrompt = "Retry with compact output. Write a concise description in 1 to 2 sentences and keep it under about 80 words. If there is important visible text, transcribe only the main visible text needed for search or understanding. Be explicit about visible NSFW content when present. Return 3 to 10 unique lowercase tags and set is_nsfw accurately. Return JSON only."

const gemmaAppAnnotationMaxTokens = 384
const gemmaRetryAnnotationMaxTokens = 256
const gemmaGenerationOutputBufferSize = 32 * 1024

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
}

type AnnotatorConfig = nativeGemmaRuntimeConfig

type nativeGemmaRuntime struct {
	mu           sync.Mutex
	handle       *C.imgsearch_llama_handle
	imageMaxSide int
}

type Annotator = nativeGemmaRuntime

func (e *Embedder) AnnotateImage(ctx context.Context, imagePath string) (coreembedder.ImageAnnotation, error) {
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
		gemmaAppAnnotationUserPrompt,
		gemmaAppAnnotationJSONSchema,
		gemmaAppAnnotationMaxTokens,
	)
	if err != nil {
		return coreembedder.ImageAnnotation{}, err
	}

	return coreembedder.ImageAnnotation{Description: annotation.Description, Tags: annotation.Tags}, nil
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
		gemmaAppAnnotationUserPrompt,
		gemmaAppAnnotationJSONSchema,
		gemmaAppAnnotationMaxTokens,
	)
	if err != nil {
		return coreembedder.ImageAnnotation{}, err
	}

	return coreembedder.ImageAnnotation{Description: annotation.Description, Tags: annotation.Tags}, nil
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

func describeAndTagImageWithHandle(ctx context.Context, handle *C.imgsearch_llama_handle, imageMaxSide int, imagePath string, systemPrompt string, userPrompt string, jsonSchema string, maxTokens int) (gemmaAppAnnotation, string, error) {
	raw, err := generateImageJSONForHandle(ctx, handle, imageMaxSide, imagePath, systemPrompt, userPrompt, jsonSchema, maxTokens)
	if err != nil {
		return gemmaAppAnnotation{}, "", err
	}

	var result gemmaAppAnnotation
	if err := decodeGemmaJSONObject(raw, &result, "description+tags"); err != nil {
		retryRaw, retryErr := generateImageJSONForHandle(ctx, handle, imageMaxSide, imagePath, gemmaAppAnnotationRetrySystemPrompt, gemmaAppAnnotationRetryUserPrompt, jsonSchema, gemmaRetryAnnotationMaxTokens)
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
