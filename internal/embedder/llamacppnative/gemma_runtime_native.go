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
	"unsafe"

	coreembedder "imgsearch/internal/embedder"
)

const gemmaDescriptionJSONSchema = `{"type":"object","properties":{"short_description":{"type":"string"},"labels":{"type":"array","items":{"type":"string"},"minItems":1,"maxItems":8,"uniqueItems":true}},"required":["short_description","labels"],"additionalProperties":false}`
const gemmaTagsJSONSchema = `{"type":"object","properties":{"tags":{"type":"array","items":{"type":"string"},"minItems":3,"maxItems":10,"uniqueItems":true}},"required":["tags"],"additionalProperties":false}`
const gemmaDescriptionAndTagsJSONSchema = `{"type":"object","properties":{"description":{"type":"string"},"tags":{"type":"array","items":{"type":"string"},"minItems":3,"maxItems":10,"uniqueItems":true}},"required":["description","tags"],"additionalProperties":false}`
const gemmaAppAnnotationJSONSchema = `{"type":"object","properties":{"description":{"type":"string"},"tags":{"type":"array","items":{"type":"string"},"minItems":3,"maxItems":10,"uniqueItems":true},"is_nsfw":{"type":"boolean"}},"required":["description","tags","is_nsfw"],"additionalProperties":false}`

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
	handle       *C.imgsearch_llama_handle
	imageMaxSide int
}

type Annotator = nativeGemmaRuntime

func (e *Embedder) AnnotateImage(_ context.Context, imagePath string) (coreembedder.ImageAnnotation, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.handle == nil {
		return coreembedder.ImageAnnotation{}, fmt.Errorf("llama-cpp-native embedder is closed")
	}

	annotation, _, err := describeAndTagImageWithHandle(
		e.handle,
		e.imageMaxSide,
		imagePath,
		"You are a precise vision model. Return only valid JSON with a useful medium-length description and concise tags for visible image content.",
		"Describe this image in one medium-length paragraph of 2 to 4 sentences. Focus on the main subject, visible attributes, composition, and setting. Then return 3 to 10 unique lowercase tags that are most relevant for search. If the image contains explicit nudity, sexual content, fetish content, graphic gore, or other clearly not-safe-for-work material, set is_nsfw to true and include the tag nsfw. Otherwise set is_nsfw to false and do not include the tag nsfw. Avoid speculation, avoid repetition, and keep the description grounded in what is visible.",
		gemmaAppAnnotationJSONSchema,
		220,
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
	if r == nil || r.handle == nil {
		return nil
	}
	C.imgsearch_llama_free(r.handle)
	r.handle = nil
	return nil
}

func (r *nativeGemmaRuntime) AnnotateImage(_ context.Context, imagePath string) (coreembedder.ImageAnnotation, error) {
	if r == nil || r.handle == nil {
		return coreembedder.ImageAnnotation{}, fmt.Errorf("native Gemma runtime is closed")
	}

	annotation, _, err := describeAndTagImageWithHandle(
		r.handle,
		r.imageMaxSide,
		imagePath,
		"You are a precise vision model. Return only valid JSON with a useful medium-length description and concise tags for visible image content.",
		"Describe this image in one medium-length paragraph of 2 to 4 sentences. Focus on the main subject, visible attributes, composition, and setting. Then return 3 to 10 unique lowercase tags that are most relevant for search. If the image contains explicit nudity, sexual content, fetish content, graphic gore, or other clearly not-safe-for-work material, set is_nsfw to true and include the tag nsfw. Otherwise set is_nsfw to false and do not include the tag nsfw. Avoid speculation, avoid repetition, and keep the description grounded in what is visible.",
		gemmaAppAnnotationJSONSchema,
		220,
	)
	if err != nil {
		return coreembedder.ImageAnnotation{}, err
	}

	return coreembedder.ImageAnnotation{Description: annotation.Description, Tags: annotation.Tags}, nil
}

func (r *nativeGemmaRuntime) DescribeImage(_ context.Context, imagePath string) (gemmaImageDescription, string, error) {
	if r == nil || r.handle == nil {
		return gemmaImageDescription{}, "", fmt.Errorf("native Gemma runtime is closed")
	}

	systemPrompt := "You are a precise vision model. Return only valid JSON that describes visible image content."
	userPrompt := "Describe this image in concrete visual terms. Return JSON with short_description as one sentence and labels as 3-8 lowercase tags. Mention the main subject and setting when obvious. Avoid speculation."
	raw, err := r.generateImageJSON(context.Background(), imagePath, systemPrompt, userPrompt, gemmaDescriptionJSONSchema, 160)
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

func (r *nativeGemmaRuntime) AutoTagImage(_ context.Context, imagePath string) (gemmaImageTags, string, error) {
	if r == nil || r.handle == nil {
		return gemmaImageTags{}, "", fmt.Errorf("native Gemma runtime is closed")
	}

	systemPrompt := "You are a precise image autotagger. Return only valid JSON with concise tags for visible content."
	userPrompt := "Generate the most relevant autotags for this image. Return JSON with tags as an array of 3 to 10 unique lowercase tags. Prioritize the main subject, obvious attributes, and the setting. Use short search-friendly tags only. Avoid speculation, full sentences, and duplicate or overly generic tags unless they are clearly useful."
	raw, err := r.generateImageJSON(context.Background(), imagePath, systemPrompt, userPrompt, gemmaTagsJSONSchema, 120)
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

func (r *nativeGemmaRuntime) DescribeAndTagImage(_ context.Context, imagePath string) (gemmaImageDescriptionAndTags, string, error) {
	if r == nil || r.handle == nil {
		return gemmaImageDescriptionAndTags{}, "", fmt.Errorf("native Gemma runtime is closed")
	}

	systemPrompt := "You are a precise vision model. Return only valid JSON with a useful medium-length description and concise tags for visible image content."
	userPrompt := "Describe this image in one medium-length paragraph of 2 to 4 sentences. Focus on the main subject, visible attributes, composition, and setting. Then return 3 to 10 unique lowercase tags that are most relevant for search. Avoid speculation, avoid repetition, and keep the description grounded in what is visible."
	raw, err := r.generateImageJSON(context.Background(), imagePath, systemPrompt, userPrompt, gemmaDescriptionAndTagsJSONSchema, 220)
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

func describeAndTagImageWithHandle(handle *C.imgsearch_llama_handle, imageMaxSide int, imagePath string, systemPrompt string, userPrompt string, jsonSchema string, maxTokens int) (gemmaAppAnnotation, string, error) {
	raw, err := generateImageJSONForHandle(handle, imageMaxSide, imagePath, systemPrompt, userPrompt, jsonSchema, maxTokens)
	if err != nil {
		return gemmaAppAnnotation{}, "", err
	}

	var result gemmaAppAnnotation
	if err := decodeGemmaJSONObject(raw, &result, "description+tags"); err != nil {
		return gemmaAppAnnotation{}, raw, err
	}

	result.Description = strings.TrimSpace(result.Description)
	result.Tags = normalizeUniqueTags(result.Tags)
	result.Tags = applyNSFWTag(result.Tags, result.IsNSFW)

	return result, raw, nil
}

func (r *nativeGemmaRuntime) generateImageJSON(_ context.Context, imagePath string, systemPrompt string, userPrompt string, jsonSchema string, maxTokens int) (string, error) {
	if r == nil || r.handle == nil {
		return "", fmt.Errorf("native Gemma runtime is closed")
	}
	return generateImageJSONForHandle(r.handle, r.imageMaxSide, imagePath, systemPrompt, userPrompt, jsonSchema, maxTokens)
}

func generateImageJSONForHandle(handle *C.imgsearch_llama_handle, imageMaxSide int, imagePath string, systemPrompt string, userPrompt string, jsonSchema string, maxTokens int) (string, error) {
	if handle == nil {
		return "", fmt.Errorf("native Gemma runtime is closed")
	}

	preprocessedPath, cleanup, err := preprocessImageForEmbeddingWithVipsgen(imagePath, imageMaxSide)
	if err != nil {
		return "", err
	}
	defer cleanup()

	cImagePath := C.CString(preprocessedPath)
	defer C.free(unsafe.Pointer(cImagePath))
	cSystemPrompt := C.CString(systemPrompt)
	defer C.free(unsafe.Pointer(cSystemPrompt))
	cUserPrompt := C.CString(userPrompt)
	defer C.free(unsafe.Pointer(cUserPrompt))
	cSchema := C.CString(jsonSchema)
	defer C.free(unsafe.Pointer(cSchema))

	buf := make([]byte, 16*1024)
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
	if err := json.Unmarshal([]byte(raw), out); err != nil {
		jsonOnly, ok := extractJSONObject(raw)
		if !ok {
			return fmt.Errorf("decode %s JSON: %w; raw=%q", label, err, raw)
		}
		if err := json.Unmarshal([]byte(jsonOnly), out); err != nil {
			return fmt.Errorf("decode extracted %s JSON: %w; raw=%q", label, err, raw)
		}
	}
	return nil
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
