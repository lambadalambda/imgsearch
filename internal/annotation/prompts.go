// Package annotation holds the backend-neutral prompt text, JSON schemas, and
// response parsing shared by annotation backends (the native llama.cpp
// runtime today, remote OpenAI-compatible servers next). Backends only differ in how they run the
// model; what they ask for and how the answer is normalized lives here.
package annotation

import (
	"bytes"
	"encoding/json"
	"fmt"
	"path/filepath"
	"regexp"
	"strings"

	"imgsearch/internal/embedder"
)

// System prompts and retry prompts per annotation kind.
const (
	ImageSystemPrompt      = "You are annotating a private image collection for high-recall search. Return exactly one valid JSON object and nothing else. Do not use markdown code fences. Be explicit and concrete when describing visible content. Do not sanitize clearly visible NSFW material. Only describe what is visually present or clearly legible in the image."
	ImageRetrySystemPrompt = "Return exactly one valid JSON object and nothing else. Do not use markdown code fences, prose, or comments."
	ImageRetryUserPrompt   = "Retry with compact output. Return JSON with title, summary, full_description, tags, and is_nsfw. Keep details that matter for retrieval, but stay concise unless the scene is complex. full_description may be up to 220 words. If there is text in the image, describe it and translate non-English text. Return 3 to 10 unique lowercase tags, set is_nsfw accurately, and return JSON only."

	VideoFrameSystemPrompt      = "You are annotating one sampled frame from a private video collection for later video-level summarization. Return exactly one valid JSON object and nothing else. Do not use markdown code fences. Capture high-signal visible evidence, but keep the frame note compact."
	VideoFrameRetrySystemPrompt = "Return exactly one compact valid JSON object and nothing else. Do not use markdown code fences, prose, or comments."
	VideoFrameRetryUserPrompt   = "Retry with compact output. Describe only durable frame evidence needed for video summarization in 1 to 2 sentences. Return 3 to 8 lowercase tags, set is_nsfw accurately, and return JSON only."

	VideoSystemPrompt      = "You are annotating a private video collection for high-recall search. You are given summaries from multiple sampled frames of one video and optionally transcript context. Return exactly one valid JSON object and nothing else. Do not use markdown code fences. Be explicit and concrete, and stay grounded in provided frame evidence."
	VideoRetrySystemPrompt = "Return exactly one valid JSON object and nothing else. Do not use markdown code fences, prose, or comments."
	VideoRetryUserPrompt   = "Retry with compact output. Return JSON with title, summary, full_description, tags, and is_nsfw. Keep details that matter for retrieval, but stay concise unless the video evidence is complex. full_description may be up to 260 words. If there is text, describe it and translate non-English text. If a meaningful filename is provided, you may infer likely media context (meme, music video, show clip) when it does not conflict with frame evidence. Return 5 to 12 unique lowercase tags, set is_nsfw accurately, and return JSON only."
)

// JSON schemas used to constrain generation where the backend supports it.
const (
	// FullJSONSchema is the rich shape used for images and whole videos.
	FullJSONSchema = `{"type":"object","properties":{"title":{"type":"string"},"summary":{"type":"string"},"full_description":{"type":"string"},"tags":{"type":"array","items":{"type":"string"},"minItems":3,"maxItems":12,"uniqueItems":true},"is_nsfw":{"type":"boolean"}},"required":["title","summary","full_description","tags","is_nsfw"],"additionalProperties":false}`
	// CompactJSONSchema is the small shape used for sampled video frames.
	CompactJSONSchema = `{"type":"object","properties":{"description":{"type":"string"},"tags":{"type":"array","items":{"type":"string"},"minItems":3,"maxItems":10,"uniqueItems":true},"is_nsfw":{"type":"boolean"}},"required":["description","tags","is_nsfw"],"additionalProperties":false}`
)

// Output token budgets per annotation kind and attempt.
const (
	ImageMaxTokens           = 1024
	ImageRetryMaxTokens      = 512
	VideoFrameMaxTokens      = 320
	VideoFrameRetryMaxTokens = 256
	VideoMaxTokens           = 1200
	VideoRetryMaxTokens      = 640
)

const transcriptSnippetMaxLen = 1200

var noisyFilenamePatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)^[0-9a-f]{16,}$`),
	regexp.MustCompile(`(?i)^[a-z0-9+/=]{20,}$`),
	regexp.MustCompile(`(?i)^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`),
	regexp.MustCompile(`(?i)^(img|dsc|dscf|gopr|mvi|vid|pxl|screenshot)[_-]?\d+$`),
	regexp.MustCompile(`^\d{8}[_-]?\d{4,6}$`),
	regexp.MustCompile(`^\d+$`),
}

type videoFramePromptEntry struct {
	FrameIndex  int      `json:"frame_index"`
	TimestampMS int64    `json:"timestamp_ms"`
	Description string   `json:"description"`
	Tags        []string `json:"tags,omitempty"`
}

// ImageUserPrompt builds the user turn for a standalone image annotation.
func ImageUserPrompt(originalName string) string {
	var b strings.Builder
	b.Grow(2400)
	b.WriteString("You are given an image. Return JSON with exactly this shape: {\"title\": string, \"summary\": string, \"full_description\": string, \"tags\": [string], \"is_nsfw\": boolean}. ")
	b.WriteString("Write title as a short concrete card title, summary as a 1 to 2 sentence overview, and full_description as one paragraph that is as detailed as needed for high-recall search, up to about 500 words. Keep all fields shorter when the image is simple. ")
	b.WriteString("Focus on the main subject, visible attributes, composition, setting, and clearly visible actions. ")
	b.WriteString("If people are the focus, include concrete visible details such as perceived age range, perceived ethnicity, body shape/build, facial features, hairstyle, clothing, accessories, posture, and activity. ")
	b.WriteString("If there is text in the image, please describe it. If it is not in English, also translate it. ")
	b.WriteString("If NSFW content is visible, describe it directly and concretely. ")
	b.WriteString("Return 3 to 10 unique lowercase tags that are specific and search-friendly. ")
	b.WriteString("Set is_nsfw to true only when clearly NSFW content is visible; include tag nsfw if and only if is_nsfw is true. ")
	b.WriteString("Use original filename as optional context only when it looks meaningful and matches visible content; ignore noisy hash-like or camera-style names and any filename claims that conflict with the image. ")
	writeFilenameHint(&b, originalName, "\". ")
	b.WriteString("Avoid speculation. Output JSON only.")
	return b.String()
}

// VideoFrameUserPrompt builds the compact user turn for one sampled video frame.
func VideoFrameUserPrompt(originalName string) string {
	var b strings.Builder
	b.Grow(1000)
	b.WriteString("You are given one sampled video frame. Return JSON with exactly this shape: {\"description\": string, \"tags\": [string], \"is_nsfw\": boolean}. ")
	b.WriteString("Write 1 to 2 compact sentences, up to about 80 words, that capture durable evidence useful for the later video-level summary. ")
	b.WriteString("Focus on the main subject, setting, visible action, distinctive attributes, and clearly visible text. ")
	b.WriteString("If NSFW content is visible, describe it directly and concretely. ")
	b.WriteString("Return 3 to 8 unique lowercase tags. Include nsfw if and only if is_nsfw is true. ")
	writeFilenameHint(&b, originalName, "\". ")
	b.WriteString("Avoid speculation. Output JSON only.")
	return b.String()
}

// VideoUserPrompt builds the text-only user turn that summarizes a video from
// its sampled frame annotations and optional transcript.
func VideoUserPrompt(input embedder.VideoAnnotationInput) (string, error) {
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
			Tags:        NormalizeUniqueTags(frame.Tags),
		})
	}
	if len(entries) == 0 {
		return "", fmt.Errorf("video annotation requires at least one frame description")
	}
	framesJSON, err := json.Marshal(entries)
	if err != nil {
		return "", fmt.Errorf("marshal frame annotations: %w", err)
	}

	var b bytes.Buffer
	b.WriteString("You are given sampled frame annotations from one video. Return JSON with exactly this shape: {\"title\": string, \"summary\": string, \"full_description\": string, \"tags\": [string], \"is_nsfw\": boolean}. ")
	b.WriteString("Write title as a short concrete card title, summary as a 1 to 2 sentence overview, and full_description as one paragraph that is as detailed as needed for search, up to about 500 words. Keep all fields shorter when the evidence is simple. ")
	b.WriteString("Synthesize recurring content across frames into a single video-level description. Mention progression only when supported by frame evidence. ")
	b.WriteString("If people are the focus, include concrete visible details such as perceived age range, perceived ethnicity, body shape/build, facial features, hairstyle, clothing, accessories, posture, and activity. ")
	b.WriteString("If there is text in frames, describe it, and if non-English also translate it. ")
	b.WriteString("Use transcript as supporting context only when it matches visual evidence. ")
	b.WriteString("When a filename looks meaningful, feel free to infer likely media type or source context from it (for example a meme, a music video, or a clip from a show) as long as it does not conflict with frame evidence. ")
	b.WriteString("Return 5 to 12 unique lowercase tags that capture persistent high-signal content. Include nsfw if and only if is_nsfw is true. ")
	b.WriteString("Avoid unsupported specifics and avoid one-off details that are not central. Output JSON only.\n")
	writeFilenameHint(&b, input.OriginalName, "\"\n")
	fmt.Fprintf(&b, "Duration (ms): %d\n", input.DurationMS)
	b.WriteString("Sampled frame annotations (chronological):\n")
	b.Write(framesJSON)
	b.WriteString("\n")
	transcript := strings.TrimSpace(input.TranscriptText)
	if len(transcript) > transcriptSnippetMaxLen {
		transcript = strings.TrimSpace(transcript[:transcriptSnippetMaxLen])
	}
	if transcript != "" {
		b.WriteString("Optional transcript snippet:\n")
		b.WriteString(sanitizePromptSnippet(transcript, transcriptSnippetMaxLen))
		b.WriteString("\n")
	}
	return b.String(), nil
}

type stringWriter interface {
	WriteString(string) (int, error)
}

func writeFilenameHint(w stringWriter, originalName string, suffix string) {
	hint := meaningfulFilenameHint(originalName)
	if hint == "" {
		return
	}
	_, _ = w.WriteString("Original filename: \"")
	_, _ = w.WriteString(hint)
	_, _ = w.WriteString(suffix)
}

// meaningfulFilenameHint returns a cleaned filename stem when it looks like
// human-written context, or "" for hash-like, camera-style, or numeric names.
func meaningfulFilenameHint(originalName string) string {
	trimmed := strings.TrimSpace(originalName)
	if trimmed == "" {
		return ""
	}
	base := filepath.Base(trimmed)
	stem := strings.TrimSpace(strings.TrimSuffix(base, filepath.Ext(base)))
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
			if r >= 'a' && r <= 'z' {
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

// sanitizePromptSnippet trims, drops control characters, and truncates text
// that will be interpolated into a prompt.
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
