package annotation

import (
	"encoding/json"
	"fmt"
	"strings"

	"imgsearch/internal/annotationtext"
	"imgsearch/internal/embedder"
)

const maxTagsWithNSFW = 10

// Response is the JSON object every annotation prompt asks the model for. The
// compact video-frame shape only fills Description, Tags, and IsNSFW.
type Response struct {
	Title           string   `json:"title"`
	Summary         string   `json:"summary"`
	FullDescription string   `json:"full_description"`
	Description     string   `json:"description"`
	Tags            []string `json:"tags"`
	IsNSFW          bool     `json:"is_nsfw"`
}

// Parse decodes raw model output into a normalized Response. It tolerates
// markdown code fences and prose around the JSON object.
func Parse(raw string) (Response, error) {
	var r Response
	if err := DecodeJSONObject(raw, &r); err != nil {
		return Response{}, err
	}
	return Normalize(r), nil
}

// DecodeJSONObject unmarshals raw model output into out, first as-is (after
// stripping code fences), then by extracting the first balanced JSON object.
func DecodeJSONObject(raw string, out any) error {
	normalizedRaw := stripMarkdownCodeFences(raw)
	if err := json.Unmarshal([]byte(normalizedRaw), out); err != nil {
		jsonOnly, ok := ExtractJSONObject(normalizedRaw)
		if !ok {
			return fmt.Errorf("decode annotation JSON: %w; raw=%q", err, raw)
		}
		if err := json.Unmarshal([]byte(jsonOnly), out); err != nil {
			return fmt.Errorf("decode extracted annotation JSON: %w; raw=%q", err, raw)
		}
	}
	return nil
}

// Normalize trims text, derives missing title/summary from the description,
// lowercases and de-duplicates tags, and makes the nsfw tag agree with IsNSFW.
func Normalize(r Response) Response {
	r.Title = strings.TrimSpace(r.Title)
	r.Summary = strings.TrimSpace(r.Summary)
	r.Description = strings.TrimSpace(r.Description)
	r.FullDescription = strings.TrimSpace(r.FullDescription)
	if r.FullDescription == "" {
		r.FullDescription = r.Description
	}
	text := annotationtext.Build(r.Title, r.Summary, r.FullDescription)
	r.Title = text.Title
	r.Summary = text.Summary
	r.FullDescription = text.FullDescription
	r.Description = text.FullDescription
	r.Tags = applyNSFWTag(NormalizeUniqueTags(r.Tags), r.IsNSFW)
	return r
}

// ImageAnnotation converts a normalized Response to the core image shape.
func (r Response) ImageAnnotation() embedder.ImageAnnotation {
	return embedder.ImageAnnotation{
		Title:       r.Title,
		Summary:     r.Summary,
		Description: r.Description,
		Tags:        r.Tags,
	}
}

// VideoAnnotation converts a normalized Response to the core video shape.
func (r Response) VideoAnnotation() embedder.VideoAnnotation {
	return embedder.VideoAnnotation{
		Title:       r.Title,
		Summary:     r.Summary,
		Description: r.Description,
		Tags:        r.Tags,
		IsNSFW:      r.IsNSFW,
	}
}

// stripMarkdownCodeFences removes a surrounding ``` fence pair if present.
func stripMarkdownCodeFences(raw string) string {
	trimmed := strings.TrimSpace(raw)
	if !strings.HasPrefix(trimmed, "```") {
		return trimmed
	}
	lines := strings.Split(trimmed, "\n")
	if strings.HasPrefix(strings.TrimSpace(lines[0]), "```") {
		lines = lines[1:]
	}
	if len(lines) > 0 && strings.TrimSpace(lines[len(lines)-1]) == "```" {
		lines = lines[:len(lines)-1]
	}
	return strings.TrimSpace(strings.Join(lines, "\n"))
}

// ExtractJSONObject returns the first balanced {...} object in raw, honouring
// braces inside JSON strings.
func ExtractJSONObject(raw string) (string, bool) {
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
			switch {
			case escaped:
				escaped = false
			case ch == '\\':
				escaped = true
			case ch == '"':
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

// NormalizeUniqueTags lowercases, trims, drops blanks, and de-duplicates
// while preserving first-seen order.
func NormalizeUniqueTags(tags []string) []string {
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

func applyNSFWTag(normalized []string, isNSFW bool) []string {
	filtered := normalized[:0]
	for _, tag := range normalized {
		if tag != "nsfw" {
			filtered = append(filtered, tag)
		}
	}
	if isNSFW {
		if len(filtered) >= maxTagsWithNSFW {
			filtered = filtered[:maxTagsWithNSFW-1]
		}
		filtered = append(filtered, "nsfw")
	}
	return filtered
}
