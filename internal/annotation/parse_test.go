package annotation

import (
	"strings"
	"testing"
)

func TestParseStripsMarkdownCodeFences(t *testing.T) {
	raw := "```json\n{\n  \"description\": \"example\",\n  \"tags\": [\"sample\", \"nsfw\"],\n  \"is_nsfw\": true\n}\n```"

	got, err := Parse(raw)
	if err != nil {
		t.Fatalf("parse fenced JSON: %v", err)
	}
	if got.Description != "example" {
		t.Fatalf("unexpected description: %q", got.Description)
	}
	if len(got.Tags) != 2 || got.Tags[0] != "sample" || got.Tags[1] != "nsfw" {
		t.Fatalf("unexpected tags: %v", got.Tags)
	}
	if !got.IsNSFW {
		t.Fatal("expected is_nsfw to be true")
	}
}

func TestParseNormalizesMultiLevelText(t *testing.T) {
	raw := `{
  "title": "  Screenshot post  ",
  "summary": " A post about orbital simulation. ",
  "full_description": " A screenshot shows a social post discussing drawing circles and making a full n-body simulation. ",
  "tags": ["Screenshot", "post", "nsfw", "post"],
  "is_nsfw": false
}`

	got, err := Parse(raw)
	if err != nil {
		t.Fatalf("parse multi-level JSON: %v", err)
	}
	if got.Title != "Screenshot post" {
		t.Fatalf("title: got=%q", got.Title)
	}
	if got.Summary != "A post about orbital simulation." {
		t.Fatalf("summary: got=%q", got.Summary)
	}
	if got.Description != "A screenshot shows a social post discussing drawing circles and making a full n-body simulation." {
		t.Fatalf("description should mirror full_description: got=%q", got.Description)
	}
	if got.FullDescription != got.Description {
		t.Fatalf("full_description=%q description=%q", got.FullDescription, got.Description)
	}
	if len(got.Tags) != 2 || got.Tags[0] != "screenshot" || got.Tags[1] != "post" {
		t.Fatalf("expected normalized non-nsfw tags, got %v", got.Tags)
	}
}

func TestParseExtractsJSONFromProse(t *testing.T) {
	raw := "Sure! Here is the annotation:\n{\"title\": \"Cat\", \"summary\": \"A cat.\", \"full_description\": \"A cat on a {sofa}.\", \"tags\": [\"cat\"], \"is_nsfw\": false}\nHope that helps."

	got, err := Parse(raw)
	if err != nil {
		t.Fatalf("parse prose-wrapped JSON: %v", err)
	}
	if got.Title != "Cat" || got.FullDescription != "A cat on a {sofa}." {
		t.Fatalf("unexpected result: %+v", got)
	}
}

func TestParseRejectsInvalidOutput(t *testing.T) {
	for _, raw := range []string{"", "no json here", "{\"title\": ", "```json\n```"} {
		if _, err := Parse(raw); err == nil {
			t.Fatalf("expected error for %q", raw)
		}
	}
}

func TestNormalizeFallsBackFromLegacyDescription(t *testing.T) {
	got := Normalize(Response{
		Description: "A legacy detailed description. More details follow.",
		Tags:        []string{"legacy"},
	})

	if got.Title != "A legacy detailed description." {
		t.Fatalf("fallback title: got=%q", got.Title)
	}
	if got.Summary != "A legacy detailed description. More details follow." {
		t.Fatalf("fallback summary: got=%q", got.Summary)
	}
	if got.FullDescription != "A legacy detailed description. More details follow." {
		t.Fatalf("fallback full description: got=%q", got.FullDescription)
	}
}

func TestNormalizeAppliesNSFWTagAndCapsAtTen(t *testing.T) {
	tags := make([]string, 0, 12)
	for i := 0; i < 12; i++ {
		tags = append(tags, "tag"+strings.Repeat("x", i))
	}
	got := Normalize(Response{Description: "d", Tags: tags, IsNSFW: true})
	if len(got.Tags) != 10 {
		t.Fatalf("expected 10 tags, got %d: %v", len(got.Tags), got.Tags)
	}
	if got.Tags[9] != "nsfw" {
		t.Fatalf("expected nsfw as final tag, got %v", got.Tags)
	}
	notNSFW := Normalize(Response{Description: "d", Tags: []string{"a", "NSFW"}, IsNSFW: false})
	if len(notNSFW.Tags) != 1 || notNSFW.Tags[0] != "a" {
		t.Fatalf("expected nsfw tag stripped when is_nsfw is false, got %v", notNSFW.Tags)
	}
}

func TestResponseConvertsToCoreAnnotations(t *testing.T) {
	r := Normalize(Response{Title: "T", Summary: "S", FullDescription: "F", Tags: []string{"x"}, IsNSFW: true})
	img := r.ImageAnnotation()
	if img.Title != "T" || img.Summary != "S" || img.Description != "F" || len(img.Tags) != 2 {
		t.Fatalf("unexpected image annotation: %+v", img)
	}
	vid := r.VideoAnnotation()
	if vid.Title != "T" || !vid.IsNSFW || len(vid.Tags) != 2 {
		t.Fatalf("unexpected video annotation: %+v", vid)
	}
}

func TestStripMarkdownCodeFencesLeavesPlainJSONUntouched(t *testing.T) {
	raw := "{\"description\":\"plain\"}"
	if got := stripMarkdownCodeFences(raw); got != raw {
		t.Fatalf("unexpected stripped value: %q", got)
	}
}

func TestExtractJSONObjectHandlesBracesInsideStrings(t *testing.T) {
	raw := `prefix {"a": "x } y", "b": {"c": 1}} suffix`
	got, ok := ExtractJSONObject(raw)
	if !ok || got != `{"a": "x } y", "b": {"c": 1}}` {
		t.Fatalf("unexpected extraction: ok=%v got=%q", ok, got)
	}
	if _, ok := ExtractJSONObject("no braces"); ok {
		t.Fatal("expected no object")
	}
}
