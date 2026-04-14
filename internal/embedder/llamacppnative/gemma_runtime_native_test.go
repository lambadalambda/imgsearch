//go:build cgo

package llamacppnative

import "testing"

func TestDecodeGemmaJSONObjectStripsMarkdownCodeFences(t *testing.T) {
	raw := "```json\n{\n  \"description\": \"example\",\n  \"tags\": [\"sample\", \"nsfw\"],\n  \"is_nsfw\": true\n}\n```"

	var got gemmaAppAnnotation
	if err := decodeGemmaJSONObject(raw, &got, "description+tags"); err != nil {
		t.Fatalf("decode fenced JSON: %v", err)
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

func TestStripMarkdownCodeFencesLeavesPlainJSONUntouched(t *testing.T) {
	raw := "{\"description\":\"plain\"}"
	if got := stripMarkdownCodeFences(raw); got != raw {
		t.Fatalf("unexpected stripped value: %q", got)
	}
}
