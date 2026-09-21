package tagutil

import (
	"strings"
	"testing"
)

func TestDecodeJSON(t *testing.T) {
	tags, err := DecodeJSON("")
	if err != nil {
		t.Fatalf("decode empty tags: %v", err)
	}
	if tags != nil {
		t.Fatalf("expected nil tags for empty input, got %#v", tags)
	}

	tags, err = DecodeJSON(`["alpha","beta"]`)
	if err != nil {
		t.Fatalf("decode valid tags: %v", err)
	}
	if len(tags) != 2 || tags[0] != "alpha" || tags[1] != "beta" {
		t.Fatalf("unexpected decoded tags: %#v", tags)
	}

	tags, err = DecodeJSON(`null`)
	if err != nil {
		t.Fatalf("decode null tags: %v", err)
	}
	if tags != nil {
		t.Fatalf("expected nil tags for null input, got %#v", tags)
	}

	_, err = DecodeJSON(`{"tag":"oops"}`)
	if err == nil {
		t.Fatal("expected error for malformed tags payload")
	}
}

func TestToggleTagAddsMissingTag(t *testing.T) {
	tags, isPresent := ToggleTag([]string{"portrait", "subject"}, "nsfw")
	if !isPresent {
		t.Fatal("expected nsfw to be present after toggle")
	}
	if len(tags) != 3 || tags[2] != "nsfw" {
		t.Fatalf("unexpected toggled tags: %#v", tags)
	}
}

func TestToggleTagRemovesExistingTagCaseInsensitive(t *testing.T) {
	tags, isPresent := ToggleTag([]string{"portrait", "NSFW", "sample"}, "nsfw")
	if isPresent {
		t.Fatal("expected nsfw to be removed after toggle")
	}
	if len(tags) != 2 || tags[0] != "portrait" || tags[1] != "sample" {
		t.Fatalf("unexpected toggled tags: %#v", tags)
	}
}

func TestToggleTagJSONEncodesUpdatedTags(t *testing.T) {
	encoded, isPresent, err := ToggleTagJSON(`["portrait"]`, "nsfw")
	if err != nil {
		t.Fatalf("toggle tag json: %v", err)
	}
	if !isPresent {
		t.Fatal("expected nsfw to be present after toggle")
	}
	if encoded != `["portrait","nsfw"]` {
		t.Fatalf("encoded tags: got=%s", encoded)
	}
}

func TestNormalizeTrimsAndDedupesCaseInsensitively(t *testing.T) {
	got := Normalize([]string{" Cat ", "", "cat", "dog", "DOG ", "bird"})
	want := []string{"Cat", "dog", "bird"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("normalize: got=%v want=%v", got, want)
	}
	if EncodeJSON(nil) != "[]" || EncodeJSON([]string{"a"}) != `["a"]` {
		t.Fatalf("encode: %q %q", EncodeJSON(nil), EncodeJSON([]string{"a"}))
	}
}

func TestMergeAndDiffRoundTrip(t *testing.T) {
	annotator := []string{"cat", "indoor", "blurry"}
	served := []string{"cat", "indoor", "my-trip", "Holiday"}
	user, removed := Diff(annotator, served)
	if strings.Join(user, ",") != "my-trip,Holiday" || strings.Join(removed, ",") != "blurry" {
		t.Fatalf("diff: user=%v removed=%v", user, removed)
	}
	// A fresh annotation keeps the user's additions and removals.
	merged := Merge([]string{"cat", "blurry", "outdoor"}, user, removed)
	if strings.Join(merged, ",") != "cat,outdoor,my-trip,Holiday" {
		t.Fatalf("merge: %v", merged)
	}
	// No edits: merge is the normalized annotator list.
	if got := Merge([]string{"a", "A", " b "}, nil, nil); strings.Join(got, ",") != "a,b" {
		t.Fatalf("merge without edits: %v", got)
	}
}
