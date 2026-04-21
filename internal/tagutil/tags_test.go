package tagutil

import "testing"

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
