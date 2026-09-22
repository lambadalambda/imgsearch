package annotation

import (
	"strings"
	"testing"

	"imgsearch/internal/embedder"
)

func TestImageUserPromptIncludesMeaningfulFilename(t *testing.T) {
	prompt := ImageUserPrompt("the cover of the album kid a by radiohead.jpg", nil)
	if !strings.Contains(prompt, "Original filename") {
		t.Fatalf("expected prompt to include original filename guidance")
	}
	if !strings.Contains(strings.ToLower(prompt), "kid a") {
		t.Fatalf("expected prompt to include meaningful filename content")
	}
	if !strings.Contains(strings.ToLower(prompt), "500 words") {
		t.Fatalf("expected prompt to allow up to 500 words")
	}
	if !strings.Contains(strings.ToLower(prompt), "if there is text in the image") {
		t.Fatalf("expected prompt to explicitly request text description and translation")
	}
}

func TestImageUserPromptSkipsNoisyFilename(t *testing.T) {
	for _, name := range []string{"34254745943.jpg", "IMG_1234.jpg", "3f2a9c1e8b7d6a5f4e3d2c1b.png", "20240101_120000.jpg"} {
		if prompt := ImageUserPrompt(name, nil); strings.Contains(prompt, "Original filename") {
			t.Fatalf("expected noisy filename %q to be omitted from prompt", name)
		}
	}
}

func TestVideoFrameUserPromptIsCompact(t *testing.T) {
	prompt := VideoFrameUserPrompt("concert stage clip.mp4", nil)
	if !strings.Contains(prompt, "80 words") {
		t.Fatalf("expected compact frame prompt word budget")
	}
	if strings.Contains(prompt, "500 words") {
		t.Fatalf("video frame prompt should not use full rich image budget")
	}
	if !strings.Contains(prompt, "Original filename") {
		t.Fatalf("expected meaningful filename context")
	}
}

func TestVideoUserPromptIncludesFrameEvidence(t *testing.T) {
	prompt, err := VideoUserPrompt(embedder.VideoAnnotationInput{
		OriginalName:            "party_clip.mp4",
		DurationMS:              12000,
		RepresentativeFramePath: "images/frame-01.jpg",
		TranscriptText:          "hello from the stage",
		Frames: []embedder.VideoFrameAnnotation{
			{FrameIndex: 0, TimestampMS: 500, Description: "a singer on stage", Tags: []string{"concert", "singer"}},
			{FrameIndex: 1, TimestampMS: 1500, Description: "crowd waving", Tags: []string{"crowd", "concert"}},
		},
	})
	if err != nil {
		t.Fatalf("VideoUserPrompt error: %v", err)
	}
	if !strings.Contains(prompt, "Sampled frame annotations") {
		t.Fatalf("expected prompt to include sampled frame annotations section")
	}
	if !strings.Contains(prompt, "hello from the stage") {
		t.Fatalf("expected prompt to include transcript context")
	}
	lower := strings.ToLower(prompt)
	if !strings.Contains(lower, "meme") || !strings.Contains(lower, "music video") || !strings.Contains(lower, "clip from a show") {
		t.Fatalf("expected prompt to include filename-based media type inference guidance")
	}
	if !strings.Contains(lower, "500 words") {
		t.Fatalf("expected video prompt to allow up to 500 words")
	}
}

func TestVideoUserPromptRequiresFrameDescriptions(t *testing.T) {
	if _, err := VideoUserPrompt(embedder.VideoAnnotationInput{}); err == nil {
		t.Fatal("expected error without frames")
	}
	if _, err := VideoUserPrompt(embedder.VideoAnnotationInput{Frames: []embedder.VideoFrameAnnotation{{Description: "  "}}}); err == nil {
		t.Fatal("expected error when all frame descriptions are blank")
	}
}

func TestSanitizePromptSnippetDropsControlCharsAndTruncates(t *testing.T) {
	got := sanitizePromptSnippet("  ab\x01c\ndef  ", 4)
	if got != "abcd" {
		t.Fatalf("unexpected snippet: %q", got)
	}
}

func TestVideoUserPromptTruncatesTranscriptBeforeSanitizing(t *testing.T) {
	long := strings.Repeat("ab\n", 500) // 1500 bytes, 500 newlines
	prompt, err := VideoUserPrompt(embedder.VideoAnnotationInput{
		TranscriptText: long,
		Frames:         []embedder.VideoFrameAnnotation{{Description: "x"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	marker := "Optional transcript snippet:\n"
	i := strings.Index(prompt, marker)
	if i < 0 {
		t.Fatal("expected transcript section")
	}
	snippet := strings.TrimSuffix(prompt[i+len(marker):], "\n")
	// 1200 bytes of "ab\n" hold 400 newlines; sanitizing after the cut leaves 800 bytes.
	if len(snippet) != 800 {
		t.Fatalf("expected 800-byte sanitized snippet, got %d", len(snippet))
	}
}

func TestPromptsCarryTheKnownTagVocabulary(t *testing.T) {
	known := []string{" Cat ", "indoor", "nsfw", "bad,comma", "", "warm-tone"}
	for name, prompt := range map[string]string{
		"image": ImageUserPrompt("x.jpg", known),
		"frame": VideoFrameUserPrompt("x.jpg", known),
	} {
		if !strings.Contains(prompt, "reuse them whenever they fit") || !strings.Contains(prompt, "cat, indoor, warm-tone. ") {
			t.Fatalf("%s prompt lacks the vocabulary hint: %s", name, prompt)
		}
		if strings.Contains(prompt, "bad,comma") || strings.Contains(prompt, "nsfw, ") {
			t.Fatalf("%s prompt should drop unusable tags: %s", name, prompt)
		}
	}
	if strings.Contains(ImageUserPrompt("x.jpg", nil), "library already uses") {
		t.Fatal("no hint expected without known tags")
	}
	many := make([]string, MaxKnownTagsInPrompt+20)
	for i := range many {
		many[i] = "t" + strings.Repeat("x", i%5) + string(rune('a'+i%26)) + strings.Repeat("y", i/26)
	}
	prompt := ImageUserPrompt("x.jpg", many)
	start := strings.Index(prompt, "do not cover: ")
	if start < 0 {
		t.Fatalf("hint missing: %s", prompt)
	}
	hint := prompt[start+len("do not cover: "):]
	hint = hint[:strings.Index(hint, ". ")]
	if got := len(strings.Split(hint, ", ")); got != MaxKnownTagsInPrompt {
		t.Fatalf("vocabulary hint not capped: %d tags", got)
	}
	video, err := VideoUserPrompt(embedder.VideoAnnotationInput{
		OriginalName: "clip.mp4",
		DurationMS:   1000,
		Frames:       []embedder.VideoFrameAnnotation{{Description: "A stage."}},
		KnownTags:    []string{"concert", "stage"},
	})
	if err != nil || !strings.Contains(video, "concert, stage. ") {
		t.Fatalf("video prompt: %v %s", err, video)
	}
}
