package llamacppnative

import "testing"

func TestParsePromptFormat(t *testing.T) {
	cases := map[string]PromptFormat{
		"":        PromptFormatQwen3VL,
		"qwen3vl": PromptFormatQwen3VL,
		" WeMM ":  PromptFormatWeMM,
		"wemm":    PromptFormatWeMM,
	}
	for input, want := range cases {
		got, err := ParsePromptFormat(input)
		if err != nil || got != want {
			t.Fatalf("ParsePromptFormat(%q) = %q, %v; want %q", input, got, err, want)
		}
	}
	if _, err := ParsePromptFormat("llama"); err == nil {
		t.Fatal("expected error for unknown format")
	}
}

func TestQwen3VLPromptsMatchLegacyBridgeFormat(t *testing.T) {
	got := textPrompt(PromptFormatQwen3VL, " Find images. ", " a dog ")
	want := "<|im_start|>system\nFind images.<|im_end|>\n<|im_start|>user\na dog<|im_end|>\n<|im_start|>assistant\n"
	if got != want {
		t.Fatalf("text prompt = %q, want %q", got, want)
	}

	got = imagePrompt(PromptFormatQwen3VL, "", "<__media__>")
	want = "<|im_start|>system\nRepresent the user's input.<|im_end|>\n<|im_start|>user\n<__media__><|im_end|>\n<|im_start|>assistant\n"
	if got != want {
		t.Fatalf("image prompt = %q, want %q", got, want)
	}
}

func TestWeMMPromptsMatchReferenceTemplate(t *testing.T) {
	// Token sequence produced by the model card's transformers processor: no
	// system turn, pooled at the trailing <embedding> token.
	got := textPrompt(PromptFormatWeMM, "ignored instruction", " a dog ")
	want := "<|im_start|>user\na dog<|im_end|>\n<embedding>"
	if got != want {
		t.Fatalf("text prompt = %q, want %q", got, want)
	}

	got = imagePrompt(PromptFormatWeMM, "ignored instruction", "<__media__>")
	want = "<|im_start|>user\n<__media__><|im_end|>\n<embedding>"
	if got != want {
		t.Fatalf("image prompt = %q, want %q", got, want)
	}
}

func TestTextPromptTrimsLikeLegacyBridge(t *testing.T) {
	// The C++ bridge trimmed only " \t\r\n" and stopped at the first NUL.
	got := textPrompt(PromptFormatQwen3VL, "i", "\u00a0dog\t\x00tail")
	want := "<|im_start|>system\ni<|im_end|>\n<|im_start|>user\n\u00a0dog<|im_end|>\n<|im_start|>assistant\n"
	if got != want {
		t.Fatalf("text prompt = %q, want %q", got, want)
	}
}
