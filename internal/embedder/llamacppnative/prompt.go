package llamacppnative

import (
	"fmt"
	"strings"
)

// PromptFormat selects how embedding inputs are wrapped for the model.
type PromptFormat string

const (
	// PromptFormatQwen3VL wraps inputs in a system instruction and an open
	// assistant turn, pooled at the last token (Qwen3-VL-Embedding).
	PromptFormatQwen3VL PromptFormat = "qwen3vl"
	// PromptFormatWeMM uses a bare user turn closed by the <embedding> token
	// (WeMM-Embedding). Instructions are not used.
	PromptFormatWeMM PromptFormat = "wemm"
)

const defaultSystemMessage = "Represent the user's input."

// ParsePromptFormat accepts a format name; empty selects Qwen3-VL.
func ParsePromptFormat(s string) (PromptFormat, error) {
	switch PromptFormat(strings.ToLower(strings.TrimSpace(s))) {
	case "", PromptFormatQwen3VL:
		return PromptFormatQwen3VL, nil
	case PromptFormatWeMM:
		return PromptFormatWeMM, nil
	}
	return "", fmt.Errorf("unknown prompt format %q (want qwen3vl or wemm)", s)
}

func textPrompt(format PromptFormat, instruction, text string) string {
	text = legacyTrim(text)
	if format == PromptFormatWeMM {
		return "<|im_start|>user\n" + text + "<|im_end|>\n<embedding>"
	}
	return qwen3VLPrompt(instruction, text)
}

func imagePrompt(format PromptFormat, instruction, mediaMarker string) string {
	if format == PromptFormatWeMM {
		return "<|im_start|>user\n" + mediaMarker + "<|im_end|>\n<embedding>"
	}
	return qwen3VLPrompt(instruction, mediaMarker)
}

func qwen3VLPrompt(instruction, content string) string {
	system := strings.TrimSpace(instruction)
	if system == "" {
		system = defaultSystemMessage
	}
	return "<|im_start|>system\n" + system + "<|im_end|>\n<|im_start|>user\n" + content +
		"<|im_end|>\n<|im_start|>assistant\n"
}

// legacyTrim matches the former C++ bridge, which saw the text as a C string
// and trimmed only ASCII whitespace; keeping it preserves stored embeddings.
func legacyTrim(s string) string {
	if i := strings.IndexByte(s, 0); i >= 0 {
		s = s[:i]
	}
	return strings.Trim(s, " \t\r\n")
}
