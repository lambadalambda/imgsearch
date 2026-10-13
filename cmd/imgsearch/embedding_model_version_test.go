package main

import (
	"strings"
	"testing"
)

func TestLlamaNativeModelVersionChangesWithConfig(t *testing.T) {
	base := llamaNativeModelVersion("/models/qwen.gguf", "/models/mmproj.gguf", 2048, 512, 0, "query", "passage", "qwen3vl")
	if base == "" {
		t.Fatal("expected non-empty native model version")
	}
	if !strings.HasPrefix(base, "native-v5-chatml-vipsjpeg-") {
		t.Fatalf("expected native version prefix to include chatml-vipsjpeg, got %q", base)
	}

	if got := llamaNativeModelVersion("/models/qwen.gguf", "/models/mmproj.gguf", 2048, 512, 0, "query", "passage", "qwen3vl"); got != base {
		t.Fatalf("expected stable native model version, got %q want %q", got, base)
	}

	if got := llamaNativeModelVersion("/models/qwen.gguf", "/models/mmproj.gguf", 2048, 768, 0, "query", "passage", "qwen3vl"); got == base {
		t.Fatal("expected image max side to affect native model version")
	}

	if got := llamaNativeModelVersion("/models/qwen.gguf", "/models/mmproj.gguf", 2048, 512, 256, "query", "passage", "qwen3vl"); got == base {
		t.Fatal("expected image max tokens to affect native model version")
	}

	if got := llamaNativeModelVersion("/models/other.gguf", "/models/mmproj.gguf", 2048, 512, 0, "query", "passage", "qwen3vl"); got == base {
		t.Fatal("expected model filename to affect native model version")
	}

	if got := llamaNativeModelVersion("/models/qwen.gguf", "/models/other-mmproj.gguf", 2048, 512, 0, "query", "passage", "qwen3vl"); got == base {
		t.Fatal("expected mmproj filename to affect native model version")
	}

	if got := llamaNativeModelVersion("/models/qwen.gguf", "/models/mmproj.gguf", 4096, 512, 0, "query", "passage", "qwen3vl"); got == base {
		t.Fatal("expected dimensions to affect native model version")
	}

	if got := llamaNativeModelVersion("/models/qwen.gguf", "/models/mmproj.gguf", 2048, 512, 0, "other query", "passage", "qwen3vl"); got == base {
		t.Fatal("expected query instruction to affect native model version")
	}

	if got := llamaNativeModelVersion("/models/qwen.gguf", "/models/mmproj.gguf", 2048, 512, 0, "query", "other passage", "qwen3vl"); got == base {
		t.Fatal("expected passage instruction to affect native model version")
	}
}

func TestResolveLlamaCPPNativeImageLimits(t *testing.T) {
	side, tokens, err := resolveLlamaCPPNativeImageLimits(0, 0)
	if err != nil {
		t.Fatalf("resolve defaults: %v", err)
	}
	if side != defaultLlamaNativeImageMaxSide || tokens != 0 {
		t.Fatalf("unexpected default limits: side=%d tokens=%d", side, tokens)
	}

	if _, _, err := resolveLlamaCPPNativeImageLimits(-1, 0); err == nil {
		t.Fatal("expected negative image max side error")
	}

	if _, _, err := resolveLlamaCPPNativeImageLimits(512, -1); err == nil {
		t.Fatal("expected negative image max tokens error")
	}
}

func TestLlamaNativeModelVersionPromptFormat(t *testing.T) {
	qwen := llamaNativeModelVersion("/m/a.gguf", "/m/b.gguf", 2048, 384, 0, "q", "p", "qwen3vl")
	legacy := "native-v5-chatml-vipsjpeg-a.gguf-b.gguf-d2048-s384-t0-q" + instructionVersionToken("q") + "-p" + instructionVersionToken("p")
	if qwen != legacy {
		t.Fatalf("qwen3vl version must stay unchanged to avoid re-embedding: got %q want %q", qwen, legacy)
	}
	if got := llamaNativeModelVersion("/m/a.gguf", "/m/b.gguf", 2048, 384, 0, "q", "p", ""); got != legacy {
		t.Fatalf("empty prompt format must match qwen3vl: got %q", got)
	}
	if got := llamaNativeModelVersion("/m/a.gguf", "/m/b.gguf", 2048, 384, 0, "q", "p", "wemm"); got != legacy+"-fwemm" {
		t.Fatalf("expected wemm format suffix, got %q", got)
	}
}
