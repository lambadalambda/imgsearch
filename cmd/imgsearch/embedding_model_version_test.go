package main

import (
	"strings"
	"testing"
)

func TestLlamaNativeModelVersionChangesWithConfig(t *testing.T) {
	base := llamaNativeModelVersion("/models/qwen.gguf", "/models/mmproj.gguf", 2048, 512, 0, "query", "passage")
	if base == "" {
		t.Fatal("expected non-empty native model version")
	}
	if !strings.HasPrefix(base, "native-v5-chatml-vipsjpeg-") {
		t.Fatalf("expected native version prefix to include chatml-vipsjpeg, got %q", base)
	}

	if got := llamaNativeModelVersion("/models/qwen.gguf", "/models/mmproj.gguf", 2048, 512, 0, "query", "passage"); got != base {
		t.Fatalf("expected stable native model version, got %q want %q", got, base)
	}

	if got := llamaNativeModelVersion("/models/qwen.gguf", "/models/mmproj.gguf", 2048, 768, 0, "query", "passage"); got == base {
		t.Fatal("expected image max side to affect native model version")
	}

	if got := llamaNativeModelVersion("/models/qwen.gguf", "/models/mmproj.gguf", 2048, 512, 256, "query", "passage"); got == base {
		t.Fatal("expected image max tokens to affect native model version")
	}

	if got := llamaNativeModelVersion("/models/other.gguf", "/models/mmproj.gguf", 2048, 512, 0, "query", "passage"); got == base {
		t.Fatal("expected model filename to affect native model version")
	}

	if got := llamaNativeModelVersion("/models/qwen.gguf", "/models/other-mmproj.gguf", 2048, 512, 0, "query", "passage"); got == base {
		t.Fatal("expected mmproj filename to affect native model version")
	}

	if got := llamaNativeModelVersion("/models/qwen.gguf", "/models/mmproj.gguf", 4096, 512, 0, "query", "passage"); got == base {
		t.Fatal("expected dimensions to affect native model version")
	}

	if got := llamaNativeModelVersion("/models/qwen.gguf", "/models/mmproj.gguf", 2048, 512, 0, "other query", "passage"); got == base {
		t.Fatal("expected query instruction to affect native model version")
	}

	if got := llamaNativeModelVersion("/models/qwen.gguf", "/models/mmproj.gguf", 2048, 512, 0, "query", "other passage"); got == base {
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
