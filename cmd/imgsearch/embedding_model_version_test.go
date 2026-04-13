package main

import "testing"

func TestLlamaCPPModelVersion(t *testing.T) {
	if got := llamaCPPModelVersion("", 2048); got != "server-v2-default-d2048" {
		t.Fatalf("unexpected default llama-cpp model version: %q", got)
	}

	if got := llamaCPPModelVersion("Qwen 3/VL", 4096); got != "server-v2-Qwen_3_VL-d4096" {
		t.Fatalf("unexpected sanitized llama-cpp model version: %q", got)
	}
}

func TestLlamaNativeModelVersionChangesWithConfig(t *testing.T) {
	base := llamaNativeModelVersion("/models/qwen.gguf", "/models/mmproj.gguf", 2048, 512, 0)
	if base == "" {
		t.Fatal("expected non-empty native model version")
	}

	if got := llamaNativeModelVersion("/models/qwen.gguf", "/models/mmproj.gguf", 2048, 512, 0); got != base {
		t.Fatalf("expected stable native model version, got %q want %q", got, base)
	}

	if got := llamaNativeModelVersion("/models/qwen.gguf", "/models/mmproj.gguf", 2048, 768, 0); got == base {
		t.Fatal("expected image max side to affect native model version")
	}

	if got := llamaNativeModelVersion("/models/qwen.gguf", "/models/mmproj.gguf", 2048, 512, 256); got == base {
		t.Fatal("expected image max tokens to affect native model version")
	}

	if got := llamaNativeModelVersion("/models/other.gguf", "/models/mmproj.gguf", 2048, 512, 0); got == base {
		t.Fatal("expected model filename to affect native model version")
	}

	if got := llamaNativeModelVersion("/models/qwen.gguf", "/models/other-mmproj.gguf", 2048, 512, 0); got == base {
		t.Fatal("expected mmproj filename to affect native model version")
	}

	if got := llamaNativeModelVersion("/models/qwen.gguf", "/models/mmproj.gguf", 4096, 512, 0); got == base {
		t.Fatal("expected dimensions to affect native model version")
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
