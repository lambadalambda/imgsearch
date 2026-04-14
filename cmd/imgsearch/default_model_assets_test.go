package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestEnsureDefaultModelAssetDownloadsMissingDefault(t *testing.T) {
	tmp := t.TempDir()
	targetPath := filepath.Join(tmp, "models", "Qwen", "model.gguf")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/model.gguf" {
			t.Fatalf("unexpected request path: %s", r.URL.Path)
		}
		_, _ = w.Write([]byte("model-bytes"))
	}))
	defer server.Close()

	resolved, err := ensureDefaultModelAsset(context.Background(), server.Client(), targetPath, targetPath, server.URL+"/model.gguf")
	if err != nil {
		t.Fatalf("ensure default asset: %v", err)
	}
	if resolved != targetPath {
		t.Fatalf("resolved path: got=%q want=%q", resolved, targetPath)
	}

	content, err := os.ReadFile(targetPath)
	if err != nil {
		t.Fatalf("read downloaded asset: %v", err)
	}
	if string(content) != "model-bytes" {
		t.Fatalf("unexpected asset content: %q", string(content))
	}
}

func TestEnsureDefaultAssetPairDownloadsMissingDefaults(t *testing.T) {
	tmp := t.TempDir()
	modelPath := filepath.Join(tmp, "models", "Gemma", "model.gguf")
	mmprojPath := filepath.Join(tmp, "models", "Gemma", "mmproj.gguf")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/model.gguf":
			_, _ = w.Write([]byte("model-bytes"))
		case "/mmproj.gguf":
			_, _ = w.Write([]byte("mmproj-bytes"))
		default:
			t.Fatalf("unexpected request path: %s", r.URL.Path)
		}
	}))
	defer server.Close()

	resolvedModelPath, resolvedMMProjPath, err := ensureDefaultAssetPair(
		context.Background(),
		server.Client(),
		"",
		"",
		modelPath,
		server.URL+"/model.gguf",
		mmprojPath,
		server.URL+"/mmproj.gguf",
	)
	if err != nil {
		t.Fatalf("ensure default asset pair: %v", err)
	}
	if resolvedModelPath != modelPath {
		t.Fatalf("resolved model path: got=%q want=%q", resolvedModelPath, modelPath)
	}
	if resolvedMMProjPath != mmprojPath {
		t.Fatalf("resolved mmproj path: got=%q want=%q", resolvedMMProjPath, mmprojPath)
	}

	modelContent, err := os.ReadFile(modelPath)
	if err != nil {
		t.Fatalf("read downloaded model asset: %v", err)
	}
	if string(modelContent) != "model-bytes" {
		t.Fatalf("unexpected model asset content: %q", string(modelContent))
	}

	mmprojContent, err := os.ReadFile(mmprojPath)
	if err != nil {
		t.Fatalf("read downloaded mmproj asset: %v", err)
	}
	if string(mmprojContent) != "mmproj-bytes" {
		t.Fatalf("unexpected mmproj asset content: %q", string(mmprojContent))
	}
}

func TestEnsureDefaultLlamaNativeAnnotatorAssetsForVariantRejectsUnknownVariant(t *testing.T) {
	_, _, err := ensureDefaultLlamaNativeAnnotatorAssetsForVariant(context.Background(), "weird", "", "")
	if err == nil {
		t.Fatal("expected error for unknown annotator variant")
	}
}

func TestEnsureDefaultModelAssetSkipsCustomPath(t *testing.T) {
	tmp := t.TempDir()
	customPath := filepath.Join(tmp, "custom.gguf")
	defaultPath := filepath.Join(tmp, "default.gguf")

	resolved, err := ensureDefaultModelAsset(context.Background(), nil, customPath, defaultPath, "https://example.invalid/model.gguf")
	if err != nil {
		t.Fatalf("ensure custom asset: %v", err)
	}
	if resolved != customPath {
		t.Fatalf("resolved path: got=%q want=%q", resolved, customPath)
	}
	if _, err := os.Stat(customPath); !os.IsNotExist(err) {
		t.Fatalf("expected custom path to remain untouched, stat err=%v", err)
	}
}

func TestFormatByteCount(t *testing.T) {
	if got := formatByteCount(999); got != "999 B" {
		t.Fatalf("unexpected bytes format: %q", got)
	}
	if got := formatByteCount(1536); got != "1.5 KiB" {
		t.Fatalf("unexpected kib format: %q", got)
	}
}
