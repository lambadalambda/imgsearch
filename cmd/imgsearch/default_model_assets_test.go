package main

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

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

func TestEnsureKnownAssetPairDownloadsAlternateKnownPair(t *testing.T) {
	tmp := t.TempDir()
	defaultPair := knownModelAssetPair{
		modelPath:  filepath.Join(tmp, "8b", "model.gguf"),
		modelURL:   "http://example.invalid/8b-model.gguf",
		mmprojPath: filepath.Join(tmp, "8b", "mmproj.gguf"),
		mmprojURL:  "http://example.invalid/8b-mmproj.gguf",
	}
	altPair := knownModelAssetPair{
		modelPath:  filepath.Join(tmp, "2b", "model.gguf"),
		modelURL:   "",
		mmprojPath: filepath.Join(tmp, "2b", "mmproj.gguf"),
		mmprojURL:  "",
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/2b-model.gguf":
			_, _ = w.Write([]byte("2b-model-bytes"))
		case "/2b-mmproj.gguf":
			_, _ = w.Write([]byte("2b-mmproj-bytes"))
		default:
			t.Fatalf("unexpected request path: %s", r.URL.Path)
		}
	}))
	defer server.Close()
	altPair.modelURL = server.URL + "/2b-model.gguf"
	altPair.mmprojURL = server.URL + "/2b-mmproj.gguf"

	resolvedModelPath, resolvedMMProjPath, err := ensureKnownAssetPair(
		context.Background(),
		server.Client(),
		altPair.modelPath,
		defaultPair.mmprojPath,
		[]knownModelAssetPair{defaultPair, altPair},
	)
	if err != nil {
		t.Fatalf("ensure known asset pair: %v", err)
	}
	if resolvedModelPath != altPair.modelPath {
		t.Fatalf("resolved model path: got=%q want=%q", resolvedModelPath, altPair.modelPath)
	}
	if resolvedMMProjPath != altPair.mmprojPath {
		t.Fatalf("resolved mmproj path: got=%q want=%q", resolvedMMProjPath, altPair.mmprojPath)
	}

	modelContent, err := os.ReadFile(altPair.modelPath)
	if err != nil {
		t.Fatalf("read downloaded model asset: %v", err)
	}
	if string(modelContent) != "2b-model-bytes" {
		t.Fatalf("unexpected model asset content: %q", string(modelContent))
	}
	mmprojContent, err := os.ReadFile(altPair.mmprojPath)
	if err != nil {
		t.Fatalf("read downloaded mmproj asset: %v", err)
	}
	if string(mmprojContent) != "2b-mmproj-bytes" {
		t.Fatalf("unexpected mmproj asset content: %q", string(mmprojContent))
	}
}

func TestSelectKnownAssetPairRecognizes2BSearchModelPath(t *testing.T) {
	pair, resolvedModelPath, resolvedMMProjPath := selectKnownAssetPair(
		llamaNativeSearch2BModelPath,
		defaultLlamaNativeMMProjPath,
		llamaNativeSearchAssetPairs,
	)
	if pair.modelPath != llamaNativeSearch2BModelPath {
		t.Fatalf("selected model path: got=%q want=%q", pair.modelPath, llamaNativeSearch2BModelPath)
	}
	if resolvedModelPath != llamaNativeSearch2BModelPath {
		t.Fatalf("resolved model path: got=%q want=%q", resolvedModelPath, llamaNativeSearch2BModelPath)
	}
	if resolvedMMProjPath != llamaNativeSearch2BMMProjPath {
		t.Fatalf("resolved mmproj path: got=%q want=%q", resolvedMMProjPath, llamaNativeSearch2BMMProjPath)
	}
}

func TestSelectKnownAssetPairRecognizesAbsolute2BSearchPaths(t *testing.T) {
	absModelPath, err := filepath.Abs(llamaNativeSearch2BModelPath)
	if err != nil {
		t.Fatalf("absolute model path: %v", err)
	}
	absMMProjPath, err := filepath.Abs(llamaNativeSearch2BMMProjPath)
	if err != nil {
		t.Fatalf("absolute mmproj path: %v", err)
	}

	pair, resolvedModelPath, resolvedMMProjPath := selectKnownAssetPair(
		absModelPath,
		absMMProjPath,
		llamaNativeSearchAssetPairs,
	)
	if pair.modelPath != llamaNativeSearch2BModelPath {
		t.Fatalf("selected model path: got=%q want=%q", pair.modelPath, llamaNativeSearch2BModelPath)
	}
	if resolvedModelPath != absModelPath {
		t.Fatalf("resolved model path: got=%q want=%q", resolvedModelPath, absModelPath)
	}
	if resolvedMMProjPath != absMMProjPath {
		t.Fatalf("resolved mmproj path: got=%q want=%q", resolvedMMProjPath, absMMProjPath)
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

func TestResolveDownloadHTTPClientAppliesDefaultTimeout(t *testing.T) {
	client := resolveDownloadHTTPClient(nil)
	if client == nil {
		t.Fatal("expected default download client")
	}
	if client.Timeout != defaultDownloadTimeout {
		t.Fatalf("client timeout: got=%s want=%s", client.Timeout, defaultDownloadTimeout)
	}

	existing := &http.Client{Timeout: time.Second}
	if got := resolveDownloadHTTPClient(existing); got != existing {
		t.Fatal("expected existing client to be preserved")
	}
}

func TestDownloadFileRejectsContentLengthMismatch(t *testing.T) {
	tmp := t.TempDir()
	destPath := filepath.Join(tmp, "models", "bad.gguf")
	client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode:    http.StatusOK,
			Status:        "200 OK",
			Body:          io.NopCloser(strings.NewReader("abc")),
			ContentLength: 10,
			Header:        make(http.Header),
			Request:       req,
		}, nil
	})}

	err := downloadFile(context.Background(), client, "https://example.invalid/bad.gguf", destPath)
	if err == nil {
		t.Fatal("expected content-length mismatch error")
	}
	if !strings.Contains(err.Error(), "content length") {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, statErr := os.Stat(destPath); !os.IsNotExist(statErr) {
		t.Fatalf("expected no final file on failure, stat err=%v", statErr)
	}
}
