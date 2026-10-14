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

	"imgsearch/internal/embedder/llamacppnative"
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
		modelPath:  filepath.Join(tmp, "8b", "8b-model.gguf"),
		modelURL:   "http://example.invalid/8b-model.gguf",
		mmprojPath: filepath.Join(tmp, "8b", "8b-mmproj.gguf"),
		mmprojURL:  "http://example.invalid/8b-mmproj.gguf",
	}
	altPair := knownModelAssetPair{
		modelPath:  filepath.Join(tmp, "2b", "2b-model.gguf"),
		modelURL:   "",
		mmprojPath: filepath.Join(tmp, "2b", "2b-mmproj.gguf"),
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

func TestSelectKnownAssetPairRecognizes8BSearchModelPath(t *testing.T) {
	pair, resolvedModelPath, resolvedMMProjPath := selectKnownAssetPair(
		llamaNativeSearch8BModelPath,
		defaultLlamaNativeMMProjPath,
		llamaNativeSearchAssetPairs,
	)
	if pair.modelPath != llamaNativeSearch8BModelPath {
		t.Fatalf("selected model path: got=%q want=%q", pair.modelPath, llamaNativeSearch8BModelPath)
	}
	if resolvedModelPath != llamaNativeSearch8BModelPath {
		t.Fatalf("resolved model path: got=%q want=%q", resolvedModelPath, llamaNativeSearch8BModelPath)
	}
	if resolvedMMProjPath != llamaNativeSearch8BMMProjPath {
		t.Fatalf("resolved mmproj path: got=%q want=%q", resolvedMMProjPath, llamaNativeSearch8BMMProjPath)
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

func TestInferLlamaNativePromptFormatFromKnownModelPaths(t *testing.T) {
	absWeMM, err := filepath.Abs(llamaNativeSearchWeMMModelPath)
	if err != nil {
		t.Fatalf("absolute model path: %v", err)
	}
	cases := map[string]llamacppnative.PromptFormat{
		llamaNativeSearchWeMMModelPath: llamacppnative.PromptFormatWeMM,
		absWeMM:                        llamacppnative.PromptFormatWeMM,
		llamaNativeSearch2BModelPath:   llamacppnative.PromptFormatQwen3VL,
		llamaNativeSearch8BModelPath:   llamacppnative.PromptFormatQwen3VL,
		"/models/custom/model.gguf":    llamacppnative.PromptFormatQwen3VL,
		"/models/tencent/WeMM-Embedding-2B-GGUF/WeMM-Embedding-2B-Q4_K_M.gguf": llamacppnative.PromptFormatWeMM,
		"/models/Qwen3-VL-Embedding-2B-Q6_K.gguf":                              llamacppnative.PromptFormatQwen3VL,
	}
	for path, want := range cases {
		if got := inferLlamaNativePromptFormat(path); got != want {
			t.Fatalf("inferLlamaNativePromptFormat(%q) = %q, want %q", path, got, want)
		}
	}
}

func TestEnsureDefaultModelAssetDownloadsKnownFileNameInOtherDirectory(t *testing.T) {
	tmp := t.TempDir()
	defaultPath := filepath.Join(tmp, "models", "known", "model.gguf")
	targetPath := filepath.Join(tmp, "mounted", "known", "model.gguf")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("model-bytes"))
	}))
	defer server.Close()

	resolved, err := ensureDefaultModelAsset(context.Background(), server.Client(), targetPath, defaultPath, server.URL+"/model.gguf")
	if err != nil {
		t.Fatalf("ensure asset: %v", err)
	}
	if resolved != targetPath {
		t.Fatalf("resolved path: got=%q want=%q", resolved, targetPath)
	}
	if content, err := os.ReadFile(targetPath); err != nil || string(content) != "model-bytes" {
		t.Fatalf("expected download at mounted path, content=%q err=%v", content, err)
	}
}

func TestSelectKnownAssetPairMatchesModelOutsideModelsDirByFileName(t *testing.T) {
	modelPath := filepath.Join("/data", "qwen", filepath.Base(llamaNativeSearch2BModelPath))
	pair, resolvedModelPath, resolvedMMProjPath := selectKnownAssetPair(modelPath, defaultLlamaNativeMMProjPath, llamaNativeSearchAssetPairs)
	if pair.modelPath != llamaNativeSearch2BModelPath {
		t.Fatalf("selected pair: got=%q want Qwen 2B", pair.modelPath)
	}
	if resolvedModelPath != modelPath {
		t.Fatalf("resolved model path: got=%q want=%q", resolvedModelPath, modelPath)
	}
	wantMMProj := filepath.Join("/data", "qwen", filepath.Base(llamaNativeSearch2BMMProjPath))
	if resolvedMMProjPath != wantMMProj {
		t.Fatalf("resolved mmproj path: got=%q want=%q", resolvedMMProjPath, wantMMProj)
	}
}

func TestValidateKnownAssetPairRejectsMixedKnownFiles(t *testing.T) {
	if err := validateKnownAssetPair(llamaNativeSearchWeMMModelPath, llamaNativeSearch2BMMProjPath, llamaNativeSearchAssetPairs); err == nil {
		t.Fatal("expected error for WeMM model with Qwen 2B mmproj")
	}
	if err := validateKnownAssetPair(llamaNativeSearch2BModelPath, llamaNativeSearch2BMMProjPath, llamaNativeSearchAssetPairs); err != nil {
		t.Fatalf("matched pair rejected: %v", err)
	}
	if err := validateKnownAssetPair("/m/custom.gguf", llamaNativeSearch2BMMProjPath, llamaNativeSearchAssetPairs); err != nil {
		t.Fatalf("custom model rejected: %v", err)
	}
}

func TestKnownSearchAssetFileNamesAreUnique(t *testing.T) {
	seen := map[string]bool{}
	for _, pair := range llamaNativeSearchAssetPairs {
		for _, path := range []string{pair.modelPath, pair.mmprojPath} {
			name := filepath.Base(path)
			if seen[name] {
				t.Fatalf("known asset file name %q is not unique; pairs are matched by file name", name)
			}
			seen[name] = true
		}
	}
}

func TestLlamaNativeAssetPairWarning(t *testing.T) {
	custom := "/m/Qwen3-VL-Embedding-2B-Q8_0.gguf"
	if w := llamaNativeAssetPairWarning(custom, defaultLlamaNativeMMProjPath, llamaNativeSearchAssetPairs); !strings.Contains(w, "mmproj") {
		t.Fatalf("expected warning for unknown model with default mmproj, got %q", w)
	}
	if w := llamaNativeAssetPairWarning(custom, "/m/mmproj-custom.gguf", llamaNativeSearchAssetPairs); w != "" {
		t.Fatalf("unexpected warning with explicit mmproj: %q", w)
	}
	if w := llamaNativeAssetPairWarning(defaultLlamaNativeModelPath, defaultLlamaNativeMMProjPath, llamaNativeSearchAssetPairs); w != "" {
		t.Fatalf("unexpected warning for default pair: %q", w)
	}
}
