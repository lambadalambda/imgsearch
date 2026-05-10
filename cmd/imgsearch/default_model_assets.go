package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	defaultLlamaNativeModelPath                    = "./models/Qwen/Qwen3-VL-Embedding-8B-Q4_K_M.gguf"
	defaultLlamaNativeMMProjPath                   = "./models/Qwen/mmproj-Qwen3-VL-Embedding-8B-f16.gguf"
	defaultLlamaNativeDimensions                   = 4096
	llamaNativeSearch2BModelPath                   = "./models/VesNFF/Qwen3-VL-Embedding-2B-GGUF/Qwen3-VL-Embedding-2B-Q6_K.gguf"
	llamaNativeSearch2BMMProjPath                  = "./models/VesNFF/Qwen3-VL-Embedding-2B-GGUF/mmproj-Qwen3-VL-Embedding-2B-f16.gguf"
	llamaNativeSearch2BModelURL                    = "https://huggingface.co/VesNFF/Qwen3-VL-Embedding-2B-GGUF/resolve/main/Qwen3-VL-Embedding-2B-Q6_K.gguf"
	llamaNativeSearch2BMMProjURL                   = "https://huggingface.co/VesNFF/Qwen3-VL-Embedding-2B-GGUF/resolve/main/mmproj-Qwen3-VL-Embedding-2B-f16.gguf"
	defaultLlamaNativeEmbedderContextSize          = 512
	defaultLlamaNativeAnnotationTemperature        = 1.0
	defaultLlamaNativeAnnotationSeed         int64 = -1
	defaultLlamaNativeMaxAnnotationSeed      int64 = 4294967295
	defaultLlamaNativeFlashAttnType                = -1
	defaultLlamaNativeCacheTypeK                   = -1
	defaultLlamaNativeCacheTypeV                   = -1
	defaultLlamaNativeModelURL                     = "https://huggingface.co/lainsoykaf/Qwen3-VL-Embedding-8B-GGUF/resolve/main/Qwen3-VL-Embedding-8B-Q4_K_M.gguf"
	defaultLlamaNativeMMProjURL                    = "https://huggingface.co/lainsoykaf/Qwen3-VL-Embedding-8B-GGUF/resolve/main/mmproj-Qwen3-VL-Embedding-8B-f16.gguf"
	defaultLlamaNativeAnnotatorVariant             = "e4b"
	llamaNativeAnnotatorVariant26B                 = "26b"
	defaultLlamaNativeAnnotatorModelPath           = "./models/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf"
	defaultLlamaNativeAnnotatorMMProjPath          = "./models/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive/mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf"
	defaultLlamaNativeAnnotatorModelURL            = "https://huggingface.co/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive/resolve/main/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf"
	defaultLlamaNativeAnnotatorMMProjURL           = "https://huggingface.co/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive/resolve/main/mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf"
	defaultLlamaNativeAnnotator26BModelPath        = "./models/nohurry/gemma-4-26B-A4B-it-heretic-GUFF/gemma-4-26b-a4b-it-heretic.q4_k_m.gguf"
	defaultLlamaNativeAnnotator26BMMProjPath       = "./models/nohurry/gemma-4-26B-A4B-it-heretic-GUFF/gemma-4-26B-A4B-it-heretic-mmproj.f16.gguf"
	defaultLlamaNativeAnnotator26BModelURL         = "https://huggingface.co/nohurry/gemma-4-26B-A4B-it-heretic-GUFF/resolve/main/gemma-4-26b-a4b-it-heretic.q4_k_m.gguf"
	defaultLlamaNativeAnnotator26BMMProjURL        = "https://huggingface.co/nohurry/gemma-4-26B-A4B-it-heretic-GUFF/resolve/main/gemma-4-26B-A4B-it-heretic-mmproj.f16.gguf"
	defaultParakeetOnnxBundleDir                   = "./models/Parakeet/parakeet-tdt-0.6b-v3-onnx"
	defaultParakeetOnnxConfigURL                   = "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/config.json"
	defaultParakeetOnnxVocabURL                    = "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/vocab.txt"
	defaultParakeetOnnxDecoderURL                  = "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/decoder_joint-model.int8.onnx"
	defaultParakeetOnnxEncoderURL                  = "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/encoder-model.int8.onnx"
	defaultDownloadUserAgent                       = "imgsearch/1.0"
	defaultDownloadTimeout                         = 30 * time.Minute
	defaultDownloadProgressPeriod                  = 5 * time.Second
)

type knownModelAssetPair struct {
	modelPath  string
	modelURL   string
	mmprojPath string
	mmprojURL  string
}

var llamaNativeSearchAssetPairs = []knownModelAssetPair{
	{
		modelPath:  defaultLlamaNativeModelPath,
		modelURL:   defaultLlamaNativeModelURL,
		mmprojPath: defaultLlamaNativeMMProjPath,
		mmprojURL:  defaultLlamaNativeMMProjURL,
	},
	{
		modelPath:  llamaNativeSearch2BModelPath,
		modelURL:   llamaNativeSearch2BModelURL,
		mmprojPath: llamaNativeSearch2BMMProjPath,
		mmprojURL:  llamaNativeSearch2BMMProjURL,
	},
}

func ensureDefaultLlamaNativeAssets(ctx context.Context, modelPath string, mmprojPath string) (string, string, error) {
	return ensureKnownAssetPair(
		ctx,
		nil,
		modelPath,
		mmprojPath,
		llamaNativeSearchAssetPairs,
	)
}

func ensureKnownAssetPair(ctx context.Context, httpClient *http.Client, modelPath string, mmprojPath string, pairs []knownModelAssetPair) (string, string, error) {
	pair, resolvedModelPath, resolvedMMProjPath := selectKnownAssetPair(modelPath, mmprojPath, pairs)
	return ensureDefaultAssetPair(
		ctx,
		httpClient,
		resolvedModelPath,
		resolvedMMProjPath,
		pair.modelPath,
		pair.modelURL,
		pair.mmprojPath,
		pair.mmprojURL,
	)
}

func selectKnownAssetPair(modelPath string, mmprojPath string, pairs []knownModelAssetPair) (knownModelAssetPair, string, string) {
	if len(pairs) == 0 {
		return knownModelAssetPair{}, strings.TrimSpace(modelPath), strings.TrimSpace(mmprojPath)
	}
	defaultPair := pairs[0]
	resolvedModelPath := strings.TrimSpace(modelPath)
	if resolvedModelPath == "" {
		resolvedModelPath = defaultPair.modelPath
	}
	resolvedMMProjPath := strings.TrimSpace(mmprojPath)
	if resolvedMMProjPath == "" {
		resolvedMMProjPath = defaultPair.mmprojPath
	}

	for _, pair := range pairs {
		modelMatches := sameAssetPath(resolvedModelPath, pair.modelPath)
		mmprojMatches := sameAssetPath(resolvedMMProjPath, pair.mmprojPath)
		modelIsDefault := sameAssetPath(resolvedModelPath, defaultPair.modelPath)
		mmprojIsDefault := sameAssetPath(resolvedMMProjPath, defaultPair.mmprojPath)
		switch {
		case modelMatches && mmprojMatches:
			return pair, resolvedModelPath, resolvedMMProjPath
		case modelMatches && mmprojIsDefault:
			return pair, resolvedModelPath, pair.mmprojPath
		case modelIsDefault && mmprojMatches:
			return pair, pair.modelPath, resolvedMMProjPath
		}
	}

	return defaultPair, resolvedModelPath, resolvedMMProjPath
}

func sameAssetPath(left string, right string) bool {
	left = strings.TrimSpace(left)
	right = strings.TrimSpace(right)
	if left == "" || right == "" {
		return left == right
	}
	return cleanAbsAssetPath(left) == cleanAbsAssetPath(right)
}

func cleanAbsAssetPath(path string) string {
	cleaned := filepath.Clean(strings.TrimSpace(path))
	abs, err := filepath.Abs(cleaned)
	if err != nil {
		return cleaned
	}
	return abs
}

func ensureDefaultLlamaNativeAnnotatorAssets(ctx context.Context, modelPath string, mmprojPath string) (string, string, error) {
	return ensureDefaultAssetPair(
		ctx,
		nil,
		modelPath,
		mmprojPath,
		defaultLlamaNativeAnnotatorModelPath,
		defaultLlamaNativeAnnotatorModelURL,
		defaultLlamaNativeAnnotatorMMProjPath,
		defaultLlamaNativeAnnotatorMMProjURL,
	)
}

func ensureDefaultLlamaNativeAnnotator26BAssets(ctx context.Context, modelPath string, mmprojPath string) (string, string, error) {
	return ensureDefaultAssetPair(
		ctx,
		nil,
		modelPath,
		mmprojPath,
		defaultLlamaNativeAnnotator26BModelPath,
		defaultLlamaNativeAnnotator26BModelURL,
		defaultLlamaNativeAnnotator26BMMProjPath,
		defaultLlamaNativeAnnotator26BMMProjURL,
	)
}

func ensureDefaultLlamaNativeAnnotatorAssetsForVariant(ctx context.Context, variant string, modelPath string, mmprojPath string) (string, string, error) {
	switch strings.ToLower(strings.TrimSpace(variant)) {
	case "", defaultLlamaNativeAnnotatorVariant:
		return ensureDefaultLlamaNativeAnnotatorAssets(ctx, modelPath, mmprojPath)
	case llamaNativeAnnotatorVariant26B:
		return ensureDefaultLlamaNativeAnnotator26BAssets(ctx, modelPath, mmprojPath)
	default:
		return "", "", fmt.Errorf("unsupported llama-cpp-native annotator variant %q (expected %q or %q)", variant, defaultLlamaNativeAnnotatorVariant, llamaNativeAnnotatorVariant26B)
	}
}

func ensureDefaultParakeetOnnxAssets(ctx context.Context, bundleDir string) (string, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	bundleDir = strings.TrimSpace(bundleDir)
	if bundleDir == "" {
		bundleDir = defaultParakeetOnnxBundleDir
	}
	if bundleDir != defaultParakeetOnnxBundleDir {
		return bundleDir, nil
	}

	files := map[string]string{
		"config.json":                   defaultParakeetOnnxConfigURL,
		"vocab.txt":                     defaultParakeetOnnxVocabURL,
		"decoder_joint-model.int8.onnx": defaultParakeetOnnxDecoderURL,
		"encoder-model.int8.onnx":       defaultParakeetOnnxEncoderURL,
	}
	for name, url := range files {
		if _, err := ensureDefaultModelAsset(ctx, nil, filepath.Join(bundleDir, name), filepath.Join(defaultParakeetOnnxBundleDir, name), url); err != nil {
			return "", err
		}
	}
	return bundleDir, nil
}

func ensureDefaultAssetPair(ctx context.Context, httpClient *http.Client, modelPath string, mmprojPath string, defaultModelPath string, defaultModelURL string, defaultMMProjPath string, defaultMMProjURL string) (string, string, error) {
	if ctx == nil {
		ctx = context.Background()
	}

	modelPath = strings.TrimSpace(modelPath)
	if modelPath == "" {
		modelPath = defaultModelPath
	}
	mmprojPath = strings.TrimSpace(mmprojPath)
	if mmprojPath == "" {
		mmprojPath = defaultMMProjPath
	}

	resolvedModelPath, err := ensureDefaultModelAsset(ctx, httpClient, modelPath, defaultModelPath, defaultModelURL)
	if err != nil {
		return "", "", err
	}
	resolvedMMProjPath, err := ensureDefaultModelAsset(ctx, httpClient, mmprojPath, defaultMMProjPath, defaultMMProjURL)
	if err != nil {
		return "", "", err
	}

	return resolvedModelPath, resolvedMMProjPath, nil
}

func ensureDefaultModelAsset(ctx context.Context, httpClient *http.Client, path string, defaultPath string, url string) (string, error) {
	resolvedPath := strings.TrimSpace(path)
	if resolvedPath == "" {
		resolvedPath = defaultPath
	}
	if !sameAssetPath(resolvedPath, defaultPath) {
		return resolvedPath, nil
	}
	if err := ensureRegularFile(resolvedPath); err == nil {
		return resolvedPath, nil
	} else if !os.IsNotExist(err) {
		return "", err
	}

	if err := downloadFile(ctx, httpClient, url, resolvedPath); err != nil {
		return "", err
	}
	return resolvedPath, nil
}

func downloadFile(ctx context.Context, httpClient *http.Client, url string, destPath string) error {
	httpClient = resolveDownloadHTTPClient(httpClient)
	if err := os.MkdirAll(filepath.Dir(destPath), 0o755); err != nil {
		return fmt.Errorf("create model directory: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("build download request: %w", err)
	}
	req.Header.Set("User-Agent", defaultDownloadUserAgent)

	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("download %s: %w", filepath.Base(destPath), err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download %s: unexpected HTTP status %s", filepath.Base(destPath), resp.Status)
	}

	tmpFile, err := os.CreateTemp(filepath.Dir(destPath), filepath.Base(destPath)+".*.part")
	if err != nil {
		return fmt.Errorf("create temporary download file: %w", err)
	}
	tmpPath := tmpFile.Name()
	tmpClosed := false
	defer func() {
		if !tmpClosed {
			_ = tmpFile.Close()
		}
		if tmpPath != "" {
			_ = os.Remove(tmpPath)
		}
	}()

	log.Printf("downloading default model asset %s", filepath.Base(destPath))
	written, err := copyWithProgress(tmpFile, resp.Body, filepath.Base(destPath), resp.ContentLength)
	if err != nil {
		return fmt.Errorf("write %s: %w", filepath.Base(destPath), err)
	}
	// For known default assets we currently trust HTTPS plus an exact body length
	// match when the server advertises Content-Length.
	if resp.ContentLength >= 0 && written != resp.ContentLength {
		return fmt.Errorf("write %s: content length mismatch: wrote %d bytes, expected %d", filepath.Base(destPath), written, resp.ContentLength)
	}
	if err := tmpFile.Close(); err != nil {
		return fmt.Errorf("close temporary download file: %w", err)
	}
	tmpClosed = true
	if err := os.Rename(tmpPath, destPath); err != nil {
		return fmt.Errorf("move downloaded file into place: %w", err)
	}
	tmpPath = ""
	log.Printf("downloaded default model asset %s", filepath.Base(destPath))
	return nil
}

func resolveDownloadHTTPClient(httpClient *http.Client) *http.Client {
	if httpClient != nil {
		return httpClient
	}
	return &http.Client{Timeout: defaultDownloadTimeout}
}

func copyWithProgress(dst io.Writer, src io.Reader, name string, total int64) (int64, error) {
	buf := make([]byte, 1024*1024)
	var written int64
	nextLog := time.Now().Add(defaultDownloadProgressPeriod)

	for {
		n, err := src.Read(buf)
		if n > 0 {
			if _, writeErr := dst.Write(buf[:n]); writeErr != nil {
				return written, writeErr
			}
			written += int64(n)
			if time.Now().After(nextLog) {
				if total > 0 {
					log.Printf("downloading %s: %.1f%% (%s/%s)", name, float64(written)*100/float64(total), formatByteCount(written), formatByteCount(total))
				} else {
					log.Printf("downloading %s: %s", name, formatByteCount(written))
				}
				nextLog = time.Now().Add(defaultDownloadProgressPeriod)
			}
		}
		if err == io.EOF {
			return written, nil
		}
		if err != nil {
			return written, err
		}
	}
}

func formatByteCount(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for value := n / unit; value >= unit; value /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(n)/float64(div), "KMGTPE"[exp])
}
