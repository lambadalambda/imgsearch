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
	defaultLlamaNativeModelPath           = "./models/Qwen/Qwen3-VL-Embedding-8B-Q4_K_M.gguf"
	defaultLlamaNativeMMProjPath          = "./models/Qwen/mmproj-Qwen3-VL-Embedding-8B-f16.gguf"
	defaultLlamaNativeDimensions          = 4096
	defaultLlamaNativeModelURL            = "https://huggingface.co/lainsoykaf/Qwen3-VL-Embedding-8B-GGUF/resolve/main/Qwen3-VL-Embedding-8B-Q4_K_M.gguf"
	defaultLlamaNativeMMProjURL           = "https://huggingface.co/lainsoykaf/Qwen3-VL-Embedding-8B-GGUF/resolve/main/mmproj-Qwen3-VL-Embedding-8B-f16.gguf"
	defaultLlamaNativeAnnotatorModelPath  = "./models/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf"
	defaultLlamaNativeAnnotatorMMProjPath = "./models/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive/mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf"
	defaultLlamaNativeAnnotatorModelURL   = "https://huggingface.co/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive/resolve/main/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf"
	defaultLlamaNativeAnnotatorMMProjURL  = "https://huggingface.co/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive/resolve/main/mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf"
	defaultDownloadUserAgent              = "imgsearch/1.0"
	defaultDownloadProgressPeriod         = 5 * time.Second
)

func ensureDefaultLlamaNativeAssets(ctx context.Context, modelPath string, mmprojPath string) (string, string, error) {
	return ensureDefaultAssetPair(
		ctx,
		nil,
		modelPath,
		mmprojPath,
		defaultLlamaNativeModelPath,
		defaultLlamaNativeModelURL,
		defaultLlamaNativeMMProjPath,
		defaultLlamaNativeMMProjURL,
	)
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
	if resolvedPath != defaultPath {
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
	if httpClient == nil {
		httpClient = &http.Client{}
	}
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
	defer func() {
		_ = tmpFile.Close()
		_ = os.Remove(tmpPath)
	}()

	log.Printf("downloading default model asset %s", filepath.Base(destPath))
	if err := copyWithProgress(tmpFile, resp.Body, filepath.Base(destPath), resp.ContentLength); err != nil {
		return fmt.Errorf("write %s: %w", filepath.Base(destPath), err)
	}
	if err := tmpFile.Close(); err != nil {
		return fmt.Errorf("close temporary download file: %w", err)
	}
	if err := os.Rename(tmpPath, destPath); err != nil {
		return fmt.Errorf("move downloaded file into place: %w", err)
	}
	log.Printf("downloaded default model asset %s", filepath.Base(destPath))
	return nil
}

func copyWithProgress(dst io.Writer, src io.Reader, name string, total int64) error {
	buf := make([]byte, 1024*1024)
	var written int64
	nextLog := time.Now().Add(defaultDownloadProgressPeriod)

	for {
		n, err := src.Read(buf)
		if n > 0 {
			if _, writeErr := dst.Write(buf[:n]); writeErr != nil {
				return writeErr
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
			return nil
		}
		if err != nil {
			return err
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
