package jinamlx

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

type ImageMode string

const (
	ImageModePath  ImageMode = "path"
	ImageModeBytes ImageMode = "bytes"
	ImageModeAuto  ImageMode = "auto"
)

type HTTPClient struct {
	baseURL string
	client  *http.Client

	imageMode ImageMode
	autoMode  ImageMode
	mu        sync.RWMutex
}

func NewHTTPClient(baseURL string) *HTTPClient {
	return NewHTTPClientWithImageMode(baseURL, string(ImageModePath))
}

func NewHTTPClientWithImageMode(baseURL string, imageMode string) *HTTPClient {
	mode := parseImageMode(imageMode)
	return &HTTPClient{
		baseURL: strings.TrimRight(baseURL, "/"),
		client: &http.Client{
			Timeout: 60 * time.Second,
		},
		imageMode: mode,
		autoMode:  ImageModePath,
	}
}

func (c *HTTPClient) EmbedText(ctx context.Context, text string) ([]float32, error) {
	return c.embed(ctx, "/embed/text", map[string]string{"text": text})
}

func (c *HTTPClient) EmbedImage(ctx context.Context, path string) ([]float32, error) {
	configured := c.configuredImageMode()
	switch configured {
	case ImageModeBytes:
		return c.embedImageBytes(ctx, path)
	case ImageModeAuto:
		if c.effectiveAutoMode() == ImageModeBytes {
			return c.embedImageBytes(ctx, path)
		}

		vec, err := c.embedImagePath(ctx, path)
		if err == nil {
			return vec, nil
		}
		if !shouldFallbackToImageBytes(err) {
			return nil, err
		}

		vec, bytesErr := c.embedImageBytes(ctx, path)
		if bytesErr != nil {
			return nil, bytesErr
		}
		c.setAutoMode(ImageModeBytes)
		return vec, nil
	default:
		return c.embedImagePath(ctx, path)
	}
}

type embedResponse struct {
	Embedding []float32 `json:"embedding"`
}

type embedHTTPError struct {
	StatusCode int
	Body       string
}

func (e *embedHTTPError) Error() string {
	return fmt.Sprintf("embedder status %d: %s", e.StatusCode, e.Body)
}

func parseImageMode(v string) ImageMode {
	switch ImageMode(strings.ToLower(strings.TrimSpace(v))) {
	case ImageModeBytes:
		return ImageModeBytes
	case ImageModeAuto:
		return ImageModeAuto
	default:
		return ImageModePath
	}
}

func (c *HTTPClient) configuredImageMode() ImageMode {
	if c == nil {
		return ImageModePath
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.imageMode
}

func (c *HTTPClient) effectiveAutoMode() ImageMode {
	if c == nil {
		return ImageModePath
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.autoMode
}

func (c *HTTPClient) setAutoMode(mode ImageMode) {
	if c == nil || c.imageMode != ImageModeAuto {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.autoMode = mode
}

func shouldFallbackToImageBytes(err error) bool {
	var httpErr *embedHTTPError
	if !errors.As(err, &httpErr) {
		return false
	}
	if httpErr.StatusCode == http.StatusForbidden {
		return true
	}
	if httpErr.StatusCode != http.StatusNotFound {
		return false
	}
	body := strings.ToLower(httpErr.Body)
	return strings.Contains(body, "image path not found") || strings.Contains(body, "outside allowed")
}

func (c *HTTPClient) embedImagePath(ctx context.Context, path string) ([]float32, error) {
	return c.embed(ctx, "/embed/image", map[string]string{"path": path})
}

func (c *HTTPClient) embedImageBytes(ctx context.Context, path string) ([]float32, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read image file: %w", err)
	}
	return c.embed(ctx, "/embed/image-bytes", map[string]string{
		"filename":  filepath.Base(path),
		"image_b64": base64.StdEncoding.EncodeToString(raw),
	})
}

func (c *HTTPClient) embed(ctx context.Context, path string, payload any) ([]float32, error) {
	if c == nil {
		return nil, fmt.Errorf("client is nil")
	}
	if c.baseURL == "" {
		return nil, fmt.Errorf("base url is empty")
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal payload: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request embedder: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return nil, &embedHTTPError{StatusCode: resp.StatusCode, Body: strings.TrimSpace(string(raw))}
	}

	var out embedResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("decode embedder response: %w", err)
	}
	if len(out.Embedding) == 0 {
		return nil, fmt.Errorf("empty embedding")
	}

	return out.Embedding, nil
}
