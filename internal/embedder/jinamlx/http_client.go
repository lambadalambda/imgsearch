package jinamlx

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type HTTPClient struct {
	baseURL string
	client  *http.Client
}

func NewHTTPClient(baseURL string) *HTTPClient {
	return &HTTPClient{
		baseURL: strings.TrimRight(baseURL, "/"),
		client: &http.Client{
			Timeout: 60 * time.Second,
		},
	}
}

func (c *HTTPClient) EmbedText(ctx context.Context, text string) ([]float32, error) {
	return c.embed(ctx, "/embed/text", map[string]string{"text": text})
}

func (c *HTTPClient) EmbedImage(ctx context.Context, path string) ([]float32, error) {
	return c.embed(ctx, "/embed/image", map[string]string{"path": path})
}

type embedResponse struct {
	Embedding []float32 `json:"embedding"`
}

func (c *HTTPClient) embed(ctx context.Context, path string, payload map[string]string) ([]float32, error) {
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
		return nil, fmt.Errorf("embedder status %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
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
