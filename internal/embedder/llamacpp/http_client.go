package llamacpp

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

const (
	defaultQueryInstruction   = "Retrieve images or text relevant to the user's query."
	defaultPassageInstruction = "Represent this image or text for retrieval."
	defaultMediaMarker        = "<__media__>"
)

type Config struct {
	BaseURL            string
	Dimensions         int
	Model              string
	QueryInstruction   string
	PassageInstruction string
	MediaMarker        string
	HTTPClient         *http.Client
}

type HTTPClient struct {
	baseURL            string
	client             *http.Client
	dimensions         int
	model              string
	queryInstruction   string
	passageInstruction string
	mediaMarker        string
}

type embedHTTPError struct {
	StatusCode int
	Body       string
}

func (e *embedHTTPError) Error() string {
	return fmt.Sprintf("llama.cpp embedder status %d: %s", e.StatusCode, e.Body)
}

func New(cfg Config) (*HTTPClient, error) {
	baseURL := strings.TrimRight(strings.TrimSpace(cfg.BaseURL), "/")
	if baseURL == "" {
		return nil, fmt.Errorf("llama.cpp base URL is required")
	}
	if cfg.Dimensions <= 0 {
		return nil, fmt.Errorf("llama.cpp dimensions must be positive")
	}

	queryInstruction := strings.TrimSpace(cfg.QueryInstruction)
	if queryInstruction == "" {
		queryInstruction = defaultQueryInstruction
	}
	passageInstruction := strings.TrimSpace(cfg.PassageInstruction)
	if passageInstruction == "" {
		passageInstruction = defaultPassageInstruction
	}
	mediaMarker := strings.TrimSpace(cfg.MediaMarker)
	if mediaMarker == "" {
		mediaMarker = defaultMediaMarker
	}

	client := cfg.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: 120 * time.Second}
	}

	return &HTTPClient{
		baseURL:            baseURL,
		client:             client,
		dimensions:         cfg.Dimensions,
		model:              strings.TrimSpace(cfg.Model),
		queryInstruction:   queryInstruction,
		passageInstruction: passageInstruction,
		mediaMarker:        mediaMarker,
	}, nil
}

func (c *HTTPClient) EmbedText(ctx context.Context, text string) ([]float32, error) {
	payload := map[string]any{
		"input":           instructionPrefix(c.queryInstruction, text),
		"encoding_format": "float",
	}
	if c.model != "" {
		payload["model"] = c.model
	}
	return c.embed(ctx, payload)
}

func (c *HTTPClient) EmbedImage(ctx context.Context, path string) ([]float32, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read image file: %w", err)
	}

	payload := map[string]any{
		"input": map[string]any{
			"prompt_string":   instructionPrefix(c.passageInstruction, c.mediaMarker),
			"multimodal_data": []string{base64.StdEncoding.EncodeToString(raw)},
		},
		"encoding_format": "float",
	}
	if c.model != "" {
		payload["model"] = c.model
	}

	return c.embed(ctx, payload)
}

func (c *HTTPClient) embed(ctx context.Context, payload any) ([]float32, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal payload: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request llama.cpp embeddings API: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, &embedHTTPError{StatusCode: resp.StatusCode, Body: strings.TrimSpace(string(raw))}
	}

	var decoded any
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return nil, fmt.Errorf("decode llama.cpp embeddings response: %w", err)
	}

	vec, err := firstEmbedding(decoded)
	if err != nil {
		return nil, err
	}
	if len(vec) != c.dimensions {
		return nil, fmt.Errorf("llama.cpp embedding dimension mismatch: got %d want %d", len(vec), c.dimensions)
	}

	return vec, nil
}

func instructionPrefix(instruction string, content string) string {
	instruction = strings.TrimSpace(instruction)
	content = strings.TrimSpace(content)
	if instruction == "" {
		return content
	}
	if content == "" {
		return instruction
	}
	return instruction + "\n\n" + content
}

func firstEmbedding(decoded any) ([]float32, error) {
	switch body := decoded.(type) {
	case map[string]any:
		if data, ok := body["data"]; ok {
			return firstEmbeddingFromCollection(data)
		}
		if emb, ok := body["embedding"]; ok {
			return parseEmbeddingValue(emb)
		}
	case []any:
		return firstEmbeddingFromCollection(body)
	}
	return nil, fmt.Errorf("llama.cpp embeddings response did not contain an embedding")
}

func firstEmbeddingFromCollection(collection any) ([]float32, error) {
	entries, ok := collection.([]any)
	if !ok || len(entries) == 0 {
		return nil, fmt.Errorf("llama.cpp embeddings response returned no data")
	}
	entry, ok := entries[0].(map[string]any)
	if !ok {
		return nil, fmt.Errorf("llama.cpp embeddings response entry is malformed")
	}
	value, ok := entry["embedding"]
	if !ok {
		return nil, fmt.Errorf("llama.cpp embeddings response entry is missing embedding")
	}
	return parseEmbeddingValue(value)
}

func parseEmbeddingValue(value any) ([]float32, error) {
	values, ok := value.([]any)
	if !ok {
		return nil, fmt.Errorf("llama.cpp embedding payload is malformed")
	}
	if len(values) == 0 {
		return nil, fmt.Errorf("llama.cpp embedding payload is empty")
	}

	if firstNested, ok := values[0].([]any); ok {
		values = firstNested
	}

	vec := make([]float32, len(values))
	for i, v := range values {
		f, ok := v.(float64)
		if !ok {
			return nil, fmt.Errorf("llama.cpp embedding value at index %d is not a number", i)
		}
		vec[i] = float32(f)
	}
	return vec, nil
}
