// Package openaicompat annotates media through any OpenAI-compatible
// chat-completions server with image input (llama-server, Ollama, LM Studio,
// vLLM, OpenAI, OpenRouter, ...). Prompts and parsing come from
// internal/annotation so results match the native backend.
package openaicompat

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
	"strconv"
	"strings"
	"sync"
	"time"

	"imgsearch/internal/annotation"
	"imgsearch/internal/embedder"
)

const (
	defaultTimeout       = 120 * time.Second
	defaultConcurrency   = 2
	defaultImageMaxSide  = 1024
	defaultRetryDelay    = 2 * time.Second
	maxRetryDelay        = 30 * time.Second
	maxTransportAttempts = 3
	maxErrorBodyBytes    = 2048
	maxRawImageBytes     = 20 << 20
	defaultTemperature   = 0.2
)

// ImagePreparer returns encoded image bytes and their MIME type, resized so
// the longest side is at most maxSide. The default reads the file as-is.
type ImagePreparer func(ctx context.Context, path string, maxSide int) ([]byte, string, error)

// Config configures a remote annotator.
type Config struct {
	BaseURL      string
	APIKey       string
	Model        string
	Timeout      time.Duration
	Concurrency  int
	ImageMaxSide int
	PrepareImage ImagePreparer
	HTTPClient   *http.Client
	// RetryBaseDelay is the first backoff after a 429/5xx; tests shorten it.
	RetryBaseDelay time.Duration
	// Temperature defaults to 0.2; use a negative value to request 0.
	Temperature float64
	Logf        func(format string, args ...any)
}

func (c Config) normalized() (Config, error) {
	c.BaseURL = strings.TrimRight(strings.TrimSpace(c.BaseURL), "/")
	c.Model = strings.TrimSpace(c.Model)
	c.APIKey = strings.TrimSpace(c.APIKey)
	if c.BaseURL == "" {
		return c, fmt.Errorf("openai-compatible base URL is required")
	}
	if c.Model == "" {
		return c, fmt.Errorf("openai-compatible model is required")
	}
	if c.Timeout <= 0 {
		c.Timeout = defaultTimeout
	}
	if c.Concurrency < 1 {
		c.Concurrency = defaultConcurrency
	}
	if c.ImageMaxSide <= 0 {
		c.ImageMaxSide = defaultImageMaxSide
	}
	if c.PrepareImage == nil {
		c.PrepareImage = ReadImageFile
	}
	// Timeout only applies to the default client; a caller-supplied
	// HTTPClient keeps its own timeout.
	if c.HTTPClient == nil {
		c.HTTPClient = &http.Client{Timeout: c.Timeout}
	}
	if c.Temperature == 0 {
		c.Temperature = defaultTemperature
	} else if c.Temperature < 0 {
		c.Temperature = 0
	}
	if c.RetryBaseDelay <= 0 {
		c.RetryBaseDelay = defaultRetryDelay
	}
	if c.Logf == nil {
		c.Logf = func(string, ...any) {}
	}
	return c, nil
}

// Annotator implements embedder.ImageAnnotatorWithOptions,
// embedder.VideoFrameAnnotator, and embedder.VideoAnnotator.
type Annotator struct {
	cfg Config
	sem chan struct{}

	mu                     sync.Mutex
	responseFormatRejected bool
}

// New validates cfg and returns a ready annotator. No network call is made.
func New(cfg Config) (*Annotator, error) {
	cfg, err := cfg.normalized()
	if err != nil {
		return nil, err
	}
	return &Annotator{cfg: cfg, sem: make(chan struct{}, cfg.Concurrency)}, nil
}

// Close satisfies app.Closer; there is nothing to release.
func (a *Annotator) Close() error { return nil }

// Model reports the configured model name for status displays.
func (a *Annotator) Model() string { return a.cfg.Model }

// BaseURL reports the configured server for status displays.
func (a *Annotator) BaseURL() string { return a.cfg.BaseURL }

func (a *Annotator) AnnotateImage(ctx context.Context, path string) (embedder.ImageAnnotation, error) {
	return a.AnnotateImageWithOptions(ctx, path, embedder.ImageAnnotationOptions{})
}

func (a *Annotator) AnnotateImageWithOptions(ctx context.Context, path string, opts embedder.ImageAnnotationOptions) (embedder.ImageAnnotation, error) {
	resp, err := a.annotate(ctx, path, opts.ImageMaxSideMultiplier, annotation.ImageRequest(opts.OriginalName, opts.KnownTags))
	if err != nil {
		return embedder.ImageAnnotation{}, err
	}
	return resp.ImageAnnotation(), nil
}

func (a *Annotator) AnnotateVideoFrame(ctx context.Context, path string, opts embedder.ImageAnnotationOptions) (embedder.ImageAnnotation, error) {
	resp, err := a.annotate(ctx, path, opts.ImageMaxSideMultiplier, annotation.VideoFrameRequest(opts.OriginalName, opts.KnownTags))
	if err != nil {
		return embedder.ImageAnnotation{}, err
	}
	return resp.ImageAnnotation(), nil
}

func (a *Annotator) AnnotateVideo(ctx context.Context, input embedder.VideoAnnotationInput) (embedder.VideoAnnotation, error) {
	framePath := strings.TrimSpace(input.RepresentativeFramePath)
	if framePath == "" {
		return embedder.VideoAnnotation{}, fmt.Errorf("representative frame path is required")
	}
	req, err := annotation.VideoRequest(input)
	if err != nil {
		return embedder.VideoAnnotation{}, err
	}
	resp, err := a.annotate(ctx, framePath, input.ImageMaxSideMultiplier, req)
	if err != nil {
		return embedder.VideoAnnotation{}, err
	}
	return resp.VideoAnnotation(), nil
}

// annotate runs the primary request and, if the answer does not parse, one
// compact retry, mirroring the native backend.
func (a *Annotator) annotate(ctx context.Context, imagePath string, maxSideMultiplier int, req annotation.Request) (annotation.Response, error) {
	if err := ctx.Err(); err != nil {
		return annotation.Response{}, err
	}
	maxSide := a.cfg.ImageMaxSide
	if maxSideMultiplier > 1 {
		maxSide *= maxSideMultiplier
	}
	imageBytes, mime, err := a.cfg.PrepareImage(ctx, imagePath, maxSide)
	if err != nil {
		return annotation.Response{}, fmt.Errorf("prepare image for remote annotation: %w", err)
	}
	dataURL := "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(imageBytes)

	raw, err := a.complete(ctx, req.SystemPrompt, req.UserPrompt, dataURL, req.MaxTokens)
	if err != nil {
		return annotation.Response{}, err
	}
	resp, parseErr := annotation.Parse(raw)
	if parseErr == nil {
		return resp, nil
	}
	a.cfg.Logf("remote annotation output did not parse, retrying compact: %v", parseErr)
	retryRaw, err := a.complete(ctx, req.RetrySystemPrompt, req.RetryUserPrompt, dataURL, req.RetryMaxTokens)
	if err != nil {
		return annotation.Response{}, fmt.Errorf("%v; retry failed: %w", parseErr, err)
	}
	resp, err = annotation.Parse(retryRaw)
	if err != nil {
		return annotation.Response{}, fmt.Errorf("%v; retry decode failed: %w", parseErr, err)
	}
	return resp, nil
}

type chatMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

type contentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *imageURL `json:"image_url,omitempty"`
}

type imageURL struct {
	URL string `json:"url"`
}

type chatRequest struct {
	Model          string          `json:"model"`
	Messages       []chatMessage   `json:"messages"`
	MaxTokens      int             `json:"max_tokens"`
	Temperature    float64         `json:"temperature"`
	ResponseFormat *responseFormat `json:"response_format,omitempty"`
}

type responseFormat struct {
	Type string `json:"type"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content json.RawMessage `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Error *apiError `json:"error"`
}

type apiError struct {
	Message string `json:"message"`
}

// complete performs one chat completion with transport retries and the
// response_format fallback. The concurrency slot is deliberately held while
// backing off: a 429 means the server wants less load, so the slot must not
// be handed to another job in the meantime.
func (a *Annotator) complete(ctx context.Context, systemPrompt, userPrompt, imageDataURL string, maxTokens int) (string, error) {
	select {
	case a.sem <- struct{}{}:
		defer func() { <-a.sem }()
	case <-ctx.Done():
		return "", ctx.Err()
	}

	body := chatRequest{
		Model: a.cfg.Model,
		Messages: []chatMessage{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: []contentPart{
				{Type: "text", Text: userPrompt},
				{Type: "image_url", ImageURL: &imageURL{URL: imageDataURL}},
			}},
		},
		MaxTokens:   maxTokens,
		Temperature: a.cfg.Temperature,
	}
	if !a.responseFormatIsRejected() {
		body.ResponseFormat = &responseFormat{Type: "json_object"}
	}

	for attempt := 1; ; attempt++ {
		status, respBody, err := doJSON(ctx, a.cfg, http.MethodPost, "/chat/completions", body)
		if err != nil {
			return "", err
		}
		switch {
		case status == http.StatusOK:
			return extractContent(respBody)
		case status == http.StatusBadRequest && body.ResponseFormat != nil && mentionsResponseFormat(respBody):
			a.markResponseFormatRejected()
			body.ResponseFormat = nil
			attempt-- // the fallback is not a transport retry
			continue
		case (status == http.StatusTooManyRequests || status >= 500) && attempt < maxTransportAttempts:
			delay := a.backoff(attempt, retryAfterFrom(respBody.header))
			a.cfg.Logf("remote annotation server returned %d, retrying in %s (attempt %d/%d)", status, delay, attempt, maxTransportAttempts)
			if err := sleepCtx(ctx, delay); err != nil {
				return "", err
			}
			continue
		default:
			return "", fmt.Errorf("remote annotation server returned HTTP %d: %s", status, summarizeError(respBody.bytes, a.cfg.APIKey))
		}
	}
}

type responseBytes struct {
	bytes  []byte
	header http.Header
}

func doJSON(ctx context.Context, cfg Config, method, path string, payload any) (int, responseBytes, error) {
	var reqBody io.Reader
	if payload != nil {
		encoded, err := json.Marshal(payload)
		if err != nil {
			return 0, responseBytes{}, fmt.Errorf("encode request: %w", err)
		}
		reqBody = bytes.NewReader(encoded)
	}
	req, err := http.NewRequestWithContext(ctx, method, cfg.BaseURL+path, reqBody)
	if err != nil {
		return 0, responseBytes{}, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Accept", "application/json")
	if payload != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
	}
	resp, err := cfg.HTTPClient.Do(req)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return 0, responseBytes{}, ctxErr
		}
		return 0, responseBytes{}, fmt.Errorf("remote annotation request failed: %w", err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return 0, responseBytes{}, ctxErr
		}
		return 0, responseBytes{}, fmt.Errorf("read remote annotation response: %w", err)
	}
	return resp.StatusCode, responseBytes{bytes: data, header: resp.Header}, nil
}

func extractContent(rb responseBytes) (string, error) {
	var parsed chatResponse
	if err := json.Unmarshal(rb.bytes, &parsed); err != nil {
		return "", fmt.Errorf("decode chat completion: %w", err)
	}
	if parsed.Error != nil && parsed.Error.Message != "" {
		return "", fmt.Errorf("remote annotation server error: %s", parsed.Error.Message)
	}
	if len(parsed.Choices) == 0 {
		return "", errors.New("chat completion returned no choices")
	}
	content := parsed.Choices[0].Message.Content
	if len(content) == 0 || string(content) == "null" {
		return "", errors.New("chat completion returned empty content")
	}
	var text string
	if err := json.Unmarshal(content, &text); err == nil {
		return strings.TrimSpace(text), nil
	}
	// Some servers return content as an array of parts.
	var parts []contentPart
	if err := json.Unmarshal(content, &parts); err != nil {
		return "", fmt.Errorf("unexpected chat completion content shape: %s", summarizeError(content, ""))
	}
	var b strings.Builder
	for _, p := range parts {
		if p.Type == "text" {
			b.WriteString(p.Text)
		}
	}
	return strings.TrimSpace(b.String()), nil
}

func mentionsResponseFormat(rb responseBytes) bool {
	return strings.Contains(strings.ToLower(string(rb.bytes)), "response_format")
}

func (a *Annotator) responseFormatIsRejected() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.responseFormatRejected
}

func (a *Annotator) markResponseFormatRejected() {
	a.mu.Lock()
	a.responseFormatRejected = true
	a.mu.Unlock()
	a.cfg.Logf("remote annotation server rejected response_format=json_object; continuing without it")
}

func (a *Annotator) backoff(attempt int, retryAfter time.Duration) time.Duration {
	if retryAfter > 0 {
		if retryAfter > maxRetryDelay {
			return maxRetryDelay
		}
		return retryAfter
	}
	delay := a.cfg.RetryBaseDelay << (attempt - 1)
	if delay > maxRetryDelay {
		delay = maxRetryDelay
	}
	return delay
}

func retryAfterFrom(h http.Header) time.Duration {
	value := strings.TrimSpace(h.Get("Retry-After"))
	if value == "" {
		return 0
	}
	if secs, err := strconv.Atoi(value); err == nil && secs > 0 {
		return time.Duration(secs) * time.Second
	}
	if when, err := http.ParseTime(value); err == nil {
		if d := time.Until(when); d > 0 {
			return d
		}
	}
	return 0
}

func sleepCtx(ctx context.Context, d time.Duration) error {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// summarizeError extracts the server's error message when the body is the
// usual {"error":{"message":...}} envelope, else returns a truncated body.
// The API key is redacted in case a proxy echoes the Authorization header.
func summarizeError(body []byte, apiKey string) string {
	var envelope struct {
		Error *apiError `json:"error"`
	}
	text := strings.TrimSpace(string(body))
	if err := json.Unmarshal(body, &envelope); err == nil && envelope.Error != nil && envelope.Error.Message != "" {
		text = envelope.Error.Message
	} else if len(text) > maxErrorBodyBytes {
		text = text[:maxErrorBodyBytes] + "..."
	}
	if apiKey != "" {
		text = strings.ReplaceAll(text, apiKey, "[redacted]")
	}
	if text == "" {
		return "empty response body"
	}
	return text
}

// ReadImageFile is the default ImagePreparer: it sends the stored bytes
// unchanged (up to 20 MiB) and sniffs the MIME type.
func ReadImageFile(_ context.Context, path string, _ int) ([]byte, string, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, "", err
	}
	if info.Size() > maxRawImageBytes {
		return nil, "", fmt.Errorf("image %s is %d bytes, above the %d byte limit for unresized upload", path, info.Size(), maxRawImageBytes)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, "", err
	}
	mime := http.DetectContentType(data)
	if !strings.HasPrefix(mime, "image/") {
		mime = "image/jpeg"
	}
	return data, mime, nil
}

// ListModels queries GET /models and returns the advertised model IDs.
func ListModels(ctx context.Context, cfg Config) ([]string, error) {
	cfg, err := cfg.normalized()
	if err != nil {
		return nil, err
	}
	status, rb, err := doJSON(ctx, cfg, http.MethodGet, "/models", nil)
	if err != nil {
		return nil, err
	}
	if status != http.StatusOK {
		return nil, fmt.Errorf("remote annotation server returned HTTP %d for /models: %s", status, summarizeError(rb.bytes, cfg.APIKey))
	}
	var parsed struct {
		Data []struct {
			ID     string   `json:"id"`
			Labels []string `json:"labels"`
		} `json:"data"`
	}
	if err := json.Unmarshal(rb.bytes, &parsed); err != nil {
		return nil, fmt.Errorf("decode /models: %w", err)
	}
	// Servers such as Lemonade label every model; keep the ones that can
	// answer a chat completion and put vision-capable ones first, since the
	// annotator sends images. Unlabelled entries pass through unchanged.
	var vision, other []string
	for _, m := range parsed.Data {
		id := strings.TrimSpace(m.ID)
		if id == "" {
			continue
		}
		switch classifyModelLabels(m.Labels) {
		case modelVision:
			vision = append(vision, id)
		case modelChat:
			other = append(other, id)
		}
	}
	return append(vision, other...), nil
}

type modelClass int

const (
	modelChat modelClass = iota
	modelVision
	modelOther
)

// classifyModelLabels maps a server's labels to what the annotator can use.
// No labels means "unknown, assume chat".
func classifyModelLabels(labels []string) modelClass {
	if len(labels) == 0 {
		return modelChat
	}
	chat, vision, excluded := false, false, false
	for _, label := range labels {
		switch strings.ToLower(strings.TrimSpace(label)) {
		case "chat":
			chat = true
		case "vision":
			vision = true
		case "image", "upscaling", "transcription", "realtime-transcription", "embedding", "embeddings", "reranking", "audio", "tts":
			excluded = true
		}
	}
	switch {
	case vision:
		return modelVision
	case chat:
		return modelChat
	case excluded:
		return modelOther
	default:
		return modelChat
	}
}

// TestConnection verifies the server is reachable and knows the configured
// model. Servers without a /models endpoint fall back to a minimal text-only
// chat completion.
func TestConnection(ctx context.Context, cfg Config) error {
	cfg, err := cfg.normalized()
	if err != nil {
		return err
	}
	models, err := ListModels(ctx, cfg)
	if err == nil {
		for _, id := range models {
			if id == cfg.Model {
				return nil
			}
		}
		if len(models) > 0 {
			return fmt.Errorf("model %q not offered by server (available: %s)", cfg.Model, strings.Join(truncateList(models, 8), ", "))
		}
	}
	// No usable model list: try a tiny completion instead.
	status, rb, err := doJSON(ctx, cfg, http.MethodPost, "/chat/completions", chatRequest{
		Model:     cfg.Model,
		Messages:  []chatMessage{{Role: "user", Content: "Reply with the single word: ok"}},
		MaxTokens: 4,
	})
	if err != nil {
		return err
	}
	if status != http.StatusOK {
		return fmt.Errorf("remote annotation server returned HTTP %d: %s", status, summarizeError(rb.bytes, cfg.APIKey))
	}
	if _, err := extractContent(rb); err != nil {
		return err
	}
	return nil
}

func truncateList(items []string, n int) []string {
	if len(items) <= n {
		return items
	}
	return append(append([]string{}, items[:n]...), "...")
}
