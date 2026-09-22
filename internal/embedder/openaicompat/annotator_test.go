package openaicompat

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"imgsearch/internal/embedder"
)

type recordedRequest struct {
	Path   string
	Auth   string
	Body   map[string]any
	Header http.Header
}

type fakeServer struct {
	t        *testing.T
	mu       sync.Mutex
	requests []recordedRequest
	// respond decides the reply for the nth chat request (0-based).
	respond func(n int, req map[string]any) (status int, body string)
	// replyHeaders are added to every chat reply.
	replyHeaders http.Header
	models       []string
	modelLabels  map[string][]string
	srv          *httptest.Server
}

func newFakeServer(t *testing.T, respond func(n int, req map[string]any) (int, string)) *fakeServer {
	t.Helper()
	f := &fakeServer{t: t, respond: respond, models: []string{"vision-a", "vision-b"}}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" && r.Method == http.MethodGet {
			f.mu.Lock()
			f.requests = append(f.requests, recordedRequest{Path: r.URL.Path, Auth: r.Header.Get("Authorization"), Header: r.Header.Clone()})
			f.mu.Unlock()
			data := make([]map[string]any, 0, len(f.models))
			for _, m := range f.models {
				entry := map[string]any{"id": m}
				if labels, ok := f.modelLabels[m]; ok {
					entry["labels"] = labels
				}
				data = append(data, entry)
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
			return
		}
		if r.URL.Path != "/v1/chat/completions" || r.Method != http.MethodPost {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		if err := json.Unmarshal(raw, &body); err != nil {
			http.Error(w, "bad json", http.StatusBadRequest)
			return
		}
		f.mu.Lock()
		n := len(f.chatRequestsLocked())
		f.requests = append(f.requests, recordedRequest{Path: r.URL.Path, Auth: r.Header.Get("Authorization"), Body: body, Header: r.Header.Clone()})
		f.mu.Unlock()
		status, reply := f.respond(n, body)
		w.Header().Set("Content-Type", "application/json")
		for k, vs := range f.replyHeaders {
			for _, v := range vs {
				w.Header().Add(k, v)
			}
		}
		w.WriteHeader(status)
		_, _ = io.WriteString(w, reply)
	}))
	t.Cleanup(f.srv.Close)
	return f
}

func (f *fakeServer) chatRequestsLocked() []recordedRequest {
	out := make([]recordedRequest, 0, len(f.requests))
	for _, r := range f.requests {
		if r.Path == "/v1/chat/completions" {
			out = append(out, r)
		}
	}
	return out
}

func (f *fakeServer) chatRequests() []recordedRequest {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.chatRequestsLocked()
}

func chatReply(content string) string {
	b, _ := json.Marshal(map[string]any{
		"choices": []map[string]any{{"message": map[string]any{"role": "assistant", "content": content}}},
	})
	return string(b)
}

const goodImageJSON = `{"title":"Cat","summary":"A cat.","full_description":"A tabby cat on a sofa.","tags":["Cat","sofa"],"is_nsfw":false}`

func writeTempImage(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "cat.jpg")
	if err := os.WriteFile(path, []byte("\xff\xd8\xff\xe0fakejpegbytes"), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func newAnnotator(t *testing.T, f *fakeServer, mutate func(*Config)) *Annotator {
	t.Helper()
	cfg := Config{BaseURL: f.srv.URL + "/v1", APIKey: "sk-test", Model: "vision-a", Timeout: 5 * time.Second, Concurrency: 2}
	if mutate != nil {
		mutate(&cfg)
	}
	a, err := New(cfg)
	if err != nil {
		t.Fatalf("new annotator: %v", err)
	}
	return a
}

func TestAnnotateImageSendsChatCompletionWithImagePart(t *testing.T) {
	f := newFakeServer(t, func(int, map[string]any) (int, string) { return 200, chatReply(goodImageJSON) })
	a := newAnnotator(t, f, nil)

	got, err := a.AnnotateImageWithOptions(context.Background(), writeTempImage(t), embedder.ImageAnnotationOptions{OriginalName: "kid a album cover.jpg"})
	if err != nil {
		t.Fatalf("annotate: %v", err)
	}
	if got.Title != "Cat" || got.Description != "A tabby cat on a sofa." || len(got.Tags) != 2 || got.Tags[0] != "cat" {
		t.Fatalf("unexpected annotation: %+v", got)
	}

	reqs := f.chatRequests()
	if len(reqs) != 1 {
		t.Fatalf("expected 1 chat request, got %d", len(reqs))
	}
	r := reqs[0]
	if r.Auth != "Bearer sk-test" {
		t.Fatalf("auth header: %q", r.Auth)
	}
	if r.Body["model"] != "vision-a" {
		t.Fatalf("model: %v", r.Body["model"])
	}
	if r.Body["max_tokens"].(float64) != 1024 {
		t.Fatalf("max_tokens: %v", r.Body["max_tokens"])
	}
	if rf, ok := r.Body["response_format"].(map[string]any); !ok || rf["type"] != "json_object" {
		t.Fatalf("expected json_object response_format, got %v", r.Body["response_format"])
	}
	msgs := r.Body["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("expected system+user messages, got %d", len(msgs))
	}
	sys := msgs[0].(map[string]any)
	if sys["role"] != "system" || !strings.Contains(sys["content"].(string), "high-recall search") {
		t.Fatalf("unexpected system message: %v", sys)
	}
	user := msgs[1].(map[string]any)
	parts := user["content"].([]any)
	if len(parts) != 2 {
		t.Fatalf("expected text+image parts, got %d", len(parts))
	}
	text := parts[0].(map[string]any)
	if text["type"] != "text" || !strings.Contains(text["text"].(string), "kid a album cover") {
		t.Fatalf("unexpected text part: %v", text)
	}
	img := parts[1].(map[string]any)
	url := img["image_url"].(map[string]any)["url"].(string)
	if img["type"] != "image_url" || !strings.HasPrefix(url, "data:image/jpeg;base64,") {
		t.Fatalf("unexpected image part: %v", img)
	}
}

func TestAnnotateImageUsesInjectedImagePreparer(t *testing.T) {
	f := newFakeServer(t, func(int, map[string]any) (int, string) { return 200, chatReply(goodImageJSON) })
	var gotMaxSide int
	a := newAnnotator(t, f, func(c *Config) {
		c.ImageMaxSide = 640
		c.PrepareImage = func(_ context.Context, path string, maxSide int) ([]byte, string, error) {
			gotMaxSide = maxSide
			return []byte("png-bytes"), "image/png", nil
		}
	})
	if _, err := a.AnnotateImageWithOptions(context.Background(), writeTempImage(t), embedder.ImageAnnotationOptions{ImageMaxSideMultiplier: 2}); err != nil {
		t.Fatal(err)
	}
	if gotMaxSide != 1280 {
		t.Fatalf("expected max side 640*2, got %d", gotMaxSide)
	}
	url := imageURLFromRequest(t, f.chatRequests()[0])
	if !strings.HasPrefix(url, "data:image/png;base64,cG5nLWJ5dGVz") {
		t.Fatalf("expected prepared png bytes, got %q", url)
	}
}

func imageURLFromRequest(t *testing.T, r recordedRequest) string {
	t.Helper()
	parts := r.Body["messages"].([]any)[1].(map[string]any)["content"].([]any)
	for _, p := range parts {
		m := p.(map[string]any)
		if m["type"] == "image_url" {
			return m["image_url"].(map[string]any)["url"].(string)
		}
	}
	t.Fatal("no image part")
	return ""
}

func TestAnnotateImageRetriesWithCompactPromptOnUnparsableOutput(t *testing.T) {
	f := newFakeServer(t, func(n int, _ map[string]any) (int, string) {
		if n == 0 {
			return 200, chatReply("I cannot produce JSON right now.")
		}
		return 200, chatReply(goodImageJSON)
	})
	a := newAnnotator(t, f, nil)
	got, err := a.AnnotateImage(context.Background(), writeTempImage(t))
	if err != nil {
		t.Fatalf("expected retry to succeed: %v", err)
	}
	if got.Title != "Cat" {
		t.Fatalf("unexpected: %+v", got)
	}
	reqs := f.chatRequests()
	if len(reqs) != 2 {
		t.Fatalf("expected 2 requests, got %d", len(reqs))
	}
	retryText := reqs[1].Body["messages"].([]any)[1].(map[string]any)["content"].([]any)[0].(map[string]any)["text"].(string)
	if !strings.Contains(retryText, "Retry with compact output") {
		t.Fatalf("expected compact retry prompt, got %q", retryText)
	}
	if reqs[1].Body["max_tokens"].(float64) != 512 {
		t.Fatalf("expected retry token budget, got %v", reqs[1].Body["max_tokens"])
	}
}

func TestAnnotateImageFailsAfterRetryStillUnparsable(t *testing.T) {
	f := newFakeServer(t, func(int, map[string]any) (int, string) { return 200, chatReply("nope") })
	a := newAnnotator(t, f, nil)
	if _, err := a.AnnotateImage(context.Background(), writeTempImage(t)); err == nil {
		t.Fatal("expected error")
	}
	if n := len(f.chatRequests()); n != 2 {
		t.Fatalf("expected exactly 2 attempts, got %d", n)
	}
}

func TestResponseFormatFallsBackWhenServerRejectsIt(t *testing.T) {
	f := newFakeServer(t, func(_ int, req map[string]any) (int, string) {
		if _, has := req["response_format"]; has {
			return 400, `{"error":{"message":"response_format is not supported"}}`
		}
		return 200, chatReply(goodImageJSON)
	})
	a := newAnnotator(t, f, nil)
	if _, err := a.AnnotateImage(context.Background(), writeTempImage(t)); err != nil {
		t.Fatalf("expected fallback to succeed: %v", err)
	}
	if _, err := a.AnnotateImage(context.Background(), writeTempImage(t)); err != nil {
		t.Fatal(err)
	}
	reqs := f.chatRequests()
	if len(reqs) != 3 {
		t.Fatalf("expected 3 requests (fail, retry without, remembered), got %d", len(reqs))
	}
	if _, has := reqs[2].Body["response_format"]; has {
		t.Fatal("expected response_format to be remembered as unsupported")
	}
}

func TestRetriesOn429HonoringRetryAfter(t *testing.T) {
	var n atomic.Int32
	f := newFakeServer(t, func(int, map[string]any) (int, string) {
		if n.Add(1) == 1 {
			return 429, `{"error":{"message":"slow down"}}`
		}
		return 200, chatReply(goodImageJSON)
	})
	f.replyHeaders = http.Header{"Retry-After": []string{"1"}}
	// Base delay is tiny so a measured wait of >= 1s proves Retry-After won.
	a := newAnnotator(t, f, func(c *Config) { c.RetryBaseDelay = time.Millisecond })
	started := time.Now()
	if _, err := a.AnnotateImage(context.Background(), writeTempImage(t)); err != nil {
		t.Fatal(err)
	}
	if len(f.chatRequests()) != 2 {
		t.Fatalf("expected retry after 429, got %d requests", len(f.chatRequests()))
	}
	if elapsed := time.Since(started); elapsed < time.Second {
		t.Fatalf("expected Retry-After of 1s to be honoured, waited only %s", elapsed)
	}
}

func TestRetryAfterParsingAndCap(t *testing.T) {
	a := &Annotator{cfg: Config{RetryBaseDelay: time.Second}}
	if got := a.backoff(1, 0); got != time.Second {
		t.Fatalf("attempt 1 backoff: %s", got)
	}
	if got := a.backoff(2, 0); got != 2*time.Second {
		t.Fatalf("attempt 2 backoff: %s", got)
	}
	if got := a.backoff(1, 5*time.Minute); got != maxRetryDelay {
		t.Fatalf("expected Retry-After capped at %s, got %s", maxRetryDelay, got)
	}
	if got := retryAfterFrom(http.Header{"Retry-After": []string{"7"}}); got != 7*time.Second {
		t.Fatalf("seconds form: %s", got)
	}
	future := time.Now().Add(3 * time.Second).UTC().Format(http.TimeFormat)
	if got := retryAfterFrom(http.Header{"Retry-After": []string{future}}); got <= 0 || got > 3*time.Second {
		t.Fatalf("http-date form: %s", got)
	}
	if got := retryAfterFrom(http.Header{"Retry-After": []string{"garbage"}}); got != 0 {
		t.Fatalf("garbage should yield 0, got %s", got)
	}
}

func TestResponseFormatFallbackDoesNotConsumeTransportAttempts(t *testing.T) {
	f := newFakeServer(t, func(n int, req map[string]any) (int, string) {
		if _, has := req["response_format"]; has {
			return 400, `{"error":{"message":"response_format unsupported"}}`
		}
		if n < 1+maxTransportAttempts-1 {
			return 503, `{"error":{"message":"warming up"}}`
		}
		return 200, chatReply(goodImageJSON)
	})
	a := newAnnotator(t, f, func(c *Config) { c.RetryBaseDelay = time.Millisecond })
	if _, err := a.AnnotateImage(context.Background(), writeTempImage(t)); err != nil {
		t.Fatalf("expected all transport attempts to remain after fallback: %v", err)
	}
	if n := len(f.chatRequests()); n != 1+maxTransportAttempts {
		t.Fatalf("expected %d requests, got %d", 1+maxTransportAttempts, n)
	}
}

func TestPlain400IsNotTreatedAsResponseFormatFallback(t *testing.T) {
	f := newFakeServer(t, func(int, map[string]any) (int, string) { return 400, `{"error":{"message":"image too large"}}` })
	a := newAnnotator(t, f, nil)
	_, err := a.AnnotateImage(context.Background(), writeTempImage(t))
	if err == nil || !strings.Contains(err.Error(), "image too large") {
		t.Fatalf("expected surfaced 400 error, got %v", err)
	}
	if n := len(f.chatRequests()); n != 1 {
		t.Fatalf("expected no retry on plain 400, got %d", n)
	}
}

func TestExtractContentHandlesArrayPartsAndEmpty(t *testing.T) {
	arr := `{"choices":[{"message":{"content":[{"type":"text","text":"{\"title\":\"A\","},{"type":"text","text":"\"full_description\":\"b\",\"tags\":[]}"}]}}]}`
	got, err := extractContent(responseBytes{bytes: []byte(arr)})
	if err != nil || got != `{"title":"A","full_description":"b","tags":[]}` {
		t.Fatalf("array parts: got=%q err=%v", got, err)
	}
	if _, err := extractContent(responseBytes{bytes: []byte(`{"choices":[{"message":{"content":null}}]}`)}); err == nil {
		t.Fatal("expected error for null content")
	}
	if _, err := extractContent(responseBytes{bytes: []byte(`{"choices":[]}`)}); err == nil {
		t.Fatal("expected error for no choices")
	}
}

func TestSummarizeErrorRedactsAPIKey(t *testing.T) {
	got := summarizeError([]byte(`{"error":{"message":"bad token sk-test in Authorization: Bearer sk-test"}}`), "sk-test")
	if strings.Contains(got, "sk-test") || !strings.Contains(got, "[redacted]") {
		t.Fatalf("expected key redacted, got %q", got)
	}
}

func TestReadImageFileRejectsOversizedFiles(t *testing.T) {
	path := filepath.Join(t.TempDir(), "big.bin")
	fh, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := fh.Truncate(maxRawImageBytes + 1); err != nil {
		t.Fatal(err)
	}
	_ = fh.Close()
	if _, _, err := ReadImageFile(context.Background(), path, 0); err == nil {
		t.Fatal("expected size error")
	}
}

func TestGivesUpAfterMaxTransportAttempts(t *testing.T) {
	f := newFakeServer(t, func(int, map[string]any) (int, string) { return 503, `{"error":{"message":"down"}}` })
	a := newAnnotator(t, f, func(c *Config) { c.RetryBaseDelay = time.Millisecond })
	_, err := a.AnnotateImage(context.Background(), writeTempImage(t))
	if err == nil || !strings.Contains(err.Error(), "503") {
		t.Fatalf("expected 503 error, got %v", err)
	}
	if n := len(f.chatRequests()); n != maxTransportAttempts {
		t.Fatalf("expected %d attempts, got %d", maxTransportAttempts, n)
	}
}

func TestErrorsDoNotLeakAPIKey(t *testing.T) {
	f := newFakeServer(t, func(int, map[string]any) (int, string) { return 401, `{"error":{"message":"bad key"}}` })
	a := newAnnotator(t, f, nil)
	_, err := a.AnnotateImage(context.Background(), writeTempImage(t))
	if err == nil || strings.Contains(err.Error(), "sk-test") {
		t.Fatalf("error must not contain the key: %v", err)
	}
	if len(f.chatRequests()) != 1 {
		t.Fatal("401 must not be retried")
	}
}

func TestContextCancellationAbortsRequest(t *testing.T) {
	release := make(chan struct{})
	f := newFakeServer(t, func(int, map[string]any) (int, string) { <-release; return 200, chatReply(goodImageJSON) })
	t.Cleanup(func() { close(release) })
	a := newAnnotator(t, f, nil)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { _, err := a.AnnotateImage(ctx, writeTempImage(t)); done <- err }()
	time.Sleep(20 * time.Millisecond)
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("expected context.Canceled, got %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("annotate did not return after cancel")
	}
}

func TestConcurrencyLimitIsEnforced(t *testing.T) {
	var inFlight, maxSeen atomic.Int32
	f := newFakeServer(t, func(int, map[string]any) (int, string) {
		cur := inFlight.Add(1)
		for {
			m := maxSeen.Load()
			if cur <= m || maxSeen.CompareAndSwap(m, cur) {
				break
			}
		}
		time.Sleep(15 * time.Millisecond)
		inFlight.Add(-1)
		return 200, chatReply(goodImageJSON)
	})
	a := newAnnotator(t, f, func(c *Config) { c.Concurrency = 2 })
	var wg sync.WaitGroup
	for i := 0; i < 6; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); _, _ = a.AnnotateImage(context.Background(), writeTempImage(t)) }()
	}
	wg.Wait()
	if maxSeen.Load() > 2 || maxSeen.Load() < 2 {
		t.Fatalf("expected exactly up to 2 concurrent requests in flight, saw max %d", maxSeen.Load())
	}
}

func TestAnnotateVideoFrameUsesCompactPrompt(t *testing.T) {
	f := newFakeServer(t, func(int, map[string]any) (int, string) {
		return 200, chatReply(`{"description":"a singer on stage","tags":["concert"],"is_nsfw":false}`)
	})
	a := newAnnotator(t, f, nil)
	got, err := a.AnnotateVideoFrame(context.Background(), writeTempImage(t), embedder.ImageAnnotationOptions{OriginalName: "concert stage clip.mp4"})
	if err != nil {
		t.Fatal(err)
	}
	if got.Description != "a singer on stage" || got.Title == "" {
		t.Fatalf("unexpected frame annotation: %+v", got)
	}
	r := f.chatRequests()[0]
	if r.Body["max_tokens"].(float64) != 320 {
		t.Fatalf("expected frame token budget, got %v", r.Body["max_tokens"])
	}
	if !strings.Contains(r.Body["messages"].([]any)[0].(map[string]any)["content"].(string), "one sampled frame") {
		t.Fatal("expected frame system prompt")
	}
}

func TestAnnotateVideoSendsFrameEvidenceAndRepresentativeImage(t *testing.T) {
	f := newFakeServer(t, func(int, map[string]any) (int, string) {
		return 200, chatReply(`{"title":"Concert","summary":"s","full_description":"d","tags":["concert"],"is_nsfw":true}`)
	})
	a := newAnnotator(t, f, nil)
	got, err := a.AnnotateVideo(context.Background(), embedder.VideoAnnotationInput{
		OriginalName:            "party clip live.mp4",
		DurationMS:              12000,
		RepresentativeFramePath: writeTempImage(t),
		TranscriptText:          "hello from the stage",
		Frames:                  []embedder.VideoFrameAnnotation{{FrameIndex: 0, Description: "a singer on stage", Tags: []string{"concert"}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.Title != "Concert" || !got.IsNSFW || got.Tags[len(got.Tags)-1] != "nsfw" {
		t.Fatalf("unexpected video annotation: %+v", got)
	}
	r := f.chatRequests()[0]
	text := r.Body["messages"].([]any)[1].(map[string]any)["content"].([]any)[0].(map[string]any)["text"].(string)
	if !strings.Contains(text, "Sampled frame annotations") || !strings.Contains(text, "hello from the stage") {
		t.Fatalf("expected frame evidence and transcript in prompt: %q", text)
	}
	if !strings.HasPrefix(imageURLFromRequest(t, r), "data:image/") {
		t.Fatal("expected representative frame image part")
	}
	if _, err := a.AnnotateVideo(context.Background(), embedder.VideoAnnotationInput{RepresentativeFramePath: writeTempImage(t)}); err == nil {
		t.Fatal("expected error without frames")
	}
}

func TestNewValidatesConfig(t *testing.T) {
	if _, err := New(Config{Model: "m"}); err == nil {
		t.Fatal("expected error without base url")
	}
	if _, err := New(Config{BaseURL: "http://x"}); err == nil {
		t.Fatal("expected error without model")
	}
	a, err := New(Config{BaseURL: "http://x/v1/", Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	if a.cfg.Concurrency < 1 || a.cfg.Timeout <= 0 || a.cfg.ImageMaxSide <= 0 {
		t.Fatalf("expected defaults filled: %+v", a.cfg)
	}
	if a.cfg.BaseURL != "http://x/v1" {
		t.Fatalf("expected trailing slash trimmed, got %q", a.cfg.BaseURL)
	}
}

func TestListModelsAndTestConnection(t *testing.T) {
	f := newFakeServer(t, func(int, map[string]any) (int, string) { return 200, chatReply(goodImageJSON) })
	cfg := Config{BaseURL: f.srv.URL + "/v1", APIKey: "sk-test", Model: "vision-a", Timeout: time.Second}
	models, err := ListModels(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 2 || models[0] != "vision-a" {
		t.Fatalf("unexpected models: %v", models)
	}
	if err := TestConnection(context.Background(), cfg); err != nil {
		t.Fatalf("expected connection test to pass: %v", err)
	}
	f.mu.Lock()
	auth := f.requests[0].Auth
	f.mu.Unlock()
	if auth != "Bearer sk-test" {
		t.Fatalf("models request should carry auth, got %q", auth)
	}

	missing := cfg
	missing.Model = "not-there"
	if err := TestConnection(context.Background(), missing); err == nil || !strings.Contains(err.Error(), "not-there") {
		t.Fatalf("expected missing model error, got %v", err)
	}

	down := cfg
	down.BaseURL = "http://127.0.0.1:1/v1"
	if err := TestConnection(context.Background(), down); err == nil {
		t.Fatal("expected connection error")
	}
}

func TestTestConnectionToleratesServersWithoutModelsEndpoint(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/chat/completions" {
			_, _ = io.WriteString(w, chatReply("ok"))
			return
		}
		http.Error(w, "nope", http.StatusNotFound)
	}))
	t.Cleanup(srv.Close)
	cfg := Config{BaseURL: srv.URL + "/v1", Model: "m", Timeout: time.Second}
	if err := TestConnection(context.Background(), cfg); err != nil {
		t.Fatalf("expected chat fallback to succeed: %v", err)
	}
}

// Lemonade labels every model; only chat-capable ones are offered, with
// vision models first so the auto-filled default can see images
// (meta/issues/120).
func TestListModelsFiltersLabelledNonChatModels(t *testing.T) {
	f := newFakeServer(t, func(int, map[string]any) (int, string) { return 200, chatReply(goodImageJSON) })
	f.mu.Lock()
	f.models = []string{"Dark-Beast-Krea2", "Whisper-Large", "chat-only", "vision-a", "plain", "LSDIR-4x"}
	f.modelLabels = map[string][]string{
		"Dark-Beast-Krea2": {"custom", "image"},
		"Whisper-Large":    {"transcription", "realtime-transcription"},
		"chat-only":        {"chat", "reasoning"},
		"vision-a":         {"chat", "vision", "tool-calling"},
		"LSDIR-4x":         {"upscaling", "image"},
	}
	f.mu.Unlock()
	cfg := Config{BaseURL: f.srv.URL + "/v1", APIKey: "sk-test", Model: "vision-a", Timeout: time.Second}
	models, err := ListModels(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(models, ",") != "vision-a,chat-only,plain" {
		t.Fatalf("unexpected models: %v", models)
	}
}
