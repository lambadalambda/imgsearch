package integration

import (
	"bytes"
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"io"
	"math"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"sync"
	"testing"
	"time"

	_ "github.com/mattn/go-sqlite3"

	"imgsearch/internal/db"
	"imgsearch/internal/embedder/jinamlx"
	"imgsearch/internal/search"
	"imgsearch/internal/upload"
	"imgsearch/internal/vectorindex"
	"imgsearch/internal/worker"
)

func TestAPIEndToEndWithJinaSidecar(t *testing.T) {
	if os.Getenv("RUN_JINA_MLX_INTEGRATION") != "1" {
		t.Skip("set RUN_JINA_MLX_INTEGRATION=1 and run `mise run jina-serve`, `mise run jina-torch-serve`, or `mise run qwen3-serve` to enable")
	}

	baseURL := os.Getenv("JINA_MLX_URL")
	if baseURL == "" {
		baseURL = "http://127.0.0.1:9009"
	}
	if err := waitForSidecar(baseURL, 90*time.Second); err != nil {
		t.Fatalf("sidecar not ready at %s: %v", baseURL, err)
	}

	repoRoot := findRepoRoot(t)
	dataDir := filepath.Join(repoRoot, "data")
	if err := os.MkdirAll(filepath.Join(dataDir, "images"), 0o755); err != nil {
		t.Fatalf("mkdir data/images: %v", err)
	}

	dbConn, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	t.Cleanup(func() { _ = dbConn.Close() })

	if err := db.RunMigrations(context.Background(), dbConn); err != nil {
		t.Fatalf("migrate db: %v", err)
	}

	modelSpec := embeddingModelSpecFromEnv(t)
	modelID, err := db.EnsureEmbeddingModel(context.Background(), dbConn, modelSpec)
	if err != nil {
		t.Fatalf("ensure model: %v", err)
	}

	embed := jinamlx.NewHTTPClientWithImageMode(baseURL, string(jinamlx.ImageModeAuto))
	index := newMemoryIndex()

	uploadSvc := &upload.Service{DB: dbConn, DataDir: dataDir, ModelID: modelID}
	queue := &worker.Queue{
		DB:            dbConn,
		DataDir:       dataDir,
		LeaseDuration: 30 * time.Second,
		Embedder:      embed,
		Index:         index,
	}

	searchHandler := search.NewHandler(&search.Handler{
		DB:       dbConn,
		ModelID:  modelID,
		DataDir:  dataDir,
		Embedder: embed,
		Index:    index,
	})

	mux := http.NewServeMux()
	mux.Handle("/api/upload", upload.NewHandler(uploadSvc))
	mux.Handle("/api/search/", searchHandler)
	server := httptest.NewServer(mux)
	defer server.Close()

	type fixture struct {
		Name string
		Path string
	}
	fixtures := []fixture{
		{Name: "cat_1.jpg", Path: filepath.Join(repoRoot, "fixtures", "images", "cat_1.jpg")},
		{Name: "dog_1.jpg", Path: filepath.Join(repoRoot, "fixtures", "images", "dog_1.jpg")},
		{Name: "woman_2.jpg", Path: filepath.Join(repoRoot, "fixtures", "images", "woman_2.jpg")},
		{Name: "woman_office.jpg", Path: filepath.Join(repoRoot, "fixtures", "images", "woman_office.jpg")},
	}

	uploadedByName := make(map[string]upload.UploadResponse, len(fixtures))
	for _, fx := range fixtures {
		raw, err := os.ReadFile(fx.Path)
		if err != nil {
			t.Fatalf("read fixture %s: %v", fx.Name, err)
		}

		digest := sha256.Sum256(raw)
		sha := hex.EncodeToString(digest[:])
		target := filepath.Join(dataDir, "images", sha)
		_, statErr := os.Stat(target)
		existedBefore := statErr == nil

		resp := uploadImage(t, server.Client(), server.URL, fx.Name, raw)
		uploadedByName[fx.Name] = resp

		if !existedBefore {
			path := target
			t.Cleanup(func() { _ = os.Remove(path) })
		}
	}

	drainQueue(t, queue)

	assertJobStates(t, dbConn, len(fixtures))

	textResp := searchText(t, server.Client(), server.URL, "dog", 3)
	if len(textResp.Results) == 0 {
		t.Fatal("expected text search results")
	}
	if textResp.Results[0].OriginalName != "dog_1.jpg" {
		t.Fatalf("expected top text result dog_1.jpg, got %s", textResp.Results[0].OriginalName)
	}

	womanID := uploadedByName["woman_2.jpg"].ImageID
	simResp := searchSimilar(t, server.Client(), server.URL, womanID, 2)
	if len(simResp.Results) == 0 {
		t.Fatal("expected similar search results")
	}
	if simResp.Results[0].OriginalName != "woman_office.jpg" {
		t.Fatalf("expected top similar result woman_office.jpg, got %s", simResp.Results[0].OriginalName)
	}
}

func embeddingModelSpecFromEnv(t *testing.T) db.EmbeddingModelSpec {
	t.Helper()

	name := envStringOrDefault("JINA_MLX_MODEL_NAME", "jina-embeddings-v4")
	version := envStringOrDefault("JINA_MLX_MODEL_VERSION", "mlx-8bit")
	dimensions := envIntOrDefault(t, "JINA_MLX_MODEL_DIMS", 2048)

	return db.EmbeddingModelSpec{
		Name:       name,
		Version:    version,
		Dimensions: dimensions,
		Metric:     "cosine",
		Normalized: true,
	}
}

func envStringOrDefault(key string, fallback string) string {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	return v
}

func envIntOrDefault(t *testing.T, key string, fallback int) int {
	t.Helper()
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		t.Fatalf("parse %s as int: %v", key, err)
	}
	return n
}

func uploadImage(t *testing.T, client *http.Client, baseURL string, filename string, content []byte) upload.UploadResponse {
	t.Helper()

	body := &bytes.Buffer{}
	mw := multipart.NewWriter(body)
	fw, err := mw.CreateFormFile("file", filename)
	if err != nil {
		t.Fatalf("create form file: %v", err)
	}
	if _, err := fw.Write(content); err != nil {
		t.Fatalf("write form content: %v", err)
	}
	if err := mw.Close(); err != nil {
		t.Fatalf("close multipart: %v", err)
	}

	req, err := http.NewRequest(http.MethodPost, baseURL+"/api/upload", body)
	if err != nil {
		t.Fatalf("new upload request: %v", err)
	}
	req.Header.Set("Content-Type", mw.FormDataContentType())

	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("upload request failed: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusCreated && resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		t.Fatalf("upload status=%d body=%s", resp.StatusCode, string(raw))
	}

	var out upload.UploadResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode upload response: %v", err)
	}
	if out.ImageID <= 0 || out.SHA256 == "" {
		t.Fatalf("invalid upload response: %+v", out)
	}

	return out
}

func searchText(t *testing.T, client *http.Client, baseURL, q string, limit int) search.SearchResponse {
	t.Helper()

	params := url.Values{}
	params.Set("q", q)
	params.Set("limit", strconv.Itoa(limit))
	req, err := http.NewRequest(http.MethodGet, baseURL+"/api/search/text?"+params.Encode(), nil)
	if err != nil {
		t.Fatalf("new text search request: %v", err)
	}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("text search request failed: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		t.Fatalf("text search status=%d body=%s", resp.StatusCode, string(raw))
	}

	var out search.SearchResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode text search response: %v", err)
	}
	return out
}

func searchSimilar(t *testing.T, client *http.Client, baseURL string, imageID int64, limit int) search.SearchResponse {
	t.Helper()

	params := url.Values{}
	params.Set("image_id", strconv.FormatInt(imageID, 10))
	params.Set("limit", strconv.Itoa(limit))
	req, err := http.NewRequest(http.MethodGet, baseURL+"/api/search/similar?"+params.Encode(), nil)
	if err != nil {
		t.Fatalf("new similar search request: %v", err)
	}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("similar search request failed: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		t.Fatalf("similar search status=%d body=%s", resp.StatusCode, string(raw))
	}

	var out search.SearchResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode similar search response: %v", err)
	}
	return out
}

func drainQueue(t *testing.T, q *worker.Queue) {
	t.Helper()

	for i := 0; i < 50; i++ {
		processed, err := q.ProcessOne(context.Background(), "api-itest")
		if err != nil {
			t.Fatalf("process queue job: %v", err)
		}
		if !processed {
			return
		}
	}
	t.Fatal("queue did not drain after 50 iterations")
}

func assertJobStates(t *testing.T, dbConn *sql.DB, expectedDone int) {
	t.Helper()

	var done int
	if err := dbConn.QueryRow(`SELECT COUNT(*) FROM index_jobs WHERE state = 'done'`).Scan(&done); err != nil {
		t.Fatalf("count done jobs: %v", err)
	}
	if done != expectedDone {
		t.Fatalf("expected %d done jobs, got %d", expectedDone, done)
	}

	var bad int
	if err := dbConn.QueryRow(`SELECT COUNT(*) FROM index_jobs WHERE state IN ('pending', 'leased', 'failed')`).Scan(&bad); err != nil {
		t.Fatalf("count non-done jobs: %v", err)
	}
	if bad != 0 {
		t.Fatalf("expected no non-done jobs, got %d", bad)
	}
}

func waitForSidecar(baseURL string, timeout time.Duration) error {
	client := &http.Client{Timeout: 2 * time.Second}
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		resp, err := client.Get(baseURL + "/healthz")
		if err == nil {
			_, _ = io.Copy(io.Discard, resp.Body)
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
		time.Sleep(500 * time.Millisecond)
	}
	return context.DeadlineExceeded
}

func findRepoRoot(t *testing.T) string {
	t.Helper()

	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("cannot determine caller path")
	}
	dir := filepath.Dir(file)
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("could not find repository root")
		}
		dir = parent
	}
}

type memoryIndex struct {
	mu   sync.Mutex
	data map[int64]map[int64][]float32
}

func newMemoryIndex() *memoryIndex {
	return &memoryIndex{data: make(map[int64]map[int64][]float32)}
}

func (m *memoryIndex) Upsert(_ context.Context, imageID int64, modelID int64, vec []float32) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	bucket, ok := m.data[modelID]
	if !ok {
		bucket = make(map[int64][]float32)
		m.data[modelID] = bucket
	}
	cp := make([]float32, len(vec))
	copy(cp, vec)
	bucket[imageID] = cp
	return nil
}

func (m *memoryIndex) Delete(_ context.Context, imageID int64, modelID int64) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if bucket, ok := m.data[modelID]; ok {
		delete(bucket, imageID)
	}
	return nil
}

func (m *memoryIndex) Search(_ context.Context, modelID int64, query []float32, limit int) ([]vectorindex.SearchHit, error) {
	m.mu.Lock()
	bucket := m.data[modelID]
	hits := make([]vectorindex.SearchHit, 0, len(bucket))
	for imageID, vec := range bucket {
		sim := cosine(query, vec)
		hits = append(hits, vectorindex.SearchHit{
			ImageID:  imageID,
			ModelID:  modelID,
			Distance: 1 - sim,
		})
	}
	m.mu.Unlock()

	sort.Slice(hits, func(i, j int) bool { return hits[i].Distance < hits[j].Distance })
	if limit > 0 && len(hits) > limit {
		hits = hits[:limit]
	}
	return hits, nil
}

func (m *memoryIndex) SearchByImageID(ctx context.Context, modelID int64, imageID int64, limit int) ([]vectorindex.SearchHit, error) {
	m.mu.Lock()
	bucket := m.data[modelID]
	query, ok := bucket[imageID]
	m.mu.Unlock()
	if !ok {
		return nil, vectorindex.ErrNotFound
	}

	hits, err := m.Search(ctx, modelID, query, 0)
	if err != nil {
		return nil, err
	}
	out := make([]vectorindex.SearchHit, 0, len(hits))
	for _, h := range hits {
		if h.ImageID == imageID {
			continue
		}
		out = append(out, h)
		if limit > 0 && len(out) >= limit {
			break
		}
	}
	return out, nil
}

func cosine(a, b []float32) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	if n == 0 {
		return 0
	}

	var dot float64
	var na float64
	var nb float64
	for i := 0; i < n; i++ {
		av := float64(a[i])
		bv := float64(b[i])
		dot += av * bv
		na += av * av
		nb += bv * bv
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
