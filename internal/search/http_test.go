package search

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	_ "github.com/mattn/go-sqlite3"

	"imgsearch/internal/db"
	"imgsearch/internal/vectorindex"
)

type fakeEmbedder struct {
	textVec          []float32
	imgVec           []float32
	textByPrompt     map[string][]float32
	textErrByPrompt  map[string]error
	textPromptInputs []string
	mu               sync.Mutex
}

func (f *fakeEmbedder) EmbedText(_ context.Context, prompt string) ([]float32, error) {
	f.mu.Lock()
	f.textPromptInputs = append(f.textPromptInputs, prompt)
	f.mu.Unlock()

	if err := f.textErrByPrompt[prompt]; err != nil {
		return nil, err
	}
	if vec, ok := f.textByPrompt[prompt]; ok {
		return vec, nil
	}
	return f.textVec, nil
}

func (f *fakeEmbedder) EmbedImage(context.Context, string) ([]float32, error) { return f.imgVec, nil }

func (f *fakeEmbedder) prompts() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]string, len(f.textPromptInputs))
	copy(out, f.textPromptInputs)
	return out
}

type fakeIndex struct {
	hits        []vectorindex.SearchHit
	similarHits []vectorindex.SearchHit
	similarErr  error
	searchVec   []float32
	searchCalls int
}

func (f *fakeIndex) Upsert(context.Context, int64, int64, []float32) error { return nil }
func (f *fakeIndex) Delete(context.Context, int64, int64) error            { return nil }
func (f *fakeIndex) Search(_ context.Context, _ int64, query []float32, _ int) ([]vectorindex.SearchHit, error) {
	f.searchCalls++
	f.searchVec = append([]float32(nil), query...)
	return f.hits, nil
}
func (f *fakeIndex) SearchByImageID(context.Context, int64, int64, int) ([]vectorindex.SearchHit, error) {
	if f.similarErr != nil {
		return nil, f.similarErr
	}
	return f.similarHits, nil
}

func setupSearchDB(t *testing.T) *sql.DB {
	t.Helper()
	dbConn, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() { _ = dbConn.Close() })

	if err := db.RunMigrations(context.Background(), dbConn); err != nil {
		t.Fatalf("migrate: %v", err)
	}

	_, err = dbConn.Exec(`
INSERT INTO images(id, sha256, original_name, storage_path, mime_type, width, height)
VALUES
  (1, 'a', 'cat.jpg', 'images/a', 'image/jpeg', 100, 100),
  (2, 'b', 'dog.jpg', 'images/b', 'image/jpeg', 100, 100)
`)
	if err != nil {
		t.Fatalf("seed images: %v", err)
	}
	return dbConn
}

func TestTextSearchReturnsRankedResults(t *testing.T) {
	dbConn := setupSearchDB(t)
	h := NewHandler(&Handler{
		DB:       dbConn,
		ModelID:  1,
		DataDir:  "/tmp",
		Embedder: &fakeEmbedder{textVec: []float32{1, 0}},
		Index: &fakeIndex{hits: []vectorindex.SearchHit{
			{ImageID: 2, ModelID: 1, Distance: 0.1},
			{ImageID: 1, ModelID: 1, Distance: 0.2},
		}},
	})

	req := httptest.NewRequest(http.MethodGet, "/api/search/text?q=dog", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d body=%s", rr.Code, rr.Body.String())
	}

	var resp SearchResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(resp.Results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(resp.Results))
	}
	if resp.Results[0].ImageID != 2 || resp.Results[1].ImageID != 1 {
		t.Fatalf("unexpected ordering: %+v", resp.Results)
	}
}

func TestSimilarSearchReturnsResults(t *testing.T) {
	dbConn := setupSearchDB(t)
	h := NewHandler(&Handler{
		DB:       dbConn,
		ModelID:  1,
		DataDir:  "/tmp",
		Embedder: &fakeEmbedder{imgVec: []float32{1, 0}},
		Index: &fakeIndex{similarHits: []vectorindex.SearchHit{
			{ImageID: 2, ModelID: 1, Distance: 0.1},
		}},
	})

	req := httptest.NewRequest(http.MethodGet, "/api/search/similar?image_id=1", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d body=%s", rr.Code, rr.Body.String())
	}

	var resp SearchResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(resp.Results) != 1 || resp.Results[0].ImageID != 2 {
		t.Fatalf("unexpected results: %+v", resp.Results)
	}
}

func TestSimilarSearchReturnsNotFoundWhenImageNotIndexed(t *testing.T) {
	dbConn := setupSearchDB(t)
	h := NewHandler(&Handler{
		DB:       dbConn,
		ModelID:  1,
		DataDir:  "/tmp",
		Embedder: &fakeEmbedder{imgVec: []float32{1, 0}},
		Index:    &fakeIndex{similarErr: vectorindex.ErrNotFound},
	})

	req := httptest.NewRequest(http.MethodGet, "/api/search/similar?image_id=1", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusNotFound {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusNotFound, rr.Body.String())
	}
}

func TestTextSearchWithNegativePromptCombinesEmbeddings(t *testing.T) {
	dbConn := setupSearchDB(t)
	embed := &fakeEmbedder{
		textByPrompt: map[string][]float32{
			"dog": {1, 0},
			"cat": {0, 1},
		},
	}
	index := &fakeIndex{hits: []vectorindex.SearchHit{{ImageID: 2, ModelID: 1, Distance: 0.1}}}
	h := NewHandler(&Handler{DB: dbConn, ModelID: 1, DataDir: "/tmp", Embedder: embed, Index: index})

	req := httptest.NewRequest(http.MethodGet, "/api/search/text?q=dog&neg=cat", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d body=%s", rr.Code, rr.Body.String())
	}
	if index.searchCalls != 1 {
		t.Fatalf("expected one search call, got %d", index.searchCalls)
	}

	expected := []float32{float32(1 / math.Sqrt2), float32(-1 / math.Sqrt2)}
	assertVectorClose(t, index.searchVec, expected, 1e-5)

	prompts := strings.Join(embed.prompts(), ",")
	if !strings.Contains(prompts, "dog") || !strings.Contains(prompts, "cat") {
		t.Fatalf("expected dog and cat prompts to be embedded, got %q", prompts)
	}
}

func TestTextSearchWithNegativePromptFallsBackWhenVectorsCancel(t *testing.T) {
	dbConn := setupSearchDB(t)
	embed := &fakeEmbedder{
		textByPrompt: map[string][]float32{
			"dog":    {1, 0},
			"canine": {1, 0},
		},
	}
	index := &fakeIndex{hits: []vectorindex.SearchHit{{ImageID: 2, ModelID: 1, Distance: 0.1}}}
	h := NewHandler(&Handler{DB: dbConn, ModelID: 1, DataDir: "/tmp", Embedder: embed, Index: index})

	req := httptest.NewRequest(http.MethodGet, "/api/search/text?q=dog&neg=canine", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d body=%s", rr.Code, rr.Body.String())
	}
	assertVectorClose(t, index.searchVec, []float32{1, 0}, 1e-6)
}

func TestTextSearchRejectsNegativePromptThatIsTooLong(t *testing.T) {
	dbConn := setupSearchDB(t)
	h := NewHandler(&Handler{
		DB:       dbConn,
		ModelID:  1,
		DataDir:  "/tmp",
		Embedder: &fakeEmbedder{textVec: []float32{1, 0}},
		Index:    &fakeIndex{hits: []vectorindex.SearchHit{{ImageID: 2, ModelID: 1, Distance: 0.1}}},
	})

	neg := strings.Repeat("n", maxNegativePromptChars+1)
	req := httptest.NewRequest(http.MethodGet, fmt.Sprintf("/api/search/text?q=dog&neg=%s", neg), nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusBadRequest {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusBadRequest, rr.Body.String())
	}
}

func TestTextSearchReturnsErrorWhenNegativeEmbeddingFails(t *testing.T) {
	dbConn := setupSearchDB(t)
	embed := &fakeEmbedder{
		textByPrompt: map[string][]float32{"dog": {1, 0}},
		textErrByPrompt: map[string]error{
			"cat": fmt.Errorf("boom"),
		},
	}
	h := NewHandler(&Handler{
		DB:       dbConn,
		ModelID:  1,
		DataDir:  "/tmp",
		Embedder: embed,
		Index:    &fakeIndex{hits: []vectorindex.SearchHit{{ImageID: 2, ModelID: 1, Distance: 0.1}}},
	})

	req := httptest.NewRequest(http.MethodGet, "/api/search/text?q=dog&neg=cat", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusInternalServerError {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusInternalServerError, rr.Body.String())
	}
	if !strings.Contains(rr.Body.String(), "negative embedding failed") {
		t.Fatalf("expected negative embedding error message, got %s", rr.Body.String())
	}
}

func assertVectorClose(t *testing.T, got []float32, want []float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("vector length mismatch: got=%d want=%d", len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > tolerance {
			t.Fatalf("vector mismatch at index %d: got=%.7f want=%.7f", i, got[i], want[i])
		}
	}
}
