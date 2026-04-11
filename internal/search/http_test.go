package search

import (
	"context"
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	_ "github.com/mattn/go-sqlite3"

	"imgsearch/internal/db"
	"imgsearch/internal/vectorindex"
)

type fakeEmbedder struct {
	textVec []float32
	imgVec  []float32
}

func (f *fakeEmbedder) EmbedText(context.Context, string) ([]float32, error)  { return f.textVec, nil }
func (f *fakeEmbedder) EmbedImage(context.Context, string) ([]float32, error) { return f.imgVec, nil }

type fakeIndex struct {
	hits        []vectorindex.SearchHit
	similarHits []vectorindex.SearchHit
	similarErr  error
}

func (f *fakeIndex) Upsert(context.Context, int64, int64, []float32) error { return nil }
func (f *fakeIndex) Delete(context.Context, int64, int64) error            { return nil }
func (f *fakeIndex) Search(context.Context, int64, []float32, int) ([]vectorindex.SearchHit, error) {
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
