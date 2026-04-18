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
	"time"

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

type concurrencyCheckingEmbedder struct {
	textByPrompt map[string][]float32
	mu           sync.Mutex
	active       int
	sawOverlap   bool
}

func (f *concurrencyCheckingEmbedder) EmbedText(_ context.Context, prompt string) ([]float32, error) {
	f.mu.Lock()
	f.active++
	if f.active > 1 {
		f.sawOverlap = true
	}
	f.mu.Unlock()
	time.Sleep(20 * time.Millisecond)
	f.mu.Lock()
	f.active--
	sawOverlap := f.sawOverlap
	f.mu.Unlock()
	if sawOverlap {
		return nil, fmt.Errorf("concurrent EmbedText call")
	}
	return f.textByPrompt[prompt], nil
}

func (f *concurrencyCheckingEmbedder) EmbedImage(context.Context, string) ([]float32, error) {
	return nil, nil
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
	if _, err := dbConn.Exec(`
UPDATE images
SET description = 'Friendly dog in the library.', tags_json = '["dog","library"]'
WHERE id = 2
`); err != nil {
		t.Fatalf("seed annotations: %v", err)
	}
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
	if resp.Results[0].Description != "Friendly dog in the library." {
		t.Fatalf("unexpected description: %q", resp.Results[0].Description)
	}
	if len(resp.Results[0].Tags) != 2 || resp.Results[0].Tags[0] != "dog" {
		t.Fatalf("unexpected tags: %v", resp.Results[0].Tags)
	}
}

func TestTextSearchGroupsVideoFrameHitsIntoSingleVideoResult(t *testing.T) {
	dbConn := setupSearchDB(t)
	if _, err := dbConn.Exec(`
INSERT INTO images(id, sha256, original_name, storage_path, mime_type, width, height)
VALUES
  (10, 'vf1', 'frame-1.jpg', 'images/vf1', 'image/jpeg', 100, 100),
  (11, 'vf2', 'frame-2.jpg', 'images/vf2', 'image/jpeg', 100, 100)
`); err != nil {
		t.Fatalf("seed frame images: %v", err)
	}
	if _, err := dbConn.Exec(`
INSERT INTO videos(id, sha256, original_name, storage_path, mime_type, duration_ms, width, height, frame_count)
VALUES (7, 'vid', 'clip.mp4', 'videos/vid', 'video/mp4', 12000, 1920, 1080, 2)
`); err != nil {
		t.Fatalf("seed video: %v", err)
	}
	if _, err := dbConn.Exec(`
INSERT INTO video_frames(video_id, image_id, frame_index, timestamp_ms)
VALUES
  (7, 10, 0, 1000),
  (7, 11, 1, 7000)
`); err != nil {
		t.Fatalf("seed video frames: %v", err)
	}

	h := NewHandler(&Handler{
		DB:       dbConn,
		ModelID:  1,
		DataDir:  "/tmp",
		Embedder: &fakeEmbedder{textVec: []float32{1, 0}},
		Index: &fakeIndex{hits: []vectorindex.SearchHit{
			{ImageID: 11, ModelID: 1, Distance: 0.05},
			{ImageID: 10, ModelID: 1, Distance: 0.06},
			{ImageID: 2, ModelID: 1, Distance: 0.20},
		}},
	})

	req := httptest.NewRequest(http.MethodGet, "/api/search/text?q=clip", nil)
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
		t.Fatalf("expected 2 grouped results, got %d: %+v", len(resp.Results), resp.Results)
	}
	if resp.Results[0].MediaType != "video" || resp.Results[0].VideoID != 7 {
		t.Fatalf("expected first result to be video 7, got %+v", resp.Results[0])
	}
	if resp.Results[0].OriginalName != "clip.mp4" || resp.Results[0].StoragePath != "videos/vid" {
		t.Fatalf("unexpected video metadata: %+v", resp.Results[0])
	}
	if resp.Results[0].PreviewPath != "images/vf2" || resp.Results[0].MatchTimestampMS != 7000 {
		t.Fatalf("unexpected video preview/timestamp: %+v", resp.Results[0])
	}
	if resp.Results[1].MediaType != "image" || resp.Results[1].ImageID != 2 {
		t.Fatalf("expected second result to stay as image 2, got %+v", resp.Results[1])
	}
}

func TestTextSearchReturnsVideoResultFromTranscriptEmbedding(t *testing.T) {
	dbConn := setupSearchDB(t)
	if _, err := dbConn.Exec(`
INSERT INTO images(id, sha256, original_name, storage_path, mime_type, width, height)
VALUES (20, 'vf20', 'frame.jpg', 'images/vf20', 'image/jpeg', 100, 100)
`); err != nil {
		t.Fatalf("seed frame image: %v", err)
	}
	if _, err := dbConn.Exec(`
INSERT INTO videos(id, sha256, original_name, storage_path, mime_type, duration_ms, width, height, frame_count, transcript_text)
VALUES (8, 'vid8', 'speech.mp4', 'videos/vid8', 'video/mp4', 4000, 1280, 720, 1, 'tis better to remain silent and be thought a fool')
`); err != nil {
		t.Fatalf("seed video: %v", err)
	}
	if _, err := dbConn.Exec(`
INSERT INTO video_frames(video_id, image_id, frame_index, timestamp_ms)
VALUES (8, 20, 0, 1000)
`); err != nil {
		t.Fatalf("seed preview frame: %v", err)
	}
	if _, err := dbConn.Exec(`
INSERT INTO video_transcript_embeddings(video_id, model_id, dim, vector_blob)
VALUES (8, 1, 2, ?)
`, vectorindex.FloatsToBlob([]float32{1, 0})); err != nil {
		t.Fatalf("seed transcript embedding: %v", err)
	}

	h := NewHandler(&Handler{
		DB:       dbConn,
		ModelID:  1,
		DataDir:  "/tmp",
		Embedder: &fakeEmbedder{textVec: []float32{1, 0}},
		Index:    &fakeIndex{hits: nil},
	})

	req := httptest.NewRequest(http.MethodGet, "/api/search/text?q=silent", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d body=%s", rr.Code, rr.Body.String())
	}

	var resp SearchResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(resp.Results) != 1 {
		t.Fatalf("expected 1 result from transcript embedding, got %d: %+v", len(resp.Results), resp.Results)
	}
	if resp.Results[0].MediaType != "video" || resp.Results[0].VideoID != 8 {
		t.Fatalf("expected video 8 transcript result, got %+v", resp.Results[0])
	}
	if !strings.Contains(resp.Results[0].TranscriptText, "remain silent") {
		t.Fatalf("expected transcript text on result, got %+v", resp.Results[0])
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

func TestTagSearchReturnsMatchingResults(t *testing.T) {
	dbConn := setupSearchDB(t)
	if _, err := dbConn.Exec(`
UPDATE images SET description = 'A playful cat outside.', tags_json = '["cat","outdoors"]' WHERE id = 1;
UPDATE images SET description = 'A friendly dog outside.', tags_json = '["dog","outdoors"]' WHERE id = 2;
`); err != nil {
		t.Fatalf("seed tags: %v", err)
	}

	h := NewHandler(&Handler{DB: dbConn, ModelID: 1, DataDir: "/tmp", Embedder: &fakeEmbedder{}, Index: &fakeIndex{}})

	req := httptest.NewRequest(http.MethodGet, "/api/search/tags?tag=outdoors&limit=10", nil)
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
		t.Fatalf("expected 2 tag results, got %d: %+v", len(resp.Results), resp.Results)
	}
	seen := map[int64]SearchResult{}
	for _, result := range resp.Results {
		seen[result.ImageID] = result
		if result.SearchSource != "tag" {
			t.Fatalf("expected tag search source, got %+v", result)
		}
	}
	if _, ok := seen[1]; !ok {
		t.Fatalf("expected image 1 in tag search results: %+v", resp.Results)
	}
	if _, ok := seen[2]; !ok {
		t.Fatalf("expected image 2 in tag search results: %+v", resp.Results)
	}
}

func TestTagSearchSupportsAllMode(t *testing.T) {
	dbConn := setupSearchDB(t)
	if _, err := dbConn.Exec(`
UPDATE images SET tags_json = '["cat","outdoors"]' WHERE id = 1;
UPDATE images SET tags_json = '["dog","outdoors"]' WHERE id = 2;
`); err != nil {
		t.Fatalf("seed tags: %v", err)
	}

	h := NewHandler(&Handler{DB: dbConn, ModelID: 1, DataDir: "/tmp", Embedder: &fakeEmbedder{}, Index: &fakeIndex{}})

	req := httptest.NewRequest(http.MethodGet, "/api/search/tags?tag=dog&tag=outdoors&mode=all", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d body=%s", rr.Code, rr.Body.String())
	}

	var resp SearchResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(resp.Results) != 1 {
		t.Fatalf("expected 1 tag result for all-mode, got %d: %+v", len(resp.Results), resp.Results)
	}
	if resp.Results[0].ImageID != 2 {
		t.Fatalf("expected image 2 for all-mode tag search, got %+v", resp.Results)
	}
}

func TestTagSearchSupportsOffsetPaginationAndTotal(t *testing.T) {
	dbConn := setupSearchDB(t)
	if _, err := dbConn.Exec(`
INSERT INTO images(id, sha256, original_name, storage_path, mime_type, width, height)
VALUES (3, 'c', 'bird.jpg', 'images/c', 'image/jpeg', 100, 100);

UPDATE images SET tags_json = '["outdoors"]' WHERE id = 1;
UPDATE images SET tags_json = '["outdoors"]' WHERE id = 2;
UPDATE images SET tags_json = '["outdoors"]' WHERE id = 3;
`); err != nil {
		t.Fatalf("seed tags: %v", err)
	}

	h := NewHandler(&Handler{DB: dbConn, ModelID: 1, DataDir: "/tmp", Embedder: &fakeEmbedder{}, Index: &fakeIndex{}})

	req := httptest.NewRequest(http.MethodGet, "/api/search/tags?tag=outdoors&limit=1&offset=1", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d body=%s", rr.Code, rr.Body.String())
	}

	var resp SearchResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Total != 3 {
		t.Fatalf("expected total=3 for paginated tag search, got %d", resp.Total)
	}
	if len(resp.Results) != 1 {
		t.Fatalf("expected one paginated result, got %d: %+v", len(resp.Results), resp.Results)
	}
	if resp.Results[0].ImageID != 2 {
		t.Fatalf("expected second result to be image 2, got %+v", resp.Results[0])
	}
}

func TestTagSearchIncludesVideoResultsFromTaggedFrames(t *testing.T) {
	dbConn := setupSearchDB(t)
	if _, err := dbConn.Exec(`
INSERT INTO images(id, sha256, original_name, storage_path, mime_type, width, height, tags_json)
VALUES (10, 'vf1', 'frame.jpg', 'images/vf1', 'image/jpeg', 100, 100, '["outdoors","trail"]');

INSERT INTO videos(id, sha256, original_name, storage_path, mime_type, duration_ms, width, height, frame_count)
VALUES (7, 'vid', 'clip.mp4', 'videos/vid', 'video/mp4', 12000, 1920, 1080, 1);

INSERT INTO video_frames(video_id, image_id, frame_index, timestamp_ms)
VALUES (7, 10, 0, 1000);
`); err != nil {
		t.Fatalf("seed tagged video frame: %v", err)
	}

	h := NewHandler(&Handler{DB: dbConn, ModelID: 1, DataDir: "/tmp", Embedder: &fakeEmbedder{}, Index: &fakeIndex{}})

	req := httptest.NewRequest(http.MethodGet, "/api/search/tags?tag=trail", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d body=%s", rr.Code, rr.Body.String())
	}

	var resp SearchResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(resp.Results) != 1 {
		t.Fatalf("expected one video result for tag, got %d: %+v", len(resp.Results), resp.Results)
	}
	if resp.Results[0].MediaType != "video" || resp.Results[0].VideoID != 7 {
		t.Fatalf("expected video result for tag search, got %+v", resp.Results[0])
	}
	if resp.Results[0].PreviewPath != "images/vf1" {
		t.Fatalf("expected preview path from matched frame, got %+v", resp.Results[0])
	}
}

func TestTagSearchRejectsMissingTag(t *testing.T) {
	dbConn := setupSearchDB(t)
	h := NewHandler(&Handler{DB: dbConn, ModelID: 1, DataDir: "/tmp", Embedder: &fakeEmbedder{}, Index: &fakeIndex{}})

	req := httptest.NewRequest(http.MethodGet, "/api/search/tags", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusBadRequest {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusBadRequest, rr.Body.String())
	}
}

func TestTagCloudReturnsRankedTags(t *testing.T) {
	dbConn := setupSearchDB(t)
	if _, err := dbConn.Exec(`
UPDATE images SET tags_json = '["cat","outdoors","sun"]' WHERE id = 1;
UPDATE images SET tags_json = '["dog","outdoors"]' WHERE id = 2;
`); err != nil {
		t.Fatalf("seed tags: %v", err)
	}

	h := NewHandler(&Handler{DB: dbConn, ModelID: 1, DataDir: "/tmp", Embedder: &fakeEmbedder{}, Index: &fakeIndex{}})

	req := httptest.NewRequest(http.MethodGet, "/api/search/tag-cloud?limit=3", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d body=%s", rr.Code, rr.Body.String())
	}

	var resp TagCloudResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(resp.Tags) != 3 {
		t.Fatalf("expected 3 cloud tags, got %d: %+v", len(resp.Tags), resp.Tags)
	}
	if resp.Tags[0].Tag != "outdoors" || resp.Tags[0].Count != 2 {
		t.Fatalf("expected outdoors to lead with count 2, got %+v", resp.Tags[0])
	}
}

func TestTagCloudSupportsPrefixQuery(t *testing.T) {
	dbConn := setupSearchDB(t)
	if _, err := dbConn.Exec(`
UPDATE images SET tags_json = '["dog","outdoors"]' WHERE id = 1;
UPDATE images SET tags_json = '["cat","indoor"]' WHERE id = 2;
`); err != nil {
		t.Fatalf("seed tags: %v", err)
	}

	h := NewHandler(&Handler{DB: dbConn, ModelID: 1, DataDir: "/tmp", Embedder: &fakeEmbedder{}, Index: &fakeIndex{}})

	req := httptest.NewRequest(http.MethodGet, "/api/search/tag-cloud?limit=10&q=do", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d body=%s", rr.Code, rr.Body.String())
	}

	var resp TagCloudResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(resp.Tags) != 1 {
		t.Fatalf("expected one prefixed tag, got %d: %+v", len(resp.Tags), resp.Tags)
	}
	if resp.Tags[0].Tag != "dog" {
		t.Fatalf("expected prefix query to return dog, got %+v", resp.Tags[0])
	}
}

func TestTextSearchSupportsTagRestriction(t *testing.T) {
	dbConn := setupSearchDB(t)
	if _, err := dbConn.Exec(`
UPDATE images SET tags_json = '["dog","outdoors"]' WHERE id = 1;
UPDATE images SET tags_json = '["dog","indoor"]' WHERE id = 2;
`); err != nil {
		t.Fatalf("seed tags: %v", err)
	}

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

	req := httptest.NewRequest(http.MethodGet, "/api/search/text?q=dog&tag=outdoors&tag_mode=all", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d body=%s", rr.Code, rr.Body.String())
	}

	var resp SearchResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(resp.Results) != 1 {
		t.Fatalf("expected one text result after tag restriction, got %d: %+v", len(resp.Results), resp.Results)
	}
	if resp.Results[0].ImageID != 1 {
		t.Fatalf("expected image 1 to match outdoors restriction, got %+v", resp.Results[0])
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

func TestTextSearchWithNegativePromptAvoidsConcurrentEmbedCalls(t *testing.T) {
	dbConn := setupSearchDB(t)
	embed := &concurrencyCheckingEmbedder{textByPrompt: map[string][]float32{
		"dog": {1, 0},
		"cat": {0, 1},
	}}
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

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusOK, rr.Body.String())
	}
}

func TestEnrichPreservesHitOrderAndDuplicates(t *testing.T) {
	dbConn := setupSearchDB(t)
	h := &Handler{DB: dbConn, ModelID: 1, DataDir: "/tmp"}

	hits := []vectorindex.SearchHit{
		{ImageID: 2, ModelID: 1, Distance: 0.1},
		{ImageID: 1, ModelID: 1, Distance: 0.2},
		{ImageID: 2, ModelID: 1, Distance: 0.3},
	}

	results, err := h.enrich(context.Background(), hits)
	if err != nil {
		t.Fatalf("enrich: %v", err)
	}
	if len(results) != 3 {
		t.Fatalf("results length: got=%d want=3", len(results))
	}
	if results[0].ImageID != 2 || results[1].ImageID != 1 || results[2].ImageID != 2 {
		t.Fatalf("unexpected result order: %+v", results)
	}
	if results[0].Distance != 0.1 || results[2].Distance != 0.3 {
		t.Fatalf("expected distances to stay aligned with hit order, got %+v", results)
	}
}

func TestEnrichReturnsEmptyForNoHits(t *testing.T) {
	dbConn := setupSearchDB(t)
	h := &Handler{DB: dbConn, ModelID: 1, DataDir: "/tmp"}

	results, err := h.enrich(context.Background(), nil)
	if err != nil {
		t.Fatalf("enrich: %v", err)
	}
	if len(results) != 0 {
		t.Fatalf("expected empty results, got %d", len(results))
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
