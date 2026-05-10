package search

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"

	"imgsearch/internal/vectorindex"
)

func TestSimilarVideoSearchUsesSessionVectorFeedbackQuery(t *testing.T) {
	dbConn := setupSearchDB(t)
	if _, err := dbConn.Exec(`
INSERT INTO images(id, sha256, original_name, storage_path, mime_type, width, height)
VALUES
  (10, 'seed-frame', 'seed-frame.jpg', 'images/seed-frame', 'image/jpeg', 640, 360),
  (20, 'positive-frame', 'positive-frame.jpg', 'images/positive-frame', 'image/jpeg', 640, 360),
  (40, 'candidate-frame', 'candidate-frame.jpg', 'images/candidate-frame', 'image/jpeg', 640, 360);

INSERT INTO videos(id, sha256, original_name, storage_path, mime_type, duration_ms, width, height, frame_count)
VALUES
  (1, 'seed-video', 'seed.mp4', 'videos/seed.mp4', 'video/mp4', 9000, 1280, 720, 1),
  (2, 'positive-video', 'positive.mp4', 'videos/positive.mp4', 'video/mp4', 9000, 1280, 720, 1),
  (4, 'candidate-video', 'candidate.mp4', 'videos/candidate.mp4', 'video/mp4', 9000, 1280, 720, 1);

INSERT INTO video_frames(video_id, image_id, frame_index, timestamp_ms)
VALUES
  (1, 10, 0, 0),
  (2, 20, 0, 1000),
  (4, 40, 0, 1000);
`); err != nil {
		t.Fatalf("seed videos: %v", err)
	}
	if _, err := dbConn.Exec(`
INSERT INTO image_embeddings(image_id, model_id, dim, vector_blob)
VALUES
  (10, 1, 2, ?),
  (20, 1, 2, ?)
`, vectorindex.FloatsToBlob([]float32{1, 0}), vectorindex.FloatsToBlob([]float32{0, 1})); err != nil {
		t.Fatalf("seed embeddings: %v", err)
	}
	idx := &fakeIndex{hits: []vectorindex.SearchHit{
		{ImageID: 10, ModelID: 1, Distance: 0.01},
		{ImageID: 40, ModelID: 1, Distance: 0.03},
	}}
	h := NewHandler(&Handler{DB: dbConn, ModelID: 1, DataDir: "/tmp", Embedder: &fakeEmbedder{}, Index: idx})

	req := httptest.NewRequest(http.MethodGet, "/api/search/similar-videos?video_id=1&seen=2&positive_image_ids=20&limit=10", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d body=%s", rr.Code, rr.Body.String())
	}
	if idx.searchCalls != 1 {
		t.Fatalf("expected one adaptive vector search call, got %d", idx.searchCalls)
	}
	if len(idx.similarImageIDs) != 0 {
		t.Fatalf("expected SearchByImageID to be bypassed, got seed image calls %v", idx.similarImageIDs)
	}
	expected := []float32{1, similarVideoFeedbackPositiveWeight}
	norm := float32(1 / math.Sqrt(float64(expected[0]*expected[0]+expected[1]*expected[1])))
	expected[0] *= norm
	expected[1] *= norm
	assertVectorClose(t, idx.searchVec, expected, 1e-5)

	var resp SearchResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(resp.Results) != 1 || resp.Results[0].VideoID != 4 {
		t.Fatalf("expected seed frame filtered and candidate returned, got %+v", resp.Results)
	}
}

func TestSimilarVideoSearchUsesSoftNegativeVectorFeedback(t *testing.T) {
	dbConn := setupSearchDB(t)
	if _, err := dbConn.Exec(`
INSERT INTO images(id, sha256, original_name, storage_path, mime_type, width, height)
VALUES
  (10, 'seed-frame', 'seed-frame.jpg', 'images/seed-frame', 'image/jpeg', 640, 360),
  (20, 'negative-frame', 'negative-frame.jpg', 'images/negative-frame', 'image/jpeg', 640, 360),
  (40, 'candidate-frame', 'candidate-frame.jpg', 'images/candidate-frame', 'image/jpeg', 640, 360);

INSERT INTO videos(id, sha256, original_name, storage_path, mime_type, duration_ms, width, height, frame_count)
VALUES
  (1, 'seed-video', 'seed.mp4', 'videos/seed.mp4', 'video/mp4', 9000, 1280, 720, 1),
  (2, 'negative-video', 'negative.mp4', 'videos/negative.mp4', 'video/mp4', 9000, 1280, 720, 1),
  (4, 'candidate-video', 'candidate.mp4', 'videos/candidate.mp4', 'video/mp4', 9000, 1280, 720, 1);

INSERT INTO video_frames(video_id, image_id, frame_index, timestamp_ms)
VALUES
  (1, 10, 0, 0),
  (2, 20, 0, 1000),
  (4, 40, 0, 1000);
`); err != nil {
		t.Fatalf("seed videos: %v", err)
	}
	if _, err := dbConn.Exec(`
INSERT INTO image_embeddings(image_id, model_id, dim, vector_blob)
VALUES
  (10, 1, 2, ?),
  (20, 1, 2, ?)
`, vectorindex.FloatsToBlob([]float32{1, 0}), vectorindex.FloatsToBlob([]float32{0, 1})); err != nil {
		t.Fatalf("seed embeddings: %v", err)
	}
	idx := &fakeIndex{hits: []vectorindex.SearchHit{{ImageID: 40, ModelID: 1, Distance: 0.03}}}
	h := NewHandler(&Handler{DB: dbConn, ModelID: 1, DataDir: "/tmp", Embedder: &fakeEmbedder{}, Index: idx})

	req := httptest.NewRequest(http.MethodGet, "/api/search/similar-videos?video_id=1&seen=2&soft_negative_image_ids=20&limit=10", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d body=%s", rr.Code, rr.Body.String())
	}
	if idx.searchCalls != 1 {
		t.Fatalf("expected one adaptive vector search call, got %d", idx.searchCalls)
	}
	expected := []float32{1, -similarVideoFeedbackSoftNegativeWeight}
	norm := float32(1 / math.Sqrt(float64(expected[0]*expected[0]+expected[1]*expected[1])))
	expected[0] *= norm
	expected[1] *= norm
	assertVectorClose(t, idx.searchVec, expected, 1e-5)
}

func TestSimilarVideoSearchPositiveFeedbackWinsOverOverlappingSoftNegative(t *testing.T) {
	dbConn := setupSearchDB(t)
	if _, err := dbConn.Exec(`
INSERT INTO images(id, sha256, original_name, storage_path, mime_type, width, height)
VALUES
  (10, 'seed-frame', 'seed-frame.jpg', 'images/seed-frame', 'image/jpeg', 640, 360),
  (20, 'feedback-frame', 'feedback-frame.jpg', 'images/feedback-frame', 'image/jpeg', 640, 360),
  (40, 'candidate-frame', 'candidate-frame.jpg', 'images/candidate-frame', 'image/jpeg', 640, 360);

INSERT INTO videos(id, sha256, original_name, storage_path, mime_type, duration_ms, width, height, frame_count)
VALUES
  (1, 'seed-video', 'seed.mp4', 'videos/seed.mp4', 'video/mp4', 9000, 1280, 720, 1),
  (2, 'feedback-video', 'feedback.mp4', 'videos/feedback.mp4', 'video/mp4', 9000, 1280, 720, 1),
  (4, 'candidate-video', 'candidate.mp4', 'videos/candidate.mp4', 'video/mp4', 9000, 1280, 720, 1);

INSERT INTO video_frames(video_id, image_id, frame_index, timestamp_ms)
VALUES
  (1, 10, 0, 0),
  (2, 20, 0, 1000),
  (4, 40, 0, 1000);
`); err != nil {
		t.Fatalf("seed videos: %v", err)
	}
	if _, err := dbConn.Exec(`
INSERT INTO image_embeddings(image_id, model_id, dim, vector_blob)
VALUES
  (10, 1, 2, ?),
  (20, 1, 2, ?)
`, vectorindex.FloatsToBlob([]float32{1, 0}), vectorindex.FloatsToBlob([]float32{0, 1})); err != nil {
		t.Fatalf("seed embeddings: %v", err)
	}
	idx := &fakeIndex{hits: []vectorindex.SearchHit{{ImageID: 40, ModelID: 1, Distance: 0.03}}}
	h := NewHandler(&Handler{DB: dbConn, ModelID: 1, DataDir: "/tmp", Embedder: &fakeEmbedder{}, Index: idx})

	req := httptest.NewRequest(http.MethodGet, "/api/search/similar-videos?video_id=1&seen=2&positive_image_ids=20&soft_negative_image_ids=20&limit=10", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d body=%s", rr.Code, rr.Body.String())
	}
	expected := []float32{1, similarVideoFeedbackPositiveWeight}
	norm := float32(1 / math.Sqrt(float64(expected[0]*expected[0]+expected[1]*expected[1])))
	expected[0] *= norm
	expected[1] *= norm
	assertVectorClose(t, idx.searchVec, expected, 1e-5)
}

func TestSimilarVideoSearchIgnoresFeedbackImageIDsOutsideSeenVideos(t *testing.T) {
	dbConn := setupSearchDB(t)
	if _, err := dbConn.Exec(`
INSERT INTO images(id, sha256, original_name, storage_path, mime_type, width, height)
VALUES
  (10, 'seed-frame', 'seed-frame.jpg', 'images/seed-frame', 'image/jpeg', 640, 360),
  (20, 'unseen-feedback-frame', 'unseen-feedback-frame.jpg', 'images/unseen-feedback-frame', 'image/jpeg', 640, 360),
  (40, 'candidate-frame', 'candidate-frame.jpg', 'images/candidate-frame', 'image/jpeg', 640, 360);

INSERT INTO videos(id, sha256, original_name, storage_path, mime_type, duration_ms, width, height, frame_count)
VALUES
  (1, 'seed-video', 'seed.mp4', 'videos/seed.mp4', 'video/mp4', 9000, 1280, 720, 1),
  (2, 'unseen-feedback-video', 'unseen-feedback.mp4', 'videos/unseen-feedback.mp4', 'video/mp4', 9000, 1280, 720, 1),
  (4, 'candidate-video', 'candidate.mp4', 'videos/candidate.mp4', 'video/mp4', 9000, 1280, 720, 1);

INSERT INTO video_frames(video_id, image_id, frame_index, timestamp_ms)
VALUES
  (1, 10, 0, 0),
  (2, 20, 0, 1000),
  (4, 40, 0, 1000);
`); err != nil {
		t.Fatalf("seed videos: %v", err)
	}
	if _, err := dbConn.Exec(`
INSERT INTO image_embeddings(image_id, model_id, dim, vector_blob)
VALUES
  (10, 1, 2, ?),
  (20, 1, 2, ?)
`, vectorindex.FloatsToBlob([]float32{1, 0}), vectorindex.FloatsToBlob([]float32{0, 1})); err != nil {
		t.Fatalf("seed embeddings: %v", err)
	}
	idx := &fakeIndex{similarHitsByImageID: map[int64][]vectorindex.SearchHit{
		10: {{ImageID: 40, ModelID: 1, Distance: 0.03}},
	}}
	h := NewHandler(&Handler{DB: dbConn, ModelID: 1, DataDir: "/tmp", Embedder: &fakeEmbedder{}, Index: idx})

	req := httptest.NewRequest(http.MethodGet, "/api/search/similar-videos?video_id=1&positive_image_ids=20&limit=10", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d body=%s", rr.Code, rr.Body.String())
	}
	if idx.searchCalls != 0 {
		t.Fatalf("expected invalid feedback to fall back to SearchByImageID, got %d raw searches", idx.searchCalls)
	}
	if len(idx.similarImageIDs) != 1 || idx.similarImageIDs[0] != 10 {
		t.Fatalf("expected SearchByImageID seed 10 fallback, got %v", idx.similarImageIDs)
	}
}

func TestSimilarVideoSearchVectorFeedbackPreservesFilteringAndTagReranking(t *testing.T) {
	dbConn := setupSearchDB(t)
	if _, err := dbConn.Exec(`
INSERT INTO images(id, sha256, original_name, storage_path, mime_type, width, height)
VALUES
  (10, 'seed-frame', 'seed-frame.jpg', 'images/seed-frame', 'image/jpeg', 640, 360),
  (20, 'avoided-frame', 'avoided-frame.jpg', 'images/avoided-frame', 'image/jpeg', 640, 360),
  (30, 'preferred-frame', 'preferred-frame.jpg', 'images/preferred-frame', 'image/jpeg', 640, 360),
  (40, 'hidden-frame', 'hidden-frame.jpg', 'images/hidden-frame', 'image/jpeg', 640, 360),
  (50, 'feedback-frame', 'feedback-frame.jpg', 'images/feedback-frame', 'image/jpeg', 640, 360);

INSERT INTO videos(id, sha256, original_name, storage_path, mime_type, duration_ms, width, height, frame_count, tags_json)
VALUES
  (1, 'seed-video', 'seed.mp4', 'videos/seed.mp4', 'video/mp4', 9000, 1280, 720, 1, '["seed"]'),
  (2, 'avoided-video', 'avoided.mp4', 'videos/avoided.mp4', 'video/mp4', 9000, 1280, 720, 1, '["avoid-me"]'),
  (3, 'preferred-video', 'preferred.mp4', 'videos/preferred.mp4', 'video/mp4', 9000, 1280, 720, 1, '["liked"]'),
  (4, 'hidden-video', 'hidden.mp4', 'videos/hidden.mp4', 'video/mp4', 9000, 1280, 720, 1, '["nsfw"]'),
  (5, 'feedback-video', 'feedback.mp4', 'videos/feedback.mp4', 'video/mp4', 9000, 1280, 720, 1, '["feedback"]');

INSERT INTO video_frames(video_id, image_id, frame_index, timestamp_ms)
VALUES
  (1, 10, 0, 0),
  (2, 20, 0, 1000),
  (3, 30, 0, 1000),
  (4, 40, 0, 1000),
  (5, 50, 0, 1000);
`); err != nil {
		t.Fatalf("seed videos: %v", err)
	}
	if _, err := dbConn.Exec(`
INSERT INTO image_embeddings(image_id, model_id, dim, vector_blob)
VALUES
  (10, 1, 2, ?),
  (50, 1, 2, ?)
`, vectorindex.FloatsToBlob([]float32{1, 0}), vectorindex.FloatsToBlob([]float32{0, 1})); err != nil {
		t.Fatalf("seed embeddings: %v", err)
	}
	idx := &fakeIndex{hits: []vectorindex.SearchHit{
		{ImageID: 10, ModelID: 1, Distance: 0.01},
		{ImageID: 50, ModelID: 1, Distance: 0.02},
		{ImageID: 20, ModelID: 1, Distance: 0.03},
		{ImageID: 40, ModelID: 1, Distance: 0.04},
		{ImageID: 30, ModelID: 1, Distance: 0.05},
	}}
	h := NewHandler(&Handler{DB: dbConn, ModelID: 1, DataDir: "/tmp", Embedder: &fakeEmbedder{}, Index: idx})

	req := httptest.NewRequest(http.MethodGet, "/api/search/similar-videos?video_id=1&seen=5&positive_image_ids=50&prefer_tags=liked&avoid_tags=avoid-me&limit=10", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d body=%s", rr.Code, rr.Body.String())
	}
	if idx.searchCalls != 1 {
		t.Fatalf("expected adaptive vector search, got %d raw searches", idx.searchCalls)
	}
	var resp SearchResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	got := make([]int64, 0, len(resp.Results))
	for _, result := range resp.Results {
		got = append(got, result.VideoID)
	}
	want := []int64{3, 2}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("expected seen/nsfw filtering and tag rerank order: got=%v want=%v results=%+v", got, want, resp.Results)
	}
}

func TestSimilarVideoFeedbackQueryIsBoundedAndNormalized(t *testing.T) {
	query, used, err := buildSimilarVideoFeedbackQuery([]float32{2, 0}, [][]float32{{0, 2}}, [][]float32{{0, -2}})
	if err != nil {
		t.Fatalf("build feedback query: %v", err)
	}
	if !used {
		t.Fatalf("expected feedback vectors to be used")
	}
	norm := math.Sqrt(float64(query[0]*query[0] + query[1]*query[1]))
	if math.Abs(norm-1) > 1e-5 {
		t.Fatalf("expected normalized query, norm=%.7f query=%v", norm, query)
	}
	if query[1] <= 0 {
		t.Fatalf("expected query to move toward positive and away from negative feedback, got %v", query)
	}
	drift := math.Sqrt(math.Pow(float64(query[0]-1), 2) + math.Pow(float64(query[1]), 2))
	if drift > similarVideoFeedbackMaxDeltaNorm+1e-5 {
		t.Fatalf("expected drift <= %.3f, got %.7f query=%v", similarVideoFeedbackMaxDeltaNorm, drift, query)
	}
}
