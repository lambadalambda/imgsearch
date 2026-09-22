package search

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"imgsearch/internal/vectorindex"
)

func TestSearchExposesImageThumbnails(t *testing.T) {
	conn := setupSearchDB(t)
	if _, err := conn.Exec(`UPDATE images SET tags_json='["sample"]'; UPDATE images SET thumbnail_path='images/custom.webp' WHERE id=2`); err != nil {
		t.Fatal(err)
	}
	hits := []vectorindex.SearchHit{{ImageID: 1}, {ImageID: 2}}
	h := NewHandler(&Handler{DB: conn, ModelID: 1, Embedder: &fakeEmbedder{textVec: []float32{1, 0}}, Index: &fakeIndex{hits: hits, similarHits: hits}})
	for _, url := range []string{"/api/search/text?q=sample", "/api/search/similar?image_id=1", "/api/search/tags?tags=sample"} {
		t.Run(url, func(t *testing.T) {
			rr := httptest.NewRecorder()
			h.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, url, nil))
			if rr.Code != 200 {
				t.Fatalf("status=%d %s", rr.Code, rr.Body.String())
			}
			var resp struct {
				Results []struct {
					ImageID       int64  `json:"image_id"`
					StoragePath   string `json:"storage_path"`
					ThumbnailPath string `json:"thumbnail_path"`
				}
			}
			if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
				t.Fatal(err)
			}
			if len(resp.Results) != 2 {
				t.Fatalf("expected 2 results: %s", rr.Body.String())
			}
			for _, item := range resp.Results {
				want := "images/thumbnails/1.webp"
				if item.ImageID == 2 {
					want = "images/custom.webp"
				}
				if item.ThumbnailPath != want || item.ThumbnailPath == item.StoragePath {
					t.Errorf("result=%+v want thumbnail=%s", item, want)
				}
			}
		})
	}
}
