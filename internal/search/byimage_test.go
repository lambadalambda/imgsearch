package search

import (
	"bytes"
	"encoding/json"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"imgsearch/internal/vectorindex"
)

func queryImageRequest(t *testing.T, path string, content []byte) *http.Request {
	t.Helper()
	body := &bytes.Buffer{}
	mw := multipart.NewWriter(body)
	fw, err := mw.CreateFormFile("file", "query.jpg")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fw.Write(content); err != nil {
		t.Fatal(err)
	}
	_ = mw.Close()
	req := httptest.NewRequest(http.MethodPost, path, body)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	return req
}

func TestSearchByImageEmbedsUploadWithoutStoringIt(t *testing.T) {
	dbConn := setupSearchDB(t)
	dataDir := t.TempDir()
	index := &fakeIndex{hits: []vectorindex.SearchHit{
		{ImageID: 2, ModelID: 1, Distance: 0.2},
		{ImageID: 1, ModelID: 1, Distance: 0.4},
	}}
	h := NewHandler(&Handler{DB: dbConn, ModelID: 1, DataDir: dataDir, Embedder: &fakeEmbedder{imgVec: []float32{0.5, 0.5}}, Index: index})

	_, thisFile, _, _ := runtime.Caller(0)
	jpeg, err := os.ReadFile(filepath.Join(filepath.Dir(thisFile), "..", "..", "fixtures", "images", "cat_1.jpg"))
	if err != nil {
		t.Fatal(err)
	}
	var before int
	if err := dbConn.QueryRow(`SELECT COUNT(*) FROM images`).Scan(&before); err != nil {
		t.Fatal(err)
	}
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, queryImageRequest(t, "/api/search/by-image?limit=1", jpeg))
	if rr.Code != http.StatusOK {
		t.Fatalf("status: got=%d body=%s", rr.Code, rr.Body.String())
	}
	var resp SearchResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Results) != 1 || resp.Results[0].ImageID != 2 || resp.Results[0].Distance != 0.2 {
		t.Fatalf("results: %+v", resp.Results)
	}
	if index.searchCalls != 1 || len(index.searchVec) != 2 || index.searchVec[0] != 0.5 {
		t.Fatalf("expected the image embedding to drive the search, got calls=%d vec=%v", index.searchCalls, index.searchVec)
	}
	// Nothing persisted, and the spooled query file is gone.
	var count int
	if err := dbConn.QueryRow(`SELECT COUNT(*) FROM images`).Scan(&count); err != nil {
		t.Fatal(err)
	}
	if count != before {
		t.Fatalf("expected the library untouched (%d rows), got %d", before, count)
	}
	leftovers, _ := filepath.Glob(filepath.Join(dataDir, "tmp", "query-*"))
	if len(leftovers) != 0 {
		t.Fatalf("query temp files left behind: %v", leftovers)
	}
}

func TestSearchByImageRejectsBadInput(t *testing.T) {
	dbConn := setupSearchDB(t)
	h := NewHandler(&Handler{DB: dbConn, ModelID: 1, DataDir: t.TempDir(), Embedder: &fakeEmbedder{imgVec: []float32{1}}, Index: &fakeIndex{}})

	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, queryImageRequest(t, "/api/search/by-image", []byte("this is not an image")))
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("text payload: got=%d body=%s", rr.Code, rr.Body.String())
	}
	rr = httptest.NewRecorder()
	h.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/api/search/by-image", nil))
	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("get: got=%d", rr.Code)
	}
	req := httptest.NewRequest(http.MethodPost, "/api/search/by-image", bytes.NewReader([]byte("nope")))
	req.Header.Set("Content-Type", "multipart/form-data; boundary=x")
	rr = httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("broken multipart: got=%d", rr.Code)
	}
}
