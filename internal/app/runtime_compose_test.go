package app

import (
	"context"
	"database/sql"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"imgsearch/internal/vectorindex"
	"imgsearch/internal/vectorindex/bruteforce"
)

type fakeRuntimeEmbedder struct{}

func (fakeRuntimeEmbedder) EmbedText(context.Context, string) ([]float32, error) {
	return []float32{1, 0}, nil
}
func (fakeRuntimeEmbedder) EmbedImage(context.Context, string) ([]float32, error) {
	return []float32{1, 0}, nil
}

type recordingCloser struct {
	name  string
	calls *[]string
}

func (c recordingCloser) Close() error {
	*c.calls = append(*c.calls, c.name)
	return nil
}

func TestNewRuntimeBuildsMuxUploadServiceAndQueue(t *testing.T) {
	dbConn := openDB(t)
	if err := Bootstrap(context.Background(), dbConn, func(context.Context, *sql.DB) error { return nil }); err != nil {
		t.Fatalf("bootstrap: %v", err)
	}
	data := &DataRuntime{DB: dbConn, Index: bruteforce.NewIndex(dbConn), VectorBackend: "bruteforce"}

	runtime, err := NewRuntime(RuntimeOptions{
		Data:                 data,
		DataDir:              t.TempDir(),
		ModelID:              7,
		Embedder:             fakeRuntimeEmbedder{},
		Index:                data.Index,
		WorkerLeaseDuration:  30 * time.Second,
		WorkerRetryBaseDelay: 5 * time.Second,
		LiveInterval:         time.Second,
		LiveImagesLimit:      10,
		LiveImagesOffset:     0,
		VideoFrameCount:      3,
		VideoTranscriptsOn:   true,
	})
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}

	if runtime.Mux == nil || runtime.Queue == nil || runtime.UploadService == nil {
		t.Fatalf("expected mux, queue, and upload service: %+v", runtime)
	}
	if runtime.UploadService.ModelID != 7 || !runtime.UploadService.EnableVideoTranscripts || runtime.UploadService.VideoFrameCount != 3 {
		t.Fatalf("unexpected upload service: %+v", runtime.UploadService)
	}
	if runtime.Queue.DB != dbConn || runtime.Queue.DataDir == "" || runtime.Queue.LeaseDuration != 30*time.Second {
		t.Fatalf("unexpected queue: %+v", runtime.Queue)
	}

	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	rr := httptest.NewRecorder()
	runtime.Mux.ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("health status: got=%d want=%d", rr.Code, http.StatusOK)
	}
}

func TestRuntimeCloseClosesOwnedResourcesInReverseOrder(t *testing.T) {
	calls := []string{}
	runtime := &Runtime{Closers: []Closer{
		recordingCloser{name: "first", calls: &calls},
		recordingCloser{name: "second", calls: &calls},
	}}

	if err := runtime.Close(); err != nil {
		t.Fatalf("close runtime: %v", err)
	}
	if fmt.Sprint(calls) != "[second first]" {
		t.Fatalf("unexpected close order: %v", calls)
	}
}

func TestNewRuntimeRejectsMissingDependencies(t *testing.T) {
	_, err := NewRuntime(RuntimeOptions{Data: &DataRuntime{}, Index: vectorindex.VectorIndex(nil)})
	if err == nil {
		t.Fatal("expected missing dependency error")
	}
}
