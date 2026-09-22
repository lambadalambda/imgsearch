package app

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"imgsearch/internal/db"
	"imgsearch/internal/embedder"
	"imgsearch/internal/stats"
	"imgsearch/internal/transcribe"
	"net/http"
	"net/http/httptest"
	"strings"
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
		UploadMaxImageBytes:  1 << 20,
		UploadMaxVideoBytes:  8 << 20,
		UploadTimeout:        time.Minute,
		DeduplicateStrict:    true,
	})
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}

	if runtime.Mux == nil || runtime.Queue == nil || runtime.UploadService == nil {
		t.Fatalf("expected mux, queue, and upload service: %+v", runtime)
	}
	if !runtime.UploadService.DeduplicateStrict || runtime.UploadService.ModelID != 7 || !runtime.UploadService.EnableVideoTranscripts || runtime.UploadService.VideoFrameCount != 3 {
		t.Fatalf("unexpected upload service: %+v", runtime.UploadService)
	}
	if runtime.UploadService.MaxImageBytes != 1<<20 || runtime.UploadService.MaxVideoBytes != 8<<20 || runtime.UploadService.RequestTimeout != time.Minute {
		t.Fatalf("unexpected upload limits: %+v", runtime.UploadService)
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

	req = httptest.NewRequest(http.MethodGet, "/api/settings", nil)
	rr = httptest.NewRecorder()
	runtime.Mux.ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("settings status: got=%d want=%d body=%s", rr.Code, http.StatusOK, rr.Body.String())
	}
	if !strings.Contains(rr.Body.String(), `"backend":"native"`) {
		t.Fatalf("expected default native annotation settings, got %s", rr.Body.String())
	}

	req = httptest.NewRequest(http.MethodPost, "/api/jobs/reannotate-all", nil)
	rr = httptest.NewRecorder()
	runtime.Mux.ServeHTTP(rr, req)
	if rr.Code != http.StatusOK || !strings.Contains(rr.Body.String(), `"queued_images"`) {
		t.Fatalf("reannotate-all: got=%d body=%s", rr.Code, rr.Body.String())
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

type healthAnnotator struct{}

func (healthAnnotator) AnnotateImage(context.Context, string) (embedder.ImageAnnotation, error) {
	return embedder.ImageAnnotation{}, nil
}

type healthTranscriber struct{}

func (healthTranscriber) TranscribeVideo(context.Context, string) (transcribe.Transcript, error) {
	return transcribe.Transcript{}, nil
}

func TestRuntimeHealthCapabilities(t *testing.T) {
	for _, tc := range []struct {
		name                                   string
		local, off, on, transcriber, annotator bool
		annotation, transcription, worker      string
	}{
		{"API cannot know worker config", false, true, false, false, false, "unknown", "unknown", "unknown"},
		{"disabled", true, true, false, false, true, "disabled", "disabled", "local"},
		{"unconfigured", true, false, true, false, false, "unconfigured", "unconfigured", "local"},
		{"transcriber configured", true, false, true, true, false, "unconfigured", "configured", "local"},
		{"annotations configured", true, false, false, false, true, "configured", "disabled", "local"},
		{"API even with local backends", false, false, true, true, true, "unknown", "unknown", "unknown"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			conn := openDB(t)
			if err := db.RunMigrations(context.Background(), conn); err != nil {
				t.Fatal(err)
			}
			opts := RuntimeOptions{Data: &DataRuntime{DB: conn}, DataDir: t.TempDir(), ModelID: 1, Embedder: fakeRuntimeEmbedder{}, Index: bruteforce.NewIndex(conn), WorkerEnabled: tc.local, AnnotationsDisabled: tc.off, VideoTranscriptsOn: tc.on}
			if tc.annotator {
				opts.Annotator = healthAnnotator{}
			}
			if tc.transcriber {
				opts.Transcriber = healthTranscriber{}
			}
			runtime, err := NewRuntime(opts)
			if err != nil {
				t.Fatal(err)
			}
			rr := httptest.NewRecorder()
			runtime.Mux.ServeHTTP(rr, httptest.NewRequest("GET", "/api/stats", nil))
			var got stats.Response
			if err := json.Unmarshal(rr.Body.Bytes(), &got); err != nil {
				t.Fatal(err)
			}
			want := stats.Capabilities{Worker: tc.worker, Annotations: tc.annotation, Transcription: tc.transcription}
			if got.Capabilities != want {
				t.Fatalf("got %+v want %+v", got.Capabilities, want)
			}
			if tc.local && tc.transcription != "configured" {
				rr = httptest.NewRecorder()
				runtime.Mux.ServeHTTP(rr, httptest.NewRequest("POST", "/api/jobs/retry-failed?kind=transcribe_video", nil))
				if rr.Code != 409 {
					t.Fatalf("disabled retry status %d", rr.Code)
				}
			}
		})
	}
}
