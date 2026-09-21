package app

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"imgsearch/internal/embedder"
	"imgsearch/internal/images"
	"imgsearch/internal/jobs"
	"imgsearch/internal/live"
	"imgsearch/internal/search"
	"imgsearch/internal/settings"
	"imgsearch/internal/stats"
	"imgsearch/internal/transcribe"
	"imgsearch/internal/upload"
	"imgsearch/internal/vectorindex"
	"imgsearch/internal/videos"
	"imgsearch/internal/webui"
	"imgsearch/internal/worker"
)

type Closer interface {
	Close() error
}

type RuntimeOptions struct {
	Data                 *DataRuntime
	DataDir              string
	ModelID              int64
	Embedder             embedder.Embedder
	Annotator            embedder.ImageAnnotator
	Transcriber          transcribe.VideoTranscriber
	Index                vectorindex.VectorIndex
	Closers              []Closer
	WorkerLeaseDuration  time.Duration
	WorkerRetryBaseDelay time.Duration
	LiveInterval         time.Duration
	LiveImagesLimit      int
	LiveImagesOffset     int
	VideoFrameCount      int
	VideoSampler         upload.VideoSampler
	VideoTranscriptsOn   bool
	// AnnotationDefaults is served by /api/settings until the user saves;
	// main seeds it from the annotator flags.
	AnnotationDefaults settings.AnnotationSettings
	// AnnotationConnectionTester probes a remote annotation backend for the
	// settings "Test connection" action. Nil disables that endpoint.
	AnnotationConnectionTester func(ctx context.Context, s settings.AnnotationSettings) error
	// AnnotationStatus reports the backend in use for the settings page.
	AnnotationStatus func(ctx context.Context) (settings.ActiveAnnotation, error)
	// NativeVariantLocked is true when native model paths are pinned by flags.
	NativeVariantLocked bool
	// AnnotationsDisabled is true when annotations are switched off by flag.
	AnnotationsDisabled bool
	// AnnotationModelLister lists a remote backend's models for the settings page.
	AnnotationModelLister func(ctx context.Context, s settings.AnnotationSettings) ([]string, error)
}

type Runtime struct {
	Data          *DataRuntime
	Mux           *http.ServeMux
	Queue         *worker.Queue
	UploadService *upload.Service
	Closers       []Closer
}

func NewRuntime(opts RuntimeOptions) (*Runtime, error) {
	if opts.Data == nil || opts.Data.DB == nil {
		return nil, fmt.Errorf("data runtime is nil")
	}
	if opts.DataDir == "" {
		return nil, fmt.Errorf("data dir is empty")
	}
	if opts.ModelID <= 0 {
		return nil, fmt.Errorf("model id must be positive")
	}
	if opts.Embedder == nil {
		return nil, fmt.Errorf("embedder is nil")
	}
	if opts.Index == nil {
		return nil, fmt.Errorf("vector index is nil")
	}
	if opts.WorkerLeaseDuration <= 0 {
		opts.WorkerLeaseDuration = 30 * time.Second
	}
	if opts.WorkerRetryBaseDelay <= 0 {
		opts.WorkerRetryBaseDelay = 5 * time.Second
	}
	if opts.LiveInterval <= 0 {
		opts.LiveInterval = 2 * time.Second
	}
	if opts.LiveImagesLimit <= 0 {
		opts.LiveImagesLimit = 120
	}
	if opts.AnnotationDefaults == (settings.AnnotationSettings{}) {
		opts.AnnotationDefaults = settings.DefaultAnnotation()
	}

	uploadSvc := &upload.Service{
		DB:                     opts.Data.DB,
		DataDir:                opts.DataDir,
		ModelID:                opts.ModelID,
		VideoFrameCount:        opts.VideoFrameCount,
		VideoSampler:           opts.VideoSampler,
		EnableVideoTranscripts: opts.VideoTranscriptsOn,
	}

	queue := &worker.Queue{
		DB:             opts.Data.DB,
		DataDir:        opts.DataDir,
		LeaseDuration:  opts.WorkerLeaseDuration,
		RetryBaseDelay: opts.WorkerRetryBaseDelay,
		Embedder:       opts.Embedder,
		TextEmbedder:   opts.Embedder,
		Annotator:      opts.Annotator,
		Transcriber:    opts.Transcriber,
		Index:          opts.Index,
	}

	mux := http.NewServeMux()
	mux.Handle("/api/upload", upload.NewHandler(uploadSvc))
	imageHandler := images.NewHandler(&images.Handler{DB: opts.Data.DB, ModelID: opts.ModelID, DataDir: opts.DataDir})
	videoHandler := videos.NewHandler(&videos.Handler{DB: opts.Data.DB, ModelID: opts.ModelID, DataDir: opts.DataDir})
	mux.Handle("/api/images", imageHandler)
	mux.Handle("/api/images/", imageHandler)
	mux.Handle("/api/videos", videoHandler)
	mux.Handle("/api/videos/", videoHandler)
	mux.Handle("/api/stats", stats.NewHandler(&stats.Handler{DB: opts.Data.DB, ModelID: opts.ModelID}))
	mux.Handle("/api/live", live.NewHandler(&live.Handler{DB: opts.Data.DB, ModelID: opts.ModelID, Interval: opts.LiveInterval, ImagesLimit: opts.LiveImagesLimit, ImagesOffset: opts.LiveImagesOffset}))
	mux.Handle("/api/jobs/retry-failed", jobs.NewRetryFailedHandler(&jobs.RetryFailedHandler{DB: opts.Data.DB, ModelID: opts.ModelID}))
	mux.Handle("/api/jobs/reannotate-all", jobs.NewReannotateAllHandler(&jobs.ReannotateAllHandler{DB: opts.Data.DB, ModelID: opts.ModelID}))
	settingsHandler := settings.NewHandler(&settings.Handler{
		DB:                  opts.Data.DB,
		Defaults:            opts.AnnotationDefaults,
		TestConnection:      opts.AnnotationConnectionTester,
		Status:              opts.AnnotationStatus,
		NativeVariantLocked: opts.NativeVariantLocked,
		AnnotationsDisabled: opts.AnnotationsDisabled,
		ListModels:          opts.AnnotationModelLister,
	})
	mux.Handle("/api/settings", settingsHandler)
	mux.Handle("/api/settings/", settingsHandler)
	mux.Handle("/api/search/", search.NewHandler(&search.Handler{
		DB:       opts.Data.DB,
		ModelID:  opts.ModelID,
		DataDir:  opts.DataDir,
		Embedder: opts.Embedder,
		Index:    opts.Index,
	}))
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})
	mux.Handle("/", webui.NewHandler(opts.DataDir))

	return &Runtime{
		Data:          opts.Data,
		Mux:           mux,
		Queue:         queue,
		UploadService: uploadSvc,
		Closers:       opts.Closers,
	}, nil
}

func (r *Runtime) Close() error {
	if r == nil {
		return nil
	}
	var firstErr error
	for i := len(r.Closers) - 1; i >= 0; i-- {
		if r.Closers[i] == nil {
			continue
		}
		if err := r.Closers[i].Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if err := r.Data.Close(); err != nil && firstErr == nil {
		firstErr = err
	}
	return firstErr
}
