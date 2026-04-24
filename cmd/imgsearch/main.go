package main

import (
	"context"
	"database/sql"
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"imgsearch/internal/app"
	"imgsearch/internal/db"
	"imgsearch/internal/embedder"
	"imgsearch/internal/httputil"
	"imgsearch/internal/images"
	"imgsearch/internal/jobs"
	"imgsearch/internal/live"
	"imgsearch/internal/search"
	"imgsearch/internal/stats"
	"imgsearch/internal/transcribe"
	"imgsearch/internal/transcribe/parakeetonnx"
	"imgsearch/internal/upload"
	"imgsearch/internal/vectorindex"
	"imgsearch/internal/vectorindex/sqlitevector"
	"imgsearch/internal/videos"
	"imgsearch/internal/webui"
	"imgsearch/internal/worker"
)

const (
	defaultHTTPReadHeaderTimeout = 5 * time.Second
	defaultHTTPReadTimeout       = 30 * time.Second
	defaultAPIKey                = "imgsearch-dev-default-api-key"
	// WebSocket handlers manage per-message deadlines after upgrade.
	defaultHTTPWriteTimeout    = 60 * time.Second
	defaultHTTPIdleTimeout     = 120 * time.Second
	defaultHTTPMaxHeaderBytes  = 1 << 20
	defaultHTTPShutdownTimeout = 10 * time.Second
)

func main() {
	cfg := defaultRuntimeConfig(os.Getenv)
	registerRuntimeFlags(flag.CommandLine, &cfg)
	flag.Parse()
	if err := cfg.Resolve(); err != nil {
		log.Fatalf("%v", err)
	}
	rootCtx, stopRoot := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stopRoot()

	mode := cfg.Mode
	if mode.startsHTTP() {
		if cfg.UsingDefaultAPIKey {
			log.Printf("WARNING: using built-in default API key; set -api-key or IMGSEARCH_API_KEY for non-development use")
		}
	}
	if cfg.AnnotationModeWarning != "" {
		log.Printf("%s", cfg.AnnotationModeWarning)
	}

	if err := os.MkdirAll(cfg.DataDir, 0o755); err != nil {
		log.Fatalf("create data directory: %v", err)
	}

	dbPath := filepath.Join(cfg.DataDir, "imgsearch.sqlite")
	dsn := fmt.Sprintf("%s?_busy_timeout=30000", dbPath)

	resolvedSQLiteVectorPath, err := discoverSQLiteVectorPath(cfg.SQLiteVectorPath)
	if err != nil {
		log.Fatalf("discover sqlite-vector extension: %v", err)
	}

	autoValidationErr := error(nil)
	openWithVector := ""
	if cfg.VectorBackend == vectorBackendAuto {
		if resolvedSQLiteVectorPath == "" {
			autoValidationErr = fmt.Errorf("sqlite-vector extension path not found (run `mise run sqlite-vector-setup` or set SQLITE_VECTOR_PATH)")
		} else {
			openWithVector = resolvedSQLiteVectorPath
		}
	}
	if cfg.VectorBackend == vectorBackendSQLiteVector {
		if resolvedSQLiteVectorPath == "" {
			log.Fatalf("sqlite-vector backend requested but extension path was not found (set -sqlite-vector-path or SQLITE_VECTOR_PATH)")
		}
		openWithVector = resolvedSQLiteVectorPath
	}

	sqlDB, err := openSQLiteDB(dsn, openWithVector)
	if err != nil {
		if cfg.VectorBackend == vectorBackendAuto && openWithVector != "" {
			autoValidationErr = err
			sqlDB, err = openSQLiteDB(dsn, "")
			if err != nil {
				log.Fatalf("open sqlite database: %v", err)
			}
			openWithVector = ""
		} else {
			log.Fatalf("open sqlite database: %v", err)
		}
	}

	if cfg.VectorBackend == vectorBackendAuto && openWithVector != "" && autoValidationErr == nil {
		autoValidationErr = sqlitevector.ValidateAvailable(rootCtx, sqlDB)
	}

	resolvedVectorBackend, backendWarning, err := resolveVectorBackend(cfg.VectorBackend, autoValidationErr)
	if err != nil {
		log.Fatalf("configure vector backend: %v", err)
	}
	if resolvedVectorBackend == vectorBackendBruteForce && openWithVector != "" {
		_ = sqlDB.Close()
		sqlDB, err = openSQLiteDB(dsn, "")
		if err != nil {
			log.Fatalf("open sqlite database: %v", err)
		}
	}

	if backendWarning != "" {
		log.Printf("%s", backendWarning)
	}
	sqlDB.SetMaxOpenConns(1)
	sqlDB.SetMaxIdleConns(1)
	defer func() { _ = sqlDB.Close() }()

	index, validateVectorBackend, err := buildVectorBackend(resolvedVectorBackend, sqlDB)
	if err != nil {
		log.Fatalf("build vector backend: %v", err)
	}

	if err := app.Bootstrap(rootCtx, sqlDB, validateVectorBackend); err != nil {
		log.Fatalf("bootstrap app: %v", err)
	}

	cfg.LlamaNativeModelPath, cfg.LlamaNativeMMProjPath, err = ensureDefaultLlamaNativeAssets(rootCtx, cfg.LlamaNativeModelPath, cfg.LlamaNativeMMProjPath)
	if err != nil {
		log.Fatalf("resolve llama-cpp-native model assets: %v", err)
	}
	loadAnnotator := cfg.LoadAnnotator
	cfg.AnnotatorModelPath, cfg.AnnotatorMMProjPath, err = resolveAnnotatorAssetPaths(
		rootCtx,
		cfg.EnableAnnotations,
		loadAnnotator,
		cfg.AnnotatorVariant,
		cfg.AnnotatorModelPath,
		cfg.AnnotatorMMProjPath,
		ensureDefaultLlamaNativeAnnotatorAssetsForVariant,
	)
	if err != nil {
		log.Fatalf("resolve llama-cpp-native annotator model assets: %v", err)
	}

	embedDimensions := cfg.LlamaNativeDimensions

	resolvedLlamaNativeImageMaxSide := cfg.ResolvedLlamaNativeImageMaxSide
	resolvedLlamaNativeImageMaxTokens := cfg.ResolvedLlamaNativeImageMaxTokens

	modelSpec, err := llamaNativeEmbeddingModelSpec(embedDimensions)
	if err != nil {
		log.Fatalf("configure model spec: %v", err)
	}
	modelSpec.Version = llamaNativeModelVersion(
		cfg.LlamaNativeModelPath,
		cfg.LlamaNativeMMProjPath,
		modelSpec.Dimensions,
		resolvedLlamaNativeImageMaxSide,
		resolvedLlamaNativeImageMaxTokens,
		cfg.LlamaNativeQueryInstruction,
		cfg.LlamaNativePassageInstruction,
	)

	modelID, err := db.EnsureEmbeddingModel(rootCtx, sqlDB, modelSpec)
	if err != nil {
		log.Fatalf("ensure embedding model: %v", err)
	}
	purgedEmbeddings, err := db.PurgeOtherModelEmbeddings(rootCtx, sqlDB, modelID)
	if err != nil {
		log.Fatalf("purge old embeddings: %v", err)
	}
	if purgedEmbeddings > 0 {
		log.Printf("purged %d embeddings for inactive model versions", purgedEmbeddings)
	}
	purgedVideoTranscriptEmbeddings, err := db.PurgeOtherModelVideoTranscriptEmbeddings(rootCtx, sqlDB, modelID)
	if err != nil {
		log.Fatalf("purge old video transcript embeddings: %v", err)
	}
	if purgedVideoTranscriptEmbeddings > 0 {
		log.Printf("purged %d video transcript embeddings for inactive model versions", purgedVideoTranscriptEmbeddings)
	}
	purgedJobs, err := db.PurgeOtherModelIndexJobs(rootCtx, sqlDB, modelID)
	if err != nil {
		log.Fatalf("purge old index jobs: %v", err)
	}
	if purgedJobs > 0 {
		log.Printf("purged %d index jobs for inactive model versions", purgedJobs)
	}

	enqueuedMissing, err := db.EnsureIndexJobsForModel(rootCtx, sqlDB, modelID)
	if err != nil {
		log.Fatalf("ensure model index jobs: %v", err)
	}
	if enqueuedMissing > 0 {
		log.Printf("enqueued %d missing index jobs for model_id=%d", enqueuedMissing, modelID)
	}
	enqueuedAnnotationJobs, err := db.EnsureAnnotationJobsForModel(rootCtx, sqlDB, modelID)
	if err != nil {
		log.Fatalf("ensure annotation jobs: %v", err)
	}
	if enqueuedAnnotationJobs > 0 {
		log.Printf("enqueued %d annotation jobs for model_id=%d", enqueuedAnnotationJobs, modelID)
	}
	enqueuedVideoAnnotationJobs, err := db.EnsureVideoAnnotationJobsForModel(rootCtx, sqlDB, modelID)
	if err != nil {
		log.Fatalf("ensure video annotation jobs: %v", err)
	}
	if enqueuedVideoAnnotationJobs > 0 {
		log.Printf("enqueued %d video annotation jobs for model_id=%d", enqueuedVideoAnnotationJobs, modelID)
	}

	embedderOpts := llamaCPPNativeEmbedderOptions{
		ModelPath:             cfg.LlamaNativeModelPath,
		VisionModelPath:       cfg.LlamaNativeMMProjPath,
		Dimensions:            modelSpec.Dimensions,
		GPULayers:             cfg.LlamaNativeGPULayers,
		UseGPU:                cfg.LlamaNativeUseGPU,
		ContextSize:           cfg.LlamaNativeContextSize,
		BatchSize:             cfg.LlamaNativeBatchSize,
		MaxSequences:          cfg.LlamaNativeMaxSequences,
		Threads:               cfg.LlamaNativeThreads,
		ImageMaxSide:          resolvedLlamaNativeImageMaxSide,
		ImageMaxTokens:        resolvedLlamaNativeImageMaxTokens,
		AnnotationTemperature: cfg.LlamaNativeAnnotationTemperature,
		AnnotationSeed:        cfg.LlamaNativeAnnotationSeed,
		QueryInstruction:      cfg.LlamaNativeQueryInstruction,
		PassageInstruction:    cfg.LlamaNativePassageInstruction,
		FlashAttnType:         cfg.LlamaNativeFlashAttnType,
		CacheTypeK:            cfg.LlamaNativeCacheTypeK,
		CacheTypeV:            cfg.LlamaNativeCacheTypeV,
	}
	activeEmbedder, err := newLlamaCPPNativeEmbedder(embedderOpts)
	if err != nil {
		log.Fatalf("configure embedder: %v", err)
	}

	var videoTranscriber transcribe.VideoTranscriber
	resolvedParakeetBundleDir := strings.TrimSpace(cfg.ParakeetONNXBundleDir)
	if strings.TrimSpace(cfg.ParakeetONNXRuntimeLib) != "" {
		resolvedParakeetBundleDir, err = ensureDefaultParakeetOnnxAssets(rootCtx, resolvedParakeetBundleDir)
		if err != nil {
			log.Fatalf("ensure Parakeet ONNX assets: %v", err)
		}
	}
	if resolvedParakeetBundleDir != "" && strings.TrimSpace(cfg.ParakeetONNXRuntimeLib) != "" {
		videoTranscriber = &parakeetonnx.Recognizer{Config: parakeetonnx.RecognizerConfig{
			ONNXRuntimeLib: strings.TrimSpace(cfg.ParakeetONNXRuntimeLib),
			BundleDir:      resolvedParakeetBundleDir,
			UseCoreML:      cfg.ParakeetONNXCoreML,
		}}
		if closer, ok := videoTranscriber.(interface{ Close() error }); ok {
			defer func() { _ = closer.Close() }()
		}
		enqueuedTranscriptJobs, err := db.EnsureVideoTranscriptJobsForModel(rootCtx, sqlDB, modelID)
		if err != nil {
			log.Fatalf("ensure video transcript jobs: %v", err)
		}
		if enqueuedTranscriptJobs > 0 {
			log.Printf("enqueued %d video transcript jobs for model_id=%d", enqueuedTranscriptJobs, modelID)
		}
		log.Printf("video transcription enabled via Parakeet ONNX bundle %s", resolvedParakeetBundleDir)
	} else {
		log.Printf("video transcription disabled")
	}

	var imageAnnotator embedder.ImageAnnotator
	canAnnotateImages := false
	annotatorModelPath := strings.TrimSpace(cfg.AnnotatorModelPath)
	annotatorVisionPath := strings.TrimSpace(cfg.AnnotatorMMProjPath)
	var modelSwitchboard *llamaModelSwitchboard
	if loadAnnotator {
		imageAnnotator, canAnnotateImages = activeEmbedder.(embedder.ImageAnnotator)
	}
	if loadAnnotator && (annotatorModelPath != "" || annotatorVisionPath != "") {
		if annotatorModelPath == "" || annotatorVisionPath == "" {
			log.Fatalf("configure annotator: both -llama-native-annotator-model-path and -llama-native-annotator-mmproj-path must be set together")
		}
		annotatorOpts := llamaCPPNativeAnnotatorOptions{
			ModelPath:             annotatorModelPath,
			VisionModelPath:       annotatorVisionPath,
			GPULayers:             cfg.AnnotatorGPULayers,
			UseGPU:                cfg.AnnotatorUseGPU,
			ContextSize:           cfg.AnnotatorContextSize,
			BatchSize:             cfg.AnnotatorBatchSize,
			Threads:               cfg.AnnotatorThreads,
			ImageMaxSide:          cfg.AnnotatorImageMaxSide,
			ImageMaxTokens:        cfg.AnnotatorImageMaxTokens,
			AnnotationTemperature: cfg.LlamaNativeAnnotationTemperature,
			AnnotationSeed:        cfg.LlamaNativeAnnotationSeed,
			FlashAttnType:         cfg.AnnotatorFlashAttnType,
			CacheTypeK:            cfg.AnnotatorCacheTypeK,
			CacheTypeV:            cfg.AnnotatorCacheTypeV,
		}
		modelSwitchboard = newLlamaModelSwitchboard(
			activeEmbedder,
			func(context.Context) (embedder.Embedder, error) {
				return newLlamaCPPNativeEmbedder(embedderOpts)
			},
			func(context.Context) (embedder.ImageAnnotator, error) {
				return newLlamaCPPNativeAnnotator(annotatorOpts)
			},
			true,
		)
		activeEmbedder = modelSwitchboard.Embedder()
		imageAnnotator = modelSwitchboard.Annotator()
		canAnnotateImages = true
		log.Printf("image annotations enabled via separate native annotator model %s", annotatorModelPath)
	} else if loadAnnotator && canAnnotateImages {
		log.Printf("image annotations enabled via active embedder model")
	} else {
		log.Printf("image annotations disabled")
	}

	if modelSwitchboard != nil {
		defer func() { _ = modelSwitchboard.Close() }()
	} else if closer, ok := activeEmbedder.(interface{ Close() error }); ok {
		defer func() { _ = closer.Close() }()
	}

	uploadSvc := &upload.Service{
		DB:                     sqlDB,
		DataDir:                cfg.DataDir,
		ModelID:                modelID,
		EnableVideoTranscripts: videoTranscriber != nil,
	}

	queue := &worker.Queue{
		DB:             sqlDB,
		DataDir:        cfg.DataDir,
		LeaseDuration:  30 * time.Second,
		RetryBaseDelay: 5 * time.Second,
		Embedder:       activeEmbedder,
		TextEmbedder:   activeEmbedder,
		Annotator:      imageAnnotator,
		Transcriber:    videoTranscriber,
		Index:          index,
	}
	log.Printf("imgsearch initialized with database %s", dbPath)
	log.Printf("using vector backend %s", resolvedVectorBackend)
	log.Printf("running in %s mode", mode)

	var workerDone chan struct{}
	if mode.startsWorker() {
		if mode.startsHTTP() {
			workerDone = make(chan struct{})
			go func() {
				defer close(workerDone)
				worker.RunLoopBatch(rootCtx, queue, "main-worker", 500*time.Millisecond, cfg.WorkerBatchSize)
			}()
		} else {
			worker.RunLoopBatch(rootCtx, queue, "main-worker", 500*time.Millisecond, cfg.WorkerBatchSize)
			return
		}
	}

	if mode.startsHTTP() {
		mux := newServerMux(sqlDB, cfg.DataDir, modelID, activeEmbedder, index, uploadSvc)
		handler := withAPISecurity(mux, cfg.ResolvedAPIKey)
		server := configuredHTTPServer(cfg.Addr, handler)
		log.Printf(
			"http limits read_header_timeout=%s read_timeout=%s write_timeout=%s idle_timeout=%s max_header_bytes=%d",
			server.ReadHeaderTimeout,
			server.ReadTimeout,
			server.WriteTimeout,
			server.IdleTimeout,
			server.MaxHeaderBytes,
		)
		log.Printf("listening on http://%s", cfg.Addr)
		if err := serveHTTPWithShutdown(rootCtx, server); err != nil {
			stopRoot()
			waitForWorkerShutdown(workerDone, defaultHTTPShutdownTimeout)
			log.Fatalf("serve http: %v", err)
		}
		stopRoot()
		waitForWorkerShutdown(workerDone, defaultHTTPShutdownTimeout)
	}
}

func serveHTTPWithShutdown(ctx context.Context, server *http.Server) error {
	if server == nil {
		return fmt.Errorf("http server is nil")
	}
	errCh := make(chan error, 1)
	go func() {
		errCh <- server.ListenAndServe()
	}()

	select {
	case err := <-errCh:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return err
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), defaultHTTPShutdownTimeout)
		defer cancel()
		if err := server.Shutdown(shutdownCtx); err != nil {
			return fmt.Errorf("shutdown http server: %w", err)
		}
		err := <-errCh
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return err
	}
}

func waitForWorkerShutdown(done <-chan struct{}, timeout time.Duration) {
	if done == nil {
		return
	}
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case <-done:
	case <-timer.C:
		log.Printf("worker shutdown timed out after %s", timeout)
	}
}

func resolveAPIKey(raw string) (string, bool) {
	apiKey := strings.TrimSpace(raw)
	if apiKey != "" {
		return apiKey, false
	}
	return defaultAPIKey, true
}

func configuredHTTPServer(addr string, handler http.Handler) *http.Server {
	return &http.Server{
		Addr:              addr,
		Handler:           handler,
		ReadHeaderTimeout: defaultHTTPReadHeaderTimeout,
		ReadTimeout:       defaultHTTPReadTimeout,
		WriteTimeout:      defaultHTTPWriteTimeout,
		IdleTimeout:       defaultHTTPIdleTimeout,
		MaxHeaderBytes:    defaultHTTPMaxHeaderBytes,
	}
}

func withAPISecurity(handler http.Handler, apiKey string) http.Handler {
	h := httputil.NewAPIAuthMiddleware(apiKey)(handler)
	h = httputil.NewAPIKeyCookieMiddleware(apiKey)(h)
	return h
}

func validateHTTPExposure(addr string, apiKey string, usingDefaultAPIKey bool) error {
	if isLoopbackListenAddress(addr) {
		return nil
	}
	if strings.TrimSpace(apiKey) == "" || usingDefaultAPIKey {
		return fmt.Errorf("non-loopback listen address %q requires an explicit -api-key (or IMGSEARCH_API_KEY); built-in development key is not allowed", addr)
	}
	return nil
}

func isLoopbackListenAddress(addr string) bool {
	addr = strings.TrimSpace(addr)
	if addr == "" {
		return false
	}

	host := addr
	if strings.HasPrefix(addr, ":") {
		host = ""
	} else if parsedHost, _, err := net.SplitHostPort(addr); err == nil {
		host = parsedHost
	}
	host = strings.Trim(strings.TrimSpace(host), "[]")
	if host == "" {
		return false
	}
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	if ip == nil {
		return false
	}
	return ip.IsLoopback()
}

func newServerMux(
	sqlDB *sql.DB,
	dataDir string,
	modelID int64,
	embedder embedder.Embedder,
	index vectorindex.VectorIndex,
	uploadSvc *upload.Service,
) *http.ServeMux {
	mux := http.NewServeMux()
	mux.Handle("/api/upload", upload.NewHandler(uploadSvc))
	imageHandler := images.NewHandler(&images.Handler{DB: sqlDB, ModelID: modelID, DataDir: dataDir})
	videoHandler := videos.NewHandler(&videos.Handler{DB: sqlDB, ModelID: modelID, DataDir: dataDir})
	mux.Handle("/api/images", imageHandler)
	mux.Handle("/api/images/", imageHandler)
	mux.Handle("/api/videos", videoHandler)
	mux.Handle("/api/videos/", videoHandler)
	mux.Handle("/api/stats", stats.NewHandler(&stats.Handler{DB: sqlDB, ModelID: modelID}))
	mux.Handle("/api/live", live.NewHandler(&live.Handler{DB: sqlDB, ModelID: modelID, Interval: 2 * time.Second, ImagesLimit: 120, ImagesOffset: 0}))
	mux.Handle("/api/jobs/retry-failed", jobs.NewRetryFailedHandler(&jobs.RetryFailedHandler{DB: sqlDB, ModelID: modelID}))
	searchHandler := search.NewHandler(&search.Handler{
		DB:       sqlDB,
		ModelID:  modelID,
		DataDir:  dataDir,
		Embedder: embedder,
		Index:    index,
	})
	mux.Handle("/api/search/", searchHandler)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})
	mux.Handle("/", webui.NewHandler(dataDir))
	return mux
}
