package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"imgsearch/internal/app"
	"imgsearch/internal/db"
	"imgsearch/internal/embedder"
	"imgsearch/internal/httputil"
	"imgsearch/internal/transcribe"
	"imgsearch/internal/transcribe/parakeetonnx"
	"imgsearch/internal/vectorindex/sqlitevector"
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

	dataRuntime, err := app.InitializeDataRuntime(rootCtx, app.DataRuntimeConfig{
		DataDir:                cfg.DataDir,
		SQLiteVectorPath:       cfg.SQLiteVectorPath,
		RequestedVectorBackend: cfg.VectorBackend,
	}, app.DataRuntimeDeps{
		DiscoverSQLiteVectorPath: discoverSQLiteVectorPath,
		OpenSQLiteDB:             openSQLiteDB,
		ValidateSQLiteVector:     sqlitevector.ValidateAvailable,
		ResolveVectorBackend:     resolveVectorBackend,
		BuildVectorBackend:       buildVectorBackend,
	})
	if err != nil {
		log.Fatalf("initialize data runtime: %v", err)
	}
	sqlDB := dataRuntime.DB
	index := dataRuntime.Index
	dataRuntimeOwned := false
	defer func() {
		if !dataRuntimeOwned {
			_ = dataRuntime.Close()
		}
	}()

	if dataRuntime.BackendWarning != "" {
		log.Printf("%s", dataRuntime.BackendWarning)
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

	var closers []app.Closer
	if closer, ok := videoTranscriber.(app.Closer); ok {
		closers = append(closers, closer)
	}
	if modelSwitchboard != nil {
		closers = append(closers, modelSwitchboard)
	} else if closer, ok := activeEmbedder.(app.Closer); ok {
		closers = append(closers, closer)
	}

	runtime, err := app.NewRuntime(app.RuntimeOptions{
		Data:                 dataRuntime,
		DataDir:              cfg.DataDir,
		ModelID:              modelID,
		Embedder:             activeEmbedder,
		Annotator:            imageAnnotator,
		Transcriber:          videoTranscriber,
		Index:                index,
		Closers:              closers,
		WorkerLeaseDuration:  30 * time.Second,
		WorkerRetryBaseDelay: 5 * time.Second,
		LiveInterval:         2 * time.Second,
		LiveImagesLimit:      120,
		LiveImagesOffset:     0,
		VideoTranscriptsOn:   videoTranscriber != nil,
	})
	if err != nil {
		log.Fatalf("compose runtime: %v", err)
	}
	dataRuntimeOwned = true
	defer func() { _ = runtime.Close() }()

	log.Printf("imgsearch initialized with database %s", dataRuntime.DBPath)
	log.Printf("using vector backend %s", dataRuntime.VectorBackend)
	log.Printf("running in %s mode", mode)

	var workerDone chan struct{}
	if mode.startsWorker() {
		if mode.startsHTTP() {
			workerDone = make(chan struct{})
			go func() {
				defer close(workerDone)
				worker.RunLoopBatch(rootCtx, runtime.Queue, "main-worker", 500*time.Millisecond, cfg.WorkerBatchSize)
			}()
		} else {
			worker.RunLoopBatch(rootCtx, runtime.Queue, "main-worker", 500*time.Millisecond, cfg.WorkerBatchSize)
			return
		}
	}

	if mode.startsHTTP() {
		handler := withAPISecurity(runtime.Mux, cfg.ResolvedAPIKey)
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
