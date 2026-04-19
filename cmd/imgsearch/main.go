package main

import (
	"context"
	"database/sql"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"imgsearch/internal/app"
	"imgsearch/internal/db"
	"imgsearch/internal/embedder"
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

func main() {
	dataDir := flag.String("data-dir", "./data", "data directory")
	addr := flag.String("addr", "127.0.0.1:8080", "http listen address")
	modeFlag := flag.String("mode", string(runtimeModeAll), "process mode: all, api, or worker")
	workerBatchSize := flag.Int("worker-batch-size", 1, "number of jobs to claim per batch (1 disables batching)")
	llamaNativeModelPath := flag.String("llama-native-model-path", defaultLlamaNativeModelPath, "path to the llama.cpp GGUF embedding model")
	llamaNativeMMProjPath := flag.String("llama-native-mmproj-path", defaultLlamaNativeMMProjPath, "path to the llama.cpp GGUF mmproj model")
	llamaNativeDimensions := flag.Int("llama-native-dimensions", defaultLlamaNativeDimensions, "embedding dimensions for llama-cpp-native model metadata")
	llamaNativeGPULayers := flag.Int("llama-native-gpu-layers", 99, "number of layers to offload for llama-cpp-native")
	llamaNativeUseGPU := flag.Bool("llama-native-use-gpu", true, "whether llama-cpp-native should use GPU for mtmd/mmproj")
	llamaNativeContextSize := flag.Int("llama-native-context-size", defaultLlamaNativeEmbedderContextSize, "context size for llama-cpp-native runtime")
	llamaNativeBatchSize := flag.Int("llama-native-batch-size", 512, "batch size for llama-cpp-native runtime")
	llamaNativeMaxSequences := flag.Int("llama-native-max-sequences", 1, "maximum independent sequences for llama-cpp-native embedding batches")
	llamaNativeThreads := flag.Int("llama-native-threads", 0, "thread count for llama-cpp-native runtime (0 uses backend default)")
	llamaNativeFlashAttnType := flag.Int("llama-native-flash-attn", defaultLlamaNativeFlashAttnType, "flash attention type for llama-cpp-native (-1=auto, 0=off, 1=on)")
	llamaNativeCacheTypeK := flag.Int("llama-native-cache-type-k", defaultLlamaNativeCacheTypeK, "K cache type for llama-cpp-native (-1=default, 0=f32, 1=f16, 8=q8_0)")
	llamaNativeCacheTypeV := flag.Int("llama-native-cache-type-v", defaultLlamaNativeCacheTypeV, "V cache type for llama-cpp-native (-1=default, 0=f32, 1=f16, 8=q8_0)")
	llamaNativeImageMaxSide := flag.Int("llama-native-image-max-side", 384, "maximum image side length used before llama-cpp-native embedding")
	llamaNativeImageMaxTokens := flag.Int("llama-native-image-max-tokens", 0, "optional maximum image tokens override for llama-cpp-native mtmd preprocessing (0 uses model default)")
	llamaNativeQueryInstruction := flag.String("llama-native-query-instruction", "Retrieve images or text relevant to the user's query.", "instruction used for llama-cpp-native text query embeddings")
	llamaNativePassageInstruction := flag.String("llama-native-passage-instruction", "Represent this image or text for retrieval.", "instruction used for llama-cpp-native image/document embeddings")
	parakeetBundleDefault := strings.TrimSpace(os.Getenv("PARAKEET_ONNX_BUNDLE_DIR"))
	if parakeetBundleDefault == "" {
		parakeetBundleDefault = defaultParakeetOnnxBundleDir
	}
	parakeetOnnxBundleDir := flag.String("parakeet-onnx-bundle-dir", parakeetBundleDefault, "directory containing the Parakeet ONNX bundle for video transcription")
	parakeetOnnxRuntimeLib := flag.String("parakeet-onnxruntime-lib", strings.TrimSpace(os.Getenv("PARAKEET_ONNXRUNTIME_LIB")), "path to the ONNX Runtime shared library used for Parakeet video transcription")
	parakeetOnnxCoreML := flag.Bool("parakeet-onnx-coreml", os.Getenv("PARAKEET_ONNX_COREML") == "1", "whether to request the CoreML execution provider for Parakeet ONNX video transcription")
	enableAnnotations := flag.Bool("enable-annotations", true, "whether to load and run image annotation models")
	llamaNativeAnnotatorModelPath := flag.String("llama-native-annotator-model-path", "", "optional path to a separate llama.cpp GGUF vision model used only for image descriptions/tags")
	llamaNativeAnnotatorMMProjPath := flag.String("llama-native-annotator-mmproj-path", "", "optional path to a separate llama.cpp GGUF mmproj used only for image descriptions/tags")
	llamaNativeAnnotatorVariant := flag.String("llama-native-annotator-variant", defaultLlamaNativeAnnotatorVariant, "default separate llama.cpp annotator model variant when explicit annotator paths are not set: e4b or 26b")
	llamaNativeAnnotatorGPULayers := flag.Int("llama-native-annotator-gpu-layers", 99, "number of layers to offload for the separate llama-cpp-native annotator")
	llamaNativeAnnotatorUseGPU := flag.Bool("llama-native-annotator-use-gpu", true, "whether the separate llama-cpp-native annotator should use GPU for mtmd/mmproj")
	llamaNativeAnnotatorContextSize := flag.Int("llama-native-annotator-context-size", 8192, "context size for the separate llama-cpp-native annotator")
	llamaNativeAnnotatorBatchSize := flag.Int("llama-native-annotator-batch-size", 512, "batch size for the separate llama-cpp-native annotator")
	llamaNativeAnnotatorThreads := flag.Int("llama-native-annotator-threads", 0, "thread count for the separate llama-cpp-native annotator (0 uses backend default)")
	llamaNativeAnnotatorFlashAttnType := flag.Int("llama-native-annotator-flash-attn", defaultLlamaNativeFlashAttnType, "flash attention type for the separate annotator (-1=auto, 0=off, 1=on)")
	llamaNativeAnnotatorCacheTypeK := flag.Int("llama-native-annotator-cache-type-k", defaultLlamaNativeCacheTypeK, "K cache type for the separate annotator (-1=default, 0=f32, 1=f16, 8=q8_0)")
	llamaNativeAnnotatorCacheTypeV := flag.Int("llama-native-annotator-cache-type-v", defaultLlamaNativeCacheTypeV, "V cache type for the separate annotator (-1=default, 0=f32, 1=f16, 8=q8_0)")
	llamaNativeAnnotatorImageMaxSide := flag.Int("llama-native-annotator-image-max-side", 1024, "maximum image side length used before separate llama-cpp-native annotation generation")
	llamaNativeAnnotatorImageMaxTokens := flag.Int("llama-native-annotator-image-max-tokens", 0, "optional maximum image tokens override for the separate llama-cpp-native annotator (0 uses model default)")
	vectorBackend := flag.String("vector-backend", vectorBackendAuto, "vector backend: auto, sqlite-vector, bruteforce")
	sqliteVectorPath := flag.String("sqlite-vector-path", "", "path to sqlite-vector extension binary (optional: defaults to SQLITE_VECTOR_PATH or tools/sqlite-vector/vector)")
	flag.Parse()

	mode, err := resolveRuntimeMode(*modeFlag)
	if err != nil {
		log.Fatalf("configure mode: %v", err)
	}
	if warning := annotationModeWarning(mode, *enableAnnotations, *llamaNativeAnnotatorModelPath, *llamaNativeAnnotatorMMProjPath); warning != "" {
		log.Printf("%s", warning)
	}

	if err := os.MkdirAll(*dataDir, 0o755); err != nil {
		log.Fatalf("create data directory: %v", err)
	}

	dbPath := filepath.Join(*dataDir, "imgsearch.sqlite")
	dsn := fmt.Sprintf("%s?_busy_timeout=30000", dbPath)

	resolvedSQLiteVectorPath, err := discoverSQLiteVectorPath(*sqliteVectorPath)
	if err != nil {
		log.Fatalf("discover sqlite-vector extension: %v", err)
	}

	autoValidationErr := error(nil)
	openWithVector := ""
	if *vectorBackend == vectorBackendAuto {
		if resolvedSQLiteVectorPath == "" {
			autoValidationErr = fmt.Errorf("sqlite-vector extension path not found (run `mise run sqlite-vector-setup` or set SQLITE_VECTOR_PATH)")
		} else {
			openWithVector = resolvedSQLiteVectorPath
		}
	}
	if *vectorBackend == vectorBackendSQLiteVector {
		if resolvedSQLiteVectorPath == "" {
			log.Fatalf("sqlite-vector backend requested but extension path was not found (set -sqlite-vector-path or SQLITE_VECTOR_PATH)")
		}
		openWithVector = resolvedSQLiteVectorPath
	}

	sqlDB, err := openSQLiteDB(dsn, openWithVector)
	if err != nil {
		if *vectorBackend == vectorBackendAuto && openWithVector != "" {
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

	if *vectorBackend == vectorBackendAuto && openWithVector != "" && autoValidationErr == nil {
		autoValidationErr = sqlitevector.ValidateAvailable(context.Background(), sqlDB)
	}

	resolvedVectorBackend, backendWarning, err := resolveVectorBackend(*vectorBackend, autoValidationErr)
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

	if err := app.Bootstrap(context.Background(), sqlDB, validateVectorBackend); err != nil {
		log.Fatalf("bootstrap app: %v", err)
	}

	*llamaNativeModelPath, *llamaNativeMMProjPath, err = ensureDefaultLlamaNativeAssets(context.Background(), *llamaNativeModelPath, *llamaNativeMMProjPath)
	if err != nil {
		log.Fatalf("resolve llama-cpp-native model assets: %v", err)
	}
	loadAnnotator := shouldLoadAnnotator(mode, *enableAnnotations)
	*llamaNativeAnnotatorModelPath, *llamaNativeAnnotatorMMProjPath, err = resolveAnnotatorAssetPaths(
		context.Background(),
		*enableAnnotations,
		loadAnnotator,
		*llamaNativeAnnotatorVariant,
		*llamaNativeAnnotatorModelPath,
		*llamaNativeAnnotatorMMProjPath,
		ensureDefaultLlamaNativeAnnotatorAssetsForVariant,
	)
	if err != nil {
		log.Fatalf("resolve llama-cpp-native annotator model assets: %v", err)
	}

	embedDimensions := *llamaNativeDimensions
	if embedDimensions <= 0 {
		log.Fatalf("configure embedder dimensions: llama-cpp-native dimensions must be positive")
	}

	resolvedLlamaNativeImageMaxSide, resolvedLlamaNativeImageMaxTokens, err := resolveLlamaCPPNativeImageLimits(*llamaNativeImageMaxSide, *llamaNativeImageMaxTokens)
	if err != nil {
		log.Fatalf("configure llama-cpp-native options: %v", err)
	}

	modelSpec, err := llamaNativeEmbeddingModelSpec(embedDimensions)
	if err != nil {
		log.Fatalf("configure model spec: %v", err)
	}
	modelSpec.Version = llamaNativeModelVersion(
		*llamaNativeModelPath,
		*llamaNativeMMProjPath,
		modelSpec.Dimensions,
		resolvedLlamaNativeImageMaxSide,
		resolvedLlamaNativeImageMaxTokens,
		*llamaNativeQueryInstruction,
		*llamaNativePassageInstruction,
	)

	modelID, err := db.EnsureEmbeddingModel(context.Background(), sqlDB, modelSpec)
	if err != nil {
		log.Fatalf("ensure embedding model: %v", err)
	}
	purgedEmbeddings, err := db.PurgeOtherModelEmbeddings(context.Background(), sqlDB, modelID)
	if err != nil {
		log.Fatalf("purge old embeddings: %v", err)
	}
	if purgedEmbeddings > 0 {
		log.Printf("purged %d embeddings for inactive model versions", purgedEmbeddings)
	}
	purgedVideoTranscriptEmbeddings, err := db.PurgeOtherModelVideoTranscriptEmbeddings(context.Background(), sqlDB, modelID)
	if err != nil {
		log.Fatalf("purge old video transcript embeddings: %v", err)
	}
	if purgedVideoTranscriptEmbeddings > 0 {
		log.Printf("purged %d video transcript embeddings for inactive model versions", purgedVideoTranscriptEmbeddings)
	}
	purgedJobs, err := db.PurgeOtherModelIndexJobs(context.Background(), sqlDB, modelID)
	if err != nil {
		log.Fatalf("purge old index jobs: %v", err)
	}
	if purgedJobs > 0 {
		log.Printf("purged %d index jobs for inactive model versions", purgedJobs)
	}

	enqueuedMissing, err := db.EnsureIndexJobsForModel(context.Background(), sqlDB, modelID)
	if err != nil {
		log.Fatalf("ensure model index jobs: %v", err)
	}
	if enqueuedMissing > 0 {
		log.Printf("enqueued %d missing index jobs for model_id=%d", enqueuedMissing, modelID)
	}
	enqueuedAnnotationJobs, err := db.EnsureAnnotationJobsForModel(context.Background(), sqlDB, modelID)
	if err != nil {
		log.Fatalf("ensure annotation jobs: %v", err)
	}
	if enqueuedAnnotationJobs > 0 {
		log.Printf("enqueued %d annotation jobs for model_id=%d", enqueuedAnnotationJobs, modelID)
	}
	enqueuedVideoAnnotationJobs, err := db.EnsureVideoAnnotationJobsForModel(context.Background(), sqlDB, modelID)
	if err != nil {
		log.Fatalf("ensure video annotation jobs: %v", err)
	}
	if enqueuedVideoAnnotationJobs > 0 {
		log.Printf("enqueued %d video annotation jobs for model_id=%d", enqueuedVideoAnnotationJobs, modelID)
	}

	embedderOpts := llamaCPPNativeEmbedderOptions{
		ModelPath:          *llamaNativeModelPath,
		VisionModelPath:    *llamaNativeMMProjPath,
		Dimensions:         modelSpec.Dimensions,
		GPULayers:          *llamaNativeGPULayers,
		UseGPU:             *llamaNativeUseGPU,
		ContextSize:        *llamaNativeContextSize,
		BatchSize:          *llamaNativeBatchSize,
		MaxSequences:       *llamaNativeMaxSequences,
		Threads:            *llamaNativeThreads,
		ImageMaxSide:       resolvedLlamaNativeImageMaxSide,
		ImageMaxTokens:     resolvedLlamaNativeImageMaxTokens,
		QueryInstruction:   *llamaNativeQueryInstruction,
		PassageInstruction: *llamaNativePassageInstruction,
		FlashAttnType:      *llamaNativeFlashAttnType,
		CacheTypeK:         *llamaNativeCacheTypeK,
		CacheTypeV:         *llamaNativeCacheTypeV,
	}
	activeEmbedder, err := newLlamaCPPNativeEmbedder(embedderOpts)
	if err != nil {
		log.Fatalf("configure embedder: %v", err)
	}

	var videoTranscriber transcribe.VideoTranscriber
	resolvedParakeetBundleDir := strings.TrimSpace(*parakeetOnnxBundleDir)
	if strings.TrimSpace(*parakeetOnnxRuntimeLib) != "" {
		resolvedParakeetBundleDir, err = ensureDefaultParakeetOnnxAssets(context.Background(), resolvedParakeetBundleDir)
		if err != nil {
			log.Fatalf("ensure Parakeet ONNX assets: %v", err)
		}
	}
	if resolvedParakeetBundleDir != "" && strings.TrimSpace(*parakeetOnnxRuntimeLib) != "" {
		videoTranscriber = &parakeetonnx.Recognizer{Config: parakeetonnx.RecognizerConfig{
			ONNXRuntimeLib: strings.TrimSpace(*parakeetOnnxRuntimeLib),
			BundleDir:      resolvedParakeetBundleDir,
			UseCoreML:      *parakeetOnnxCoreML,
		}}
		if closer, ok := videoTranscriber.(interface{ Close() error }); ok {
			defer func() { _ = closer.Close() }()
		}
		enqueuedTranscriptJobs, err := db.EnsureVideoTranscriptJobsForModel(context.Background(), sqlDB, modelID)
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
	annotatorModelPath := strings.TrimSpace(*llamaNativeAnnotatorModelPath)
	annotatorVisionPath := strings.TrimSpace(*llamaNativeAnnotatorMMProjPath)
	var modelSwitchboard *llamaModelSwitchboard
	if loadAnnotator {
		imageAnnotator, canAnnotateImages = activeEmbedder.(embedder.ImageAnnotator)
	}
	if loadAnnotator && (annotatorModelPath != "" || annotatorVisionPath != "") {
		if annotatorModelPath == "" || annotatorVisionPath == "" {
			log.Fatalf("configure annotator: both -llama-native-annotator-model-path and -llama-native-annotator-mmproj-path must be set together")
		}
		annotatorOpts := llamaCPPNativeAnnotatorOptions{
			ModelPath:       annotatorModelPath,
			VisionModelPath: annotatorVisionPath,
			GPULayers:       *llamaNativeAnnotatorGPULayers,
			UseGPU:          *llamaNativeAnnotatorUseGPU,
			ContextSize:     *llamaNativeAnnotatorContextSize,
			BatchSize:       *llamaNativeAnnotatorBatchSize,
			Threads:         *llamaNativeAnnotatorThreads,
			ImageMaxSide:    *llamaNativeAnnotatorImageMaxSide,
			ImageMaxTokens:  *llamaNativeAnnotatorImageMaxTokens,
			FlashAttnType:   *llamaNativeAnnotatorFlashAttnType,
			CacheTypeK:      *llamaNativeAnnotatorCacheTypeK,
			CacheTypeV:      *llamaNativeAnnotatorCacheTypeV,
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
		DataDir:                *dataDir,
		ModelID:                modelID,
		EnableVideoTranscripts: videoTranscriber != nil,
	}

	queue := &worker.Queue{
		DB:             sqlDB,
		DataDir:        *dataDir,
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

	if mode.startsWorker() {
		if mode.startsHTTP() {
			go worker.RunLoopBatch(context.Background(), queue, "main-worker", 500*time.Millisecond, *workerBatchSize)
		} else {
			worker.RunLoopBatch(context.Background(), queue, "main-worker", 500*time.Millisecond, *workerBatchSize)
			return
		}
	}

	if mode.startsHTTP() {
		mux := newServerMux(sqlDB, *dataDir, modelID, activeEmbedder, index, uploadSvc)
		log.Printf("listening on http://%s", *addr)
		if err := http.ListenAndServe(*addr, mux); err != nil {
			log.Fatalf("serve http: %v", err)
		}
	}
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
