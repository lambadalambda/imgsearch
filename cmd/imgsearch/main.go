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
	"imgsearch/internal/upload"
	"imgsearch/internal/vectorindex"
	"imgsearch/internal/vectorindex/sqlitevector"
	"imgsearch/internal/webui"
	"imgsearch/internal/worker"
)

func main() {
	dataDir := flag.String("data-dir", "./data", "data directory")
	addr := flag.String("addr", "127.0.0.1:8080", "http listen address")
	embedderType := flag.String("embedder", "llama-cpp-native", "embedder backend (default): llama-cpp-native; alternatives: llama-cpp, sqlite-ai (deprecated), jina-mlx, jina-torch, qwen3-vl-embedding-8b, deterministic")
	jinaURL := flag.String("jina-mlx-url", "http://127.0.0.1:9009", "embedding sidecar URL (jina-mlx, jina-torch, or qwen3-vl-embedding-8b)")
	embedImageMode := flag.String("embed-image-mode", "auto", "image transport mode for sidecar embedders: path, bytes, or auto")
	sqliteAIPath := flag.String("sqlite-ai-path", "", "[deprecated] path to sqlite-ai extension binary (optional: defaults to SQLITE_AI_PATH or ../sqlite-ai/dist/ai)")
	sqliteAIModelPath := flag.String("sqlite-ai-model-path", "", "[deprecated] path to sqlite-ai GGUF embedding model (required for -embedder sqlite-ai)")
	sqliteAIVisionModelPath := flag.String("sqlite-ai-vision-model-path", "", "[deprecated] path to sqlite-ai GGUF vision projector model (required for -embedder sqlite-ai)")
	sqliteAIModelOptions := flag.String("sqlite-ai-model-options", "gpu_layers=99", "[deprecated] llm_model_load options for sqlite-ai embedder")
	sqliteAIVisionOptions := flag.String("sqlite-ai-vision-options", "use_gpu=1", "[deprecated] llm_vision_load options for sqlite-ai embedder")
	sqliteAIContextOptions := flag.String("sqlite-ai-context-options", "embedding_type=FLOAT32,normalize_embedding=1,pooling_type=last", "[deprecated] llm_context_create_embedding options for sqlite-ai embedder")
	sqliteAIQueryInstruction := flag.String("sqlite-ai-query-instruction", "Retrieve images or text relevant to the user's query.", "[deprecated] instruction used for sqlite-ai text query embeddings")
	sqliteAIPassageInstruction := flag.String("sqlite-ai-passage-instruction", "Represent this image or text for retrieval.", "[deprecated] instruction used for sqlite-ai image/document embeddings")
	sqliteAIImageMaxSide := flag.Int("sqlite-ai-image-max-side", 512, "[deprecated] maximum image side length used before sqlite-ai embedding (resized with vips)")
	sqliteAIVipsPath := flag.String("sqlite-ai-vips-path", "", "[deprecated] path to vips binary for sqlite-ai image preprocessing (defaults to PATH lookup)")
	sqliteAIDimensions := flag.Int("sqlite-ai-dimensions", 4096, "[deprecated] embedding dimensions for sqlite-ai model metadata")
	sqliteAIModelName := flag.String("sqlite-ai-model-name", "sqlite-ai-embedding", "[deprecated] embedding model name used in metadata for sqlite-ai")
	sqliteAIModelVersion := flag.String("sqlite-ai-model-version", "", "[deprecated] embedding model version used in metadata for sqlite-ai (defaults to sqlite-ai model filename)")
	llamaCPPURL := flag.String("llama-cpp-url", "http://127.0.0.1:8081", "llama.cpp server URL for -embedder llama-cpp")
	llamaCPPDimensions := flag.Int("llama-cpp-dimensions", 2048, "embedding dimensions for llama-cpp model metadata")
	llamaCPPModel := flag.String("llama-cpp-model", "", "optional model field passed to llama.cpp /v1/embeddings")
	llamaCPPQueryInstruction := flag.String("llama-cpp-query-instruction", "Retrieve images or text relevant to the user's query.", "instruction used for llama-cpp text query embeddings")
	llamaCPPPassageInstruction := flag.String("llama-cpp-passage-instruction", "Represent this image or text for retrieval.", "instruction used for llama-cpp image/document embeddings")
	llamaNativeModelPath := flag.String("llama-native-model-path", "", "path to llama.cpp GGUF embedding model for -embedder llama-cpp-native")
	llamaNativeMMProjPath := flag.String("llama-native-mmproj-path", "", "path to llama.cpp GGUF mmproj model for -embedder llama-cpp-native")
	llamaNativeDimensions := flag.Int("llama-native-dimensions", 2048, "embedding dimensions for llama-cpp-native model metadata")
	llamaNativeGPULayers := flag.Int("llama-native-gpu-layers", 99, "number of layers to offload for llama-cpp-native")
	llamaNativeUseGPU := flag.Bool("llama-native-use-gpu", true, "whether llama-cpp-native should use GPU for mtmd/mmproj")
	llamaNativeContextSize := flag.Int("llama-native-context-size", 8192, "context size for llama-cpp-native runtime")
	llamaNativeBatchSize := flag.Int("llama-native-batch-size", 512, "batch size for llama-cpp-native runtime")
	llamaNativeThreads := flag.Int("llama-native-threads", 0, "thread count for llama-cpp-native runtime (0 uses backend default)")
	llamaNativeImageMaxSide := flag.Int("llama-native-image-max-side", 512, "maximum image side length used before llama-cpp-native embedding")
	llamaNativeImageMaxTokens := flag.Int("llama-native-image-max-tokens", 0, "optional maximum image tokens override for llama-cpp-native mtmd preprocessing (0 uses model default)")
	vectorBackend := flag.String("vector-backend", vectorBackendAuto, "vector backend: auto, sqlite-vector, bruteforce")
	sqliteVectorPath := flag.String("sqlite-vector-path", "", "path to sqlite-vector extension binary (optional: defaults to SQLITE_VECTOR_PATH or tools/sqlite-vector/vector)")
	flag.Parse()

	if *embedderType == "sqlite-ai" {
		log.Printf("warning: embedder 'sqlite-ai' is deprecated; prefer '-embedder llama-cpp-native'")
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

	resolvedSQLiteAIPath := ""
	if *embedderType == "sqlite-ai" {
		resolvedSQLiteAIPath, err = discoverSQLiteAIPath(*sqliteAIPath)
		if err != nil {
			log.Fatalf("discover sqlite-ai extension: %v", err)
		}
		if resolvedSQLiteAIPath == "" {
			log.Fatalf("sqlite-ai embedder requested but extension path was not found (set -sqlite-ai-path or SQLITE_AI_PATH)")
		}
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

	sqlDB, err := openSQLiteDB(dsn, openWithVector, "")
	if err != nil {
		if *vectorBackend == vectorBackendAuto && openWithVector != "" {
			autoValidationErr = err
			sqlDB, err = openSQLiteDB(dsn, "", "")
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
		sqlDB, err = openSQLiteDB(dsn, "", "")
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

	embedDimensions := 0
	if *embedderType == "sqlite-ai" {
		embedDimensions = *sqliteAIDimensions
		if embedDimensions <= 0 {
			log.Fatalf("configure embedder dimensions: sqlite-ai dimensions must be positive")
		}
	} else if *embedderType == "llama-cpp" {
		embedDimensions = *llamaCPPDimensions
		if embedDimensions <= 0 {
			log.Fatalf("configure embedder dimensions: llama-cpp dimensions must be positive")
		}
	} else if *embedderType == "llama-cpp-native" {
		embedDimensions = *llamaNativeDimensions
		if embedDimensions <= 0 {
			log.Fatalf("configure embedder dimensions: llama-cpp-native dimensions must be positive")
		}
	} else {
		embedDimensions, err = embedderDimensionsForType(*embedderType)
		if err != nil {
			log.Fatalf("configure embedder dimensions: %v", err)
		}
	}

	resolvedLlamaNativeImageMaxSide := *llamaNativeImageMaxSide
	resolvedLlamaNativeImageMaxTokens := *llamaNativeImageMaxTokens
	if *embedderType == "llama-cpp-native" {
		resolvedLlamaNativeImageMaxSide, resolvedLlamaNativeImageMaxTokens, err = resolveLlamaCPPNativeImageLimits(*llamaNativeImageMaxSide, *llamaNativeImageMaxTokens)
		if err != nil {
			log.Fatalf("configure llama-cpp-native options: %v", err)
		}
	}

	modelSpec := db.EmbeddingModelSpec{}
	if *embedderType == "sqlite-ai" {
		modelName := strings.TrimSpace(*sqliteAIModelName)
		if modelName == "" {
			log.Fatalf("configure model spec: sqlite-ai model name must not be empty")
		}
		modelVersion := strings.TrimSpace(*sqliteAIModelVersion)
		if modelVersion == "" {
			modelVersion = filepath.Base(strings.TrimSpace(*sqliteAIModelPath))
		}
		if modelVersion == "" {
			log.Fatalf("configure model spec: sqlite-ai model version must not be empty")
		}
		modelSpec = db.EmbeddingModelSpec{
			Name:       modelName,
			Version:    modelVersion,
			Dimensions: embedDimensions,
			Metric:     "cosine",
			Normalized: true,
		}
	} else {
		modelSpec, err = embeddingModelSpec(*embedderType, embedDimensions)
		if err != nil {
			log.Fatalf("configure model spec: %v", err)
		}
		if *embedderType == "llama-cpp" {
			modelSpec.Version = llamaCPPModelVersion(*llamaCPPModel, modelSpec.Dimensions)
		}
		if *embedderType == "llama-cpp-native" {
			modelSpec.Version = llamaNativeModelVersion(
				*llamaNativeModelPath,
				*llamaNativeMMProjPath,
				modelSpec.Dimensions,
				resolvedLlamaNativeImageMaxSide,
				resolvedLlamaNativeImageMaxTokens,
			)
		}
	}

	modelID, err := db.EnsureEmbeddingModel(context.Background(), sqlDB, modelSpec)
	if err != nil {
		log.Fatalf("ensure embedding model: %v", err)
	}

	enqueuedMissing, err := db.EnsureIndexJobsForModel(context.Background(), sqlDB, modelID)
	if err != nil {
		log.Fatalf("ensure model index jobs: %v", err)
	}
	if enqueuedMissing > 0 {
		log.Printf("enqueued %d missing index jobs for model_id=%d", enqueuedMissing, modelID)
	}

	embedder := embedder.Embedder(nil)
	if *embedderType == "sqlite-ai" {
		sqliteAIEmbedDB, err := openSQLiteAIRuntimeDB(resolvedSQLiteAIPath)
		if err != nil {
			log.Fatalf("open sqlite-ai runtime db: %v", err)
		}
		defer func() { _ = sqliteAIEmbedDB.Close() }()

		embedder, err = newSQLiteAIEmbedder(sqliteAIEmbedDB, sqliteAIEmbedderOptions{
			ModelPath:          *sqliteAIModelPath,
			ModelOptions:       *sqliteAIModelOptions,
			VisionModelPath:    *sqliteAIVisionModelPath,
			VisionModelOptions: *sqliteAIVisionOptions,
			ContextOptions:     *sqliteAIContextOptions,
			QueryInstruction:   *sqliteAIQueryInstruction,
			PassageInstruction: *sqliteAIPassageInstruction,
			ImageMaxSide:       *sqliteAIImageMaxSide,
			VipsPath:           *sqliteAIVipsPath,
		})
	} else if *embedderType == "llama-cpp" {
		embedder, err = newLlamaCPPEmbedder(llamaCPPEmbedderOptions{
			URL:                *llamaCPPURL,
			Dimensions:         modelSpec.Dimensions,
			Model:              *llamaCPPModel,
			QueryInstruction:   *llamaCPPQueryInstruction,
			PassageInstruction: *llamaCPPPassageInstruction,
		})
	} else if *embedderType == "llama-cpp-native" {
		embedder, err = newLlamaCPPNativeEmbedder(llamaCPPNativeEmbedderOptions{
			ModelPath:          *llamaNativeModelPath,
			VisionModelPath:    *llamaNativeMMProjPath,
			Dimensions:         modelSpec.Dimensions,
			GPULayers:          *llamaNativeGPULayers,
			UseGPU:             *llamaNativeUseGPU,
			ContextSize:        *llamaNativeContextSize,
			BatchSize:          *llamaNativeBatchSize,
			Threads:            *llamaNativeThreads,
			ImageMaxSide:       resolvedLlamaNativeImageMaxSide,
			ImageMaxTokens:     resolvedLlamaNativeImageMaxTokens,
			QueryInstruction:   *llamaCPPQueryInstruction,
			PassageInstruction: *llamaCPPPassageInstruction,
		})
	} else {
		embedder, err = newEmbedder(*embedderType, *jinaURL, modelSpec.Dimensions, *embedImageMode)
	}
	if err != nil {
		log.Fatalf("configure embedder: %v", err)
	}
	uploadSvc := &upload.Service{
		DB:      sqlDB,
		DataDir: *dataDir,
		ModelID: modelID,
	}

	queue := &worker.Queue{
		DB:             sqlDB,
		DataDir:        *dataDir,
		LeaseDuration:  30 * time.Second,
		RetryBaseDelay: 5 * time.Second,
		Embedder:       embedder,
		Index:          index,
	}
	go worker.RunLoop(context.Background(), queue, "main-worker", 500*time.Millisecond)

	mux := newServerMux(sqlDB, *dataDir, modelID, embedder, index, uploadSvc)

	log.Printf("imgsearch initialized with database %s", dbPath)
	log.Printf("using vector backend %s", resolvedVectorBackend)
	log.Printf("listening on http://%s", *addr)
	if err := http.ListenAndServe(*addr, mux); err != nil {
		log.Fatalf("serve http: %v", err)
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
	mux.Handle("/api/images", images.NewHandler(&images.Handler{DB: sqlDB, ModelID: modelID}))
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
