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
	embedderType := flag.String("embedder", "jina-mlx", "embedder backend: jina-mlx, jina-torch, qwen3-vl-embedding-8b, sqlite-ai, or deterministic")
	jinaURL := flag.String("jina-mlx-url", "http://127.0.0.1:9009", "embedding sidecar URL (jina-mlx, jina-torch, or qwen3-vl-embedding-8b)")
	embedImageMode := flag.String("embed-image-mode", "auto", "image transport mode for sidecar embedders: path, bytes, or auto")
	sqliteAIPath := flag.String("sqlite-ai-path", "", "path to sqlite-ai extension binary (optional: defaults to SQLITE_AI_PATH or ../sqlite-ai/dist/ai)")
	sqliteAIModelPath := flag.String("sqlite-ai-model-path", "", "path to sqlite-ai GGUF embedding model (required for -embedder sqlite-ai)")
	sqliteAIVisionModelPath := flag.String("sqlite-ai-vision-model-path", "", "path to sqlite-ai GGUF vision projector model (required for -embedder sqlite-ai)")
	sqliteAIModelOptions := flag.String("sqlite-ai-model-options", "gpu_layers=99", "llm_model_load options for sqlite-ai embedder")
	sqliteAIVisionOptions := flag.String("sqlite-ai-vision-options", "use_gpu=1", "llm_vision_load options for sqlite-ai embedder")
	sqliteAIContextOptions := flag.String("sqlite-ai-context-options", "embedding_type=FLOAT32,normalize_embedding=1,pooling_type=mean", "llm_context_create_embedding options for sqlite-ai embedder")
	sqliteAIDimensions := flag.Int("sqlite-ai-dimensions", 4096, "embedding dimensions for sqlite-ai model metadata")
	sqliteAIModelName := flag.String("sqlite-ai-model-name", "sqlite-ai-embedding", "embedding model name used in metadata for sqlite-ai")
	sqliteAIModelVersion := flag.String("sqlite-ai-model-version", "", "embedding model version used in metadata for sqlite-ai (defaults to sqlite-ai model filename)")
	vectorBackend := flag.String("vector-backend", vectorBackendAuto, "vector backend: auto, sqlite-vector, bruteforce")
	sqliteVectorPath := flag.String("sqlite-vector-path", "", "path to sqlite-vector extension binary (optional: defaults to SQLITE_VECTOR_PATH or tools/sqlite-vector/vector)")
	flag.Parse()

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

	sqlDB, err := openSQLiteDB(dsn, openWithVector, resolvedSQLiteAIPath)
	if err != nil {
		if *vectorBackend == vectorBackendAuto && openWithVector != "" {
			autoValidationErr = err
			sqlDB, err = openSQLiteDB(dsn, "", resolvedSQLiteAIPath)
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
		sqlDB, err = openSQLiteDB(dsn, "", resolvedSQLiteAIPath)
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
	} else {
		embedDimensions, err = embedderDimensionsForType(*embedderType)
		if err != nil {
			log.Fatalf("configure embedder dimensions: %v", err)
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
		embedder, err = newSQLiteAIEmbedder(sqlDB, sqliteAIEmbedderOptions{
			ModelPath:          *sqliteAIModelPath,
			ModelOptions:       *sqliteAIModelOptions,
			VisionModelPath:    *sqliteAIVisionModelPath,
			VisionModelOptions: *sqliteAIVisionOptions,
			ContextOptions:     *sqliteAIContextOptions,
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
