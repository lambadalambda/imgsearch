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
	embedderType := flag.String("embedder", "jina-mlx", "embedder backend: jina-mlx, jina-torch, or deterministic")
	jinaURL := flag.String("jina-mlx-url", "http://127.0.0.1:9009", "embedding sidecar URL (jina-mlx or jina-torch)")
	embedImageMode := flag.String("embed-image-mode", "auto", "image transport mode for sidecar embedders: path, bytes, or auto")
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

	modelSpec, err := embeddingModelSpec(*embedderType, 2048)
	if err != nil {
		log.Fatalf("configure model spec: %v", err)
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

	embedder, err := newEmbedder(*embedderType, *jinaURL, modelSpec.Dimensions, *embedImageMode)
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
