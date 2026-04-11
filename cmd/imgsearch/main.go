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

	_ "github.com/mattn/go-sqlite3"

	"imgsearch/internal/app"
	"imgsearch/internal/db"
	"imgsearch/internal/embedder/deterministic"
	"imgsearch/internal/search"
	"imgsearch/internal/upload"
	"imgsearch/internal/vectorindex/sqlitevector"
	"imgsearch/internal/worker"
)

func main() {
	dataDir := flag.String("data-dir", "./data", "data directory")
	addr := flag.String("addr", "127.0.0.1:8080", "http listen address")
	flag.Parse()

	if err := os.MkdirAll(*dataDir, 0o755); err != nil {
		log.Fatalf("create data directory: %v", err)
	}

	dbPath := filepath.Join(*dataDir, "imgsearch.sqlite")
	dsn := fmt.Sprintf("%s?_busy_timeout=5000", dbPath)
	sqlDB, err := sql.Open("sqlite3", dsn)
	if err != nil {
		log.Fatalf("open sqlite database: %v", err)
	}
	defer func() { _ = sqlDB.Close() }()

	if err := app.Bootstrap(context.Background(), sqlDB, sqlitevector.ValidateAvailable); err != nil {
		log.Fatalf("bootstrap app: %v", err)
	}

	modelID, err := db.EnsureEmbeddingModel(context.Background(), sqlDB, db.EmbeddingModelSpec{
		Name:       "jina-embeddings-v4",
		Version:    "mlx-8bit",
		Dimensions: 2048,
		Metric:     "cosine",
		Normalized: true,
	})
	if err != nil {
		log.Fatalf("ensure embedding model: %v", err)
	}

	embedder := &deterministic.Embedder{Dimensions: 2048}
	index := sqlitevector.NewIndex(sqlDB)

	uploadSvc := &upload.Service{
		DB:      sqlDB,
		DataDir: *dataDir,
		ModelID: modelID,
	}

	queue := &worker.Queue{
		DB:            sqlDB,
		DataDir:       *dataDir,
		LeaseDuration: 30 * time.Second,
		Embedder:      embedder,
		Index:         index,
	}
	go worker.RunLoop(context.Background(), queue, "main-worker", 500*time.Millisecond)

	mux := http.NewServeMux()
	mux.Handle("/api/upload", upload.NewHandler(uploadSvc))
	searchHandler := search.NewHandler(&search.Handler{
		DB:       sqlDB,
		ModelID:  modelID,
		DataDir:  *dataDir,
		Embedder: embedder,
		Index:    index,
	})
	mux.Handle("/api/search/", searchHandler)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})

	log.Printf("imgsearch initialized with database %s", dbPath)
	log.Printf("listening on http://%s", *addr)
	if err := http.ListenAndServe(*addr, mux); err != nil {
		log.Fatalf("serve http: %v", err)
	}
}
