package integration

import (
	"context"
	"database/sql"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	sqlite3 "github.com/mattn/go-sqlite3"

	"imgsearch/internal/db"
	"imgsearch/internal/embedder/sqliteai"
	"imgsearch/internal/search"
	"imgsearch/internal/upload"
	"imgsearch/internal/worker"
)

func TestAPIEndToEndWithSQLiteAIEmbedder(t *testing.T) {
	if os.Getenv("RUN_SQLITE_AI_INTEGRATION") != "1" {
		t.Skip("set RUN_SQLITE_AI_INTEGRATION=1 with SQLITE_AI_PATH, SQLITE_AI_MODEL_PATH, and SQLITE_AI_VISION_PATH to enable")
	}
	if _, err := exec.LookPath("vips"); err != nil {
		t.Skip("vips CLI is required for sqlite-ai image preprocessing")
	}

	extensionPath := strings.TrimSpace(os.Getenv("SQLITE_AI_PATH"))
	if extensionPath == "" {
		t.Skip("set SQLITE_AI_PATH to sqlite-ai extension binary path")
	}
	modelPath := strings.TrimSpace(os.Getenv("SQLITE_AI_MODEL_PATH"))
	if modelPath == "" {
		t.Skip("set SQLITE_AI_MODEL_PATH to sqlite-ai GGUF embedding model")
	}
	visionPath := strings.TrimSpace(os.Getenv("SQLITE_AI_VISION_PATH"))
	if visionPath == "" {
		t.Skip("set SQLITE_AI_VISION_PATH to sqlite-ai GGUF vision projector")
	}

	dataDir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dataDir, "images"), 0o755); err != nil {
		t.Fatalf("mkdir data/images: %v", err)
	}

	dbConn := openWithSQLiteAIForIntegration(t, extensionPath)

	if err := db.RunMigrations(context.Background(), dbConn); err != nil {
		t.Fatalf("migrate db: %v", err)
	}

	modelSpec := db.EmbeddingModelSpec{
		Name:       sqliteAIEnvOr("SQLITE_AI_MODEL_NAME", "sqlite-ai-embedding"),
		Version:    sqliteAIEnvOr("SQLITE_AI_MODEL_VERSION", filepath.Base(modelPath)),
		Dimensions: sqliteAIEnvIntOrDefault(t, "SQLITE_AI_DIMS", 4096),
		Metric:     "cosine",
		Normalized: true,
	}
	modelID, err := db.EnsureEmbeddingModel(context.Background(), dbConn, modelSpec)
	if err != nil {
		t.Fatalf("ensure model: %v", err)
	}

	embed, err := sqliteai.New(sqliteai.Config{
		DB:                 dbConn,
		ModelPath:          modelPath,
		ModelOptions:       sqliteAIEnvOr("SQLITE_AI_MODEL_OPTIONS", "gpu_layers=99"),
		VisionModelPath:    visionPath,
		VisionModelOptions: sqliteAIEnvOr("SQLITE_AI_VISION_OPTIONS", "use_gpu=1"),
		ContextOptions:     sqliteAIEnvOr("SQLITE_AI_CONTEXT_OPTIONS", "embedding_type=FLOAT32,normalize_embedding=1,pooling_type=last"),
	})
	if err != nil {
		t.Fatalf("new sqlite-ai embedder: %v", err)
	}

	index := newMemoryIndex()

	uploadSvc := &upload.Service{DB: dbConn, DataDir: dataDir, ModelID: modelID}
	queue := &worker.Queue{
		DB:            dbConn,
		DataDir:       dataDir,
		LeaseDuration: 30 * time.Second,
		Embedder:      embed,
		Index:         index,
	}

	searchHandler := search.NewHandler(&search.Handler{
		DB:       dbConn,
		ModelID:  modelID,
		DataDir:  dataDir,
		Embedder: embed,
		Index:    index,
	})

	mux := http.NewServeMux()
	mux.Handle("/api/upload", upload.NewHandler(uploadSvc))
	mux.Handle("/api/search/", searchHandler)
	server := httptest.NewServer(mux)
	defer server.Close()

	type fixture struct {
		Name string
		Path string
	}
	repoRoot := findRepoRoot(t)
	fixtures := []fixture{
		{Name: "cat_1.jpg", Path: filepath.Join(repoRoot, "fixtures", "images", "cat_1.jpg")},
		{Name: "dog_1.jpg", Path: filepath.Join(repoRoot, "fixtures", "images", "dog_1.jpg")},
		{Name: "woman_2.jpg", Path: filepath.Join(repoRoot, "fixtures", "images", "woman_2.jpg")},
		{Name: "woman_office.jpg", Path: filepath.Join(repoRoot, "fixtures", "images", "woman_office.jpg")},
	}

	uploadedByName := make(map[string]upload.UploadResponse, len(fixtures))
	for _, fx := range fixtures {
		raw, err := os.ReadFile(fx.Path)
		if err != nil {
			t.Fatalf("read fixture %s: %v", fx.Name, err)
		}
		resp := uploadImage(t, server.Client(), server.URL, fx.Name, raw)
		uploadedByName[fx.Name] = resp
	}

	drainQueue(t, queue)
	assertJobStates(t, dbConn, len(fixtures))

	textResp := searchText(t, server.Client(), server.URL, "a dog playing outdoors", 3)
	if len(textResp.Results) == 0 {
		t.Fatal("expected text search results")
	}
	if textResp.Results[0].OriginalName != "dog_1.jpg" {
		t.Fatalf("expected top text result dog_1.jpg, got %s", textResp.Results[0].OriginalName)
	}

	womanID := uploadedByName["woman_2.jpg"].ImageID
	simResp := searchSimilar(t, server.Client(), server.URL, womanID, 2)
	if len(simResp.Results) == 0 {
		t.Fatal("expected similar search results")
	}
	if simResp.Results[0].OriginalName != "woman_office.jpg" {
		t.Fatalf("expected top similar result woman_office.jpg, got %s", simResp.Results[0].OriginalName)
	}
}

func sqliteAIEnvOr(key string, fallback string) string {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	return v
}

func sqliteAIEnvIntOrDefault(t *testing.T, key string, fallback int) int {
	t.Helper()
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		t.Fatalf("parse %s as int: %v", key, err)
	}
	return n
}

var sqliteAIAPIDriverCounter uint64

func openWithSQLiteAIForIntegration(t *testing.T, extensionPath string) *sql.DB {
	t.Helper()

	abs, err := filepath.Abs(extensionPath)
	if err != nil {
		t.Fatalf("resolve extension path: %v", err)
	}
	loadTarget := abs
	ext := strings.ToLower(filepath.Ext(loadTarget))
	if ext == ".dylib" || ext == ".so" || ext == ".dll" {
		loadTarget = strings.TrimSuffix(loadTarget, filepath.Ext(loadTarget))
	}

	driverName := fmt.Sprintf("sqlite3_with_ai_api_itest_%d", atomic.AddUint64(&sqliteAIAPIDriverCounter, 1))
	sql.Register(driverName, &sqlite3.SQLiteDriver{
		ConnectHook: func(conn *sqlite3.SQLiteConn) error {
			if err := conn.LoadExtension(loadTarget, "sqlite3_ai_init"); err != nil {
				return fmt.Errorf("load sqlite-ai extension from %q: %w", loadTarget, err)
			}
			return nil
		},
	})

	dbConn, err := sql.Open(driverName, ":memory:")
	if err != nil {
		t.Fatalf("open sqlite db: %v", err)
	}
	dbConn.SetMaxOpenConns(1)
	dbConn.SetMaxIdleConns(1)
	t.Cleanup(func() { _ = dbConn.Close() })
	return dbConn
}
