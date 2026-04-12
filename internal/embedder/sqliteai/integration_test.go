package sqliteai

import (
	"context"
	"database/sql"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"

	sqlite3 "github.com/mattn/go-sqlite3"
)

func TestSQLiteAIEmbeddingsAreSemanticallyReasonable(t *testing.T) {
	if os.Getenv("RUN_SQLITE_AI_INTEGRATION") != "1" {
		t.Skip("set RUN_SQLITE_AI_INTEGRATION=1 with SQLITE_AI_PATH, SQLITE_AI_MODEL_PATH, and SQLITE_AI_VISION_PATH")
	}
	if _, err := exec.LookPath("vips"); err != nil {
		t.Skip("vips CLI is required for sqlite-ai image preprocessing")
	}

	extensionPath := os.Getenv("SQLITE_AI_PATH")
	if extensionPath == "" {
		t.Skip("set SQLITE_AI_PATH to sqlite-ai extension binary path")
	}
	modelPath := os.Getenv("SQLITE_AI_MODEL_PATH")
	if modelPath == "" {
		t.Skip("set SQLITE_AI_MODEL_PATH to sqlite-ai GGUF embedding model")
	}
	visionPath := os.Getenv("SQLITE_AI_VISION_PATH")
	if visionPath == "" {
		t.Skip("set SQLITE_AI_VISION_PATH to sqlite-ai GGUF vision projector")
	}

	dbConn := openWithSQLiteAI(t, extensionPath)

	modelOptions := envOr("SQLITE_AI_MODEL_OPTIONS", "gpu_layers=99")
	visionOptions := envOr("SQLITE_AI_VISION_OPTIONS", "use_gpu=1")
	contextOptions := envOr("SQLITE_AI_CONTEXT_OPTIONS", "embedding_type=FLOAT32,normalize_embedding=1,pooling_type=mean")
	expectedDimensions := envIntOrDefault("SQLITE_AI_DIMS", 4096)

	embedder, err := New(Config{
		DB:                 dbConn,
		ModelPath:          modelPath,
		ModelOptions:       modelOptions,
		VisionModelPath:    visionPath,
		VisionModelOptions: visionOptions,
		ContextOptions:     contextOptions,
	})
	if err != nil {
		t.Fatalf("new sqlite-ai embedder: %v", err)
	}

	repoRoot := findRepoRoot(t)
	fixtures := map[string]string{
		"cat":          filepath.Join(repoRoot, "fixtures", "images", "cat_1.jpg"),
		"dog":          filepath.Join(repoRoot, "fixtures", "images", "dog_1.jpg"),
		"woman":        filepath.Join(repoRoot, "fixtures", "images", "woman_2.jpg"),
		"woman_office": filepath.Join(repoRoot, "fixtures", "images", "woman_office.jpg"),
	}

	vectors := map[string][]float32{}
	for key, path := range fixtures {
		vec, err := embedder.EmbedImage(context.Background(), path)
		if err != nil {
			t.Fatalf("embed %s image: %v", key, err)
		}
		if len(vec) != expectedDimensions {
			t.Fatalf("embedding dimensions for %s: got=%d want=%d", key, len(vec), expectedDimensions)
		}
		vectors[key] = vec
	}

	womanWoman := cosine(vectors["woman"], vectors["woman_office"])
	womanDog := cosine(vectors["woman"], vectors["dog"])
	if womanWoman <= womanDog {
		t.Fatalf("expected woman pair similarity > woman-dog: woman-woman=%.4f woman-dog=%.4f", womanWoman, womanDog)
	}

	dogQuery, err := embedder.EmbedText(context.Background(), "a dog playing outdoors")
	if err != nil {
		t.Fatalf("embed dog query: %v", err)
	}
	if len(dogQuery) != expectedDimensions {
		t.Fatalf("dog query embedding dimensions: got=%d want=%d", len(dogQuery), expectedDimensions)
	}

	dogScore := cosine(dogQuery, vectors["dog"])
	catScore := cosine(dogQuery, vectors["cat"])
	womanScore := cosine(dogQuery, vectors["woman"])
	if dogScore <= catScore {
		t.Fatalf("expected dog query to rank dog above cat: dog=%.4f cat=%.4f woman=%.4f", dogScore, catScore, womanScore)
	}
}

func envOr(key string, fallback string) string {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	return v
}

func envIntOrDefault(key string, fallback int) int {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return fallback
	}
	return n
}

var sqliteAIDriverCounter uint64

func openWithSQLiteAI(t *testing.T, extensionPath string) *sql.DB {
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

	driverName := fmt.Sprintf("sqlite3_with_ai_test_%d", atomic.AddUint64(&sqliteAIDriverCounter, 1))
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
	t.Cleanup(func() { _ = dbConn.Close() })
	return dbConn
}

func findRepoRoot(t *testing.T) string {
	t.Helper()

	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("cannot determine caller path")
	}
	dir := filepath.Dir(file)
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("could not find repository root")
		}
		dir = parent
	}
}

func cosine(a, b []float32) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	if n == 0 {
		return 0
	}

	var dot float64
	var na float64
	var nb float64
	for i := 0; i < n; i++ {
		av := float64(a[i])
		bv := float64(b[i])
		dot += av * bv
		na += av * av
		nb += bv * bv
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
