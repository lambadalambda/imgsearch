package upload

import (
	"bytes"
	"context"
	"database/sql"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	_ "github.com/mattn/go-sqlite3"

	"imgsearch/internal/db"
)

func setupService(t *testing.T) (*Service, *sql.DB) {
	t.Helper()

	sqlDB, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	t.Cleanup(func() { _ = sqlDB.Close() })

	if err := db.RunMigrations(context.Background(), sqlDB); err != nil {
		t.Fatalf("run migrations: %v", err)
	}

	_, err = sqlDB.Exec(`
INSERT INTO embedding_models(name, version, dimensions, metric, normalized)
VALUES('test-model', 'v1', 4, 'cosine', 1)
`)
	if err != nil {
		t.Fatalf("insert model: %v", err)
	}

	var modelID int64
	if err := sqlDB.QueryRow(`SELECT id FROM embedding_models LIMIT 1`).Scan(&modelID); err != nil {
		t.Fatalf("select model id: %v", err)
	}

	dataDir := t.TempDir()
	svc := &Service{DB: sqlDB, DataDir: dataDir, ModelID: modelID}
	return svc, sqlDB
}

func pngBytes(t *testing.T) []byte {
	t.Helper()

	img := image.NewRGBA(image.Rect(0, 0, 8, 8))
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			img.Set(x, y, color.RGBA{R: uint8(x * 10), G: uint8(y * 10), B: 100, A: 255})
		}
	}

	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode png: %v", err)
	}
	return buf.Bytes()
}

func fixtureImageBytes(t *testing.T, name string) []byte {
	t.Helper()
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve caller location")
	}
	path := filepath.Join(filepath.Dir(thisFile), "..", "..", "fixtures", "images", name)
	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture %s: %v", name, err)
	}
	return content
}

func TestStoreCreatesImageAndQueueJob(t *testing.T) {
	svc, sqlDB := setupService(t)

	out, err := svc.Store(context.Background(), "sample.png", bytes.NewReader(pngBytes(t)))
	if err != nil {
		t.Fatalf("store: %v", err)
	}
	if out.Duplicate {
		t.Fatal("expected new image, got duplicate")
	}

	var imageCount int
	if err := sqlDB.QueryRow(`SELECT COUNT(*) FROM images`).Scan(&imageCount); err != nil {
		t.Fatalf("count images: %v", err)
	}
	if imageCount != 1 {
		t.Fatalf("expected 1 image, got %d", imageCount)
	}

	var jobCount int
	if err := sqlDB.QueryRow(`SELECT COUNT(*) FROM index_jobs`).Scan(&jobCount); err != nil {
		t.Fatalf("count jobs: %v", err)
	}
	if jobCount != 1 {
		t.Fatalf("expected 1 job, got %d", jobCount)
	}

	abs := filepath.Join(svc.DataDir, out.StoragePath)
	if _, err := os.Stat(abs); err != nil {
		t.Fatalf("expected file at %s: %v", abs, err)
	}
}

func TestStoreIsIdempotentByContentHash(t *testing.T) {
	svc, sqlDB := setupService(t)
	content := pngBytes(t)

	first, err := svc.Store(context.Background(), "first.png", bytes.NewReader(content))
	if err != nil {
		t.Fatalf("first store: %v", err)
	}
	second, err := svc.Store(context.Background(), "second.png", bytes.NewReader(content))
	if err != nil {
		t.Fatalf("second store: %v", err)
	}

	if second.ImageID != first.ImageID {
		t.Fatalf("expected same image id, got first=%d second=%d", first.ImageID, second.ImageID)
	}
	if !second.Duplicate {
		t.Fatal("expected duplicate=true for repeated content")
	}

	var imageCount int
	if err := sqlDB.QueryRow(`SELECT COUNT(*) FROM images`).Scan(&imageCount); err != nil {
		t.Fatalf("count images: %v", err)
	}
	if imageCount != 1 {
		t.Fatalf("expected 1 image after duplicate upload, got %d", imageCount)
	}

	var jobCount int
	if err := sqlDB.QueryRow(`SELECT COUNT(*) FROM index_jobs`).Scan(&jobCount); err != nil {
		t.Fatalf("count jobs: %v", err)
	}
	if jobCount != 1 {
		t.Fatalf("expected 1 job after duplicate upload, got %d", jobCount)
	}
}

func TestStoreRejectsUnsupportedFormat(t *testing.T) {
	svc, sqlDB := setupService(t)

	_, err := svc.Store(context.Background(), "notes.txt", bytes.NewReader([]byte("hello")))
	if err == nil {
		t.Fatal("expected unsupported format error")
	}
	if err != ErrUnsupportedFormat {
		t.Fatalf("expected ErrUnsupportedFormat, got %v", err)
	}

	var imageCount int
	if err := sqlDB.QueryRow(`SELECT COUNT(*) FROM images`).Scan(&imageCount); err != nil {
		t.Fatalf("count images: %v", err)
	}
	if imageCount != 0 {
		t.Fatalf("expected no images written, got %d", imageCount)
	}
}

func TestStoreAcceptsWEBP(t *testing.T) {
	svc, sqlDB := setupService(t)

	out, err := svc.Store(context.Background(), "cat_2.webp", bytes.NewReader(fixtureImageBytes(t, "cat_2.webp")))
	if err != nil {
		t.Fatalf("store webp: %v", err)
	}
	if out.ImageID == 0 {
		t.Fatal("expected non-zero image id")
	}

	var mime string
	if err := sqlDB.QueryRow(`SELECT mime_type FROM images WHERE id = ?`, out.ImageID).Scan(&mime); err != nil {
		t.Fatalf("load stored mime: %v", err)
	}
	if mime != "image/webp" {
		t.Fatalf("expected image/webp mime, got %q", mime)
	}
}

func TestStoreAcceptsAVIF(t *testing.T) {
	svc, sqlDB := setupService(t)

	out, err := svc.Store(context.Background(), "dog_2.avif", bytes.NewReader(fixtureImageBytes(t, "dog_2.avif")))
	if err != nil {
		t.Fatalf("store avif: %v", err)
	}
	if out.ImageID == 0 {
		t.Fatal("expected non-zero image id")
	}

	var mime string
	if err := sqlDB.QueryRow(`SELECT mime_type FROM images WHERE id = ?`, out.ImageID).Scan(&mime); err != nil {
		t.Fatalf("load stored mime: %v", err)
	}
	if mime != "image/avif" {
		t.Fatalf("expected image/avif mime, got %q", mime)
	}
}

func TestStoreRejectsFakeWEBPByExtension(t *testing.T) {
	svc, sqlDB := setupService(t)

	_, err := svc.Store(context.Background(), "fake.webp", bytes.NewReader([]byte("not an image")))
	if err == nil {
		t.Fatal("expected unsupported format error")
	}
	if err != ErrUnsupportedFormat {
		t.Fatalf("expected ErrUnsupportedFormat, got %v", err)
	}

	var imageCount int
	if err := sqlDB.QueryRow(`SELECT COUNT(*) FROM images`).Scan(&imageCount); err != nil {
		t.Fatalf("count images: %v", err)
	}
	if imageCount != 0 {
		t.Fatalf("expected no images written, got %d", imageCount)
	}
}

func TestStoreRejectsFakeAVIFByExtension(t *testing.T) {
	svc, sqlDB := setupService(t)

	_, err := svc.Store(context.Background(), "fake.avif", bytes.NewReader([]byte("not an image")))
	if err == nil {
		t.Fatal("expected unsupported format error")
	}
	if err != ErrUnsupportedFormat {
		t.Fatalf("expected ErrUnsupportedFormat, got %v", err)
	}

	var imageCount int
	if err := sqlDB.QueryRow(`SELECT COUNT(*) FROM images`).Scan(&imageCount); err != nil {
		t.Fatalf("count images: %v", err)
	}
	if imageCount != 0 {
		t.Fatalf("expected no images written, got %d", imageCount)
	}
}
