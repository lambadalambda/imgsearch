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

func mp4Bytes() []byte {
	return []byte{
		0x00, 0x00, 0x00, 0x18,
		'f', 't', 'y', 'p',
		'i', 's', 'o', 'm',
		0x00, 0x00, 0x02, 0x00,
		'i', 's', 'o', 'm',
		'm', 'p', '4', '2',
	}
}

type fakeVideoSampler struct {
	durationMS int64
	width      int
	height     int
	frames     int
}

func (f *fakeVideoSampler) Sample(ctx context.Context, videoPath string, frameCount int, tmpDir string) (VideoSample, error) {
	_ = ctx
	_ = videoPath
	out := VideoSample{DurationMS: f.durationMS, Width: f.width, Height: f.height}
	for i := 0; i < f.frames && i < frameCount; i++ {
		framePath := filepath.Join(tmpDir, "frame-"+string(rune('a'+i))+".png")
		if err := os.WriteFile(framePath, pngBytesForFrame(i), 0o644); err != nil {
			return VideoSample{}, err
		}
		out.Frames = append(out.Frames, SampledFrame{
			Path:        framePath,
			TimestampMS: int64(i+1) * 1000,
			FrameIndex:  i,
		})
	}
	return out, nil
}

func pngBytesForFrame(seed int) []byte {
	img := image.NewRGBA(image.Rect(0, 0, 4, 4))
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			img.Set(x, y, color.RGBA{R: uint8(100 + seed*20), G: uint8(10*y + seed), B: uint8(10*x + seed), A: 255})
		}
	}
	var buf bytes.Buffer
	_ = png.Encode(&buf, img)
	return buf.Bytes()
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
	if jobCount != 2 {
		t.Fatalf("expected 2 jobs, got %d", jobCount)
	}

	var annotateJobs int
	if err := sqlDB.QueryRow(`SELECT COUNT(*) FROM index_jobs WHERE kind = 'annotate_image'`).Scan(&annotateJobs); err != nil {
		t.Fatalf("count annotate jobs: %v", err)
	}
	if annotateJobs != 1 {
		t.Fatalf("expected 1 annotate job, got %d", annotateJobs)
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
	if jobCount != 2 {
		t.Fatalf("expected 2 jobs after duplicate upload, got %d", jobCount)
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

func TestStoreCreatesVideoFramesAndEmbedJobs(t *testing.T) {
	svc, sqlDB := setupService(t)
	svc.VideoSampler = &fakeVideoSampler{durationMS: 12_000, width: 1920, height: 1080, frames: 3}
	svc.VideoFrameCount = 3

	out, err := svc.Store(context.Background(), "clip.mp4", bytes.NewReader(mp4Bytes()))
	if err != nil {
		t.Fatalf("store video: %v", err)
	}
	if out.Duplicate {
		t.Fatal("expected new video, got duplicate")
	}
	if out.MediaType != "video" {
		t.Fatalf("expected video media type, got %q", out.MediaType)
	}
	if out.VideoID == 0 {
		t.Fatal("expected non-zero video id")
	}

	var videoCount int
	if err := sqlDB.QueryRow(`SELECT COUNT(*) FROM videos`).Scan(&videoCount); err != nil {
		t.Fatalf("count videos: %v", err)
	}
	if videoCount != 1 {
		t.Fatalf("expected 1 video, got %d", videoCount)
	}

	var frameCount int
	if err := sqlDB.QueryRow(`SELECT COUNT(*) FROM video_frames WHERE video_id = ?`, out.VideoID).Scan(&frameCount); err != nil {
		t.Fatalf("count video frames: %v", err)
	}
	if frameCount != 3 {
		t.Fatalf("expected 3 video frames, got %d", frameCount)
	}

	var imageCount int
	if err := sqlDB.QueryRow(`SELECT COUNT(*) FROM images`).Scan(&imageCount); err != nil {
		t.Fatalf("count images: %v", err)
	}
	if imageCount != 3 {
		t.Fatalf("expected 3 frame images, got %d", imageCount)
	}

	var jobCount int
	if err := sqlDB.QueryRow(`SELECT COUNT(*) FROM index_jobs WHERE kind = 'embed_image'`).Scan(&jobCount); err != nil {
		t.Fatalf("count embed jobs: %v", err)
	}
	if jobCount != 3 {
		t.Fatalf("expected 3 embed jobs, got %d", jobCount)
	}

	var annotateCount int
	if err := sqlDB.QueryRow(`SELECT COUNT(*) FROM index_jobs WHERE kind = 'annotate_image'`).Scan(&annotateCount); err != nil {
		t.Fatalf("count annotate jobs: %v", err)
	}
	if annotateCount != 0 {
		t.Fatalf("expected 0 annotate jobs for video frames, got %d", annotateCount)
	}

	abs := filepath.Join(svc.DataDir, out.StoragePath)
	if _, err := os.Stat(abs); err != nil {
		t.Fatalf("expected stored video at %s: %v", abs, err)
	}

	rows, err := sqlDB.Query(`
SELECT i.storage_path
FROM images i
JOIN video_frames vf ON vf.image_id = i.id
WHERE vf.video_id = ?
ORDER BY vf.frame_index ASC
`, out.VideoID)
	if err != nil {
		t.Fatalf("load frame storage paths: %v", err)
	}
	defer func() { _ = rows.Close() }()
	for rows.Next() {
		var framePath string
		if err := rows.Scan(&framePath); err != nil {
			t.Fatalf("scan frame path: %v", err)
		}
		if _, err := os.Stat(filepath.Join(svc.DataDir, framePath)); err != nil {
			t.Fatalf("expected frame file %s: %v", framePath, err)
		}
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("iterate frame paths: %v", err)
	}
}
