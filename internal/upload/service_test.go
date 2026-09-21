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
	"imgsearch/internal/exif"
	"imgsearch/internal/phash"
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
	durationMS          int64
	width               int
	height              int
	frames              int
	requestedFrameCount int
	err                 error
	// beforeReturn runs after sampling, before the caller's transaction
	// starts, to simulate a concurrent upload winning the race.
	beforeReturn func()
}

func (f *fakeVideoSampler) Sample(ctx context.Context, videoPath string, frameCount int, tmpDir string) (VideoSample, error) {
	_ = ctx
	_ = videoPath
	f.requestedFrameCount = frameCount
	if f.err != nil {
		return VideoSample{}, f.err
	}
	if f.beforeReturn != nil {
		f.beforeReturn()
	}
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

func TestStoreAppliesExifOrientationAndCaptureTime(t *testing.T) {
	svc, sqlDB := setupService(t)
	// A landscape fixture tagged as rotated 90° (orientation 6) with a
	// capture time; the stored dimensions must be the displayed ones.
	rotated := exif.InsertAPP1(fixtureImageBytes(t, "cat_1.jpg"), exif.BuildAPP1(6, "2024:05:06 07:08:09"))
	plainW, plainH, err := decodeDimensionsFromBytes(rotated, "image/jpeg")
	if err != nil {
		t.Fatalf("decode fixture: %v", err)
	}

	out, err := svc.Store(context.Background(), "rotated.jpg", bytes.NewReader(rotated))
	if err != nil {
		t.Fatalf("store: %v", err)
	}
	var width, height int
	var capturedAt string
	if err := sqlDB.QueryRow(`SELECT width, height, captured_at FROM images WHERE id = ?`, out.ImageID).Scan(&width, &height, &capturedAt); err != nil {
		t.Fatalf("load image: %v", err)
	}
	if width != plainH || height != plainW {
		t.Fatalf("expected swapped dimensions %dx%d, got %dx%d", plainH, plainW, width, height)
	}
	if capturedAt != "2024-05-06 07:08:09" {
		t.Fatalf("captured_at: got=%q", capturedAt)
	}

	// A PNG has no EXIF: dimensions stay and the row is marked scanned ("").
	png, err := svc.Store(context.Background(), "plain.png", bytes.NewReader(pngBytes(t)))
	if err != nil {
		t.Fatalf("store png: %v", err)
	}
	if err := sqlDB.QueryRow(`SELECT captured_at FROM images WHERE id = ?`, png.ImageID).Scan(&capturedAt); err != nil {
		t.Fatalf("load png: %v", err)
	}
	if capturedAt != "" {
		t.Fatalf("png captured_at: got=%q want empty", capturedAt)
	}
}

func TestStoreReportsVideoDuplicateWhenConcurrentUploadWins(t *testing.T) {
	svc, sqlDB := setupService(t)
	winner := &Service{DB: svc.DB, DataDir: svc.DataDir, ModelID: svc.ModelID, VideoFrameCount: 2,
		VideoSampler: &fakeVideoSampler{durationMS: 12_000, width: 1920, height: 1080, frames: 2}}
	var winnerOut StoreResult
	svc.VideoFrameCount = 2
	svc.VideoSampler = &fakeVideoSampler{durationMS: 12_000, width: 1920, height: 1080, frames: 2, beforeReturn: func() {
		out, err := winner.Store(context.Background(), "clip.mp4", bytes.NewReader(mp4Bytes()))
		if err != nil {
			t.Fatalf("winner store: %v", err)
		}
		winnerOut = out
	}}

	out, err := svc.Store(context.Background(), "clip.mp4", bytes.NewReader(mp4Bytes()))
	if err != nil {
		t.Fatalf("loser store: %v", err)
	}
	if !out.Duplicate {
		t.Fatal("expected loser to be reported as duplicate")
	}
	if out.VideoID != winnerOut.VideoID || out.StoragePath != winnerOut.StoragePath || out.MediaType != "video" {
		t.Fatalf("loser result mismatch: got=%+v want=%+v", out, winnerOut)
	}

	var videoCount, frameCount int
	if err := sqlDB.QueryRow(`SELECT COUNT(*) FROM videos`).Scan(&videoCount); err != nil {
		t.Fatalf("count videos: %v", err)
	}
	if err := sqlDB.QueryRow(`SELECT COUNT(*) FROM video_frames`).Scan(&frameCount); err != nil {
		t.Fatalf("count frames: %v", err)
	}
	if videoCount != 1 || frameCount != 2 {
		t.Fatalf("expected 1 video with 2 frames, got videos=%d frames=%d", videoCount, frameCount)
	}
	if _, err := os.Stat(filepath.Join(svc.DataDir, filepath.FromSlash(winnerOut.StoragePath))); err != nil {
		t.Fatalf("winner video file missing: %v", err)
	}
}

func TestStoreCreatesVideoFramesAndEmbedJobs(t *testing.T) {
	svc, sqlDB := setupService(t)
	svc.VideoSampler = &fakeVideoSampler{durationMS: 12_000, width: 1920, height: 1080, frames: 3}
	svc.VideoFrameCount = 3
	svc.EnableVideoTranscripts = true

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

	var transcribeCount int
	if err := sqlDB.QueryRow(`SELECT COUNT(*) FROM index_jobs WHERE kind = 'transcribe_video'`).Scan(&transcribeCount); err != nil {
		t.Fatalf("count transcribe jobs: %v", err)
	}
	if transcribeCount != 1 {
		t.Fatalf("expected 1 transcribe job for video, got %d", transcribeCount)
	}

	var annotateVideoCount int
	if err := sqlDB.QueryRow(`SELECT COUNT(*) FROM index_jobs WHERE kind = 'annotate_video'`).Scan(&annotateVideoCount); err != nil {
		t.Fatalf("count annotate video jobs: %v", err)
	}
	if annotateVideoCount != 1 {
		t.Fatalf("expected 1 annotate_video job for video, got %d", annotateVideoCount)
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

func TestStoreDefaultsVideoFrameCountToFive(t *testing.T) {
	svc, sqlDB := setupService(t)
	sampler := &fakeVideoSampler{durationMS: 12_000, width: 1920, height: 1080, frames: 10}
	svc.VideoSampler = sampler

	out, err := svc.Store(context.Background(), "clip.mp4", bytes.NewReader(mp4Bytes()))
	if err != nil {
		t.Fatalf("store video: %v", err)
	}
	if sampler.requestedFrameCount != 5 {
		t.Fatalf("sampler frame count: got=%d want=5", sampler.requestedFrameCount)
	}

	var frameCount int
	if err := sqlDB.QueryRow(`SELECT COUNT(*) FROM video_frames WHERE video_id = ?`, out.VideoID).Scan(&frameCount); err != nil {
		t.Fatalf("count video frames: %v", err)
	}
	if frameCount != 5 {
		t.Fatalf("stored video frames: got=%d want=5", frameCount)
	}
}

// A JPEG and its WEBP copy get hashes within the duplicate distance
// (meta/issues/111); an AVIF is stored as unhashable.
func TestStoreComputesPerceptualHash(t *testing.T) {
	svc, sqlDB := setupService(t)
	jpeg, err := svc.Store(context.Background(), "cat.jpg", bytes.NewReader(fixtureImageBytes(t, "cat_2.jpg")))
	if err != nil {
		t.Fatalf("store jpeg: %v", err)
	}
	webp, err := svc.Store(context.Background(), "cat.webp", bytes.NewReader(fixtureImageBytes(t, "cat_2.webp")))
	if err != nil {
		t.Fatalf("store webp: %v", err)
	}
	avif, err := svc.Store(context.Background(), "dog.avif", bytes.NewReader(fixtureImageBytes(t, "dog_2.avif")))
	if err != nil {
		t.Fatalf("store avif: %v", err)
	}
	var jpegHash, webpHash, avifHash int64
	if err := sqlDB.QueryRow(`SELECT phash FROM images WHERE id = ?`, jpeg.ImageID).Scan(&jpegHash); err != nil {
		t.Fatal(err)
	}
	if err := sqlDB.QueryRow(`SELECT phash FROM images WHERE id = ?`, webp.ImageID).Scan(&webpHash); err != nil {
		t.Fatal(err)
	}
	if err := sqlDB.QueryRow(`SELECT phash FROM images WHERE id = ?`, avif.ImageID).Scan(&avifHash); err != nil {
		t.Fatal(err)
	}
	if jpegHash == phash.Unhashable || webpHash == phash.Unhashable || avifHash != phash.Unhashable {
		t.Fatalf("hashes: jpeg=%d webp=%d avif=%d", jpegHash, webpHash, avifHash)
	}
	if d := phash.Distance(phash.FromInt64(jpegHash), phash.FromInt64(webpHash)); d > 6 {
		t.Fatalf("jpeg/webp distance %d, want <= 6", d)
	}
}
