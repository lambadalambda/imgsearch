package upload

import (
	"bytes"
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

var ErrUnsupportedFormat = errors.New("unsupported image format")

func isSupportedImageMime(mime string) bool {
	switch mime {
	case "image/png", "image/jpeg", "image/webp", "image/avif":
		return true
	default:
		return false
	}
}

func isSupportedVideoMime(mime string) bool {
	switch mime {
	case "video/mp4", "video/quicktime", "video/webm", "video/x-matroska":
		return true
	default:
		return false
	}
}

func isSupportedMime(mime string) bool {
	return isSupportedImageMime(mime) || isSupportedVideoMime(mime)
}

func looksLikeWEBP(header []byte) bool {
	return len(header) >= 12 && bytes.Equal(header[:4], []byte("RIFF")) && bytes.Equal(header[8:12], []byte("WEBP"))
}

func looksLikeAVIF(header []byte) bool {
	if len(header) < 12 {
		return false
	}
	if !bytes.Equal(header[4:8], []byte("ftyp")) {
		return false
	}
	if bytes.Equal(header[8:12], []byte("avif")) || bytes.Equal(header[8:12], []byte("avis")) {
		return true
	}
	for i := 16; i+4 <= len(header); i += 4 {
		if bytes.Equal(header[i:i+4], []byte("avif")) || bytes.Equal(header[i:i+4], []byte("avis")) {
			return true
		}
	}
	return false
}

func resolveMime(header []byte, detected string) string {
	if isSupportedMime(detected) {
		return detected
	}
	if looksLikeWEBP(header) {
		return "image/webp"
	}
	if looksLikeAVIF(header) {
		return "image/avif"
	}
	if looksLikeMP4(header) {
		return "video/mp4"
	}
	return ""
}

func looksLikeMP4(header []byte) bool {
	if len(header) < 12 {
		return false
	}
	if !bytes.Equal(header[4:8], []byte("ftyp")) {
		return false
	}
	brand := string(header[8:12])
	return strings.HasPrefix(brand, "mp4") || strings.HasPrefix(brand, "iso") || brand == "isom" || brand == "qt  "
}

func decodeDimensions(tmpFile *os.File, mime string) (int, int, error) {
	if _, err := tmpFile.Seek(0, io.SeekStart); err != nil {
		return 0, 0, fmt.Errorf("seek for decode: %w", err)
	}

	cfg, _, err := image.DecodeConfig(tmpFile)
	if err == nil {
		return cfg.Width, cfg.Height, nil
	}

	if mime == "image/webp" || mime == "image/avif" {
		return 0, 0, nil
	}
	return 0, 0, ErrUnsupportedFormat
}

func decodeDimensionsFromBytes(content []byte, mime string) (int, int, error) {
	reader := bytes.NewReader(content)
	cfg, _, err := image.DecodeConfig(reader)
	if err == nil {
		return cfg.Width, cfg.Height, nil
	}
	if mime == "image/webp" || mime == "image/avif" {
		return 0, 0, nil
	}
	return 0, 0, ErrUnsupportedFormat
}

type Service struct {
	DB                     *sql.DB
	DataDir                string
	ModelID                int64
	VideoFrameCount        int
	VideoSampler           VideoSampler
	EnableVideoTranscripts bool
}

type StoreResult struct {
	ImageID     int64
	VideoID     int64
	SHA256      string
	StoragePath string
	MediaType   string
	Duplicate   bool
}

func (s *Service) Store(ctx context.Context, originalName string, src io.Reader) (StoreResult, error) {
	if s == nil || s.DB == nil {
		return StoreResult{}, fmt.Errorf("service db is nil")
	}
	if s.DataDir == "" {
		return StoreResult{}, fmt.Errorf("data dir is empty")
	}
	if s.ModelID <= 0 {
		return StoreResult{}, fmt.Errorf("model id must be set")
	}

	tmpDir := filepath.Join(s.DataDir, "tmp")
	imageDir := filepath.Join(s.DataDir, "images")
	videoDir := filepath.Join(s.DataDir, "videos")
	if err := os.MkdirAll(tmpDir, 0o755); err != nil {
		return StoreResult{}, fmt.Errorf("create tmp dir: %w", err)
	}
	if err := os.MkdirAll(imageDir, 0o755); err != nil {
		return StoreResult{}, fmt.Errorf("create images dir: %w", err)
	}
	if err := os.MkdirAll(videoDir, 0o755); err != nil {
		return StoreResult{}, fmt.Errorf("create videos dir: %w", err)
	}

	tmpFile, err := os.CreateTemp(tmpDir, "upload-*")
	if err != nil {
		return StoreResult{}, fmt.Errorf("create temp file: %w", err)
	}
	tmpPath := tmpFile.Name()
	renamed := false
	defer func() {
		_ = tmpFile.Close()
		if !renamed {
			_ = os.Remove(tmpPath)
		}
	}()

	hasher := sha256.New()
	if _, err := io.Copy(io.MultiWriter(tmpFile, hasher), src); err != nil {
		return StoreResult{}, fmt.Errorf("write temp upload: %w", err)
	}

	if _, err := tmpFile.Seek(0, io.SeekStart); err != nil {
		return StoreResult{}, fmt.Errorf("seek temp file: %w", err)
	}

	header := make([]byte, 512)
	n, err := io.ReadFull(tmpFile, header)
	if err != nil && !errors.Is(err, io.ErrUnexpectedEOF) {
		return StoreResult{}, fmt.Errorf("read header bytes: %w", err)
	}
	mime := resolveMime(header[:n], http.DetectContentType(header[:n]))
	if !isSupportedMime(mime) {
		return StoreResult{}, ErrUnsupportedFormat
	}

	digest := hex.EncodeToString(hasher.Sum(nil))
	if isSupportedVideoMime(mime) {
		return s.storeVideo(ctx, originalName, tmpDir, tmpPath, digest, mime)
	}

	width, height, err := decodeDimensions(tmpFile, mime)
	if err != nil {
		if errors.Is(err, ErrUnsupportedFormat) {
			return StoreResult{}, ErrUnsupportedFormat
		}
		return StoreResult{}, err
	}

	storageRel := filepath.ToSlash(filepath.Join("images", digest))
	storageAbs := filepath.Join(s.DataDir, storageRel)

	tx, err := s.DB.BeginTx(ctx, nil)
	if err != nil {
		return StoreResult{}, fmt.Errorf("begin tx: %w", err)
	}

	res, err := tx.ExecContext(ctx, `
INSERT INTO images(sha256, original_name, storage_path, mime_type, width, height)
VALUES(?, ?, ?, ?, ?, ?)
ON CONFLICT(sha256) DO NOTHING
`, digest, originalName, storageRel, mime, width, height)
	if err != nil {
		_ = tx.Rollback()
		return StoreResult{}, fmt.Errorf("insert image: %w", err)
	}

	rows, err := res.RowsAffected()
	if err != nil {
		_ = tx.Rollback()
		return StoreResult{}, fmt.Errorf("rows affected: %w", err)
	}

	out := StoreResult{SHA256: digest, StoragePath: storageRel, MediaType: "image", Duplicate: rows == 0}
	if rows > 0 {
		out.ImageID, err = res.LastInsertId()
		if err != nil {
			_ = tx.Rollback()
			return StoreResult{}, fmt.Errorf("last insert id: %w", err)
		}
	} else {
		if err := tx.QueryRowContext(ctx,
			`SELECT id, storage_path FROM images WHERE sha256 = ?`,
			digest,
		).Scan(&out.ImageID, &out.StoragePath); err != nil {
			_ = tx.Rollback()
			return StoreResult{}, fmt.Errorf("load existing image: %w", err)
		}
	}

	if _, err := tx.ExecContext(ctx, `
INSERT INTO index_jobs(kind, image_id, model_id, state)
VALUES('embed_image', ?, ?, 'pending')
ON CONFLICT DO NOTHING
`, out.ImageID, s.ModelID); err != nil {
		_ = tx.Rollback()
		return StoreResult{}, fmt.Errorf("insert index job: %w", err)
	}
	if _, err := tx.ExecContext(ctx, `
INSERT INTO index_jobs(kind, image_id, model_id, state)
SELECT 'annotate_image', i.id, ?, 'pending'
FROM images i
WHERE i.id = ?
  AND (
    trim(COALESCE(i.description, '')) = ''
    OR COALESCE(i.tags_json, '') = ''
    OR COALESCE(i.tags_json, '[]') = '[]'
  )
ON CONFLICT DO NOTHING
`, s.ModelID, out.ImageID); err != nil {
		_ = tx.Rollback()
		return StoreResult{}, fmt.Errorf("insert annotation job: %w", err)
	}

	if !out.Duplicate {
		if err := os.Rename(tmpPath, storageAbs); err != nil {
			_ = tx.Rollback()
			return StoreResult{}, fmt.Errorf("move upload to storage: %w", err)
		}
		renamed = true
	}

	if err := tx.Commit(); err != nil {
		return StoreResult{}, fmt.Errorf("commit tx: %w", err)
	}

	return out, nil
}

func (s *Service) storeVideo(ctx context.Context, originalName string, tmpDir string, tmpPath string, digest string, mime string) (StoreResult, error) {
	storageRel := filepath.ToSlash(filepath.Join("videos", digest))
	storageAbs := filepath.Join(s.DataDir, storageRel)

	out := StoreResult{SHA256: digest, StoragePath: storageRel, MediaType: "video"}
	if err := s.DB.QueryRowContext(ctx, `SELECT id, storage_path FROM videos WHERE sha256 = ?`, digest).Scan(&out.VideoID, &out.StoragePath); err == nil {
		out.Duplicate = true
		return out, nil
	} else if !errors.Is(err, sql.ErrNoRows) {
		return StoreResult{}, fmt.Errorf("load existing video: %w", err)
	}

	frameCount := s.VideoFrameCount
	if frameCount <= 0 {
		frameCount = 10
	}

	frameTmpDir, err := os.MkdirTemp(tmpDir, "video-frames-*")
	if err != nil {
		return StoreResult{}, fmt.Errorf("create video frame temp dir: %w", err)
	}
	defer func() { _ = os.RemoveAll(frameTmpDir) }()

	sampler := s.VideoSampler
	if sampler == nil {
		sampler = execVideoSampler{}
	}
	sample, err := sampler.Sample(ctx, tmpPath, frameCount, frameTmpDir)
	if err != nil {
		return StoreResult{}, err
	}
	if len(sample.Frames) == 0 {
		return StoreResult{}, fmt.Errorf("video sampling produced no frames")
	}

	tx, err := s.DB.BeginTx(ctx, nil)
	if err != nil {
		return StoreResult{}, fmt.Errorf("begin video tx: %w", err)
	}
	committed := false
	renamedPaths := make([]string, 0, len(sample.Frames)+1)
	defer func() {
		if committed {
			return
		}
		for _, path := range renamedPaths {
			_ = os.Remove(path)
		}
	}()

	res, err := tx.ExecContext(ctx, `
INSERT INTO videos(sha256, original_name, storage_path, mime_type, duration_ms, width, height, frame_count)
VALUES(?, ?, ?, ?, ?, ?, ?, ?)
`, digest, originalName, storageRel, mime, sample.DurationMS, sample.Width, sample.Height, len(sample.Frames))
	if err != nil {
		_ = tx.Rollback()
		return StoreResult{}, fmt.Errorf("insert video: %w", err)
	}
	out.VideoID, err = res.LastInsertId()
	if err != nil {
		_ = tx.Rollback()
		return StoreResult{}, fmt.Errorf("video last insert id: %w", err)
	}

	for _, frame := range sample.Frames {
		_, renamed, err := s.storeVideoFrameTx(ctx, tx, out.VideoID, frame)
		if err != nil {
			_ = tx.Rollback()
			return StoreResult{}, err
		}
		if renamed != "" {
			renamedPaths = append(renamedPaths, renamed)
		}
	}
	if s.EnableVideoTranscripts {
		if _, err := tx.ExecContext(ctx, `
INSERT INTO index_jobs(kind, image_id, video_id, model_id, state)
VALUES('transcribe_video', NULL, ?, ?, 'pending')
ON CONFLICT DO NOTHING
`, out.VideoID, s.ModelID); err != nil {
			_ = tx.Rollback()
			return StoreResult{}, fmt.Errorf("insert transcribe job: %w", err)
		}
	}

	if err := os.Rename(tmpPath, storageAbs); err != nil {
		_ = tx.Rollback()
		return StoreResult{}, fmt.Errorf("move upload to video storage: %w", err)
	}
	renamedPaths = append(renamedPaths, storageAbs)

	if err := tx.Commit(); err != nil {
		return StoreResult{}, fmt.Errorf("commit video tx: %w", err)
	}
	committed = true
	return out, nil
}

func (s *Service) storeVideoFrameTx(ctx context.Context, tx *sql.Tx, videoID int64, frame SampledFrame) (int64, string, error) {
	content, err := os.ReadFile(frame.Path)
	if err != nil {
		return 0, "", fmt.Errorf("read sampled frame: %w", err)
	}
	header := content
	if len(header) > 512 {
		header = header[:512]
	}
	mime := resolveMime(header, http.DetectContentType(header))
	if !isSupportedImageMime(mime) {
		return 0, "", ErrUnsupportedFormat
	}
	width, height, err := decodeDimensionsFromBytes(content, mime)
	if err != nil {
		return 0, "", err
	}

	hash := sha256.Sum256(content)
	digest := hex.EncodeToString(hash[:])
	storageRel := filepath.ToSlash(filepath.Join("images", digest))
	storageAbs := filepath.Join(s.DataDir, storageRel)

	res, err := tx.ExecContext(ctx, `
INSERT INTO images(sha256, original_name, storage_path, mime_type, width, height)
VALUES(?, ?, ?, ?, ?, ?)
ON CONFLICT(sha256) DO NOTHING
`, digest, filepath.Base(frame.Path), storageRel, mime, width, height)
	if err != nil {
		return 0, "", fmt.Errorf("insert sampled frame image: %w", err)
	}
	rows, err := res.RowsAffected()
	if err != nil {
		return 0, "", fmt.Errorf("sampled frame rows affected: %w", err)
	}

	var imageID int64
	if rows > 0 {
		imageID, err = res.LastInsertId()
		if err != nil {
			return 0, "", fmt.Errorf("sampled frame last insert id: %w", err)
		}
	} else {
		if err := tx.QueryRowContext(ctx, `SELECT id FROM images WHERE sha256 = ?`, digest).Scan(&imageID); err != nil {
			return 0, "", fmt.Errorf("load sampled frame image: %w", err)
		}
	}

	if _, err := tx.ExecContext(ctx, `
INSERT INTO video_frames(video_id, image_id, frame_index, timestamp_ms)
VALUES(?, ?, ?, ?)
`, videoID, imageID, frame.FrameIndex, frame.TimestampMS); err != nil {
		return 0, "", fmt.Errorf("insert video frame: %w", err)
	}
	if _, err := tx.ExecContext(ctx, `
INSERT INTO index_jobs(kind, image_id, model_id, state)
VALUES('embed_image', ?, ?, 'pending')
ON CONFLICT DO NOTHING
`, imageID, s.ModelID); err != nil {
		return 0, "", fmt.Errorf("insert sampled frame embed job: %w", err)
	}

	if rows > 0 {
		if err := os.Rename(frame.Path, storageAbs); err != nil {
			return 0, "", fmt.Errorf("move sampled frame to storage: %w", err)
		}
		return imageID, storageAbs, nil
	}

	if err := os.Remove(frame.Path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return 0, "", fmt.Errorf("remove duplicate sampled frame temp file: %w", err)
	}
	return imageID, "", nil
}
