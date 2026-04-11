package upload

import (
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
)

var ErrUnsupportedFormat = errors.New("unsupported image format")

type Service struct {
	DB      *sql.DB
	DataDir string
	ModelID int64
}

type StoreResult struct {
	ImageID     int64
	SHA256      string
	StoragePath string
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
	if err := os.MkdirAll(tmpDir, 0o755); err != nil {
		return StoreResult{}, fmt.Errorf("create tmp dir: %w", err)
	}
	if err := os.MkdirAll(imageDir, 0o755); err != nil {
		return StoreResult{}, fmt.Errorf("create images dir: %w", err)
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
	mime := http.DetectContentType(header[:n])
	if mime != "image/png" && mime != "image/jpeg" {
		return StoreResult{}, ErrUnsupportedFormat
	}

	if _, err := tmpFile.Seek(0, io.SeekStart); err != nil {
		return StoreResult{}, fmt.Errorf("seek for decode: %w", err)
	}
	cfg, _, err := image.DecodeConfig(tmpFile)
	if err != nil {
		return StoreResult{}, ErrUnsupportedFormat
	}

	digest := hex.EncodeToString(hasher.Sum(nil))
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
`, digest, originalName, storageRel, mime, cfg.Width, cfg.Height)
	if err != nil {
		_ = tx.Rollback()
		return StoreResult{}, fmt.Errorf("insert image: %w", err)
	}

	rows, err := res.RowsAffected()
	if err != nil {
		_ = tx.Rollback()
		return StoreResult{}, fmt.Errorf("rows affected: %w", err)
	}

	out := StoreResult{SHA256: digest, StoragePath: storageRel, Duplicate: rows == 0}
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
ON CONFLICT(kind, image_id, model_id) DO NOTHING
`, out.ImageID, s.ModelID); err != nil {
		_ = tx.Rollback()
		return StoreResult{}, fmt.Errorf("insert index job: %w", err)
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
