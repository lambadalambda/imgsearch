package db

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"imgsearch/internal/exif"
)

// CapturedAtFromFile reads the EXIF capture time of a stored image. It
// returns the SQLite-formatted time, or "" when the file carries none, so
// callers can mark the row as scanned either way.
func CapturedAtFromFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer func() { _ = f.Close() }()
	info, err := exif.Parse(f)
	if err != nil {
		return "", err
	}
	if info.CapturedAt.IsZero() {
		return "", nil
	}
	return exif.SQLiteTime(info.CapturedAt), nil
}

// BackfillCapturedAt fills images.captured_at for rows that were stored
// before the column existed (captured_at IS NULL). Rows whose file is
// missing or has no EXIF are marked with "" so they are not rescanned.
// It returns how many rows received a capture time.
func BackfillCapturedAt(ctx context.Context, db *sql.DB, dataDir string) (int, error) {
	const batch = 200
	filled := 0
	for {
		rows, err := db.QueryContext(ctx, `
SELECT id, storage_path, mime_type FROM images
WHERE captured_at IS NULL
ORDER BY id
LIMIT ?`, batch)
		if err != nil {
			return filled, fmt.Errorf("select unscanned images: %w", err)
		}
		type pending struct {
			id   int64
			path string
			mime string
		}
		var todo []pending
		for rows.Next() {
			var p pending
			if err := rows.Scan(&p.id, &p.path, &p.mime); err != nil {
				_ = rows.Close()
				return filled, fmt.Errorf("scan unscanned image: %w", err)
			}
			todo = append(todo, p)
		}
		_ = rows.Close()
		if err := rows.Err(); err != nil {
			return filled, err
		}
		if len(todo) == 0 {
			return filled, nil
		}
		for _, p := range todo {
			if err := ctx.Err(); err != nil {
				return filled, err
			}
			captured := ""
			if strings.EqualFold(p.mime, "image/jpeg") {
				if value, err := CapturedAtFromFile(filepath.Join(dataDir, filepath.FromSlash(p.path))); err == nil {
					captured = value
				}
			}
			if _, err := db.ExecContext(ctx, `UPDATE images SET captured_at = ? WHERE id = ? AND captured_at IS NULL`, captured, p.id); err != nil {
				return filled, fmt.Errorf("update captured_at: %w", err)
			}
			if captured != "" {
				filled++
			}
		}
	}
}
