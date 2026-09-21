package db

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"path/filepath"

	"imgsearch/internal/phash"
)

// HashFromFile computes the perceptual hash of a stored image, reporting
// phash.Unhashable for files that cannot be decoded.
func HashFromFile(path string) int64 {
	f, err := os.Open(path)
	if err != nil {
		return phash.Unhashable
	}
	defer func() { _ = f.Close() }()
	h, err := phash.Compute(f)
	if err != nil {
		return phash.Unhashable
	}
	return phash.ToInt64(h)
}

// BackfillPhash fills images.phash for rows stored before the column
// existed. Sampled video frames are marked unhashable without decoding,
// since only standalone images take part in duplicate detection. It returns
// how many rows received a real hash.
func BackfillPhash(ctx context.Context, db *sql.DB, dataDir string) (int, error) {
	if _, err := db.ExecContext(ctx, `
UPDATE images SET phash = ?
WHERE phash IS NULL AND EXISTS (SELECT 1 FROM video_frames vf WHERE vf.image_id = images.id)`, phash.Unhashable); err != nil {
		return 0, fmt.Errorf("mark frame hashes: %w", err)
	}
	const batch = 100
	hashed := 0
	for {
		rows, err := db.QueryContext(ctx, `SELECT id, storage_path FROM images WHERE phash IS NULL ORDER BY id LIMIT ?`, batch)
		if err != nil {
			return hashed, fmt.Errorf("select unhashed images: %w", err)
		}
		type pending struct {
			id   int64
			path string
		}
		var todo []pending
		for rows.Next() {
			var p pending
			if err := rows.Scan(&p.id, &p.path); err != nil {
				_ = rows.Close()
				return hashed, fmt.Errorf("scan unhashed image: %w", err)
			}
			todo = append(todo, p)
		}
		_ = rows.Close()
		if err := rows.Err(); err != nil {
			return hashed, err
		}
		if len(todo) == 0 {
			return hashed, nil
		}
		for _, p := range todo {
			if err := ctx.Err(); err != nil {
				return hashed, err
			}
			value := HashFromFile(filepath.Join(dataDir, filepath.FromSlash(p.path)))
			if _, err := db.ExecContext(ctx, `UPDATE images SET phash = ? WHERE id = ? AND phash IS NULL`, value, p.id); err != nil {
				return hashed, fmt.Errorf("update phash: %w", err)
			}
			if value != phash.Unhashable {
				hashed++
			}
		}
	}
}
