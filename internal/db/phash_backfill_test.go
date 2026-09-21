package db

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"imgsearch/internal/phash"
)

func TestBackfillPhashHashesStandaloneImagesOnly(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()
	if err := RunMigrations(ctx, db); err != nil {
		t.Fatalf("run migrations: %v", err)
	}
	dataDir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dataDir, "images"), 0o755); err != nil {
		t.Fatal(err)
	}
	_, thisFile, _, _ := runtime.Caller(0)
	fixtures := filepath.Join(filepath.Dir(thisFile), "..", "..", "fixtures", "images")
	copyFixture := func(name, dest string) {
		data, err := os.ReadFile(filepath.Join(fixtures, name))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dataDir, "images", dest), data, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	copyFixture("cat_2.jpg", "a")
	copyFixture("cat_2.webp", "b")
	copyFixture("dog_2.avif", "c")
	copyFixture("cat_1.jpg", "frame")
	if _, err := db.ExecContext(ctx, `
INSERT INTO images(id, sha256, original_name, storage_path, mime_type, width, height) VALUES
  (1, 'a', 'a.jpg', 'images/a', 'image/jpeg', 1, 1),
  (2, 'b', 'b.webp', 'images/b', 'image/webp', 1, 1),
  (3, 'c', 'c.avif', 'images/c', 'image/avif', 1, 1),
  (4, 'f', 'frame', 'images/frame', 'image/jpeg', 1, 1),
  (5, 'm', 'missing.jpg', 'images/missing', 'image/jpeg', 1, 1);
INSERT INTO videos(id, sha256, original_name, storage_path, mime_type, duration_ms, width, height, frame_count)
  VALUES (1, 'v', 'v.mp4', 'videos/v', 'video/mp4', 1000, 1, 1, 1);
INSERT INTO video_frames(video_id, image_id, frame_index, timestamp_ms) VALUES (1, 4, 0, 0);
`); err != nil {
		t.Fatalf("seed: %v", err)
	}

	hashed, err := BackfillPhash(ctx, db, dataDir)
	if err != nil {
		t.Fatalf("backfill: %v", err)
	}
	if hashed != 2 {
		t.Fatalf("hashed: got=%d want=2", hashed)
	}
	values := map[int64]int64{}
	rows, err := db.QueryContext(ctx, `SELECT id, phash FROM images ORDER BY id`)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = rows.Close() }()
	for rows.Next() {
		var id, value int64
		if err := rows.Scan(&id, &value); err != nil {
			t.Fatalf("scan (NULL left?): %v", err)
		}
		values[id] = value
	}
	if values[3] != phash.Unhashable || values[4] != phash.Unhashable || values[5] != phash.Unhashable {
		t.Fatalf("expected avif, frame, and missing file to be unhashable: %v", values)
	}
	if d := phash.Distance(phash.FromInt64(values[1]), phash.FromInt64(values[2])); d > 6 {
		t.Fatalf("jpeg/webp copies distance %d", d)
	}
	if again, err := BackfillPhash(ctx, db, dataDir); err != nil || again != 0 {
		t.Fatalf("second pass: %d %v", again, err)
	}
}
