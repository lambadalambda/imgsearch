package db

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"imgsearch/internal/exif"
)

func TestBackfillCapturedAtFillsFromExifAndMarksOthers(t *testing.T) {
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
	jpeg, err := os.ReadFile(filepath.Join(filepath.Dir(thisFile), "..", "..", "fixtures", "images", "cat_1.jpg"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	withExif := exif.InsertAPP1(jpeg, exif.BuildAPP1(1, "2023:12:24 18:30:00"))
	if err := os.WriteFile(filepath.Join(dataDir, "images", "a"), withExif, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dataDir, "images", "b"), []byte("\x89PNG not really"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := db.ExecContext(ctx, `
INSERT INTO images(id, sha256, original_name, storage_path, mime_type, width, height, captured_at)
VALUES
  (1, 'a', 'a.jpg', 'images/a', 'image/jpeg', 1, 1, NULL),
  (2, 'b', 'b.png', 'images/b', 'image/png', 1, 1, NULL),
  (3, 'c', 'missing.jpg', 'images/c', 'image/jpeg', 1, 1, NULL),
  (4, 'd', 'done.jpg', 'images/d', 'image/jpeg', 1, 1, '2020-01-01 00:00:00')
`); err != nil {
		t.Fatalf("seed: %v", err)
	}

	filled, err := BackfillCapturedAt(ctx, db, dataDir)
	if err != nil {
		t.Fatalf("backfill: %v", err)
	}
	if filled != 1 {
		t.Fatalf("filled: got=%d want=1", filled)
	}
	rows := map[int64]string{}
	res, err := db.QueryContext(ctx, `SELECT id, captured_at FROM images ORDER BY id`)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = res.Close() }()
	for res.Next() {
		var id int64
		var captured string
		if err := res.Scan(&id, &captured); err != nil {
			t.Fatalf("scan: %v (NULL left behind?)", err)
		}
		rows[id] = captured
	}
	if rows[1] != "2023-12-24 18:30:00" || rows[2] != "" || rows[3] != "" || rows[4] != "2020-01-01 00:00:00" {
		t.Fatalf("unexpected captured_at values: %+v", rows)
	}

	// A second pass finds nothing left to scan.
	if again, err := BackfillCapturedAt(ctx, db, dataDir); err != nil || again != 0 {
		t.Fatalf("second pass: filled=%d err=%v", again, err)
	}
}
