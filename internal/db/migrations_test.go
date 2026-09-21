package db

import (
	"context"
	"database/sql"
	"strings"
	"testing"

	_ "github.com/mattn/go-sqlite3"
)

func openTestDB(t *testing.T) *sql.DB {
	t.Helper()

	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	return db
}

func TestRunMigrationsAddsImageAnnotationColumns(t *testing.T) {
	db := openTestDB(t)

	if err := RunMigrations(context.Background(), db); err != nil {
		t.Fatalf("run migrations: %v", err)
	}

	columns := []string{"description", "tags_json", "title", "summary", "reannotate_requested"}
	for _, column := range columns {
		rows, err := db.Query(`PRAGMA table_info(images)`)
		if err != nil {
			t.Fatalf("pragma table_info(images): %v", err)
		}
		var count int
		for rows.Next() {
			var (
				cid        int
				name       string
				columnType string
				notNull    int
				defaultVal any
				pk         int
			)
			if err := rows.Scan(&cid, &name, &columnType, &notNull, &defaultVal, &pk); err != nil {
				_ = rows.Close()
				t.Fatalf("scan pragma row: %v", err)
			}
			if name == column {
				count++
			}
		}
		_ = rows.Close()
		if count != 1 {
			t.Fatalf("expected images.%s column to exist exactly once, got %d", column, count)
		}
	}
}

func TestRunMigrationsAddsVideoAnnotationTextColumns(t *testing.T) {
	db := openTestDB(t)

	if err := RunMigrations(context.Background(), db); err != nil {
		t.Fatalf("run migrations: %v", err)
	}

	columns := []string{"title", "summary"}
	for _, column := range columns {
		rows, err := db.Query(`PRAGMA table_info(videos)`)
		if err != nil {
			t.Fatalf("pragma table_info(videos): %v", err)
		}
		var count int
		for rows.Next() {
			var (
				cid        int
				name       string
				columnType string
				notNull    int
				defaultVal any
				pk         int
			)
			if err := rows.Scan(&cid, &name, &columnType, &notNull, &defaultVal, &pk); err != nil {
				_ = rows.Close()
				t.Fatalf("scan pragma row: %v", err)
			}
			if name == column {
				count++
			}
		}
		_ = rows.Close()
		if count != 1 {
			t.Fatalf("expected videos.%s column to exist exactly once, got %d", column, count)
		}
	}
}

func TestRunMigrationsAddsVideoReannotateRequestedColumn(t *testing.T) {
	db := openTestDB(t)

	if err := RunMigrations(context.Background(), db); err != nil {
		t.Fatalf("run migrations: %v", err)
	}

	rows, err := db.Query(`PRAGMA table_info(videos)`)
	if err != nil {
		t.Fatalf("pragma table_info(videos): %v", err)
	}
	defer func() { _ = rows.Close() }()

	found := 0
	for rows.Next() {
		var (
			cid        int
			name       string
			columnType string
			notNull    int
			defaultVal any
			pk         int
		)
		if err := rows.Scan(&cid, &name, &columnType, &notNull, &defaultVal, &pk); err != nil {
			t.Fatalf("scan pragma row: %v", err)
		}
		if name == "reannotate_requested" {
			found++
		}
	}
	if found != 1 {
		t.Fatalf("expected videos.reannotate_requested column to exist exactly once, got %d", found)
	}
}

func TestRunMigrationsIdempotent(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()

	if err := RunMigrations(ctx, db); err != nil {
		t.Fatalf("first run migrations: %v", err)
	}

	if err := RunMigrations(ctx, db); err != nil {
		t.Fatalf("second run migrations: %v", err)
	}

	got, err := CurrentVersion(ctx, db)
	if err != nil {
		t.Fatalf("current version: %v", err)
	}
	if got != LatestVersion() {
		t.Fatalf("version mismatch: got=%d want=%d", got, LatestVersion())
	}
}

func TestRunMigrationsCreatesCoreTables(t *testing.T) {
	db := openTestDB(t)

	if err := RunMigrations(context.Background(), db); err != nil {
		t.Fatalf("run migrations: %v", err)
	}

	tables := []string{
		"schema_migrations",
		"images",
		"videos",
		"video_frames",
		"video_transcript_embeddings",
		"embedding_models",
		"image_embeddings",
		"index_jobs",
		"settings",
		"settings_version",
	}

	for _, table := range tables {
		var got string
		err := db.QueryRow(
			`SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?`,
			table,
		).Scan(&got)
		if err != nil {
			t.Fatalf("expected table %q to exist: %v", table, err)
		}
	}
}

// TestRunMigrationsAddsIndexJobsLookupIndexes pins the query plans for the
// index_jobs shapes that used to full-scan or build automatic indexes
// (meta/issues/099): media deletes, the library list joins, the video list
// CTEs, and the worker claim.
func TestRunMigrationsAddsIndexJobsLookupIndexes(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()
	if err := RunMigrations(ctx, db); err != nil {
		t.Fatalf("run migrations: %v", err)
	}

	for _, name := range []string{"idx_index_jobs_image_lookup", "idx_index_jobs_video_lookup", "idx_index_jobs_claim"} {
		var got string
		if err := db.QueryRowContext(ctx, `SELECT name FROM sqlite_master WHERE type = 'index' AND name = ?`, name).Scan(&got); err != nil {
			t.Fatalf("index %s missing: %v", name, err)
		}
	}

	queries := map[string]string{
		"delete image jobs": `DELETE FROM index_jobs WHERE image_id = 1`,
		"delete video jobs": `DELETE FROM index_jobs WHERE video_id = 1`,
		"images list join":  `SELECT i.id, COALESCE(j.state, 'pending') FROM images i LEFT JOIN index_jobs j ON j.image_id = i.id AND j.model_id = 1 AND j.kind = 'embed_image' ORDER BY i.id DESC LIMIT 10`,
		"videos list cte":   `SELECT j.video_id, COUNT(j.id) FROM index_jobs j WHERE j.video_id IS NOT NULL AND j.model_id = 1 AND j.kind = 'annotate_video' GROUP BY j.video_id`,
		"frame jobs cte":    `SELECT vf.video_id, COUNT(j.id) FROM video_frames vf LEFT JOIN index_jobs j ON j.image_id = vf.image_id AND j.model_id = 1 AND j.kind = 'embed_image' GROUP BY vf.video_id`,
		"worker claim":      `SELECT id, kind FROM index_jobs WHERE kind IN ('embed_image', 'annotate_image') AND (run_after IS NULL OR run_after <= datetime('now')) AND (state = 'pending' OR (state = 'leased' AND leased_until IS NOT NULL AND leased_until <= datetime('now'))) ORDER BY CASE kind WHEN 'embed_image' THEN 0 ELSE 1 END ASC, created_at ASC LIMIT 1`,
	}
	for name, query := range queries {
		plan := explainQueryPlan(t, db, query)
		if strings.Contains(plan, "SCAN index_jobs") || strings.Contains(plan, "AUTOMATIC") {
			t.Fatalf("%s still scans index_jobs:\n%s", name, plan)
		}
		if !strings.Contains(plan, "USING") || !strings.Contains(plan, "idx_index_jobs_") {
			t.Fatalf("%s does not use an index_jobs lookup index:\n%s", name, plan)
		}
	}
}

func explainQueryPlan(t *testing.T, db *sql.DB, query string) string {
	t.Helper()
	rows, err := db.QueryContext(context.Background(), "EXPLAIN QUERY PLAN "+query)
	if err != nil {
		t.Fatalf("explain %q: %v", query, err)
	}
	defer func() { _ = rows.Close() }()
	var lines []string
	for rows.Next() {
		var id, parent, notUsed int
		var detail string
		if err := rows.Scan(&id, &parent, &notUsed, &detail); err != nil {
			t.Fatalf("scan plan row: %v", err)
		}
		lines = append(lines, detail)
	}
	return strings.Join(lines, "\n")
}
