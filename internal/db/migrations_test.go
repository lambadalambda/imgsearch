package db

import (
	"context"
	"database/sql"
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

	columns := []string{"description", "tags_json", "reannotate_requested"}
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
