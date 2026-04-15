package app

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"
	"testing"

	_ "github.com/mattn/go-sqlite3"
)

func openDB(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	return db
}

func TestBootstrapRunsMigrationsAndValidation(t *testing.T) {
	sqlDB := openDB(t)

	called := false
	validate := func(context.Context, *sql.DB) error {
		called = true
		return nil
	}

	if err := Bootstrap(context.Background(), sqlDB, validate); err != nil {
		t.Fatalf("bootstrap: %v", err)
	}
	if !called {
		t.Fatal("expected vector validator to be called")
	}
}

func TestBootstrapEnablesForeignKeys(t *testing.T) {
	sqlDB := openDB(t)

	if err := Bootstrap(context.Background(), sqlDB, func(context.Context, *sql.DB) error { return nil }); err != nil {
		t.Fatalf("bootstrap: %v", err)
	}

	var enabled int
	if err := sqlDB.QueryRow(`PRAGMA foreign_keys`).Scan(&enabled); err != nil {
		t.Fatalf("read foreign_keys pragma: %v", err)
	}
	if enabled != 1 {
		t.Fatalf("foreign_keys pragma: got=%d want=1", enabled)
	}

	res, err := sqlDB.Exec(`
INSERT INTO images(sha256, original_name, storage_path, mime_type, width, height)
VALUES('seed', 'seed.jpg', 'images/seed', 'image/jpeg', 1, 1)
`)
	if err != nil {
		t.Fatalf("seed image: %v", err)
	}
	imageID, err := res.LastInsertId()
	if err != nil {
		t.Fatalf("last insert id: %v", err)
	}

	_, err = sqlDB.Exec(`
INSERT INTO image_embeddings(image_id, model_id, dim, vector_blob)
VALUES(?, ?, 4, X'00000000')
`, imageID, int64(9999))
	if err == nil {
		t.Fatal("expected foreign key violation for missing embedding model")
	}
	if !strings.Contains(strings.ToLower(fmt.Sprint(err)), "foreign key") {
		t.Fatalf("expected foreign key error, got %v", err)
	}
}

func TestBootstrapReturnsValidationError(t *testing.T) {
	sqlDB := openDB(t)
	want := errors.New("vector unavailable")

	err := Bootstrap(context.Background(), sqlDB, func(context.Context, *sql.DB) error { return want })
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "validate vector backend") {
		t.Fatalf("expected wrapped validation error, got %v", err)
	}
}
