package app

import (
	"context"
	"database/sql"
	"errors"
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
