package settings

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"path/filepath"
	"testing"

	_ "github.com/mattn/go-sqlite3"

	"imgsearch/internal/db"
)

func openTestDB(t *testing.T) *sql.DB {
	t.Helper()
	conn, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	if err := db.RunMigrations(context.Background(), conn); err != nil {
		t.Fatalf("run migrations: %v", err)
	}
	return conn
}

func marshalForTest(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return string(b)
}

func TestLoadAnnotationReturnsNotFoundBeforeFirstSave(t *testing.T) {
	conn := openTestDB(t)
	_, err := LoadAnnotation(context.Background(), conn)
	if !errors.Is(err, ErrNotFound) {
		t.Fatalf("expected ErrNotFound, got %v", err)
	}
	v, err := Version(context.Background(), conn)
	if err != nil || v != 0 {
		t.Fatalf("expected version 0, got %d err=%v", v, err)
	}
}

func TestSaveAndLoadAnnotationRoundTripBumpsVersion(t *testing.T) {
	conn := openTestDB(t)
	ctx := context.Background()
	in := AnnotationSettings{
		Backend: BackendOpenAI,
		OpenAI:  OpenAISettings{BaseURL: "http://127.0.0.1:8081/v1", APIKey: "k", Model: "m", TimeoutSeconds: 30, Concurrency: 3},
	}
	v1, err := SaveAnnotation(ctx, conn, in)
	if err != nil {
		t.Fatalf("save: %v", err)
	}
	if v1 != 1 {
		t.Fatalf("expected version 1 after first save, got %d", v1)
	}
	got, err := LoadAnnotation(ctx, conn)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if got != in.Normalized() {
		t.Fatalf("round trip mismatch:\n got=%+v\nwant=%+v", got, in.Normalized())
	}
	v2, err := SaveAnnotation(ctx, conn, in)
	if err != nil || v2 != 2 {
		t.Fatalf("expected version 2 after second save, got %d err=%v", v2, err)
	}
	if v, _ := Version(ctx, conn); v != 2 {
		t.Fatalf("Version() = %d, want 2", v)
	}
}

func TestSaveAnnotationRejectsInvalid(t *testing.T) {
	conn := openTestDB(t)
	if _, err := SaveAnnotation(context.Background(), conn, AnnotationSettings{Backend: "nope"}); err == nil {
		t.Fatal("expected validation error")
	}
	if v, _ := Version(context.Background(), conn); v != 0 {
		t.Fatalf("failed save must not bump version, got %d", v)
	}
}

func TestSettingsAreVisibleFromASecondConnection(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "settings.db")
	open := func() *sql.DB {
		conn, err := sql.Open("sqlite3", path)
		if err != nil {
			t.Fatalf("open sqlite: %v", err)
		}
		t.Cleanup(func() { _ = conn.Close() })
		if err := db.RunMigrations(ctx, conn); err != nil {
			t.Fatalf("run migrations: %v", err)
		}
		return conn
	}
	api := open()
	worker := open()

	if v, err := Version(ctx, worker); err != nil || v != 0 {
		t.Fatalf("worker initial version: %d err=%v", v, err)
	}
	in := AnnotationSettings{Backend: BackendOpenAI, OpenAI: OpenAISettings{BaseURL: "http://x", Model: "m"}}
	if _, err := SaveAnnotation(ctx, api, in); err != nil {
		t.Fatalf("save via api: %v", err)
	}
	if v, err := Version(ctx, worker); err != nil || v != 1 {
		t.Fatalf("worker should observe version 1, got %d err=%v", v, err)
	}
	got, err := LoadAnnotation(ctx, worker)
	if err != nil || got.OpenAI.Model != "m" {
		t.Fatalf("worker should load saved settings, got %+v err=%v", got, err)
	}
}
