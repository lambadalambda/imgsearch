package bruteforce

import (
	"context"
	"database/sql"
	"testing"

	_ "github.com/mattn/go-sqlite3"

	"imgsearch/internal/db"
	"imgsearch/internal/vectorindex"
)

func setupIndex(t *testing.T) *Index {
	t.Helper()

	dbConn, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	t.Cleanup(func() { _ = dbConn.Close() })

	if err := db.RunMigrations(context.Background(), dbConn); err != nil {
		t.Fatalf("run migrations: %v", err)
	}

	return NewIndex(dbConn)
}

func TestSearchRanksByCosineDistance(t *testing.T) {
	idx := setupIndex(t)

	if err := idx.Upsert(context.Background(), 1, 10, []float32{1, 0}); err != nil {
		t.Fatalf("upsert 1: %v", err)
	}
	if err := idx.Upsert(context.Background(), 2, 10, []float32{0.8, 0.2}); err != nil {
		t.Fatalf("upsert 2: %v", err)
	}
	if err := idx.Upsert(context.Background(), 3, 10, []float32{0, 1}); err != nil {
		t.Fatalf("upsert 3: %v", err)
	}

	hits, err := idx.Search(context.Background(), 10, []float32{1, 0}, 3)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(hits) != 3 {
		t.Fatalf("expected 3 hits, got %d", len(hits))
	}
	if hits[0].ImageID != 1 {
		t.Fatalf("expected closest image 1, got %d", hits[0].ImageID)
	}
	if hits[1].ImageID != 2 {
		t.Fatalf("expected second closest image 2, got %d", hits[1].ImageID)
	}
}

func TestSearchByImageIDExcludesSelf(t *testing.T) {
	idx := setupIndex(t)

	_ = idx.Upsert(context.Background(), 1, 10, []float32{1, 0})
	_ = idx.Upsert(context.Background(), 2, 10, []float32{0.9, 0.1})
	_ = idx.Upsert(context.Background(), 3, 10, []float32{0, 1})

	hits, err := idx.SearchByImageID(context.Background(), 10, 1, 2)
	if err != nil {
		t.Fatalf("search by image: %v", err)
	}
	if len(hits) != 2 {
		t.Fatalf("expected 2 hits, got %d", len(hits))
	}
	if hits[0].ImageID == 1 || hits[1].ImageID == 1 {
		t.Fatalf("expected self to be excluded: %+v", hits)
	}
}

func TestSearchByImageIDReturnsNotFound(t *testing.T) {
	idx := setupIndex(t)

	_, err := idx.SearchByImageID(context.Background(), 10, 999, 5)
	if err == nil {
		t.Fatal("expected not found error")
	}
	if err != vectorindex.ErrNotFound {
		t.Fatalf("expected ErrNotFound, got %v", err)
	}
}
