package sqlitevector

import (
	"context"
	"database/sql"
	"strings"
	"testing"

	_ "github.com/mattn/go-sqlite3"
)

func TestValidateAvailableReturnsHelpfulErrorWhenUnavailable(t *testing.T) {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })

	err = ValidateAvailable(context.Background(), db)
	if err == nil {
		t.Fatal("expected validation error when sqlite-vector is unavailable")
	}
	if !strings.Contains(err.Error(), "sqlite-vector extension is not available") {
		t.Fatalf("unexpected error: %v", err)
	}
}
