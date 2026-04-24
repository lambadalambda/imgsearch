package app

import (
	"context"
	"database/sql"
	"errors"
	"strings"
	"testing"

	_ "github.com/mattn/go-sqlite3"

	"imgsearch/internal/vectorindex"
	"imgsearch/internal/vectorindex/bruteforce"
)

func openRuntimeTestDB(_ string, _ string) (*sql.DB, error) {
	return sql.Open("sqlite3", ":memory:")
}

func buildRuntimeTestBackend(backend string, db *sql.DB) (vectorindex.VectorIndex, ValidateVectorFn, error) {
	return bruteforce.NewIndex(db), func(context.Context, *sql.DB) error { return nil }, nil
}

func TestInitializeDataRuntimeFallsBackFromAutoWhenVectorUnavailable(t *testing.T) {
	runtime, err := InitializeDataRuntime(context.Background(), DataRuntimeConfig{
		DataDir:                t.TempDir(),
		RequestedVectorBackend: "auto",
	}, DataRuntimeDeps{
		DiscoverSQLiteVectorPath: func(string) (string, error) { return "", nil },
		OpenSQLiteDB:             openRuntimeTestDB,
		ValidateSQLiteVector:     func(context.Context, *sql.DB) error { return errors.New("sqlite-vector unavailable") },
		ResolveVectorBackend: func(requested string, validateErr error) (string, string, error) {
			if requested != "auto" {
				t.Fatalf("requested backend: got=%q want=auto", requested)
			}
			if validateErr == nil {
				t.Fatal("expected validation error for auto fallback")
			}
			return "bruteforce", "falling back", nil
		},
		BuildVectorBackend: buildRuntimeTestBackend,
	})
	if err != nil {
		t.Fatalf("initialize runtime: %v", err)
	}
	t.Cleanup(func() { _ = runtime.Close() })

	if runtime.VectorBackend != "bruteforce" {
		t.Fatalf("backend: got=%q want=bruteforce", runtime.VectorBackend)
	}
	if runtime.BackendWarning == "" {
		t.Fatal("expected backend fallback warning")
	}
	if runtime.DB == nil || runtime.Index == nil {
		t.Fatalf("expected db and index: %+v", runtime)
	}
}

func TestInitializeDataRuntimeReopensWithoutVectorOnAutoValidationFailure(t *testing.T) {
	openPaths := []string{}
	runtime, err := InitializeDataRuntime(context.Background(), DataRuntimeConfig{
		DataDir:                t.TempDir(),
		RequestedVectorBackend: "auto",
	}, DataRuntimeDeps{
		DiscoverSQLiteVectorPath: func(string) (string, error) { return "/tmp/vector.so", nil },
		OpenSQLiteDB: func(_ string, sqliteVectorPath string) (*sql.DB, error) {
			openPaths = append(openPaths, sqliteVectorPath)
			return sql.Open("sqlite3", ":memory:")
		},
		ValidateSQLiteVector: func(context.Context, *sql.DB) error { return errors.New("wrong architecture") },
		ResolveVectorBackend: func(requested string, validateErr error) (string, string, error) {
			if requested != "auto" || validateErr == nil {
				t.Fatalf("expected auto validation error, got requested=%q err=%v", requested, validateErr)
			}
			return "bruteforce", "falling back", nil
		},
		BuildVectorBackend: buildRuntimeTestBackend,
	})
	if err != nil {
		t.Fatalf("initialize runtime: %v", err)
	}
	t.Cleanup(func() { _ = runtime.Close() })

	if runtime.VectorBackend != "bruteforce" {
		t.Fatalf("backend: got=%q want=bruteforce", runtime.VectorBackend)
	}
	if len(openPaths) != 2 || openPaths[0] != "/tmp/vector.so" || openPaths[1] != "" {
		t.Fatalf("expected open with vector then reopen without vector, got %#v", openPaths)
	}
	if err := runtime.DB.Ping(); err != nil {
		t.Fatalf("reopened db should be usable: %v", err)
	}
}

func TestInitializeDataRuntimeUsesSQLiteVectorWhenAutoValidationSucceeds(t *testing.T) {
	openPaths := []string{}
	runtime, err := InitializeDataRuntime(context.Background(), DataRuntimeConfig{
		DataDir:                t.TempDir(),
		RequestedVectorBackend: "auto",
	}, DataRuntimeDeps{
		DiscoverSQLiteVectorPath: func(string) (string, error) { return "/tmp/vector.so", nil },
		OpenSQLiteDB: func(_ string, sqliteVectorPath string) (*sql.DB, error) {
			openPaths = append(openPaths, sqliteVectorPath)
			return sql.Open("sqlite3", ":memory:")
		},
		ValidateSQLiteVector: func(context.Context, *sql.DB) error { return nil },
		ResolveVectorBackend: func(requested string, validateErr error) (string, string, error) {
			if requested != "auto" || validateErr != nil {
				t.Fatalf("expected valid auto sqlite-vector path, got requested=%q err=%v", requested, validateErr)
			}
			return "sqlite-vector", "", nil
		},
		BuildVectorBackend: buildRuntimeTestBackend,
	})
	if err != nil {
		t.Fatalf("initialize runtime: %v", err)
	}
	t.Cleanup(func() { _ = runtime.Close() })

	if runtime.VectorBackend != "sqlite-vector" {
		t.Fatalf("backend: got=%q want=sqlite-vector", runtime.VectorBackend)
	}
	if len(openPaths) != 1 || openPaths[0] != "/tmp/vector.so" {
		t.Fatalf("expected one open with vector path, got %#v", openPaths)
	}
}

func TestInitializeDataRuntimeRejectsExplicitSQLiteVectorWithoutPath(t *testing.T) {
	_, err := InitializeDataRuntime(context.Background(), DataRuntimeConfig{
		DataDir:                t.TempDir(),
		RequestedVectorBackend: "sqlite-vector",
	}, DataRuntimeDeps{
		DiscoverSQLiteVectorPath: func(string) (string, error) { return "", nil },
		OpenSQLiteDB:             openRuntimeTestDB,
		ValidateSQLiteVector:     func(context.Context, *sql.DB) error { return nil },
		ResolveVectorBackend:     func(string, error) (string, string, error) { return "sqlite-vector", "", nil },
		BuildVectorBackend:       buildRuntimeTestBackend,
	})
	if err == nil || !strings.Contains(err.Error(), "sqlite-vector backend requested") {
		t.Fatalf("expected explicit sqlite-vector path error, got %v", err)
	}
}

func TestInitializeDataRuntimeBootstrapsDatabase(t *testing.T) {
	runtime, err := InitializeDataRuntime(context.Background(), DataRuntimeConfig{
		DataDir:                t.TempDir(),
		RequestedVectorBackend: "bruteforce",
	}, DataRuntimeDeps{
		DiscoverSQLiteVectorPath: func(string) (string, error) { return "", nil },
		OpenSQLiteDB:             openRuntimeTestDB,
		ValidateSQLiteVector:     func(context.Context, *sql.DB) error { return nil },
		ResolveVectorBackend:     func(string, error) (string, string, error) { return "bruteforce", "", nil },
		BuildVectorBackend:       buildRuntimeTestBackend,
	})
	if err != nil {
		t.Fatalf("initialize runtime: %v", err)
	}
	t.Cleanup(func() { _ = runtime.Close() })

	var tableName string
	if err := runtime.DB.QueryRow(`SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'images'`).Scan(&tableName); err != nil {
		t.Fatalf("expected migrations to create images table: %v", err)
	}
}
