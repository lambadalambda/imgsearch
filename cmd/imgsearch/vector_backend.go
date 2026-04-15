package main

import (
	"context"
	"crypto/sha1"
	"database/sql"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	sqlite3 "github.com/mattn/go-sqlite3"

	"imgsearch/internal/app"
	"imgsearch/internal/vectorindex"
	"imgsearch/internal/vectorindex/bruteforce"
	"imgsearch/internal/vectorindex/sqlitevector"
)

const (
	vectorBackendAuto         = "auto"
	vectorBackendSQLiteVector = "sqlite-vector"
	vectorBackendBruteForce   = "bruteforce"
	sqliteVectorEntryPoint    = "sqlite3_vector_init"
)

var (
	vectorDriverMu       sync.Mutex
	vectorDriversByName  = make(map[string]struct{})
	vectorEnvPathKeyName = "SQLITE_VECTOR_PATH"
)

func resolveVectorBackend(requested string, sqliteValidateErr error) (backend string, warning string, err error) {
	switch requested {
	case vectorBackendAuto:
		if sqliteValidateErr == nil {
			return vectorBackendSQLiteVector, "", nil
		}
		return vectorBackendBruteForce,
			fmt.Sprintf("sqlite-vector unavailable, falling back to bruteforce backend: %v", sqliteValidateErr),
			nil
	case vectorBackendSQLiteVector:
		return vectorBackendSQLiteVector, "", nil
	case vectorBackendBruteForce:
		return vectorBackendBruteForce, "", nil
	default:
		return "", "", fmt.Errorf("unknown vector backend %q (expected: auto, sqlite-vector, bruteforce)", requested)
	}
}

func buildVectorBackend(backend string, db *sql.DB) (vectorindex.VectorIndex, app.ValidateVectorFn, error) {
	switch backend {
	case vectorBackendSQLiteVector:
		return sqlitevector.NewIndex(db), sqlitevector.ValidateAvailable, nil
	case vectorBackendBruteForce:
		return bruteforce.NewIndex(db), noVectorValidation, nil
	default:
		return nil, nil, fmt.Errorf("unknown vector backend %q", backend)
	}
}

func noVectorValidation(context.Context, *sql.DB) error {
	return nil
}

func discoverSQLiteVectorPath(explicitPath string) (string, error) {
	if explicitPath != "" {
		if resolved, ok := resolveExistingExtensionPath(explicitPath); ok {
			return resolved, nil
		}
		return "", fmt.Errorf("sqlite-vector extension not found at %q", explicitPath)
	}

	if envPath := strings.TrimSpace(os.Getenv(vectorEnvPathKeyName)); envPath != "" {
		if resolved, ok := resolveExistingExtensionPath(envPath); ok {
			return resolved, nil
		}
		return "", fmt.Errorf("sqlite-vector extension not found at %q from %s", envPath, vectorEnvPathKeyName)
	}

	for _, candidate := range sqliteVectorPathCandidates() {
		if resolved, ok := resolveExistingExtensionPath(candidate); ok {
			return resolved, nil
		}
	}
	return "", nil
}

func sqliteVectorPathCandidates() []string {
	baseNames := []string{
		filepath.Join("tools", "sqlite-vector", "vector"),
		filepath.Join(".tools", "sqlite-vector", "vector"),
	}

	exts := []string{""}
	switch runtime.GOOS {
	case "darwin":
		exts = []string{".dylib", ""}
	case "windows":
		exts = []string{".dll", ""}
	default:
		exts = []string{".so", ""}
	}

	out := make([]string, 0, len(baseNames)*len(exts))
	for _, base := range baseNames {
		for _, ext := range exts {
			out = append(out, base+ext)
		}
	}
	return out
}
func resolveExistingExtensionPath(candidate string) (string, bool) {
	if candidate == "" {
		return "", false
	}

	paths := []string{candidate}
	if filepath.Ext(candidate) == "" {
		switch runtime.GOOS {
		case "darwin":
			paths = []string{candidate + ".dylib", candidate}
		case "windows":
			paths = []string{candidate + ".dll", candidate}
		default:
			paths = []string{candidate + ".so", candidate}
		}
	}

	for _, path := range paths {
		if info, err := os.Stat(path); err == nil && !info.IsDir() {
			return path, true
		}
	}
	return "", false
}

func openSQLiteDB(dsn string, sqliteVectorPath string) (*sql.DB, error) {
	driverName, err := registerSQLiteExtensionDriver(sqliteVectorPath)
	if err != nil {
		return nil, err
	}
	return sql.Open(driverName, dsn)
}

func registerSQLiteExtensionDriver(sqliteVectorPath string) (string, error) {
	vectorAbs := ""
	vectorLoadTarget := ""
	if sqliteVectorPath != "" {
		abs, err := filepath.Abs(sqliteVectorPath)
		if err != nil {
			return "", fmt.Errorf("resolve sqlite-vector path: %w", err)
		}
		vectorAbs = abs
		vectorLoadTarget = sqliteExtensionLoadTarget(abs)
	}

	driverKey := vectorAbs
	sum := sha1.Sum([]byte(driverKey))
	driverName := "sqlite3_with_vector_" + hex.EncodeToString(sum[:8])

	vectorDriverMu.Lock()
	defer vectorDriverMu.Unlock()

	if _, ok := vectorDriversByName[driverName]; ok {
		return driverName, nil
	}

	sql.Register(driverName, &sqlite3.SQLiteDriver{
		ConnectHook: func(conn *sqlite3.SQLiteConn) error {
			if _, err := conn.Exec("PRAGMA foreign_keys = ON", nil); err != nil {
				return fmt.Errorf("enable sqlite foreign keys: %w", err)
			}
			if vectorLoadTarget != "" {
				if err := conn.LoadExtension(vectorLoadTarget, sqliteVectorEntryPoint); err != nil {
					return fmt.Errorf("load sqlite-vector extension from %q: %w", vectorLoadTarget, err)
				}
			}
			return nil
		},
	})
	vectorDriversByName[driverName] = struct{}{}

	return driverName, nil
}

func sqliteExtensionLoadTarget(path string) string {
	ext := strings.ToLower(filepath.Ext(path))
	if ext == ".dylib" || ext == ".so" || ext == ".dll" {
		return strings.TrimSuffix(path, filepath.Ext(path))
	}
	return path
}
