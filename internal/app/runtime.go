package app

import (
	"context"
	"database/sql"
	"fmt"
	"path/filepath"

	"imgsearch/internal/vectorindex"
)

type DataRuntimeConfig struct {
	DataDir                string
	SQLiteVectorPath       string
	RequestedVectorBackend string
}

const (
	VectorBackendAuto         = "auto"
	VectorBackendSQLiteVector = "sqlite-vector"
	VectorBackendBruteForce   = "bruteforce"
)

type DataRuntimeDeps struct {
	DiscoverSQLiteVectorPath func(explicitPath string) (string, error)
	OpenSQLiteDB             func(dsn string, sqliteVectorPath string) (*sql.DB, error)
	ValidateSQLiteVector     ValidateVectorFn
	ResolveVectorBackend     func(requested string, sqliteValidateErr error) (backend string, warning string, err error)
	BuildVectorBackend       func(backend string, db *sql.DB) (vectorindex.VectorIndex, ValidateVectorFn, error)
}

type DataRuntime struct {
	DB             *sql.DB
	DBPath         string
	Index          vectorindex.VectorIndex
	VectorBackend  string
	BackendWarning string
}

// Close closes the underlying database connection. The index shares the database lifetime.
func (r *DataRuntime) Close() error {
	if r == nil || r.DB == nil {
		return nil
	}
	return r.DB.Close()
}

func InitializeDataRuntime(ctx context.Context, cfg DataRuntimeConfig, deps DataRuntimeDeps) (*DataRuntime, error) {
	if deps.DiscoverSQLiteVectorPath == nil {
		return nil, fmt.Errorf("discover sqlite-vector dependency is nil")
	}
	if deps.OpenSQLiteDB == nil {
		return nil, fmt.Errorf("open sqlite dependency is nil")
	}
	if deps.ValidateSQLiteVector == nil {
		return nil, fmt.Errorf("validate sqlite-vector dependency is nil")
	}
	if deps.ResolveVectorBackend == nil {
		return nil, fmt.Errorf("resolve vector backend dependency is nil")
	}
	if deps.BuildVectorBackend == nil {
		return nil, fmt.Errorf("build vector backend dependency is nil")
	}

	dbPath := filepath.Join(cfg.DataDir, "imgsearch.sqlite")
	dsn := fmt.Sprintf("%s?_busy_timeout=30000", dbPath)

	resolvedSQLiteVectorPath, err := deps.DiscoverSQLiteVectorPath(cfg.SQLiteVectorPath)
	if err != nil {
		return nil, fmt.Errorf("discover sqlite-vector extension: %w", err)
	}

	autoValidationErr := error(nil)
	openWithVector := ""
	if cfg.RequestedVectorBackend == VectorBackendAuto {
		if resolvedSQLiteVectorPath == "" {
			autoValidationErr = fmt.Errorf("sqlite-vector extension path not found (run `mise run sqlite-vector-setup` or set SQLITE_VECTOR_PATH)")
		} else {
			openWithVector = resolvedSQLiteVectorPath
		}
	}
	if cfg.RequestedVectorBackend == VectorBackendSQLiteVector {
		if resolvedSQLiteVectorPath == "" {
			return nil, fmt.Errorf("sqlite-vector backend requested but extension path was not found (set -sqlite-vector-path or SQLITE_VECTOR_PATH)")
		}
		openWithVector = resolvedSQLiteVectorPath
	}

	sqlDB, err := deps.OpenSQLiteDB(dsn, openWithVector)
	if err != nil {
		if cfg.RequestedVectorBackend == VectorBackendAuto && openWithVector != "" {
			autoValidationErr = err
			sqlDB, err = deps.OpenSQLiteDB(dsn, "")
			if err != nil {
				return nil, fmt.Errorf("open sqlite database: %w", err)
			}
			openWithVector = ""
		} else {
			return nil, fmt.Errorf("open sqlite database: %w", err)
		}
	}

	if cfg.RequestedVectorBackend == VectorBackendAuto && openWithVector != "" && autoValidationErr == nil {
		autoValidationErr = deps.ValidateSQLiteVector(ctx, sqlDB)
	}

	resolvedVectorBackend, backendWarning, err := deps.ResolveVectorBackend(cfg.RequestedVectorBackend, autoValidationErr)
	if err != nil {
		_ = sqlDB.Close()
		return nil, fmt.Errorf("configure vector backend: %w", err)
	}
	if resolvedVectorBackend == VectorBackendBruteForce && openWithVector != "" {
		_ = sqlDB.Close()
		sqlDB, err = deps.OpenSQLiteDB(dsn, "")
		if err != nil {
			return nil, fmt.Errorf("open sqlite database: %w", err)
		}
	}

	sqlDB.SetMaxOpenConns(1)
	sqlDB.SetMaxIdleConns(1)

	index, validateVectorBackend, err := deps.BuildVectorBackend(resolvedVectorBackend, sqlDB)
	if err != nil {
		_ = sqlDB.Close()
		return nil, fmt.Errorf("build vector backend: %w", err)
	}

	if err := Bootstrap(ctx, sqlDB, validateVectorBackend); err != nil {
		_ = sqlDB.Close()
		return nil, fmt.Errorf("bootstrap app: %w", err)
	}

	return &DataRuntime{
		DB:             sqlDB,
		DBPath:         dbPath,
		Index:          index,
		VectorBackend:  resolvedVectorBackend,
		BackendWarning: backendWarning,
	}, nil
}
