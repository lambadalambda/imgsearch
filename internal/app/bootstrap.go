package app

import (
    "context"
    "database/sql"
    "fmt"

    "imgsearch/internal/db"
)

type ValidateVectorFn func(context.Context, *sql.DB) error

func Bootstrap(ctx context.Context, sqlDB *sql.DB, validateVector ValidateVectorFn) error {
    if sqlDB == nil {
        return fmt.Errorf("db is nil")
    }
    if validateVector == nil {
        return fmt.Errorf("validateVector is nil")
    }

    if _, err := sqlDB.ExecContext(ctx, `PRAGMA journal_mode = WAL`); err != nil {
        return fmt.Errorf("enable wal mode: %w", err)
    }

    if err := db.RunMigrations(ctx, sqlDB); err != nil {
        return fmt.Errorf("run migrations: %w", err)
    }

    if err := validateVector(ctx, sqlDB); err != nil {
        return fmt.Errorf("validate vector backend: %w", err)
    }

    return nil
}
