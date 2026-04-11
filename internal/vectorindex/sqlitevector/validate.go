package sqlitevector

import (
	"context"
	"database/sql"
	"fmt"
)

func ValidateAvailable(ctx context.Context, db *sql.DB) error {
	if db == nil {
		return fmt.Errorf("db is nil")
	}

	var version string
	if err := db.QueryRowContext(ctx, `SELECT vector_version()`).Scan(&version); err != nil {
		return fmt.Errorf("sqlite-vector extension is not available: %w", err)
	}
	if version == "" {
		return fmt.Errorf("sqlite-vector extension returned empty version")
	}
	return nil
}
