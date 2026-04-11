package db

import (
	"context"
	"database/sql"
	"fmt"
)

type EmbeddingModelSpec struct {
	Name       string
	Version    string
	Dimensions int
	Metric     string
	Normalized bool
}

func EnsureEmbeddingModel(ctx context.Context, db *sql.DB, spec EmbeddingModelSpec) (int64, error) {
	if db == nil {
		return 0, fmt.Errorf("db is nil")
	}
	if spec.Name == "" || spec.Version == "" || spec.Dimensions <= 0 || spec.Metric == "" {
		return 0, fmt.Errorf("invalid embedding model spec")
	}

	normalized := 0
	if spec.Normalized {
		normalized = 1
	}

	if _, err := db.ExecContext(ctx, `
INSERT INTO embedding_models(name, version, dimensions, metric, normalized)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT(name, version) DO NOTHING
`, spec.Name, spec.Version, spec.Dimensions, spec.Metric, normalized); err != nil {
		return 0, fmt.Errorf("upsert embedding model: %w", err)
	}

	var id int64
	if err := db.QueryRowContext(ctx, `
SELECT id FROM embedding_models WHERE name = ? AND version = ?
`, spec.Name, spec.Version).Scan(&id); err != nil {
		return 0, fmt.Errorf("load embedding model id: %w", err)
	}

	return id, nil
}
