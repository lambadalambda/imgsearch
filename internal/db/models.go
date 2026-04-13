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

	var (
		id             int64
		existingDims   int
		existingMetric string
		existingNorm   int
	)
	if err := db.QueryRowContext(ctx, `
SELECT id, dimensions, metric, normalized
FROM embedding_models
WHERE name = ? AND version = ?
`, spec.Name, spec.Version).Scan(&id, &existingDims, &existingMetric, &existingNorm); err != nil {
		return 0, fmt.Errorf("load embedding model id: %w", err)
	}

	if existingDims != spec.Dimensions || existingMetric != spec.Metric || existingNorm != normalized {
		return 0, fmt.Errorf(
			"existing embedding model %q version %q does not match requested spec (have dimensions=%d metric=%s normalized=%d, want dimensions=%d metric=%s normalized=%d)",
			spec.Name,
			spec.Version,
			existingDims,
			existingMetric,
			existingNorm,
			spec.Dimensions,
			spec.Metric,
			normalized,
		)
	}

	return id, nil
}
