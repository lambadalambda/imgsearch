package db

import (
	"context"
	"database/sql"
	"fmt"
)

type migration struct {
	version int
	sql     string
}

var migrations = []migration{
	{
		version: 1,
		sql: `
CREATE TABLE IF NOT EXISTS schema_migrations (
  version INTEGER PRIMARY KEY,
  applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS images (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  sha256 TEXT NOT NULL UNIQUE,
  original_name TEXT NOT NULL,
  storage_path TEXT NOT NULL,
  thumbnail_path TEXT,
  mime_type TEXT NOT NULL,
  width INTEGER NOT NULL,
  height INTEGER NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS embedding_models (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  version TEXT NOT NULL,
  dimensions INTEGER NOT NULL,
  metric TEXT NOT NULL,
  normalized INTEGER NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(name, version)
);

CREATE TABLE IF NOT EXISTS image_embeddings (
  image_id INTEGER NOT NULL,
  model_id INTEGER NOT NULL,
  dim INTEGER NOT NULL,
  vector_blob BLOB NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (image_id, model_id),
  FOREIGN KEY (image_id) REFERENCES images(id),
  FOREIGN KEY (model_id) REFERENCES embedding_models(id)
);

CREATE TABLE IF NOT EXISTS index_jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,
  image_id INTEGER NOT NULL,
  model_id INTEGER NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('pending', 'leased', 'done', 'failed')),
  run_after TEXT,
  leased_until TEXT,
  lease_owner TEXT,
  attempts INTEGER NOT NULL DEFAULT 0,
  max_attempts INTEGER NOT NULL DEFAULT 3,
  last_error TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(kind, image_id, model_id),
  FOREIGN KEY (image_id) REFERENCES images(id),
  FOREIGN KEY (model_id) REFERENCES embedding_models(id)
);

CREATE INDEX IF NOT EXISTS idx_index_jobs_state_run_after
ON index_jobs(state, run_after);
`,
	},
	{
		version: 2,
		sql: `
ALTER TABLE images ADD COLUMN description TEXT NOT NULL DEFAULT '';
ALTER TABLE images ADD COLUMN tags_json TEXT NOT NULL DEFAULT '[]';
`,
	},
}

func LatestVersion() int {
	return len(migrations)
}

func RunMigrations(ctx context.Context, db *sql.DB) error {
	if db == nil {
		return fmt.Errorf("db is nil")
	}

	if _, err := db.ExecContext(ctx, `
CREATE TABLE IF NOT EXISTS schema_migrations (
  version INTEGER PRIMARY KEY,
  applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);
`); err != nil {
		return fmt.Errorf("create schema_migrations: %w", err)
	}

	current, err := CurrentVersion(ctx, db)
	if err != nil {
		return err
	}

	for _, m := range migrations {
		if m.version <= current {
			continue
		}

		tx, err := db.BeginTx(ctx, nil)
		if err != nil {
			return fmt.Errorf("begin migration %d: %w", m.version, err)
		}

		if _, err := tx.ExecContext(ctx, m.sql); err != nil {
			_ = tx.Rollback()
			return fmt.Errorf("apply migration %d: %w", m.version, err)
		}

		if _, err := tx.ExecContext(ctx,
			`INSERT INTO schema_migrations(version) VALUES(?)`,
			m.version,
		); err != nil {
			_ = tx.Rollback()
			return fmt.Errorf("record migration %d: %w", m.version, err)
		}

		if err := tx.Commit(); err != nil {
			return fmt.Errorf("commit migration %d: %w", m.version, err)
		}
	}

	return nil
}

func CurrentVersion(ctx context.Context, db *sql.DB) (int, error) {
	if db == nil {
		return 0, fmt.Errorf("db is nil")
	}

	var version sql.NullInt64
	if err := db.QueryRowContext(ctx, `SELECT MAX(version) FROM schema_migrations`).Scan(&version); err != nil {
		return 0, fmt.Errorf("query current migration version: %w", err)
	}
	if !version.Valid {
		return 0, nil
	}

	return int(version.Int64), nil
}
