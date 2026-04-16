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
	{
		version: 3,
		sql: `
CREATE TABLE IF NOT EXISTS videos (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  sha256 TEXT NOT NULL UNIQUE,
  original_name TEXT NOT NULL,
  storage_path TEXT NOT NULL,
  mime_type TEXT NOT NULL,
  duration_ms INTEGER NOT NULL,
  width INTEGER NOT NULL,
  height INTEGER NOT NULL,
  frame_count INTEGER NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS video_frames (
  video_id INTEGER NOT NULL,
  image_id INTEGER NOT NULL,
  frame_index INTEGER NOT NULL,
  timestamp_ms INTEGER NOT NULL,
  PRIMARY KEY (video_id, frame_index),
  FOREIGN KEY (video_id) REFERENCES videos(id),
  FOREIGN KEY (image_id) REFERENCES images(id)
);

CREATE INDEX IF NOT EXISTS idx_video_frames_image_id
ON video_frames(image_id);
`,
	},
	{
		version: 4,
		sql: `
ALTER TABLE videos ADD COLUMN transcript_text TEXT NOT NULL DEFAULT '';
ALTER TABLE videos ADD COLUMN transcript_updated_at TEXT;

CREATE TABLE IF NOT EXISTS video_transcript_embeddings (
  video_id INTEGER NOT NULL,
  model_id INTEGER NOT NULL,
  dim INTEGER NOT NULL,
  vector_blob BLOB NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (video_id, model_id),
  FOREIGN KEY (video_id) REFERENCES videos(id),
  FOREIGN KEY (model_id) REFERENCES embedding_models(id)
);
`,
	},
	{
		version: 5,
		sql: `
CREATE TABLE IF NOT EXISTS index_jobs_new (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,
  image_id INTEGER,
  video_id INTEGER,
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
  CHECK ((image_id IS NOT NULL) <> (video_id IS NOT NULL)),
  FOREIGN KEY (image_id) REFERENCES images(id),
  FOREIGN KEY (video_id) REFERENCES videos(id),
  FOREIGN KEY (model_id) REFERENCES embedding_models(id)
);

INSERT INTO index_jobs_new(id, kind, image_id, video_id, model_id, state, run_after, leased_until, lease_owner, attempts, max_attempts, last_error, created_at, updated_at)
SELECT id, kind, image_id, NULL, model_id, state, run_after, leased_until, lease_owner, attempts, max_attempts, last_error, created_at, updated_at
FROM index_jobs;

DROP TABLE index_jobs;
ALTER TABLE index_jobs_new RENAME TO index_jobs;

CREATE INDEX IF NOT EXISTS idx_index_jobs_state_run_after
ON index_jobs(state, run_after);

CREATE UNIQUE INDEX IF NOT EXISTS idx_index_jobs_unique_image_target
ON index_jobs(kind, image_id, model_id)
WHERE image_id IS NOT NULL AND video_id IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_index_jobs_unique_video_target
ON index_jobs(kind, video_id, model_id)
WHERE video_id IS NOT NULL AND image_id IS NULL;
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
