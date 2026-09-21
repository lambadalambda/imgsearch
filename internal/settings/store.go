package settings

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
)

const annotationKey = "annotation"

// ErrNotFound is returned by Load* when no value has been saved yet.
var ErrNotFound = errors.New("settings: not found")

type querier interface {
	QueryRowContext(ctx context.Context, query string, args ...any) *sql.Row
}

// LoadAnnotation reads the persisted annotation settings.
func LoadAnnotation(ctx context.Context, q querier) (AnnotationSettings, error) {
	var raw string
	err := q.QueryRowContext(ctx, `SELECT value_json FROM settings WHERE key = ?`, annotationKey).Scan(&raw)
	if errors.Is(err, sql.ErrNoRows) {
		return AnnotationSettings{}, ErrNotFound
	}
	if err != nil {
		return AnnotationSettings{}, fmt.Errorf("load annotation settings: %w", err)
	}
	var s AnnotationSettings
	if err := json.Unmarshal([]byte(raw), &s); err != nil {
		return AnnotationSettings{}, fmt.Errorf("decode annotation settings: %w", err)
	}
	return s.Normalized(), nil
}

// LoadAnnotationOrDefault returns the persisted settings or fallback when
// nothing has been saved.
func LoadAnnotationOrDefault(ctx context.Context, q querier, fallback AnnotationSettings) (AnnotationSettings, error) {
	s, err := LoadAnnotation(ctx, q)
	if errors.Is(err, ErrNotFound) {
		return fallback.Normalized(), nil
	}
	return s, err
}

// SaveAnnotation validates, persists, and bumps the settings version. It
// returns the new version.
func SaveAnnotation(ctx context.Context, db *sql.DB, s AnnotationSettings) (int64, error) {
	s = s.Normalized()
	if err := s.Validate(); err != nil {
		return 0, err
	}
	raw, err := json.Marshal(s)
	if err != nil {
		return 0, fmt.Errorf("encode annotation settings: %w", err)
	}
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return 0, fmt.Errorf("begin save settings: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	if _, err := tx.ExecContext(ctx, `
INSERT INTO settings(key, value_json, updated_at) VALUES(?, ?, datetime('now'))
ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json, updated_at = excluded.updated_at`,
		annotationKey, string(raw)); err != nil {
		return 0, fmt.Errorf("save annotation settings: %w", err)
	}
	var version int64
	if err := tx.QueryRowContext(ctx, `
INSERT INTO settings_version(id, version) VALUES(1, 1)
ON CONFLICT(id) DO UPDATE SET version = settings_version.version + 1
RETURNING version`).Scan(&version); err != nil {
		return 0, fmt.Errorf("bump settings version: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return 0, fmt.Errorf("commit save settings: %w", err)
	}
	return version, nil
}

// Version returns the current settings version, bumped on every save. Other
// processes poll it to detect changes cheaply.
func Version(ctx context.Context, q querier) (int64, error) {
	var version int64
	err := q.QueryRowContext(ctx, `SELECT version FROM settings_version WHERE id = 1`).Scan(&version)
	if errors.Is(err, sql.ErrNoRows) {
		return 0, nil
	}
	if err != nil {
		return 0, fmt.Errorf("read settings version: %w", err)
	}
	return version, nil
}
