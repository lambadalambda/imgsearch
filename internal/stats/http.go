package stats

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
)

type Handler struct {
	DB      *sql.DB
	ModelID int64
}

type QueueStats struct {
	Total                    int64 `json:"total"`
	Tracked                  int64 `json:"tracked"`
	Missing                  int64 `json:"missing"`
	AnnotationsMissing       int64 `json:"annotations_missing"`
	Runnable                 int64 `json:"runnable"`
	Pending                  int64 `json:"pending"`
	Leased                   int64 `json:"leased"`
	Done                     int64 `json:"done"`
	Failed                   int64 `json:"failed"`
	OldestRunnableAgeSeconds int64 `json:"oldest_runnable_age_seconds"`
}

type JobKindStats struct {
	Tracked                  int64 `json:"tracked"`
	Runnable                 int64 `json:"runnable"`
	Pending                  int64 `json:"pending"`
	Leased                   int64 `json:"leased"`
	Done                     int64 `json:"done"`
	Failed                   int64 `json:"failed"`
	OldestRunnableAgeSeconds int64 `json:"oldest_runnable_age_seconds"`
}

type FailureItem struct {
	JobID        int64  `json:"job_id"`
	Kind         string `json:"kind"`
	ImageID      int64  `json:"image_id"`
	OriginalName string `json:"original_name"`
	Attempts     int    `json:"attempts"`
	LastError    string `json:"last_error"`
	UpdatedAt    string `json:"updated_at"`
}

type Response struct {
	ImagesTotal    int64                   `json:"images_total"`
	Queue          QueueStats              `json:"queue"`
	JobKinds       map[string]JobKindStats `json:"job_kinds,omitempty"`
	RecentFailures []FailureItem           `json:"recent_failures"`
}

func Collect(ctx context.Context, db *sql.DB, modelID int64) (Response, error) {
	if db == nil {
		return Response{}, fmt.Errorf("stats database unavailable")
	}

	var resp Response

	if err := db.QueryRowContext(ctx, `SELECT COUNT(*) FROM images`).Scan(&resp.ImagesTotal); err != nil {
		return Response{}, fmt.Errorf("count images: %w", err)
	}

	jobKinds, err := collectJobKindStats(ctx, db, modelID)
	if err != nil {
		return Response{}, err
	}
	resp.JobKinds = jobKinds
	if embedStats, ok := jobKinds["embed_image"]; ok {
		resp.Queue.Tracked = embedStats.Tracked
		resp.Queue.Runnable = embedStats.Runnable
		resp.Queue.Pending = embedStats.Pending
		resp.Queue.Leased = embedStats.Leased
		resp.Queue.Done = embedStats.Done
		resp.Queue.Failed = embedStats.Failed
		resp.Queue.OldestRunnableAgeSeconds = embedStats.OldestRunnableAgeSeconds
	}

	resp.Queue.Missing = resp.ImagesTotal - resp.Queue.Tracked
	if resp.Queue.Missing < 0 {
		resp.Queue.Missing = 0
	}
	resp.Queue.AnnotationsMissing, err = countDoneJobsMissingAnnotations(ctx, db, modelID)
	if err != nil {
		return Response{}, err
	}
	resp.Queue.Total = resp.Queue.Tracked + resp.Queue.Missing

	rows, err := db.QueryContext(ctx, `
SELECT j.id, j.kind, j.image_id, i.original_name, j.attempts, COALESCE(j.last_error, ''), j.updated_at
FROM index_jobs j
JOIN images i ON i.id = j.image_id
WHERE j.model_id = ?
  AND j.state = 'failed'
ORDER BY j.updated_at DESC, j.id DESC
LIMIT 10
`, modelID)
	if err != nil {
		return Response{}, fmt.Errorf("query recent failures: %w", err)
	}
	defer func() { _ = rows.Close() }()

	resp.RecentFailures = make([]FailureItem, 0, 10)
	for rows.Next() {
		var item FailureItem
		if err := rows.Scan(&item.JobID, &item.Kind, &item.ImageID, &item.OriginalName, &item.Attempts, &item.LastError, &item.UpdatedAt); err != nil {
			return Response{}, fmt.Errorf("decode failure row: %w", err)
		}
		resp.RecentFailures = append(resp.RecentFailures, item)
	}
	if err := rows.Err(); err != nil {
		return Response{}, fmt.Errorf("iterate failure rows: %w", err)
	}

	return resp, nil
}

func countDoneJobsMissingAnnotations(ctx context.Context, db *sql.DB, modelID int64) (int64, error) {
	var total int64
	if err := db.QueryRowContext(ctx, `
SELECT COUNT(*)
FROM index_jobs j
JOIN images i ON i.id = j.image_id
WHERE j.kind = 'embed_image'
  AND j.model_id = ?
  AND j.state = 'done'
  AND (
    trim(COALESCE(i.description, '')) = ''
    OR COALESCE(i.tags_json, '') = ''
    OR COALESCE(i.tags_json, '[]') = '[]'
  )
`, modelID).Scan(&total); err != nil {
		return 0, fmt.Errorf("count done jobs missing annotations: %w", err)
	}
	return total, nil
}

func collectJobKindStats(ctx context.Context, db *sql.DB, modelID int64) (map[string]JobKindStats, error) {
	rows, err := db.QueryContext(ctx, `
SELECT
  kind,
  COUNT(*) AS tracked,
  COALESCE(SUM(CASE
    WHEN (run_after IS NULL OR run_after <= datetime('now'))
      AND (
        state = 'pending'
        OR (state = 'leased' AND leased_until IS NOT NULL AND leased_until <= datetime('now'))
      )
    THEN 1 ELSE 0 END), 0) AS runnable,
  COALESCE(SUM(CASE WHEN state = 'pending' THEN 1 ELSE 0 END), 0) AS pending,
  COALESCE(SUM(CASE WHEN state = 'leased' THEN 1 ELSE 0 END), 0) AS leased,
  COALESCE(SUM(CASE WHEN state = 'done' THEN 1 ELSE 0 END), 0) AS done,
  COALESCE(SUM(CASE WHEN state = 'failed' THEN 1 ELSE 0 END), 0) AS failed,
  COALESCE(MAX(CASE
    WHEN state = 'pending'
      AND (run_after IS NULL OR run_after <= datetime('now'))
    THEN CAST((julianday('now') - julianday(COALESCE(run_after, created_at))) * 86400 AS INTEGER)
    WHEN state = 'leased'
      AND leased_until IS NOT NULL
      AND leased_until <= datetime('now')
      AND (run_after IS NULL OR run_after <= datetime('now'))
    THEN CAST((julianday('now') - julianday(
      CASE
        WHEN run_after IS NOT NULL AND run_after > leased_until THEN run_after
        ELSE leased_until
      END
    )) * 86400 AS INTEGER)
    ELSE 0 END), 0) AS oldest_runnable_age_seconds
FROM index_jobs
WHERE model_id = ?
GROUP BY kind
ORDER BY kind
`, modelID)
	if err != nil {
		return nil, fmt.Errorf("query queue stats: %w", err)
	}
	defer func() { _ = rows.Close() }()

	statsByKind := make(map[string]JobKindStats)
	for rows.Next() {
		var kind string
		var stats JobKindStats
		if err := rows.Scan(
			&kind,
			&stats.Tracked,
			&stats.Runnable,
			&stats.Pending,
			&stats.Leased,
			&stats.Done,
			&stats.Failed,
			&stats.OldestRunnableAgeSeconds,
		); err != nil {
			return nil, fmt.Errorf("decode queue stats row: %w", err)
		}
		statsByKind[kind] = stats
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate queue stats rows: %w", err)
	}

	return statsByKind, nil
}

func NewHandler(h *Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if h == nil || h.DB == nil {
			writeJSONError(w, http.StatusInternalServerError, "stats backend unavailable")
			return
		}

		if r.Method != http.MethodGet {
			writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		resp, err := Collect(r.Context(), h.DB, h.ModelID)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "query failed")
			return
		}

		writeJSON(w, http.StatusOK, resp)
	})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeJSONError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}
