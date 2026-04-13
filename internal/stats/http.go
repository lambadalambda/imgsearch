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
	Total   int64 `json:"total"`
	Tracked int64 `json:"tracked"`
	Missing int64 `json:"missing"`
	Pending int64 `json:"pending"`
	Leased  int64 `json:"leased"`
	Done    int64 `json:"done"`
	Failed  int64 `json:"failed"`
}

type FailureItem struct {
	JobID        int64  `json:"job_id"`
	ImageID      int64  `json:"image_id"`
	OriginalName string `json:"original_name"`
	Attempts     int    `json:"attempts"`
	LastError    string `json:"last_error"`
	UpdatedAt    string `json:"updated_at"`
}

type Response struct {
	ImagesTotal    int64         `json:"images_total"`
	Queue          QueueStats    `json:"queue"`
	RecentFailures []FailureItem `json:"recent_failures"`
}

func Collect(ctx context.Context, db *sql.DB, modelID int64) (Response, error) {
	if db == nil {
		return Response{}, fmt.Errorf("stats database unavailable")
	}

	var resp Response

	if err := db.QueryRowContext(ctx, `SELECT COUNT(*) FROM images`).Scan(&resp.ImagesTotal); err != nil {
		return Response{}, fmt.Errorf("count images: %w", err)
	}

	if err := db.QueryRowContext(ctx, `
SELECT
  COUNT(*) AS total,
  COALESCE(SUM(CASE WHEN state = 'pending' THEN 1 ELSE 0 END), 0) AS pending,
  COALESCE(SUM(CASE WHEN state = 'leased' THEN 1 ELSE 0 END), 0) AS leased,
  COALESCE(SUM(CASE WHEN state = 'done' THEN 1 ELSE 0 END), 0) AS done,
  COALESCE(SUM(CASE WHEN state = 'failed' THEN 1 ELSE 0 END), 0) AS failed
FROM index_jobs
WHERE kind = 'embed_image' AND model_id = ?
`, modelID).Scan(
		&resp.Queue.Tracked,
		&resp.Queue.Pending,
		&resp.Queue.Leased,
		&resp.Queue.Done,
		&resp.Queue.Failed,
	); err != nil {
		return Response{}, fmt.Errorf("query queue stats: %w", err)
	}

	resp.Queue.Missing = resp.ImagesTotal - resp.Queue.Tracked
	if resp.Queue.Missing < 0 {
		resp.Queue.Missing = 0
	}
	resp.Queue.Total = resp.Queue.Tracked + resp.Queue.Missing

	rows, err := db.QueryContext(ctx, `
SELECT j.id, j.image_id, i.original_name, j.attempts, COALESCE(j.last_error, ''), j.updated_at
FROM index_jobs j
JOIN images i ON i.id = j.image_id
WHERE j.kind = 'embed_image'
  AND j.model_id = ?
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
		if err := rows.Scan(&item.JobID, &item.ImageID, &item.OriginalName, &item.Attempts, &item.LastError, &item.UpdatedAt); err != nil {
			return Response{}, fmt.Errorf("decode failure row: %w", err)
		}
		resp.RecentFailures = append(resp.RecentFailures, item)
	}
	if err := rows.Err(); err != nil {
		return Response{}, fmt.Errorf("iterate failure rows: %w", err)
	}

	return resp, nil
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
