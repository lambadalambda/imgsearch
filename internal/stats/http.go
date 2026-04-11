package stats

import (
	"database/sql"
	"encoding/json"
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

		var resp Response

		if err := h.DB.QueryRowContext(r.Context(), `SELECT COUNT(*) FROM images`).Scan(&resp.ImagesTotal); err != nil {
			writeJSONError(w, http.StatusInternalServerError, "query failed")
			return
		}

		if err := h.DB.QueryRowContext(r.Context(), `
SELECT
  COUNT(*) AS total,
  COALESCE(SUM(CASE WHEN state = 'pending' THEN 1 ELSE 0 END), 0) AS pending,
  COALESCE(SUM(CASE WHEN state = 'leased' THEN 1 ELSE 0 END), 0) AS leased,
  COALESCE(SUM(CASE WHEN state = 'done' THEN 1 ELSE 0 END), 0) AS done,
  COALESCE(SUM(CASE WHEN state = 'failed' THEN 1 ELSE 0 END), 0) AS failed
FROM index_jobs
WHERE kind = 'embed_image' AND model_id = ?
`, h.ModelID).Scan(
			&resp.Queue.Tracked,
			&resp.Queue.Pending,
			&resp.Queue.Leased,
			&resp.Queue.Done,
			&resp.Queue.Failed,
		); err != nil {
			writeJSONError(w, http.StatusInternalServerError, "query failed")
			return
		}

		resp.Queue.Missing = resp.ImagesTotal - resp.Queue.Tracked
		if resp.Queue.Missing < 0 {
			resp.Queue.Missing = 0
		}
		resp.Queue.Total = resp.Queue.Tracked + resp.Queue.Missing

		rows, err := h.DB.QueryContext(r.Context(), `
SELECT j.id, j.image_id, i.original_name, j.attempts, COALESCE(j.last_error, ''), j.updated_at
FROM index_jobs j
JOIN images i ON i.id = j.image_id
WHERE j.kind = 'embed_image'
  AND j.model_id = ?
  AND j.state = 'failed'
ORDER BY j.updated_at DESC, j.id DESC
LIMIT 10
`, h.ModelID)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "query failed")
			return
		}
		defer func() { _ = rows.Close() }()

		resp.RecentFailures = make([]FailureItem, 0, 10)
		for rows.Next() {
			var item FailureItem
			if err := rows.Scan(&item.JobID, &item.ImageID, &item.OriginalName, &item.Attempts, &item.LastError, &item.UpdatedAt); err != nil {
				writeJSONError(w, http.StatusInternalServerError, "result decode failed")
				return
			}
			resp.RecentFailures = append(resp.RecentFailures, item)
		}
		if err := rows.Err(); err != nil {
			writeJSONError(w, http.StatusInternalServerError, "result iteration failed")
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
