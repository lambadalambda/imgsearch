package jobs

import (
	"database/sql"
	"encoding/json"
	"net/http"

	"imgsearch/internal/db"
)

type RetryFailedHandler struct {
	DB      *sql.DB
	ModelID int64
}

type RetryFailedResponse struct {
	Retried  int64 `json:"retried"`
	Enqueued int64 `json:"enqueued_missing"`
}

func NewRetryFailedHandler(h *RetryFailedHandler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if h == nil || h.DB == nil {
			writeJSONError(w, http.StatusInternalServerError, "jobs backend unavailable")
			return
		}
		if r.Method != http.MethodPost {
			writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		res, err := h.DB.ExecContext(r.Context(), `
UPDATE index_jobs
SET state = 'pending',
    attempts = 0,
    run_after = NULL,
    leased_until = NULL,
    lease_owner = NULL,
    last_error = NULL,
    updated_at = datetime('now')
WHERE kind = 'embed_image'
  AND model_id = ?
  AND state = 'failed'
`, h.ModelID)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "retry failed")
			return
		}

		rows, err := res.RowsAffected()
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "retry failed")
			return
		}

		requeuedAnnotations, err := db.RequeueDoneJobsMissingAnnotations(r.Context(), h.DB, h.ModelID)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "retry failed")
			return
		}

		enqueued, err := db.EnsureIndexJobsForModel(r.Context(), h.DB, h.ModelID)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "retry failed")
			return
		}

		writeJSON(w, http.StatusOK, RetryFailedResponse{Retried: rows, Enqueued: requeuedAnnotations + enqueued})
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
