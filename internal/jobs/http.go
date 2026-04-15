package jobs

import (
	"database/sql"
	"net/http"

	"imgsearch/internal/db"
	"imgsearch/internal/httputil"
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
			httputil.WriteJSONError(w, http.StatusInternalServerError, "jobs backend unavailable")
			return
		}
		if r.Method != http.MethodPost {
			httputil.WriteJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
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
WHERE model_id = ?
  AND state = 'failed'
		`, h.ModelID)
		if err != nil {
			httputil.WriteJSONError(w, http.StatusInternalServerError, "retry failed")
			return
		}

		rows, err := res.RowsAffected()
		if err != nil {
			httputil.WriteJSONError(w, http.StatusInternalServerError, "retry failed")
			return
		}

		requeuedAnnotations, err := db.RequeueDoneJobsMissingAnnotations(r.Context(), h.DB, h.ModelID)
		if err != nil {
			httputil.WriteJSONError(w, http.StatusInternalServerError, "retry failed")
			return
		}

		enqueued, err := db.EnsureIndexJobsForModel(r.Context(), h.DB, h.ModelID)
		if err != nil {
			httputil.WriteJSONError(w, http.StatusInternalServerError, "retry failed")
			return
		}
		annotationJobs, err := db.EnsureAnnotationJobsForModel(r.Context(), h.DB, h.ModelID)
		if err != nil {
			httputil.WriteJSONError(w, http.StatusInternalServerError, "retry failed")
			return
		}

		httputil.WriteJSON(w, http.StatusOK, RetryFailedResponse{Retried: rows, Enqueued: requeuedAnnotations + enqueued + annotationJobs})
	})
}
