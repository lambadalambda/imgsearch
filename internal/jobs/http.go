package jobs

import (
	"database/sql"
	"net/http"
	"net/url"

	"imgsearch/internal/db"
	"imgsearch/internal/httputil"
	"imgsearch/internal/jobkind"
)

type RetryFailedHandler struct {
	DB      *sql.DB
	ModelID int64
	// BlockedKinds applies only to the bounded, kind-filtered recovery contract.
	BlockedKinds map[string]bool
}

type RetryFailedResponse struct {
	Retried  int64 `json:"retried"`
	Enqueued int64 `json:"enqueued_missing"`
}

func NewRetryFailedHandler(h *RetryFailedHandler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			httputil.WriteMethodNotAllowed(w, http.MethodPost)
			return
		}
		if h == nil || h.DB == nil {
			httputil.WriteJSONError(w, http.StatusServiceUnavailable, "jobs backend unavailable")
			return
		}

		query, err := url.ParseQuery(r.URL.RawQuery)
		if err != nil {
			httputil.WriteJSONError(w, http.StatusBadRequest, "invalid query")
			return
		}
		// A present kind selects bounded recovery: existing failed rows only.
		// Never fall through to legacy backfill, even for an empty/invalid kind.
		if query.Has("kind") {
			kind := query.Get("kind")
			switch kind {
			case jobkind.EmbedImage, jobkind.AnnotateImage, jobkind.AnnotateVideo, jobkind.TranscribeVideo:
			default:
				httputil.WriteJSONError(w, http.StatusBadRequest, "unknown job kind")
				return
			}
			if h.BlockedKinds[kind] {
				httputil.WriteJSONError(w, http.StatusConflict, "processing capability is disabled or not configured")
				return
			}
			res, err := h.DB.ExecContext(r.Context(), `UPDATE index_jobs SET state='pending', attempts=0, run_after=NULL, leased_until=NULL, lease_owner=NULL, last_error=NULL, updated_at=datetime('now') WHERE model_id=? AND kind=? AND state='failed'`, h.ModelID, kind)
			if err != nil {
				httputil.WriteJSONError(w, http.StatusInternalServerError, "retry failed")
				return
			}
			rows, err := res.RowsAffected()
			if err != nil {
				httputil.WriteJSONError(w, http.StatusInternalServerError, "retry failed")
				return
			}
			httputil.WriteJSON(w, http.StatusOK, RetryFailedResponse{Retried: rows})
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
		videoAnnotationJobs, err := db.EnsureVideoAnnotationJobsForModel(r.Context(), h.DB, h.ModelID)
		if err != nil {
			httputil.WriteJSONError(w, http.StatusInternalServerError, "retry failed")
			return
		}
		videoTranscriptJobs, err := db.EnsureVideoTranscriptJobsForModel(r.Context(), h.DB, h.ModelID)
		if err != nil {
			httputil.WriteJSONError(w, http.StatusInternalServerError, "retry failed")
			return
		}

		httputil.WriteJSON(w, http.StatusOK, RetryFailedResponse{Retried: rows, Enqueued: requeuedAnnotations + enqueued + annotationJobs + videoAnnotationJobs + videoTranscriptJobs})
	})
}
