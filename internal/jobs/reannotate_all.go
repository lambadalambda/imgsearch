package jobs

import (
	"context"
	"database/sql"
	"fmt"
	"net/http"
	"strings"

	"imgsearch/internal/httputil"
	"imgsearch/internal/jobkind"
)

// ReannotateAllHandler queues a fresh annotation for every standalone image
// and every video, for example after switching annotation backends.
type ReannotateAllHandler struct {
	DB      *sql.DB
	ModelID int64
}

type ReannotateAllResponse struct {
	QueuedImages  int64 `json:"queued_images"`
	QueuedVideos  int64 `json:"queued_videos"`
	SkippedLeased int64 `json:"skipped_leased"`
}

const (
	reannotateMediaAll    = "all"
	reannotateMediaImages = "images"
	reannotateMediaVideos = "videos"
)

func NewReannotateAllHandler(h *ReannotateAllHandler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			httputil.WriteMethodNotAllowed(w, http.MethodPost)
			return
		}
		if h == nil || h.DB == nil {
			httputil.WriteJSONError(w, http.StatusServiceUnavailable, "jobs backend unavailable")
			return
		}
		media := strings.ToLower(strings.TrimSpace(r.URL.Query().Get("media")))
		if media == "" {
			media = reannotateMediaAll
		}
		if media != reannotateMediaAll && media != reannotateMediaImages && media != reannotateMediaVideos {
			httputil.WriteJSONError(w, http.StatusBadRequest, "media must be all, images, or videos")
			return
		}
		resp, err := ReannotateAll(r.Context(), h.DB, h.ModelID, media)
		if err != nil {
			httputil.WriteJSONError(w, http.StatusInternalServerError, "re-annotate all failed")
			return
		}
		httputil.WriteJSON(w, http.StatusOK, resp)
	})
}

// ReannotateAll flags media for re-annotation and resets their annotation
// jobs to pending in one transaction. Unlike the per-item action it keeps
// the existing text so cards stay populated until the worker replaces them;
// the worker honours reannotate_requested regardless of existing text.
// Leased jobs are left alone and counted in SkippedLeased.
func ReannotateAll(ctx context.Context, db *sql.DB, modelID int64, media string) (ReannotateAllResponse, error) {
	if db == nil {
		return ReannotateAllResponse{}, fmt.Errorf("jobs database unavailable")
	}
	if modelID <= 0 {
		return ReannotateAllResponse{}, fmt.Errorf("invalid model id")
	}
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return ReannotateAllResponse{}, fmt.Errorf("begin reannotate all tx: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	var resp ReannotateAllResponse
	if media == reannotateMediaAll || media == reannotateMediaImages {
		resp.QueuedImages, err = reannotateAllImages(ctx, tx, modelID)
		if err != nil {
			return ReannotateAllResponse{}, err
		}
	}
	if media == reannotateMediaAll || media == reannotateMediaVideos {
		resp.QueuedVideos, err = reannotateAllVideos(ctx, tx, modelID)
		if err != nil {
			return ReannotateAllResponse{}, err
		}
	}
	if err := tx.QueryRowContext(ctx, `
SELECT COUNT(*) FROM index_jobs
WHERE model_id = ? AND state = 'leased' AND kind IN (?, ?)`, modelID, jobkind.AnnotateImage, jobkind.AnnotateVideo).Scan(&resp.SkippedLeased); err != nil {
		return ReannotateAllResponse{}, fmt.Errorf("count leased annotation jobs: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return ReannotateAllResponse{}, fmt.Errorf("commit reannotate all tx: %w", err)
	}
	return resp, nil
}

// Standalone images are those not backing a sampled video frame.
const standaloneImagesWhere = `NOT EXISTS (SELECT 1 FROM video_frames vf WHERE vf.image_id = images.id)`

func reannotateAllImages(ctx context.Context, tx *sql.Tx, modelID int64) (int64, error) {
	if _, err := tx.ExecContext(ctx, `UPDATE images SET reannotate_requested = 1 WHERE `+standaloneImagesWhere); err != nil {
		return 0, fmt.Errorf("flag images for reannotation: %w", err)
	}
	if _, err := tx.ExecContext(ctx, `
INSERT INTO index_jobs(kind, image_id, model_id, state)
SELECT ?, images.id, ?, 'pending' FROM images WHERE `+standaloneImagesWhere+`
ON CONFLICT DO NOTHING`, jobkind.AnnotateImage, modelID); err != nil {
		return 0, fmt.Errorf("insert image annotation jobs: %w", err)
	}
	res, err := tx.ExecContext(ctx, `
UPDATE index_jobs
SET state = 'pending', attempts = 0, run_after = NULL, leased_until = NULL, lease_owner = NULL, last_error = NULL, updated_at = datetime('now')
WHERE kind = ? AND model_id = ? AND state <> 'leased'
  AND image_id IN (SELECT images.id FROM images WHERE `+standaloneImagesWhere+`)`, jobkind.AnnotateImage, modelID)
	if err != nil {
		return 0, fmt.Errorf("reset image annotation jobs: %w", err)
	}
	return res.RowsAffected()
}

func reannotateAllVideos(ctx context.Context, tx *sql.Tx, modelID int64) (int64, error) {
	if _, err := tx.ExecContext(ctx, `UPDATE videos SET reannotate_requested = 1`); err != nil {
		return 0, fmt.Errorf("flag videos for reannotation: %w", err)
	}
	if _, err := tx.ExecContext(ctx, `
INSERT INTO index_jobs(kind, image_id, video_id, model_id, state)
SELECT ?, NULL, videos.id, ?, 'pending' FROM videos WHERE true
ON CONFLICT DO NOTHING`, jobkind.AnnotateVideo, modelID); err != nil {
		return 0, fmt.Errorf("insert video annotation jobs: %w", err)
	}
	res, err := tx.ExecContext(ctx, `
UPDATE index_jobs
SET state = 'pending', attempts = 0, run_after = NULL, leased_until = NULL, lease_owner = NULL, last_error = NULL, updated_at = datetime('now')
WHERE kind = ? AND model_id = ? AND state <> 'leased'`, jobkind.AnnotateVideo, modelID)
	if err != nil {
		return 0, fmt.Errorf("reset video annotation jobs: %w", err)
	}
	return res.RowsAffected()
}
