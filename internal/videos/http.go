package videos

import (
	"context"
	"database/sql"
	"fmt"
	"net/http"
	"strings"

	"imgsearch/internal/httputil"
	"imgsearch/internal/jobkind"
	"imgsearch/internal/mediaops"
	"imgsearch/internal/nsfwsql"
	"imgsearch/internal/tagutil"
)

type Handler struct {
	DB      *sql.DB
	ModelID int64
	DataDir string
}

type VideoItem struct {
	VideoID        int64    `json:"video_id"`
	ImageID        int64    `json:"image_id,omitempty"`
	MediaType      string   `json:"media_type"`
	OriginalName   string   `json:"original_name"`
	StoragePath    string   `json:"storage_path"`
	PreviewPath    string   `json:"preview_path,omitempty"`
	PreviewWidth   int      `json:"preview_width,omitempty"`
	PreviewHeight  int      `json:"preview_height,omitempty"`
	TranscriptText string   `json:"transcript_text,omitempty"`
	Description    string   `json:"description,omitempty"`
	Tags           []string `json:"tags,omitempty"`
	MimeType       string   `json:"mime_type"`
	DurationMS     int64    `json:"duration_ms"`
	Width          int      `json:"width"`
	Height         int      `json:"height"`
	FrameCount     int      `json:"frame_count"`
	IndexState     string   `json:"index_state"`
	CreatedAt      string   `json:"created_at"`
}

type ListResponse struct {
	Videos []VideoItem `json:"videos"`
	Total  int64       `json:"total"`
}

const (
	listOrderNewest = "newest"
	listOrderRandom = "random"
	randomOrderMask = int64(2147483647)
)

func List(ctx context.Context, db *sql.DB, modelID int64, limit int, offset int, includeNSFW bool) (ListResponse, error) {
	return listWithOrder(ctx, db, modelID, limit, offset, includeNSFW, listOrderNewest, 0)
}

func listWithOrder(ctx context.Context, db *sql.DB, modelID int64, limit int, offset int, includeNSFW bool, order string, seed int64) (ListResponse, error) {
	if db == nil {
		return ListResponse{}, fmt.Errorf("videos database unavailable")
	}
	if limit <= 0 {
		limit = 50
	}
	if offset < 0 {
		offset = 0
	}
	includeNSFWInt := boolToInt(includeNSFW)
	videoHasNSFWCountExpr := nsfwsql.VideoHasNSFW("v.id", "v.tags_json", "tag", "vtag")
	videoHasNSFWListExpr := nsfwsql.VideoHasNSFW("v.id", "v.tags_json", "tag_nsfw", "vtag_nsfw")
	orderClause := "v.id DESC"
	args := []any{modelID, jobkind.EmbedImage, modelID, jobkind.TranscribeVideo, modelID, jobkind.AnnotateVideo, includeNSFWInt}
	if order == listOrderRandom {
		seed = seed & randomOrderMask
		orderClause = "((((v.id * 1103515245 + ?) & 2147483647) | (((v.id * 1103515245 + ?) & 2147483647) >> 16)) * 1103515245 + 12345) & 2147483647 ASC, v.id ASC"
		args = append(args, seed, seed)
	}
	args = append(args, limit, offset)

	var total int64
	if err := db.QueryRowContext(ctx, fmt.Sprintf(`
SELECT COUNT(*)
FROM videos v
WHERE (? = 1 OR NOT (%s))
`, videoHasNSFWCountExpr), includeNSFWInt).Scan(&total); err != nil {
		return ListResponse{}, fmt.Errorf("count videos: %w", err)
	}

	rows, err := db.QueryContext(ctx, fmt.Sprintf(`
WITH frame_jobs AS (
  SELECT vf.video_id,
         SUM(CASE WHEN j.state = 'failed' THEN 1 ELSE 0 END) AS failed_jobs,
         SUM(CASE WHEN j.state = 'leased' THEN 1 ELSE 0 END) AS leased_jobs,
         SUM(CASE WHEN j.state = 'done' THEN 1 ELSE 0 END) AS done_jobs,
         COUNT(j.id) AS total_jobs
  FROM video_frames vf
  LEFT JOIN index_jobs j
    ON j.image_id = vf.image_id
   AND j.model_id = ?
   AND j.kind = ?
  GROUP BY vf.video_id
), transcript_jobs AS (
  SELECT j.video_id,
         SUM(CASE WHEN j.state = 'failed' THEN 1 ELSE 0 END) AS failed_jobs,
         SUM(CASE WHEN j.state = 'leased' THEN 1 ELSE 0 END) AS leased_jobs,
         SUM(CASE WHEN j.state = 'done' THEN 1 ELSE 0 END) AS done_jobs,
         COUNT(j.id) AS total_jobs
  FROM index_jobs j
  WHERE j.video_id IS NOT NULL
    AND j.model_id = ?
    AND j.kind = ?
  GROUP BY j.video_id
), annotation_jobs AS (
  SELECT j.video_id,
         SUM(CASE WHEN j.state = 'failed' THEN 1 ELSE 0 END) AS failed_jobs,
         SUM(CASE WHEN j.state = 'leased' THEN 1 ELSE 0 END) AS leased_jobs,
         SUM(CASE WHEN j.state = 'done' THEN 1 ELSE 0 END) AS done_jobs,
         COUNT(j.id) AS total_jobs
  FROM index_jobs j
  WHERE j.video_id IS NOT NULL
    AND j.model_id = ?
    AND j.kind = ?
  GROUP BY j.video_id
), preview_frames AS (
  SELECT vf.video_id,
         vf.image_id,
         i.storage_path,
         i.width,
         i.height,
         ROW_NUMBER() OVER (PARTITION BY vf.video_id ORDER BY vf.frame_index ASC) AS rn
  FROM video_frames vf
  JOIN images i ON i.id = vf.image_id
)
SELECT v.id,
       v.original_name,
       v.storage_path,
       v.mime_type,
       COALESCE(v.transcript_text, ''),
       COALESCE(v.description, ''),
       COALESCE(v.tags_json, '[]'),
       v.duration_ms,
       v.width,
       v.height,
       v.frame_count,
         COALESCE(p.image_id, 0),
         COALESCE(p.storage_path, ''),
         COALESCE(p.width, 0),
         COALESCE(p.height, 0),
         CASE
          WHEN COALESCE(f.failed_jobs, 0) > 0 OR COALESCE(tj.failed_jobs, 0) > 0 OR COALESCE(aj.failed_jobs, 0) > 0 THEN 'failed'
          WHEN COALESCE(f.leased_jobs, 0) > 0 OR COALESCE(tj.leased_jobs, 0) > 0 OR COALESCE(aj.leased_jobs, 0) > 0 THEN 'leased'
          WHEN COALESCE(f.total_jobs, 0) > 0 AND COALESCE(f.done_jobs, 0) = COALESCE(f.total_jobs, 0)
            AND (COALESCE(tj.total_jobs, 0) = 0 OR COALESCE(tj.done_jobs, 0) = COALESCE(tj.total_jobs, 0))
            AND (COALESCE(aj.total_jobs, 0) = 0 OR COALESCE(aj.done_jobs, 0) = COALESCE(aj.total_jobs, 0)) THEN 'done'
          ELSE 'pending'
        END AS index_state,
        v.created_at
FROM videos v
LEFT JOIN frame_jobs f ON f.video_id = v.id
LEFT JOIN transcript_jobs tj ON tj.video_id = v.id
LEFT JOIN annotation_jobs aj ON aj.video_id = v.id
LEFT JOIN preview_frames p ON p.video_id = v.id AND p.rn = 1
WHERE (? = 1 OR NOT (%s))
ORDER BY %s
LIMIT ? OFFSET ?
`, videoHasNSFWListExpr, orderClause), args...)
	if err != nil {
		return ListResponse{}, fmt.Errorf("query videos: %w", err)
	}
	defer func() { _ = rows.Close() }()

	items := make([]VideoItem, 0, limit)
	for rows.Next() {
		var item VideoItem
		var tagsJSON string
		if err := rows.Scan(
			&item.VideoID,
			&item.OriginalName,
			&item.StoragePath,
			&item.MimeType,
			&item.TranscriptText,
			&item.Description,
			&tagsJSON,
			&item.DurationMS,
			&item.Width,
			&item.Height,
			&item.FrameCount,
			&item.ImageID,
			&item.PreviewPath,
			&item.PreviewWidth,
			&item.PreviewHeight,
			&item.IndexState,
			&item.CreatedAt,
		); err != nil {
			return ListResponse{}, fmt.Errorf("decode video row: %w", err)
		}
		tags, err := tagutil.DecodeJSON(tagsJSON)
		if err != nil {
			return ListResponse{}, fmt.Errorf("decode video %d tags: %w", item.VideoID, err)
		}
		item.Tags = tags
		item.MediaType = "video"
		items = append(items, item)
	}
	if err := rows.Err(); err != nil {
		return ListResponse{}, fmt.Errorf("iterate video rows: %w", err)
	}

	return ListResponse{Videos: items, Total: total}, nil
}

func NewHandler(h *Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			if h == nil || h.DB == nil {
				httputil.WriteJSONError(w, http.StatusServiceUnavailable, "service unavailable")
				return
			}
			if r.URL.Path != "/api/videos" {
				httputil.WriteJSONError(w, http.StatusNotFound, "not found")
				return
			}
			limit := httputil.ParseLimitQuery(r, 50)
			offset := httputil.ParseOffsetQuery(r, 0)
			includeNSFW := httputil.ParseIncludeNSFWQuery(r)
			order := httputil.ParseOrderQuery(r, listOrderNewest, listOrderNewest, listOrderRandom)
			seed := httputil.ParseInt64Query(r, "seed", 0)

			resp, err := listWithOrder(r.Context(), h.DB, h.ModelID, limit, offset, includeNSFW, order, seed)
			if err != nil {
				httputil.WriteJSONError(w, http.StatusInternalServerError, "query failed")
				return
			}

			httputil.WriteJSON(w, http.StatusOK, resp)
		case http.MethodDelete:
			if h == nil || h.DB == nil {
				httputil.WriteJSONError(w, http.StatusServiceUnavailable, "service unavailable")
				return
			}
			videoID, err := httputil.ParseItemIDPath(r.URL.Path, "/api/videos/")
			if err != nil {
				httputil.WriteJSONError(w, http.StatusBadRequest, "invalid video id")
				return
			}
			if err := Delete(r.Context(), h.DB, h.DataDir, videoID); err != nil {
				if err == sql.ErrNoRows {
					httputil.WriteJSONError(w, http.StatusNotFound, "video not found")
					return
				}
				httputil.WriteJSONError(w, http.StatusInternalServerError, "delete failed")
				return
			}
			w.WriteHeader(http.StatusNoContent)
		case http.MethodPost:
			if r.URL.Path == "/api/videos" {
				httputil.WriteMethodNotAllowed(w, http.MethodGet)
				return
			}
			if strings.HasSuffix(r.URL.Path, "/reannotate") {
				if h == nil || h.DB == nil {
					httputil.WriteJSONError(w, http.StatusServiceUnavailable, "service unavailable")
					return
				}
				videoID, err := parseReannotateVideoIDPath(r.URL.Path)
				if err != nil {
					httputil.WriteJSONError(w, http.StatusBadRequest, "invalid video id")
					return
				}
				if err := Reannotate(r.Context(), h.DB, h.ModelID, videoID); err != nil {
					if err == sql.ErrNoRows {
						httputil.WriteJSONError(w, http.StatusNotFound, "video not found")
						return
					}
					httputil.WriteJSONError(w, http.StatusInternalServerError, "re-annotate failed")
					return
				}
				w.WriteHeader(http.StatusAccepted)
				return
			}
			if strings.HasSuffix(r.URL.Path, "/toggle-nsfw") {
				if h == nil || h.DB == nil {
					httputil.WriteJSONError(w, http.StatusServiceUnavailable, "service unavailable")
					return
				}
				videoID, err := parseToggleNSFWVideoIDPath(r.URL.Path)
				if err != nil {
					httputil.WriteJSONError(w, http.StatusBadRequest, "invalid video id")
					return
				}
				isNSFW, err := ToggleNSFW(r.Context(), h.DB, videoID)
				if err != nil {
					if err == sql.ErrNoRows {
						httputil.WriteJSONError(w, http.StatusNotFound, "video not found")
						return
					}
					httputil.WriteJSONError(w, http.StatusInternalServerError, "toggle nsfw failed")
					return
				}
				httputil.WriteJSON(w, http.StatusOK, struct {
					IsNSFW bool `json:"is_nsfw"`
				}{IsNSFW: isNSFW})
				return
			}
			httputil.WriteJSONError(w, http.StatusNotFound, "not found")
		default:
			if r.URL.Path == "/api/videos" {
				httputil.WriteMethodNotAllowed(w, http.MethodGet)
				return
			}
			httputil.WriteMethodNotAllowed(w, http.MethodDelete, http.MethodPost)
		}
	})
}

func parseReannotateVideoIDPath(path string) (int64, error) {
	return httputil.ParseItemActionIDPath(path, "/api/videos/", "reannotate")
}

func parseToggleNSFWVideoIDPath(path string) (int64, error) {
	return httputil.ParseItemActionIDPath(path, "/api/videos/", "toggle-nsfw")
}

func Reannotate(ctx context.Context, db *sql.DB, modelID int64, videoID int64) error {
	if db == nil {
		return fmt.Errorf("videos database unavailable")
	}
	if modelID <= 0 {
		return fmt.Errorf("invalid model id")
	}
	if videoID <= 0 {
		return fmt.Errorf("invalid video id")
	}

	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin reannotate video tx: %w", err)
	}

	var existingID int64
	if err := tx.QueryRowContext(ctx, `SELECT id FROM videos WHERE id = ?`, videoID).Scan(&existingID); err != nil {
		_ = tx.Rollback()
		return err
	}
	if _, err := tx.ExecContext(ctx, `
UPDATE videos
SET description = '', tags_json = '[]', annotation_updated_at = NULL, reannotate_requested = 1
WHERE id = ?
`, videoID); err != nil {
		_ = tx.Rollback()
		return fmt.Errorf("clear video annotations: %w", err)
	}

	if err := mediaops.RequestReannotationJob(ctx, tx, mediaops.ReannotationTarget{Kind: jobkind.AnnotateVideo, VideoID: videoID, ModelID: modelID}); err != nil {
		_ = tx.Rollback()
		return err
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit reannotate video tx: %w", err)
	}
	return nil
}

func ToggleNSFW(ctx context.Context, db *sql.DB, videoID int64) (bool, error) {
	if db == nil {
		return false, fmt.Errorf("videos database unavailable")
	}
	if videoID <= 0 {
		return false, fmt.Errorf("invalid video id")
	}

	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return false, fmt.Errorf("begin toggle video nsfw tx: %w", err)
	}

	var tagsJSON string
	if err := tx.QueryRowContext(ctx, `
SELECT COALESCE(tags_json, '[]')
FROM videos
WHERE id = ?
`, videoID).Scan(&tagsJSON); err != nil {
		_ = tx.Rollback()
		return false, err
	}

	encodedTags, isNSFW, err := tagutil.ToggleTagJSON(tagsJSON, "nsfw")
	if err != nil {
		_ = tx.Rollback()
		return false, fmt.Errorf("decode video tags: %w", err)
	}

	if _, err := tx.ExecContext(ctx, `
UPDATE videos
SET tags_json = ?
WHERE id = ?
`, encodedTags, videoID); err != nil {
		_ = tx.Rollback()
		return false, fmt.Errorf("update video tags: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return false, fmt.Errorf("commit toggle video nsfw tx: %w", err)
	}

	return isNSFW, nil
}

func Delete(ctx context.Context, db *sql.DB, dataDir string, videoID int64) error {
	if db == nil {
		return fmt.Errorf("videos database unavailable")
	}
	if videoID <= 0 {
		return fmt.Errorf("invalid video id")
	}

	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin delete video tx: %w", err)
	}

	var videoPath string
	if err := tx.QueryRowContext(ctx, `SELECT storage_path FROM videos WHERE id = ?`, videoID).Scan(&videoPath); err != nil {
		_ = tx.Rollback()
		return err
	}

	type frameRef struct {
		imageID       int64
		storagePath   string
		thumbnailPath sql.NullString
	}
	frameRows, err := tx.QueryContext(ctx, `
SELECT i.id, i.storage_path, i.thumbnail_path
FROM video_frames vf
JOIN images i ON i.id = vf.image_id
WHERE vf.video_id = ?
`, videoID)
	if err != nil {
		_ = tx.Rollback()
		return fmt.Errorf("load video frames: %w", err)
	}
	frames := make([]frameRef, 0, 8)
	for frameRows.Next() {
		var ref frameRef
		if err := frameRows.Scan(&ref.imageID, &ref.storagePath, &ref.thumbnailPath); err != nil {
			_ = frameRows.Close()
			_ = tx.Rollback()
			return fmt.Errorf("scan video frame ref: %w", err)
		}
		frames = append(frames, ref)
	}
	_ = frameRows.Close()

	if _, err := tx.ExecContext(ctx, `DELETE FROM video_transcript_embeddings WHERE video_id = ?`, videoID); err != nil {
		_ = tx.Rollback()
		return fmt.Errorf("delete video transcript embeddings: %w", err)
	}
	if _, err := tx.ExecContext(ctx, `DELETE FROM index_jobs WHERE video_id = ?`, videoID); err != nil {
		_ = tx.Rollback()
		return fmt.Errorf("delete video jobs: %w", err)
	}
	if _, err := tx.ExecContext(ctx, `DELETE FROM video_frames WHERE video_id = ?`, videoID); err != nil {
		_ = tx.Rollback()
		return fmt.Errorf("delete video frames: %w", err)
	}

	pathsToDelete := []string{videoPath}
	thumbsToDelete := []string{}
	for _, frame := range frames {
		var remaining int
		if err := tx.QueryRowContext(ctx, `SELECT COUNT(*) FROM video_frames WHERE image_id = ?`, frame.imageID).Scan(&remaining); err != nil {
			_ = tx.Rollback()
			return fmt.Errorf("count remaining frame refs: %w", err)
		}
		if remaining > 0 {
			continue
		}
		if _, err := tx.ExecContext(ctx, `DELETE FROM image_embeddings WHERE image_id = ?`, frame.imageID); err != nil {
			_ = tx.Rollback()
			return fmt.Errorf("delete frame embeddings: %w", err)
		}
		if _, err := tx.ExecContext(ctx, `DELETE FROM index_jobs WHERE image_id = ?`, frame.imageID); err != nil {
			_ = tx.Rollback()
			return fmt.Errorf("delete frame jobs: %w", err)
		}
		if _, err := tx.ExecContext(ctx, `DELETE FROM images WHERE id = ?`, frame.imageID); err != nil {
			_ = tx.Rollback()
			return fmt.Errorf("delete frame image row: %w", err)
		}
		pathsToDelete = append(pathsToDelete, frame.storagePath)
		if frame.thumbnailPath.Valid {
			thumbsToDelete = append(thumbsToDelete, frame.thumbnailPath.String)
		}
	}

	if _, err := tx.ExecContext(ctx, `DELETE FROM videos WHERE id = ?`, videoID); err != nil {
		_ = tx.Rollback()
		return fmt.Errorf("delete video row: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit delete video tx: %w", err)
	}
	for _, path := range pathsToDelete {
		_ = httputil.RemoveStoredPath(dataDir, path)
	}
	for _, thumb := range thumbsToDelete {
		_ = httputil.RemoveStoredPath(dataDir, thumb)
	}
	return nil
}

func boolToInt(v bool) int {
	return httputil.BoolToInt(v)
}
