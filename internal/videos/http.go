package videos

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"imgsearch/internal/httputil"
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

func List(ctx context.Context, db *sql.DB, modelID int64, limit int, offset int, includeNSFW bool) (ListResponse, error) {
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

	var total int64
	if err := db.QueryRowContext(ctx, `
SELECT COUNT(*)
FROM videos v
WHERE (
  ? = 1
  OR NOT (
    EXISTS (
      SELECT 1
      FROM video_frames vf
      JOIN images i ON i.id = vf.image_id
      JOIN json_each(COALESCE(i.tags_json, '[]')) tag
        ON lower(trim(COALESCE(tag.value, ''))) = 'nsfw'
      WHERE vf.video_id = v.id
    )
    OR EXISTS (
      SELECT 1
      FROM json_each(COALESCE(v.tags_json, '[]')) vtag
      WHERE lower(trim(COALESCE(vtag.value, ''))) = 'nsfw'
    )
  )
)
`, includeNSFWInt).Scan(&total); err != nil {
		return ListResponse{}, fmt.Errorf("count videos: %w", err)
	}

	rows, err := db.QueryContext(ctx, `
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
   AND j.kind = 'embed_image'
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
    AND j.kind = 'transcribe_video'
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
    AND j.kind = 'annotate_video'
  GROUP BY j.video_id
), preview_frames AS (
  SELECT vf.video_id,
         vf.image_id,
         i.storage_path,
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
WHERE (
  ? = 1
  OR NOT (
    EXISTS (
      SELECT 1
      FROM video_frames vf_nsfw
      JOIN images i_nsfw ON i_nsfw.id = vf_nsfw.image_id
      JOIN json_each(COALESCE(i_nsfw.tags_json, '[]')) tag_nsfw
        ON lower(trim(COALESCE(tag_nsfw.value, ''))) = 'nsfw'
      WHERE vf_nsfw.video_id = v.id
    )
    OR EXISTS (
      SELECT 1
      FROM json_each(COALESCE(v.tags_json, '[]')) vtag_nsfw
      WHERE lower(trim(COALESCE(vtag_nsfw.value, ''))) = 'nsfw'
    )
  )
)
ORDER BY v.id DESC
LIMIT ? OFFSET ?
`, modelID, modelID, modelID, includeNSFWInt, limit, offset)
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
			&item.IndexState,
			&item.CreatedAt,
		); err != nil {
			return ListResponse{}, fmt.Errorf("decode video row: %w", err)
		}
		tags, err := decodeTagsJSON(tagsJSON)
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
			limit := parseLimit(r, 50)
			offset := parseOffset(r)
			includeNSFW := parseIncludeNSFW(r)

			resp, err := List(r.Context(), h.DB, h.ModelID, limit, offset, includeNSFW)
			if err != nil {
				httputil.WriteJSONError(w, http.StatusInternalServerError, "query failed")
				return
			}

			httputil.WriteJSON(w, http.StatusOK, resp)
		case http.MethodDelete:
			videoID, err := parseItemID(r.URL.Path, "/api/videos/")
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
		default:
			httputil.WriteJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
		}
	})
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
		removeStoredPath(dataDir, path)
	}
	for _, thumb := range thumbsToDelete {
		removeStoredPath(dataDir, thumb)
	}
	return nil
}

func parseItemID(path string, prefix string) (int64, error) {
	idText := strings.TrimPrefix(path, prefix)
	if idText == path || idText == "" || strings.ContainsRune(idText, '/') {
		return 0, fmt.Errorf("invalid id path")
	}
	return strconv.ParseInt(idText, 10, 64)
}

func removeStoredPath(dataDir string, rel string) {
	if strings.TrimSpace(dataDir) == "" || strings.TrimSpace(rel) == "" {
		return
	}
	_ = os.Remove(filepath.Join(dataDir, filepath.FromSlash(rel)))
}

func parseLimit(r *http.Request, fallback int) int {
	v := r.URL.Query().Get("limit")
	if v == "" {
		return fallback
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 || n > 200 {
		return fallback
	}
	return n
}

func parseOffset(r *http.Request) int {
	v := r.URL.Query().Get("offset")
	if v == "" {
		return 0
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < 0 {
		return 0
	}
	return n
}

func parseIncludeNSFW(r *http.Request) bool {
	v := strings.TrimSpace(r.URL.Query().Get("include_nsfw"))
	if v == "" {
		return false
	}
	parsed, err := strconv.ParseBool(v)
	if err != nil {
		return false
	}
	return parsed
}

func boolToInt(v bool) int {
	if v {
		return 1
	}
	return 0
}

func decodeTagsJSON(raw string) ([]string, error) {
	if raw == "" {
		return nil, nil
	}
	var tags []string
	if err := json.Unmarshal([]byte(raw), &tags); err != nil {
		return nil, err
	}
	return tags, nil
}
