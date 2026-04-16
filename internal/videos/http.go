package videos

import (
	"context"
	"database/sql"
	"fmt"
	"net/http"
	"strconv"

	"imgsearch/internal/httputil"
)

type Handler struct {
	DB      *sql.DB
	ModelID int64
}

type VideoItem struct {
	VideoID      int64  `json:"video_id"`
	ImageID      int64  `json:"image_id,omitempty"`
	MediaType    string `json:"media_type"`
	OriginalName string `json:"original_name"`
	StoragePath  string `json:"storage_path"`
	PreviewPath  string `json:"preview_path,omitempty"`
	MimeType     string `json:"mime_type"`
	DurationMS   int64  `json:"duration_ms"`
	Width        int    `json:"width"`
	Height       int    `json:"height"`
	FrameCount   int    `json:"frame_count"`
	IndexState   string `json:"index_state"`
	CreatedAt    string `json:"created_at"`
}

type ListResponse struct {
	Videos []VideoItem `json:"videos"`
	Total  int64       `json:"total"`
}

func List(ctx context.Context, db *sql.DB, modelID int64, limit int, offset int) (ListResponse, error) {
	if db == nil {
		return ListResponse{}, fmt.Errorf("videos database unavailable")
	}
	if limit <= 0 {
		limit = 50
	}
	if offset < 0 {
		offset = 0
	}

	var total int64
	if err := db.QueryRowContext(ctx, `SELECT COUNT(*) FROM videos`).Scan(&total); err != nil {
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
       v.duration_ms,
       v.width,
       v.height,
       v.frame_count,
       COALESCE(p.image_id, 0),
       COALESCE(p.storage_path, ''),
       CASE
         WHEN COALESCE(f.failed_jobs, 0) > 0 THEN 'failed'
         WHEN COALESCE(f.done_jobs, 0) > 0 AND COALESCE(f.done_jobs, 0) = COALESCE(f.total_jobs, 0) THEN 'done'
         WHEN COALESCE(f.leased_jobs, 0) > 0 THEN 'leased'
         ELSE 'pending'
       END AS index_state,
       v.created_at
FROM videos v
LEFT JOIN frame_jobs f ON f.video_id = v.id
LEFT JOIN preview_frames p ON p.video_id = v.id AND p.rn = 1
ORDER BY v.id DESC
LIMIT ? OFFSET ?
`, modelID, limit, offset)
	if err != nil {
		return ListResponse{}, fmt.Errorf("query videos: %w", err)
	}
	defer func() { _ = rows.Close() }()

	items := make([]VideoItem, 0, limit)
	for rows.Next() {
		var item VideoItem
		if err := rows.Scan(
			&item.VideoID,
			&item.OriginalName,
			&item.StoragePath,
			&item.MimeType,
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
		if r.Method != http.MethodGet {
			httputil.WriteJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		limit := parseLimit(r, 50)
		offset := parseOffset(r)

		resp, err := List(r.Context(), h.DB, h.ModelID, limit, offset)
		if err != nil {
			httputil.WriteJSONError(w, http.StatusInternalServerError, "query failed")
			return
		}

		httputil.WriteJSON(w, http.StatusOK, resp)
	})
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
