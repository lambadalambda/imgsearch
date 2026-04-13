package images

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
)

type Handler struct {
	DB      *sql.DB
	ModelID int64
}

type ImageItem struct {
	ImageID       int64  `json:"image_id"`
	OriginalName  string `json:"original_name"`
	StoragePath   string `json:"storage_path"`
	MimeType      string `json:"mime_type"`
	Width         int    `json:"width"`
	Height        int    `json:"height"`
	IndexState    string `json:"index_state"`
	CreatedAt     string `json:"created_at"`
	ThumbnailPath string `json:"thumbnail_path,omitempty"`
}

type ListResponse struct {
	Images []ImageItem `json:"images"`
	Total  int64       `json:"total"`
}

func List(ctx context.Context, db *sql.DB, modelID int64, limit int, offset int) (ListResponse, error) {
	if db == nil {
		return ListResponse{}, fmt.Errorf("images database unavailable")
	}
	if limit <= 0 {
		limit = 50
	}
	if offset < 0 {
		offset = 0
	}

	var total int64
	if err := db.QueryRowContext(ctx, `SELECT COUNT(*) FROM images`).Scan(&total); err != nil {
		return ListResponse{}, fmt.Errorf("count images: %w", err)
	}

	rows, err := db.QueryContext(ctx, `
SELECT i.id, i.original_name, i.storage_path, i.thumbnail_path, i.mime_type, i.width, i.height,
	COALESCE(j.state, 'pending') AS state,
	i.created_at
FROM images i
LEFT JOIN index_jobs j
	ON j.image_id = i.id
	AND j.model_id = ?
	AND j.kind = 'embed_image'
ORDER BY i.id DESC
LIMIT ? OFFSET ?
`, modelID, limit, offset)
	if err != nil {
		return ListResponse{}, fmt.Errorf("query images: %w", err)
	}
	defer func() { _ = rows.Close() }()

	items := make([]ImageItem, 0, limit)
	for rows.Next() {
		var item ImageItem
		var thumb sql.NullString
		if err := rows.Scan(
			&item.ImageID,
			&item.OriginalName,
			&item.StoragePath,
			&thumb,
			&item.MimeType,
			&item.Width,
			&item.Height,
			&item.IndexState,
			&item.CreatedAt,
		); err != nil {
			return ListResponse{}, fmt.Errorf("decode image row: %w", err)
		}
		if thumb.Valid {
			item.ThumbnailPath = thumb.String
		}
		items = append(items, item)
	}
	if err := rows.Err(); err != nil {
		return ListResponse{}, fmt.Errorf("iterate image rows: %w", err)
	}

	return ListResponse{Images: items, Total: total}, nil
}

func NewHandler(h *Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		limit := parseLimit(r, 50)
		offset := parseOffset(r)

		resp, err := List(r.Context(), h.DB, h.ModelID, limit, offset)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "query failed")
			return
		}

		writeJSON(w, http.StatusOK, resp)
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

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeJSONError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

func (h *Handler) String() string {
	return fmt.Sprintf("images.Handler(model_id=%d)", h.ModelID)
}
