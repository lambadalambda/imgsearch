package images

import (
	"context"
	"database/sql"
	"fmt"
	"net/http"
	"strings"

	"imgsearch/internal/httputil"
	"imgsearch/internal/nsfwsql"
	"imgsearch/internal/tagutil"
)

type Handler struct {
	DB      *sql.DB
	ModelID int64
	DataDir string
}

type ImageItem struct {
	ImageID       int64    `json:"image_id"`
	OriginalName  string   `json:"original_name"`
	StoragePath   string   `json:"storage_path"`
	MimeType      string   `json:"mime_type"`
	Width         int      `json:"width"`
	Height        int      `json:"height"`
	IndexState    string   `json:"index_state"`
	CreatedAt     string   `json:"created_at"`
	Description   string   `json:"description,omitempty"`
	Tags          []string `json:"tags,omitempty"`
	ThumbnailPath string   `json:"thumbnail_path,omitempty"`
}

type ListResponse struct {
	Images []ImageItem `json:"images"`
	Total  int64       `json:"total"`
}

func List(ctx context.Context, db *sql.DB, modelID int64, limit int, offset int, includeNSFW bool) (ListResponse, error) {
	if db == nil {
		return ListResponse{}, fmt.Errorf("images database unavailable")
	}
	if limit <= 0 {
		limit = 50
	}
	if offset < 0 {
		offset = 0
	}
	includeNSFWInt := boolToInt(includeNSFW)
	imageHasNSFWExpr := nsfwsql.TagsJSONHasNSFW("i.tags_json", "tag")

	var total int64
	if err := db.QueryRowContext(ctx, fmt.Sprintf(`
SELECT COUNT(*)
FROM images i
WHERE NOT EXISTS (
  SELECT 1
  FROM video_frames vf
  WHERE vf.image_id = i.id
)
  AND (? = 1 OR NOT (%s))
`, imageHasNSFWExpr), includeNSFWInt).Scan(&total); err != nil {
		return ListResponse{}, fmt.Errorf("count images: %w", err)
	}

	rows, err := db.QueryContext(ctx, fmt.Sprintf(`
SELECT i.id, i.original_name, i.storage_path, i.thumbnail_path, i.mime_type, i.width, i.height,
	COALESCE(i.description, ''), COALESCE(i.tags_json, '[]'),
	COALESCE(j.state, 'pending') AS state,
	i.created_at
FROM images i
LEFT JOIN index_jobs j
	ON j.image_id = i.id
	AND j.model_id = ?
	AND j.kind = 'embed_image'
WHERE NOT EXISTS (
	SELECT 1
	FROM video_frames vf
	WHERE vf.image_id = i.id
)
	AND (? = 1 OR NOT (%s))
ORDER BY i.id DESC
LIMIT ? OFFSET ?
`, imageHasNSFWExpr), modelID, includeNSFWInt, limit, offset)
	if err != nil {
		return ListResponse{}, fmt.Errorf("query images: %w", err)
	}
	defer func() { _ = rows.Close() }()

	items := make([]ImageItem, 0, limit)
	for rows.Next() {
		var item ImageItem
		var thumb sql.NullString
		var tagsJSON string
		if err := rows.Scan(
			&item.ImageID,
			&item.OriginalName,
			&item.StoragePath,
			&thumb,
			&item.MimeType,
			&item.Width,
			&item.Height,
			&item.Description,
			&tagsJSON,
			&item.IndexState,
			&item.CreatedAt,
		); err != nil {
			return ListResponse{}, fmt.Errorf("decode image row: %w", err)
		}
		if tags, err := tagutil.DecodeJSON(tagsJSON); err != nil {
			return ListResponse{}, fmt.Errorf("decode image tags: %w", err)
		} else {
			item.Tags = tags
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
		switch r.Method {
		case http.MethodGet:
			limit := httputil.ParseLimitQuery(r, 50)
			offset := httputil.ParseOffsetQuery(r, 0)
			includeNSFW := httputil.ParseIncludeNSFWQuery(r)

			resp, err := List(r.Context(), h.DB, h.ModelID, limit, offset, includeNSFW)
			if err != nil {
				httputil.WriteJSONError(w, http.StatusInternalServerError, "query failed")
				return
			}

			httputil.WriteJSON(w, http.StatusOK, resp)
		case http.MethodDelete:
			imageID, err := httputil.ParseItemIDPath(r.URL.Path, "/api/images/")
			if err != nil {
				httputil.WriteJSONError(w, http.StatusBadRequest, "invalid image id")
				return
			}
			if err := Delete(r.Context(), h.DB, h.DataDir, imageID); err != nil {
				switch {
				case strings.Contains(err.Error(), "derived video frame"):
					httputil.WriteJSONError(w, http.StatusConflict, err.Error())
				case err == sql.ErrNoRows:
					httputil.WriteJSONError(w, http.StatusNotFound, "image not found")
				default:
					httputil.WriteJSONError(w, http.StatusInternalServerError, "delete failed")
				}
				return
			}
			w.WriteHeader(http.StatusNoContent)
		default:
			httputil.WriteJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
		}
	})
}

func Delete(ctx context.Context, db *sql.DB, dataDir string, imageID int64) error {
	if db == nil {
		return fmt.Errorf("images database unavailable")
	}
	if imageID <= 0 {
		return fmt.Errorf("invalid image id")
	}

	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin delete image tx: %w", err)
	}

	var derivedCount int
	if err := tx.QueryRowContext(ctx, `SELECT COUNT(*) FROM video_frames WHERE image_id = ?`, imageID).Scan(&derivedCount); err != nil {
		_ = tx.Rollback()
		return fmt.Errorf("check derived video frame: %w", err)
	}
	if derivedCount > 0 {
		_ = tx.Rollback()
		return fmt.Errorf("cannot delete derived video frame image directly")
	}

	var storagePath string
	var thumbnailPath sql.NullString
	if err := tx.QueryRowContext(ctx, `SELECT storage_path, thumbnail_path FROM images WHERE id = ?`, imageID).Scan(&storagePath, &thumbnailPath); err != nil {
		_ = tx.Rollback()
		return err
	}

	if _, err := tx.ExecContext(ctx, `DELETE FROM image_embeddings WHERE image_id = ?`, imageID); err != nil {
		_ = tx.Rollback()
		return fmt.Errorf("delete image embeddings: %w", err)
	}
	if _, err := tx.ExecContext(ctx, `DELETE FROM index_jobs WHERE image_id = ?`, imageID); err != nil {
		_ = tx.Rollback()
		return fmt.Errorf("delete image jobs: %w", err)
	}
	if _, err := tx.ExecContext(ctx, `DELETE FROM images WHERE id = ?`, imageID); err != nil {
		_ = tx.Rollback()
		return fmt.Errorf("delete image row: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit delete image tx: %w", err)
	}

	_ = httputil.RemoveStoredPath(dataDir, storagePath)
	if thumbnailPath.Valid {
		_ = httputil.RemoveStoredPath(dataDir, thumbnailPath.String)
	}
	return nil
}

func boolToInt(v bool) int {
	if v {
		return 1
	}
	return 0
}

func (h *Handler) String() string {
	return fmt.Sprintf("images.Handler(model_id=%d)", h.ModelID)
}
