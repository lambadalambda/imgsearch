package images

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"imgsearch/internal/annotationtext"
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

type ImageItem struct {
	ImageID      int64  `json:"image_id"`
	OriginalName string `json:"original_name"`
	StoragePath  string `json:"storage_path"`
	MimeType     string `json:"mime_type"`
	Width        int    `json:"width"`
	Height       int    `json:"height"`
	IndexState   string `json:"index_state"`
	CreatedAt    string `json:"created_at"`
	// CapturedAt is the EXIF capture time when known, else the upload time.
	CapturedAt string `json:"captured_at"`
	// AnnotationState is one of queued, annotating, failed, done, none.
	AnnotationState     string   `json:"annotation_state"`
	AnnotationUpdatedAt string   `json:"annotation_updated_at"`
	Title               string   `json:"title,omitempty"`
	Summary             string   `json:"summary,omitempty"`
	Description         string   `json:"description,omitempty"`
	FullDescription     string   `json:"full_description,omitempty"`
	Tags                []string `json:"tags,omitempty"`
	ThumbnailPath       string   `json:"thumbnail_path,omitempty"`
}

type ListResponse struct {
	Images []ImageItem `json:"images"`
	Total  int64       `json:"total"`
}

const (
	listOrderNewest   = "newest"
	listOrderRandom   = "random"
	listOrderCaptured = "captured"
	capturedAtExpr    = "COALESCE(NULLIF(i.captured_at, ''), i.created_at)"
	randomOrderMask   = int64(2147483647)
)

func List(ctx context.Context, db *sql.DB, modelID int64, limit int, offset int, includeNSFW bool) (ListResponse, error) {
	return listWithOrder(ctx, db, modelID, limit, offset, includeNSFW, listOrderNewest, 0)
}

func listWithOrder(ctx context.Context, db *sql.DB, modelID int64, limit int, offset int, includeNSFW bool, order string, seed int64) (ListResponse, error) {
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
	orderClause := "i.id DESC"
	args := []any{modelID, jobkind.EmbedImage, modelID, jobkind.AnnotateImage, includeNSFWInt}
	if order == listOrderCaptured {
		orderClause = capturedAtExpr + " DESC, i.id DESC"
	}
	if order == listOrderRandom {
		seed = seed & randomOrderMask
		orderClause = "((((i.id * 1103515245 + ?) & 2147483647) | (((i.id * 1103515245 + ?) & 2147483647) >> 16)) * 1103515245 + 12345) & 2147483647 ASC, i.id ASC"
		args = append(args, seed, seed)
	}
	args = append(args, limit, offset)

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

	rows, err := db.QueryContext(ctx, fmt.Sprintf(imageItemSelect+`
WHERE NOT EXISTS (
	SELECT 1
	FROM video_frames vf
	WHERE vf.image_id = i.id
)
	AND (? = 1 OR NOT (%s))
ORDER BY %s
LIMIT ? OFFSET ?
`, imageHasNSFWExpr, orderClause), args...)
	if err != nil {
		return ListResponse{}, fmt.Errorf("query images: %w", err)
	}
	defer func() { _ = rows.Close() }()

	items := make([]ImageItem, 0, limit)
	for rows.Next() {
		item, err := scanImageItem(rows)
		if err != nil {
			return ListResponse{}, err
		}
		items = append(items, item)
	}
	if err := rows.Err(); err != nil {
		return ListResponse{}, fmt.Errorf("iterate image rows: %w", err)
	}

	return ListResponse{Images: items, Total: total}, nil
}

// imageItemSelect lists the columns scanImageItem expects; it takes the
// model id and embed job kind, then the model id and annotate job kind, as
// its four arguments.
const imageItemSelect = `
SELECT i.id, i.original_name, i.storage_path, i.thumbnail_path, i.mime_type, i.width, i.height,
	COALESCE(i.title, ''), COALESCE(i.summary, ''), COALESCE(i.description, ''), COALESCE(i.tags_json, '[]'),
	COALESCE(j.state, 'pending') AS state,
	i.created_at,
	COALESCE(NULLIF(i.captured_at, ''), i.created_at) AS captured_at,
	COALESCE(a.state, ''),
	i.reannotate_requested,
	COALESCE(i.annotation_updated_at, '')
FROM images i
LEFT JOIN index_jobs j
	ON j.image_id = i.id
	AND j.model_id = ?
	AND j.kind = ?
LEFT JOIN index_jobs a
	ON a.image_id = i.id
	AND a.model_id = ?
	AND a.kind = ?`

type rowScanner interface {
	Scan(dest ...any) error
}

func scanImageItem(row rowScanner) (ImageItem, error) {
	var item ImageItem
	var thumb sql.NullString
	var tagsJSON string
	var title string
	var summary string
	var fullDescription string
	var annotationJobState string
	var reannotateRequested bool
	if err := row.Scan(
		&item.ImageID,
		&item.OriginalName,
		&item.StoragePath,
		&thumb,
		&item.MimeType,
		&item.Width,
		&item.Height,
		&title,
		&summary,
		&fullDescription,
		&tagsJSON,
		&item.IndexState,
		&item.CreatedAt,
		&item.CapturedAt,
		&annotationJobState,
		&reannotateRequested,
		&item.AnnotationUpdatedAt,
	); err != nil {
		return ImageItem{}, fmt.Errorf("decode image row: %w", err)
	}
	item.AnnotationState = mediaops.AnnotationState(annotationJobState, strings.TrimSpace(fullDescription) != "", reannotateRequested)
	tags, err := tagutil.DecodeJSON(tagsJSON)
	if err != nil {
		return ImageItem{}, fmt.Errorf("decode image tags: %w", err)
	}
	item.Tags = tags
	text := annotationtext.Build(title, summary, fullDescription)
	item.Title = text.Title
	item.Summary = text.Summary
	item.Description = text.Description
	item.FullDescription = text.FullDescription
	if thumb.Valid {
		item.ThumbnailPath = thumb.String
	}
	return item, nil
}

// GetItem returns one image in the list shape. It reports sql.ErrNoRows for
// an unknown id.
func GetItem(ctx context.Context, db *sql.DB, modelID int64, imageID int64) (ImageItem, error) {
	if db == nil {
		return ImageItem{}, fmt.Errorf("images database unavailable")
	}
	row := db.QueryRowContext(ctx, imageItemSelect+`
WHERE i.id = ?`, modelID, jobkind.EmbedImage, modelID, jobkind.AnnotateImage, imageID)
	item, err := scanImageItem(row)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return ImageItem{}, sql.ErrNoRows
		}
		return ImageItem{}, err
	}
	return item, nil
}

// Update applies a metadata patch (manual title and tags) and returns the
// updated item.
func Update(ctx context.Context, db *sql.DB, modelID int64, imageID int64, patch mediaops.MetadataPatch) (ImageItem, error) {
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return ImageItem{}, fmt.Errorf("begin update image tx: %w", err)
	}
	if err := mediaops.ApplyMetadataPatch(ctx, tx, mediaops.TableImages, imageID, patch); err != nil {
		_ = tx.Rollback()
		return ImageItem{}, err
	}
	if err := tx.Commit(); err != nil {
		return ImageItem{}, fmt.Errorf("commit update image tx: %w", err)
	}
	return GetItem(ctx, db, modelID, imageID)
}

func NewHandler(h *Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			if h == nil || h.DB == nil {
				httputil.WriteJSONError(w, http.StatusServiceUnavailable, "service unavailable")
				return
			}
			if r.URL.Path != "/api/images" {
				imageID, err := httputil.ParseItemIDPath(r.URL.Path, "/api/images/")
				if err != nil {
					httputil.WriteJSONError(w, http.StatusNotFound, "not found")
					return
				}
				item, err := GetItem(r.Context(), h.DB, h.ModelID, imageID)
				writeImageItem(w, item, err)
				return
			}
			limit := httputil.ParseLimitQuery(r, 50)
			offset := httputil.ParseOffsetQuery(r, 0)
			includeNSFW := httputil.ParseIncludeNSFWQuery(r)
			order := httputil.ParseOrderQuery(r, listOrderNewest, listOrderNewest, listOrderRandom, listOrderCaptured)
			seed := httputil.ParseInt64Query(r, "seed", 0)

			resp, err := listWithOrder(r.Context(), h.DB, h.ModelID, limit, offset, includeNSFW, order, seed)
			if err != nil {
				httputil.WriteJSONError(w, http.StatusInternalServerError, "query failed")
				return
			}

			httputil.WriteJSON(w, http.StatusOK, resp)
		case http.MethodPatch:
			if h == nil || h.DB == nil {
				httputil.WriteJSONError(w, http.StatusServiceUnavailable, "service unavailable")
				return
			}
			imageID, err := httputil.ParseItemIDPath(r.URL.Path, "/api/images/")
			if err != nil {
				httputil.WriteJSONError(w, http.StatusBadRequest, "invalid image id")
				return
			}
			patch, err := mediaops.DecodeMetadataPatch(r.Body)
			if err != nil {
				httputil.WriteJSONError(w, http.StatusBadRequest, err.Error())
				return
			}
			item, err := Update(r.Context(), h.DB, h.ModelID, imageID, patch)
			writeImageItem(w, item, err)
		case http.MethodDelete:
			if h == nil || h.DB == nil {
				httputil.WriteJSONError(w, http.StatusServiceUnavailable, "service unavailable")
				return
			}
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
		case http.MethodPost:
			if r.URL.Path == "/api/images" {
				httputil.WriteMethodNotAllowed(w, http.MethodGet)
				return
			}
			if strings.HasSuffix(r.URL.Path, "/reannotate") {
				if h == nil || h.DB == nil {
					httputil.WriteJSONError(w, http.StatusServiceUnavailable, "service unavailable")
					return
				}
				imageID, err := parseReannotateImageIDPath(r.URL.Path)
				if err != nil {
					httputil.WriteJSONError(w, http.StatusBadRequest, "invalid image id")
					return
				}
				if err := Reannotate(r.Context(), h.DB, h.ModelID, imageID); err != nil {
					if err == sql.ErrNoRows {
						httputil.WriteJSONError(w, http.StatusNotFound, "image not found")
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
				imageID, err := parseToggleNSFWImageIDPath(r.URL.Path)
				if err != nil {
					httputil.WriteJSONError(w, http.StatusBadRequest, "invalid image id")
					return
				}
				isNSFW, err := ToggleNSFW(r.Context(), h.DB, imageID)
				if err != nil {
					if err == sql.ErrNoRows {
						httputil.WriteJSONError(w, http.StatusNotFound, "image not found")
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
			if r.URL.Path == "/api/images" {
				httputil.WriteMethodNotAllowed(w, http.MethodGet)
				return
			}
			httputil.WriteMethodNotAllowed(w, http.MethodGet, http.MethodPatch, http.MethodDelete, http.MethodPost)
		}
	})
}

func writeImageItem(w http.ResponseWriter, item ImageItem, err error) {
	switch {
	case err == nil:
		httputil.WriteJSON(w, http.StatusOK, item)
	case errors.Is(err, sql.ErrNoRows):
		httputil.WriteJSONError(w, http.StatusNotFound, "image not found")
	default:
		httputil.WriteJSONError(w, http.StatusInternalServerError, "query failed")
	}
}

func parseReannotateImageIDPath(path string) (int64, error) {
	return httputil.ParseItemActionIDPath(path, "/api/images/", "reannotate")
}

func parseToggleNSFWImageIDPath(path string) (int64, error) {
	return httputil.ParseItemActionIDPath(path, "/api/images/", "toggle-nsfw")
}

func Reannotate(ctx context.Context, db *sql.DB, modelID int64, imageID int64) error {
	if db == nil {
		return fmt.Errorf("images database unavailable")
	}
	if modelID <= 0 {
		return fmt.Errorf("invalid model id")
	}
	if imageID <= 0 {
		return fmt.Errorf("invalid image id")
	}

	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin reannotate image tx: %w", err)
	}

	var existingID int64
	if err := tx.QueryRowContext(ctx, `SELECT id FROM images WHERE id = ?`, imageID).Scan(&existingID); err != nil {
		_ = tx.Rollback()
		return err
	}
	if _, err := tx.ExecContext(ctx, `
UPDATE images
SET title = user_title, summary = '', description = '', tags_json = user_tags_json, reannotate_requested = 1
WHERE id = ?
`, imageID); err != nil {
		_ = tx.Rollback()
		return fmt.Errorf("clear image annotations: %w", err)
	}

	if err := mediaops.RequestReannotationJob(ctx, tx, mediaops.ReannotationTarget{Kind: jobkind.AnnotateImage, ImageID: imageID, ModelID: modelID}); err != nil {
		_ = tx.Rollback()
		return err
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit reannotate image tx: %w", err)
	}
	return nil
}

func ToggleNSFW(ctx context.Context, db *sql.DB, imageID int64) (bool, error) {
	if db == nil {
		return false, fmt.Errorf("images database unavailable")
	}
	if imageID <= 0 {
		return false, fmt.Errorf("invalid image id")
	}

	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return false, fmt.Errorf("begin toggle image nsfw tx: %w", err)
	}

	var tagsJSON string
	if err := tx.QueryRowContext(ctx, `
SELECT COALESCE(tags_json, '[]')
FROM images
WHERE id = ?
`, imageID).Scan(&tagsJSON); err != nil {
		_ = tx.Rollback()
		return false, err
	}

	tags, err := tagutil.DecodeJSON(tagsJSON)
	if err != nil {
		_ = tx.Rollback()
		return false, fmt.Errorf("decode image tags: %w", err)
	}
	toggled, isNSFW := tagutil.ToggleTag(tags, "nsfw")
	// Route through the served-tags writer so the flag counts as a manual
	// edit and survives re-annotation.
	if _, err := mediaops.SetServedTags(ctx, tx, mediaops.TableImages, imageID, toggled); err != nil {
		_ = tx.Rollback()
		return false, fmt.Errorf("update image tags: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return false, fmt.Errorf("commit toggle image nsfw tx: %w", err)
	}

	return isNSFW, nil
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
	return httputil.BoolToInt(v)
}

func (h *Handler) String() string {
	return fmt.Sprintf("images.Handler(model_id=%d)", h.ModelID)
}
