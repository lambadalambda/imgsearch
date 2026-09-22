// Package mediastatus serves GET /api/media/status: the indexing and
// annotation state of a set of visible items, so the Atelier can show
// progress badges and refresh cards without reloading the whole list.
package mediastatus

import (
	"context"
	"database/sql"
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"imgsearch/internal/httputil"
	"imgsearch/internal/jobkind"
	"imgsearch/internal/mediaops"
)

// MaxIDsPerRequest bounds one poll; the UI only asks about pins on screen.
const MaxIDsPerRequest = 500

type Handler struct {
	DB      *sql.DB
	ModelID int64
}

type ImageStatus struct {
	ImageID             int64  `json:"image_id"`
	IndexState          string `json:"index_state"`
	AnnotationState     string `json:"annotation_state"`
	AnnotationUpdatedAt string `json:"annotation_updated_at"`
}

type VideoStatus struct {
	VideoID             int64  `json:"video_id"`
	AnnotationState     string `json:"annotation_state"`
	AnnotationUpdatedAt string `json:"annotation_updated_at"`
}

type Response struct {
	Images []ImageStatus `json:"images"`
	Videos []VideoStatus `json:"videos"`
}

func NewHandler(h *Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			httputil.WriteMethodNotAllowed(w, http.MethodGet)
			return
		}
		if h == nil || h.DB == nil {
			httputil.WriteJSONError(w, http.StatusServiceUnavailable, "service unavailable")
			return
		}
		imageIDs := parseIDs(r.URL.Query().Get("images"))
		videoIDs := parseIDs(r.URL.Query().Get("videos"))
		if len(imageIDs)+len(videoIDs) > MaxIDsPerRequest {
			httputil.WriteJSONError(w, http.StatusBadRequest, fmt.Sprintf("at most %d ids per request", MaxIDsPerRequest))
			return
		}
		resp, err := Collect(r.Context(), h.DB, h.ModelID, imageIDs, videoIDs)
		if err != nil {
			httputil.WriteJSONError(w, http.StatusInternalServerError, "query failed")
			return
		}
		httputil.WriteJSON(w, http.StatusOK, resp)
	})
}

// Collect looks up the states for the given ids; unknown ids are omitted.
func Collect(ctx context.Context, db *sql.DB, modelID int64, imageIDs []int64, videoIDs []int64) (Response, error) {
	resp := Response{Images: []ImageStatus{}, Videos: []VideoStatus{}}
	if len(imageIDs) > 0 {
		args := []any{modelID, jobkind.EmbedImage, modelID, jobkind.AnnotateImage}
		rows, err := db.QueryContext(ctx, `
SELECT i.id,
       COALESCE(j.state, 'pending'),
       COALESCE(a.state, ''),
       trim(COALESCE(i.description, '')) <> '',
       i.reannotate_requested,
       COALESCE(i.annotation_updated_at, '')
FROM images i
LEFT JOIN index_jobs j ON j.image_id = i.id AND j.model_id = ? AND j.kind = ?
LEFT JOIN index_jobs a ON a.image_id = i.id AND a.model_id = ? AND a.kind = ?
WHERE i.id IN (`+placeholders(len(imageIDs), &args, imageIDs)+`)`, args...)
		if err != nil {
			return Response{}, fmt.Errorf("query image status: %w", err)
		}
		for rows.Next() {
			var s ImageStatus
			var jobState string
			var hasText, requested bool
			if err := rows.Scan(&s.ImageID, &s.IndexState, &jobState, &hasText, &requested, &s.AnnotationUpdatedAt); err != nil {
				_ = rows.Close()
				return Response{}, fmt.Errorf("scan image status: %w", err)
			}
			s.AnnotationState = mediaops.AnnotationState(jobState, hasText, requested)
			resp.Images = append(resp.Images, s)
		}
		_ = rows.Close()
		if err := rows.Err(); err != nil {
			return Response{}, err
		}
	}
	if len(videoIDs) > 0 {
		args := []any{modelID, jobkind.AnnotateVideo}
		rows, err := db.QueryContext(ctx, `
SELECT v.id,
       COALESCE(a.state, ''),
       trim(COALESCE(v.description, '')) <> '',
       v.reannotate_requested,
       COALESCE(v.annotation_updated_at, '')
FROM videos v
LEFT JOIN index_jobs a ON a.video_id = v.id AND a.model_id = ? AND a.kind = ?
WHERE v.id IN (`+placeholders(len(videoIDs), &args, videoIDs)+`)`, args...)
		if err != nil {
			return Response{}, fmt.Errorf("query video status: %w", err)
		}
		for rows.Next() {
			var s VideoStatus
			var jobState string
			var hasText, requested bool
			if err := rows.Scan(&s.VideoID, &jobState, &hasText, &requested, &s.AnnotationUpdatedAt); err != nil {
				_ = rows.Close()
				return Response{}, fmt.Errorf("scan video status: %w", err)
			}
			s.AnnotationState = mediaops.AnnotationState(jobState, hasText, requested)
			resp.Videos = append(resp.Videos, s)
		}
		_ = rows.Close()
		if err := rows.Err(); err != nil {
			return Response{}, err
		}
	}
	return resp, nil
}

func placeholders(n int, args *[]any, ids []int64) string {
	parts := make([]string, n)
	for i, id := range ids {
		parts[i] = "?"
		*args = append(*args, id)
	}
	return strings.Join(parts, ",")
}

func parseIDs(raw string) []int64 {
	if strings.TrimSpace(raw) == "" {
		return nil
	}
	seen := map[int64]struct{}{}
	var ids []int64
	for _, part := range strings.Split(raw, ",") {
		id, err := strconv.ParseInt(strings.TrimSpace(part), 10, 64)
		if err != nil || id <= 0 {
			continue
		}
		if _, dup := seen[id]; dup {
			continue
		}
		seen[id] = struct{}{}
		ids = append(ids, id)
	}
	return ids
}
