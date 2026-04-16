package search

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"imgsearch/internal/embedder"
	"imgsearch/internal/httputil"
	"imgsearch/internal/vectorindex"
)

type Handler struct {
	DB       *sql.DB
	ModelID  int64
	DataDir  string
	Embedder embedder.Embedder
	Index    vectorindex.VectorIndex
}

type SearchResult struct {
	ImageID          int64    `json:"image_id"`
	MediaType        string   `json:"media_type,omitempty"`
	VideoID          int64    `json:"video_id,omitempty"`
	PreviewPath      string   `json:"preview_path,omitempty"`
	MatchTimestampMS int64    `json:"match_timestamp_ms,omitempty"`
	TranscriptText   string   `json:"transcript_text,omitempty"`
	MimeType         string   `json:"mime_type,omitempty"`
	Distance         float64  `json:"distance"`
	OriginalName     string   `json:"original_name"`
	StoragePath      string   `json:"storage_path"`
	Description      string   `json:"description,omitempty"`
	Tags             []string `json:"tags,omitempty"`
}

type SearchResponse struct {
	Results []SearchResult `json:"results"`
}

const maxNegativePromptChars = 500

var errNegativePromptNearZero = errors.New("negative prompt too similar to query")

func NewHandler(h *Handler) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/search/text", h.handleTextSearch)
	mux.HandleFunc("/api/search/similar", h.handleSimilarSearch)
	return mux
}

func (h *Handler) handleTextSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httputil.WriteJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	q := strings.TrimSpace(r.URL.Query().Get("q"))
	if q == "" {
		httputil.WriteJSONError(w, http.StatusBadRequest, "missing query")
		return
	}
	neg := strings.TrimSpace(r.URL.Query().Get("neg"))
	if len(neg) > maxNegativePromptChars {
		httputil.WriteJSONError(w, http.StatusBadRequest, fmt.Sprintf("negative prompt too long (max %d characters)", maxNegativePromptChars))
		return
	}

	vec, err := h.embedQueryVector(r.Context(), q, neg)
	if err != nil {
		if errors.Is(err, errNegativeEmbedding) {
			httputil.WriteJSONError(w, http.StatusInternalServerError, "negative embedding failed")
			return
		}
		httputil.WriteJSONError(w, http.StatusInternalServerError, "embedding failed")
		return
	}

	limit := parseLimit(r, 20)
	hits, err := h.Index.Search(r.Context(), h.ModelID, vec, limit)
	if err != nil {
		httputil.WriteJSONError(w, http.StatusInternalServerError, "search failed")
		return
	}

	frameResults, err := h.enrich(r.Context(), hits)
	if err != nil {
		httputil.WriteJSONError(w, http.StatusInternalServerError, "result enrich failed")
		return
	}
	transcriptResults, err := h.searchTranscriptEmbeddings(r.Context(), vec, limit)
	if err != nil {
		httputil.WriteJSONError(w, http.StatusInternalServerError, "transcript search failed")
		return
	}
	results := mergeResults(frameResults, transcriptResults, limit)

	httputil.WriteJSON(w, http.StatusOK, SearchResponse{Results: results})
}

var errNegativeEmbedding = errors.New("negative embedding failed")

func (h *Handler) embedQueryVector(ctx context.Context, query string, negative string) ([]float32, error) {
	if negative == "" {
		return h.Embedder.EmbedText(ctx, query)
	}

	queryVec, err := h.Embedder.EmbedText(ctx, query)
	if err != nil {
		return nil, err
	}
	negativeVec, err := h.Embedder.EmbedText(ctx, negative)
	if err != nil {
		return nil, errNegativeEmbedding
	}

	combined, err := combineQueryWithNegative(queryVec, negativeVec)
	if err != nil {
		if errors.Is(err, errNegativePromptNearZero) {
			return queryVec, nil
		}
		return nil, err
	}

	return combined, nil
}

func combineQueryWithNegative(query []float32, negative []float32) ([]float32, error) {
	if len(query) != len(negative) {
		return nil, fmt.Errorf("query and negative dimensions differ: %d vs %d", len(query), len(negative))
	}
	if len(query) == 0 {
		return nil, fmt.Errorf("query embedding is empty")
	}

	out := make([]float32, len(query))
	var squareNorm float64
	for i := range query {
		v := float64(query[i]) - float64(negative[i])
		out[i] = float32(v)
		squareNorm += v * v
	}

	if squareNorm < 1e-12 {
		return nil, errNegativePromptNearZero
	}

	invNorm := float32(1 / math.Sqrt(squareNorm))
	for i := range out {
		out[i] *= invNorm
	}

	return out, nil
}

func (h *Handler) handleSimilarSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httputil.WriteJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	imageIDStr := r.URL.Query().Get("image_id")
	if imageIDStr == "" {
		httputil.WriteJSONError(w, http.StatusBadRequest, "missing image_id")
		return
	}
	imageID, err := strconv.ParseInt(imageIDStr, 10, 64)
	if err != nil || imageID <= 0 {
		httputil.WriteJSONError(w, http.StatusBadRequest, "invalid image_id")
		return
	}

	limit := parseLimit(r, 20)
	hits, err := h.Index.SearchByImageID(r.Context(), h.ModelID, imageID, limit)
	if err != nil {
		if errors.Is(err, vectorindex.ErrNotFound) {
			httputil.WriteJSONError(w, http.StatusNotFound, "image not indexed")
			return
		}
		httputil.WriteJSONError(w, http.StatusInternalServerError, "search failed")
		return
	}

	results, err := h.enrich(r.Context(), hits)
	if err != nil {
		httputil.WriteJSONError(w, http.StatusInternalServerError, "result enrich failed")
		return
	}

	httputil.WriteJSON(w, http.StatusOK, SearchResponse{Results: results})
}

func (h *Handler) enrich(ctx context.Context, hits []vectorindex.SearchHit) ([]SearchResult, error) {
	if len(hits) == 0 {
		return []SearchResult{}, nil
	}

	ids := make([]int64, 0, len(hits))
	seen := make(map[int64]struct{}, len(hits))
	for _, hit := range hits {
		if _, ok := seen[hit.ImageID]; ok {
			continue
		}
		seen[hit.ImageID] = struct{}{}
		ids = append(ids, hit.ImageID)
	}

	placeholders := make([]string, len(ids))
	args := make([]any, 0, len(ids))
	for i, id := range ids {
		placeholders[i] = "?"
		args = append(args, id)
	}

	rows, err := h.DB.QueryContext(ctx, fmt.Sprintf(`
	SELECT i.id,
	       i.original_name,
	       i.storage_path,
	       i.mime_type,
	       COALESCE(i.description, ''),
	       COALESCE(i.tags_json, '[]'),
	       vf.video_id,
	       v.original_name,
	       v.storage_path,
	       v.mime_type,
	       COALESCE(v.transcript_text, ''),
	       vf.timestamp_ms
	FROM images i
	LEFT JOIN video_frames vf ON vf.image_id = i.id
	LEFT JOIN videos v ON v.id = vf.video_id
	WHERE i.id IN (%s)
	`, strings.Join(placeholders, ",")), args...)
	if err != nil {
		return nil, fmt.Errorf("load images for enrich: %w", err)
	}
	defer func() { _ = rows.Close() }()

	type mediaRow struct {
		imageID             int64
		originalName        string
		storagePath         string
		mimeType            string
		description         string
		tags                []string
		videoID             sql.NullInt64
		videoOriginalName   sql.NullString
		videoStoragePath    sql.NullString
		videoMimeType       sql.NullString
		videoTranscriptText sql.NullString
		timestampMS         sql.NullInt64
	}
	byID := make(map[int64][]mediaRow, len(ids))
	for rows.Next() {
		var row mediaRow
		var tagsJSON string
		if err := rows.Scan(&row.imageID, &row.originalName, &row.storagePath, &row.mimeType, &row.description, &tagsJSON, &row.videoID, &row.videoOriginalName, &row.videoStoragePath, &row.videoMimeType, &row.videoTranscriptText, &row.timestampMS); err != nil {
			return nil, fmt.Errorf("scan enriched image row: %w", err)
		}
		tags, err := decodeTagsJSON(tagsJSON)
		if err != nil {
			return nil, fmt.Errorf("decode image %d tags: %w", row.imageID, err)
		}
		row.tags = tags
		byID[row.imageID] = append(byID[row.imageID], row)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate enriched image rows: %w", err)
	}

	seenVideos := make(map[int64]struct{}, len(hits))
	results := make([]SearchResult, 0, len(hits))
	for _, hit := range hits {
		rowsForImage, ok := byID[hit.ImageID]
		if !ok || len(rowsForImage) == 0 {
			return nil, fmt.Errorf("load image %d: %w", hit.ImageID, sql.ErrNoRows)
		}
		for _, row := range rowsForImage {
			if row.videoID.Valid {
				if _, seen := seenVideos[row.videoID.Int64]; seen {
					continue
				}
				seenVideos[row.videoID.Int64] = struct{}{}
			}
			result := SearchResult{
				ImageID:      hit.ImageID,
				MediaType:    "image",
				MimeType:     row.mimeType,
				Distance:     hit.Distance,
				OriginalName: row.originalName,
				StoragePath:  filepath.ToSlash(row.storagePath),
				Description:  row.description,
				Tags:         row.tags,
			}
			if row.videoID.Valid {
				result.MediaType = "video"
				result.VideoID = row.videoID.Int64
				result.OriginalName = row.videoOriginalName.String
				result.StoragePath = filepath.ToSlash(row.videoStoragePath.String)
				result.MimeType = row.videoMimeType.String
				result.TranscriptText = row.videoTranscriptText.String
				result.PreviewPath = filepath.ToSlash(row.storagePath)
				result.MatchTimestampMS = row.timestampMS.Int64
			}
			results = append(results, result)
			if !row.videoID.Valid {
				continue
			}
		}
	}
	return results, nil
}

func (h *Handler) searchTranscriptEmbeddings(ctx context.Context, query []float32, limit int) ([]SearchResult, error) {
	rows, err := h.DB.QueryContext(ctx, `
WITH preview_frames AS (
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
       COALESCE(p.image_id, 0),
       COALESCE(p.storage_path, ''),
       vte.vector_blob
FROM video_transcript_embeddings vte
JOIN videos v ON v.id = vte.video_id
LEFT JOIN preview_frames p ON p.video_id = v.id AND p.rn = 1
WHERE vte.model_id = ?
`, h.ModelID)
	if err != nil {
		return nil, fmt.Errorf("query transcript embeddings: %w", err)
	}
	defer func() { _ = rows.Close() }()

	results := make([]SearchResult, 0, limit)
	for rows.Next() {
		var result SearchResult
		var blob []byte
		if err := rows.Scan(&result.VideoID, &result.OriginalName, &result.StoragePath, &result.MimeType, &result.TranscriptText, &result.ImageID, &result.PreviewPath, &blob); err != nil {
			return nil, fmt.Errorf("scan transcript embedding row: %w", err)
		}
		vec := vectorindex.BlobToFloats(blob)
		if len(vec) == 0 {
			continue
		}
		result.MediaType = "video"
		result.Distance = 1 - cosine(query, vec)
		result.StoragePath = filepath.ToSlash(result.StoragePath)
		result.PreviewPath = filepath.ToSlash(result.PreviewPath)
		results = append(results, result)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate transcript embedding rows: %w", err)
	}
	sort.Slice(results, func(i, j int) bool {
		if results[i].Distance == results[j].Distance {
			return results[i].VideoID < results[j].VideoID
		}
		return results[i].Distance < results[j].Distance
	})
	if len(results) > limit {
		results = results[:limit]
	}
	return results, nil
}

func mergeResults(frameResults []SearchResult, transcriptResults []SearchResult, limit int) []SearchResult {
	merged := make(map[string]SearchResult, len(frameResults)+len(transcriptResults))
	mergeOne := func(result SearchResult) {
		key := fmt.Sprintf("image:%d", result.ImageID)
		if result.MediaType == "video" && result.VideoID > 0 {
			key = fmt.Sprintf("video:%d", result.VideoID)
		}
		existing, ok := merged[key]
		if !ok || result.Distance < existing.Distance {
			merged[key] = result
			return
		}
		if existing.TranscriptText == "" && result.TranscriptText != "" {
			existing.TranscriptText = result.TranscriptText
			merged[key] = existing
		}
	}
	for _, result := range frameResults {
		mergeOne(result)
	}
	for _, result := range transcriptResults {
		mergeOne(result)
	}
	out := make([]SearchResult, 0, len(merged))
	for _, result := range merged {
		out = append(out, result)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Distance == out[j].Distance {
			if out[i].MediaType == out[j].MediaType {
				if out[i].MediaType == "video" {
					return out[i].VideoID < out[j].VideoID
				}
				return out[i].ImageID < out[j].ImageID
			}
			return out[i].MediaType < out[j].MediaType
		}
		return out[i].Distance < out[j].Distance
	})
	if limit > 0 && len(out) > limit {
		out = out[:limit]
	}
	return out
}

func cosine(a, b []float32) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	if n == 0 {
		return 0
	}
	var dot, na, nb float64
	for i := 0; i < n; i++ {
		av := float64(a[i])
		bv := float64(b[i])
		dot += av * bv
		na += av * av
		nb += bv * bv
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
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
