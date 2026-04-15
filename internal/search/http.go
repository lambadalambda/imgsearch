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
	ImageID      int64    `json:"image_id"`
	Distance     float64  `json:"distance"`
	OriginalName string   `json:"original_name"`
	StoragePath  string   `json:"storage_path"`
	Description  string   `json:"description,omitempty"`
	Tags         []string `json:"tags,omitempty"`
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

	results, err := h.enrich(r.Context(), hits)
	if err != nil {
		httputil.WriteJSONError(w, http.StatusInternalServerError, "result enrich failed")
		return
	}

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
	SELECT id, original_name, storage_path, COALESCE(description, ''), COALESCE(tags_json, '[]')
	FROM images
	WHERE id IN (%s)
	`, strings.Join(placeholders, ",")), args...)
	if err != nil {
		return nil, fmt.Errorf("load images for enrich: %w", err)
	}
	defer func() { _ = rows.Close() }()

	type imageRow struct {
		originalName string
		storagePath  string
		description  string
		tags         []string
	}
	byID := make(map[int64]imageRow, len(ids))
	for rows.Next() {
		var imageID int64
		var row imageRow
		var tagsJSON string
		if err := rows.Scan(&imageID, &row.originalName, &row.storagePath, &row.description, &tagsJSON); err != nil {
			return nil, fmt.Errorf("scan enriched image row: %w", err)
		}
		tags, err := decodeTagsJSON(tagsJSON)
		if err != nil {
			return nil, fmt.Errorf("decode image %d tags: %w", imageID, err)
		}
		row.tags = tags
		byID[imageID] = row
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate enriched image rows: %w", err)
	}

	results := make([]SearchResult, 0, len(hits))
	for _, hit := range hits {
		row, ok := byID[hit.ImageID]
		if !ok {
			return nil, fmt.Errorf("load image %d: %w", hit.ImageID, sql.ErrNoRows)
		}
		results = append(results, SearchResult{
			ImageID:      hit.ImageID,
			Distance:     hit.Distance,
			OriginalName: row.originalName,
			StoragePath:  filepath.ToSlash(row.storagePath),
			Description:  row.description,
			Tags:         row.tags,
		})
	}
	return results, nil
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
