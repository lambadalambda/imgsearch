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
	"sync"

	"imgsearch/internal/embedder"
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
	ImageID      int64   `json:"image_id"`
	Distance     float64 `json:"distance"`
	OriginalName string  `json:"original_name"`
	StoragePath  string  `json:"storage_path"`
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
		writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	q := strings.TrimSpace(r.URL.Query().Get("q"))
	if q == "" {
		writeJSONError(w, http.StatusBadRequest, "missing query")
		return
	}
	neg := strings.TrimSpace(r.URL.Query().Get("neg"))
	if len(neg) > maxNegativePromptChars {
		writeJSONError(w, http.StatusBadRequest, fmt.Sprintf("negative prompt too long (max %d characters)", maxNegativePromptChars))
		return
	}

	vec, err := h.embedQueryVector(r.Context(), q, neg)
	if err != nil {
		if errors.Is(err, errNegativeEmbedding) {
			writeJSONError(w, http.StatusInternalServerError, "negative embedding failed")
			return
		}
		writeJSONError(w, http.StatusInternalServerError, "embedding failed")
		return
	}

	limit := parseLimit(r, 20)
	hits, err := h.Index.Search(r.Context(), h.ModelID, vec, limit)
	if err != nil {
		writeJSONError(w, http.StatusInternalServerError, "search failed")
		return
	}

	results, err := h.enrich(r.Context(), hits)
	if err != nil {
		writeJSONError(w, http.StatusInternalServerError, "result enrich failed")
		return
	}

	writeJSON(w, http.StatusOK, SearchResponse{Results: results})
}

var errNegativeEmbedding = errors.New("negative embedding failed")

func (h *Handler) embedQueryVector(ctx context.Context, query string, negative string) ([]float32, error) {
	if negative == "" {
		return h.Embedder.EmbedText(ctx, query)
	}

	var queryVec []float32
	var negativeVec []float32
	var queryErr error
	var negativeErr error

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		queryVec, queryErr = h.Embedder.EmbedText(ctx, query)
	}()
	go func() {
		defer wg.Done()
		negativeVec, negativeErr = h.Embedder.EmbedText(ctx, negative)
	}()
	wg.Wait()

	if queryErr != nil {
		return nil, queryErr
	}
	if negativeErr != nil {
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
		writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	imageIDStr := r.URL.Query().Get("image_id")
	if imageIDStr == "" {
		writeJSONError(w, http.StatusBadRequest, "missing image_id")
		return
	}
	imageID, err := strconv.ParseInt(imageIDStr, 10, 64)
	if err != nil || imageID <= 0 {
		writeJSONError(w, http.StatusBadRequest, "invalid image_id")
		return
	}

	limit := parseLimit(r, 20)
	hits, err := h.Index.SearchByImageID(r.Context(), h.ModelID, imageID, limit)
	if err != nil {
		if errors.Is(err, vectorindex.ErrNotFound) {
			writeJSONError(w, http.StatusNotFound, "image not indexed")
			return
		}
		writeJSONError(w, http.StatusInternalServerError, "search failed")
		return
	}

	results, err := h.enrich(r.Context(), hits)
	if err != nil {
		writeJSONError(w, http.StatusInternalServerError, "result enrich failed")
		return
	}

	writeJSON(w, http.StatusOK, SearchResponse{Results: results})
}

func (h *Handler) enrich(ctx context.Context, hits []vectorindex.SearchHit) ([]SearchResult, error) {
	results := make([]SearchResult, 0, len(hits))
	for _, hit := range hits {
		var originalName, storagePath string
		if err := h.DB.QueryRowContext(ctx, `
SELECT original_name, storage_path FROM images WHERE id = ?
`, hit.ImageID).Scan(&originalName, &storagePath); err != nil {
			return nil, fmt.Errorf("load image %d: %w", hit.ImageID, err)
		}
		results = append(results, SearchResult{
			ImageID:      hit.ImageID,
			Distance:     hit.Distance,
			OriginalName: originalName,
			StoragePath:  filepath.ToSlash(storagePath),
		})
	}
	return results, nil
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

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeJSONError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}
