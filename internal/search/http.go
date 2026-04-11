package search

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"path/filepath"
	"strconv"

	"imgsearch/internal/vectorindex"
)

type Embedder interface {
	EmbedText(ctx context.Context, text string) ([]float32, error)
	EmbedImage(ctx context.Context, path string) ([]float32, error)
}

type Handler struct {
	DB       *sql.DB
	ModelID  int64
	DataDir  string
	Embedder Embedder
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
	q := r.URL.Query().Get("q")
	if q == "" {
		writeJSONError(w, http.StatusBadRequest, "missing query")
		return
	}

	vec, err := h.Embedder.EmbedText(r.Context(), q)
	if err != nil {
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
