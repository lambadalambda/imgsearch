package search

import (
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"imgsearch/internal/httputil"
	"imgsearch/internal/vectorindex"
)

// MaxQueryImageBytes bounds a search-by-image upload; it matches the default
// per-file image upload limit.
const MaxQueryImageBytes int64 = 64 << 20

// handleByImageSearch answers POST /api/search/by-image: a multipart "file"
// (JPEG, PNG, WEBP, AVIF) is embedded in-process without being stored, and
// the nearest library items come back in the "similar" result shape.
func (h *Handler) handleByImageSearch(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	if r.Method != http.MethodPost {
		httputil.WriteMethodNotAllowed(w, http.MethodPost)
		return
	}
	if h == nil || h.DB == nil || h.Index == nil || h.Embedder == nil {
		httputil.WriteJSONError(w, http.StatusServiceUnavailable, "service unavailable")
		return
	}
	includeNSFW := httputil.ParseIncludeNSFWQuery(r)
	limit := httputil.ParseLimitQuery(r, 20)

	r.Body = http.MaxBytesReader(w, r.Body, MaxQueryImageBytes+1<<20)
	if err := r.ParseMultipartForm(8 << 20); err != nil {
		var maxBytesErr *http.MaxBytesError
		if errors.As(err, &maxBytesErr) {
			httputil.WriteJSONError(w, http.StatusRequestEntityTooLarge, fmt.Sprintf("query image exceeds the %d MiB limit", MaxQueryImageBytes>>20))
			return
		}
		httputil.WriteJSONError(w, http.StatusBadRequest, "invalid multipart upload")
		return
	}
	defer func() {
		if r.MultipartForm != nil {
			_ = r.MultipartForm.RemoveAll()
		}
	}()
	files := r.MultipartForm.File["file"]
	if len(files) != 1 {
		httputil.WriteJSONError(w, http.StatusBadRequest, "send exactly one file")
		return
	}
	if files[0].Size > MaxQueryImageBytes {
		httputil.WriteJSONError(w, http.StatusRequestEntityTooLarge, fmt.Sprintf("query image exceeds the %d MiB limit", MaxQueryImageBytes>>20))
		return
	}

	queryPath, err := h.spoolQueryImage(files[0])
	if err != nil {
		if errors.Is(err, errUnsupportedQueryImage) {
			httputil.WriteJSONError(w, http.StatusBadRequest, "unsupported image format")
			return
		}
		httputil.WriteJSONError(w, http.StatusInternalServerError, "could not read query image")
		return
	}
	defer func() { _ = os.Remove(queryPath) }()

	vec, err := h.Embedder.EmbedImage(r.Context(), queryPath)
	if err != nil || len(vec) == 0 {
		httputil.WriteJSONError(w, http.StatusInternalServerError, "query embedding failed")
		return
	}

	searchLimit := limit
	if !includeNSFW {
		searchLimit = limit * 8
		if searchLimit > 200 {
			searchLimit = 200
		}
	}
	indexDebug := vectorindex.SearchDebug{}
	searchCtx := vectorindex.WithSearchDebug(r.Context(), &indexDebug)
	hits, err := h.Index.Search(searchCtx, h.ModelID, vec, searchLimit)
	if err != nil {
		httputil.WriteJSONError(w, http.StatusInternalServerError, "search failed")
		return
	}
	results, err := h.enrich(r.Context(), hits, includeNSFW)
	if err != nil {
		httputil.WriteJSONError(w, http.StatusInternalServerError, "result enrich failed")
		return
	}
	if len(results) > limit {
		results = results[:limit]
	}
	httputil.WriteJSON(w, http.StatusOK, SearchResponse{Results: results, Total: int64(len(results)), Debug: buildSearchDebugResponse(start, indexDebug)})
}

var errUnsupportedQueryImage = errors.New("unsupported query image")

// spoolQueryImage writes the uploaded part to a temp file under the data
// dir (the embedder reads paths) after checking it is an image.
func (h *Handler) spoolQueryImage(header *multipart.FileHeader) (string, error) {
	part, err := header.Open()
	if err != nil {
		return "", err
	}
	defer func() { _ = part.Close() }()

	head := make([]byte, 512)
	n, _ := io.ReadFull(part, head)
	if !isSupportedQueryImage(head[:n]) {
		return "", errUnsupportedQueryImage
	}
	tmpDir := filepath.Join(h.DataDir, "tmp")
	if err := os.MkdirAll(tmpDir, 0o755); err != nil {
		return "", err
	}
	tmp, err := os.CreateTemp(tmpDir, "query-*")
	if err != nil {
		return "", err
	}
	if _, err := tmp.Write(head[:n]); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmp.Name())
		return "", err
	}
	if _, err := io.Copy(tmp, part); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmp.Name())
		return "", err
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmp.Name())
		return "", err
	}
	return tmp.Name(), nil
}

func isSupportedQueryImage(head []byte) bool {
	mime := http.DetectContentType(head)
	switch mime {
	case "image/jpeg", "image/png", "image/webp", "image/gif":
		return true
	}
	// AVIF is not sniffed by net/http; recognise the ISO BMFF brand.
	return len(head) >= 12 && string(head[4:8]) == "ftyp" && strings.HasPrefix(string(head[8:12]), "avif")
}
