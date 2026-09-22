// Package thumbnails lazily generates persistent grid derivatives. A stable URL
// is available before generation, so existing libraries need no eager backfill.
package thumbnails

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"imgsearch/internal/imageutil"
)

const URLPrefix = "/media/images/thumbnails/"
const maxSide = 640

// Path preserves explicitly stored derivatives, otherwise returning the lazy URL
// path relative to /media/. API reads never generate files or write metadata.
func Path(imageID int64, stored string) string {
	if stored = strings.TrimSpace(stored); stored != "" {
		return filepath.ToSlash(stored)
	}
	return fmt.Sprintf("images/thumbnails/%d.webp", imageID)
}

type handler struct {
	db       *sql.DB
	dataDir  string
	generate func(source, destination string, maxSide int) error
	gate     chan struct{}
}

func NewHandler(db *sql.DB, dataDir string) http.Handler {
	return &handler{db: db, dataDir: dataDir, generate: imageutil.ThumbnailWebP, gate: make(chan struct{}, 1)}
}

func (h *handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		w.Header().Set("Allow", "GET, HEAD")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	name := strings.TrimPrefix(r.URL.Path, URLPrefix)
	id, err := strconv.ParseInt(strings.TrimSuffix(name, ".webp"), 10, 64)
	if err != nil || id <= 0 || r.URL.Path != URLPrefix+strconv.FormatInt(id, 10)+".webp" {
		http.NotFound(w, r)
		return
	}
	filename, err := h.ensure(r.Context(), id)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) || errors.Is(err, os.ErrNotExist) {
			http.NotFound(w, r)
		} else {
			http.Error(w, "thumbnail unavailable", http.StatusInternalServerError)
		}
		return
	}
	file, err := os.Open(filename)
	if err != nil {
		http.NotFound(w, r)
		return
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		http.Error(w, "thumbnail unavailable", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "image/webp")
	// ID-only URLs must revalidate: a database reset can reuse an image ID.
	// Subsecond precision also distinguishes regenerated files within one second.
	w.Header().Set("Cache-Control", "private, no-cache")
	w.Header().Set("ETag", fmt.Sprintf(`W/"%x-%x"`, info.ModTime().UnixNano(), info.Size()))
	http.ServeContent(w, r, info.Name(), info.ModTime(), file)
}

func (h *handler) ensure(ctx context.Context, id int64) (string, error) {
	// Cheap cache hits do not wait for another image's resize. Always check the
	// row first so a deleted image cannot be served from an orphan cache file.
	source, cached, err := h.lookup(ctx, id)
	if err != nil {
		return "", err
	}
	if cached != "" {
		return cached, nil
	}
	// Bound simultaneous native resizes and coalesce misses for the same image.
	// Waiting callers can cancel; no DB transaction/connection is held in vips.
	select {
	case h.gate <- struct{}{}:
	case <-ctx.Done():
		return "", ctx.Err()
	}
	defer func() { <-h.gate }()
	source, cached, err = h.lookup(ctx, id)
	if err != nil {
		return "", err
	}
	if cached != "" {
		return cached, nil
	}
	rel := Path(id, "")
	imagesDir, err := existingPath(h.dataDir, "images")
	if err != nil {
		return "", err
	}
	if err := os.Mkdir(filepath.Join(imagesDir, "thumbnails"), 0755); err != nil && !errors.Is(err, os.ErrExist) {
		return "", err
	}
	dir, err := existingPath(h.dataDir, "images/thumbnails")
	if err != nil {
		return "", err
	}
	tmp, err := os.CreateTemp(dir, ".thumbnail-*.webp")
	if err != nil {
		return "", err
	}
	tempPath := tmp.Name()
	defer os.Remove(tempPath)
	if err := tmp.Close(); err != nil {
		return "", err
	}
	if err := h.generate(source, tempPath, maxSide); err != nil {
		return "", err
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	filename := filepath.Join(dir, strconv.FormatInt(id, 10)+".webp")
	if err := os.Rename(tempPath, filename); err != nil {
		return "", err
	}
	// Publish the complete file before recording it. If deletion raced generation,
	// the zero-row update removes the orphan; later deletions see thumbnail_path.
	result, err := h.db.ExecContext(ctx, `UPDATE images SET thumbnail_path=? WHERE id=?`, rel, id)
	if err == nil {
		var count int64
		count, err = result.RowsAffected()
		if err == nil && count == 0 {
			err = sql.ErrNoRows
		}
	}
	if err != nil {
		_ = os.Remove(filename)
		return "", err
	}
	return filename, nil
}

// A cache hit must be recorded in this database. Merely finding an ID-named
// file could serve a previous library's thumbnail after a database reset.
func (h *handler) lookup(ctx context.Context, id int64) (source, cached string, err error) {
	var stored, thumb string
	if err := h.db.QueryRowContext(ctx, `SELECT storage_path, COALESCE(thumbnail_path, '') FROM images WHERE id=?`, id).Scan(&stored, &thumb); err != nil {
		return "", "", err
	}
	if !strings.HasPrefix(stored, "images/") {
		return "", "", os.ErrNotExist
	}
	source, err = existingPath(h.dataDir, stored)
	if err != nil {
		return "", "", err
	}
	info, err := os.Stat(source)
	if err != nil {
		return "", "", err
	}
	if !info.Mode().IsRegular() {
		return "", "", os.ErrNotExist
	}
	if thumb == Path(id, "") {
		if filename, err := existingPath(h.dataDir, thumb); err == nil {
			if info, err := os.Stat(filename); err == nil && info.Mode().IsRegular() {
				cached = filename
			}
		}
	}
	return source, cached, nil
}

// Resolve both source and cache paths beneath dataDir, including symlinks.
// The filesystem itself is trusted (as for the existing original-media server).
func existingPath(dataDir, rel string) (string, error) {
	if !filepath.IsLocal(rel) {
		return "", os.ErrNotExist
	}
	base, err := filepath.EvalSymlinks(dataDir)
	if err != nil {
		return "", err
	}
	base, err = filepath.Abs(base)
	if err != nil {
		return "", err
	}
	target, err := filepath.EvalSymlinks(filepath.Join(base, filepath.FromSlash(rel)))
	if err != nil {
		return "", err
	}
	local, err := filepath.Rel(base, target)
	if err != nil || local == "." || !filepath.IsLocal(local) {
		return "", os.ErrNotExist
	}
	return target, nil
}
