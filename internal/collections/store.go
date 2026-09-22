// Package collections stores lists of references to existing images and videos.
package collections

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"imgsearch/internal/images"
	"imgsearch/internal/nsfwsql"
	"imgsearch/internal/videos"
)

type Collection struct {
	ID        int64  `json:"id"`
	Name      string `json:"name"`
	CreatedAt string `json:"created_at"`
}

type Item struct {
	MediaType string            `json:"media_type"`
	Image     *images.ImageItem `json:"image,omitempty"`
	Video     *videos.VideoItem `json:"video,omitempty"`
}

type ItemPage struct {
	Items []Item `json:"items"`
	Total int64  `json:"total"`
}

type target struct {
	kind string
	id   int64
}

// column and table return SQL identifiers from a validated, fixed allow-list.
func (t target) column() string {
	if t.kind == "image" {
		return "image_id"
	}
	return "video_id"
}
func (t target) table() string {
	if t.kind == "image" {
		return "images"
	}
	return "videos"
}

type rowQuerier interface {
	QueryRowContext(context.Context, string, ...any) *sql.Row
}

func getCollection(ctx context.Context, q rowQuerier, id int64) (Collection, error) {
	var c Collection
	err := q.QueryRowContext(ctx, `SELECT id,name,created_at FROM collections WHERE id=?`, id).Scan(&c.ID, &c.Name, &c.CreatedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return c, problem(404, "collection not found")
	}
	return c, err
}

func validateTarget(ctx context.Context, q rowQuerier, t target) error {
	var exists bool
	if err := q.QueryRowContext(ctx, `SELECT EXISTS(SELECT 1 FROM `+t.table()+` WHERE id=?)`, t.id).Scan(&exists); err != nil {
		return err
	}
	if !exists {
		return problem(404, "media not found")
	}
	if t.kind == "image" {
		var frame bool
		if err := q.QueryRowContext(ctx, `SELECT EXISTS(SELECT 1 FROM video_frames WHERE image_id=?)`, t.id).Scan(&frame); err != nil {
			return err
		}
		if frame {
			return problem(409, "save the video, not its derived frame")
		}
	}
	return nil
}

func (h *Handler) list(ctx context.Context, t *target) ([]Collection, error) {
	query := `SELECT c.id,c.name,c.created_at FROM collections c`
	var args []any
	if t != nil {
		if err := validateTarget(ctx, h.DB, *t); err != nil {
			return nil, err
		}
		query += ` WHERE EXISTS(SELECT 1 FROM collection_items m WHERE m.collection_id=c.id AND m.` + t.column() + `=?)`
		args = append(args, t.id)
	}
	rows, err := h.DB.QueryContext(ctx, query+` ORDER BY c.id DESC`, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	result := []Collection{}
	for rows.Next() {
		var c Collection
		if err := rows.Scan(&c.ID, &c.Name, &c.CreatedAt); err != nil {
			return nil, err
		}
		result = append(result, c)
	}
	return result, rows.Err()
}

func (h *Handler) saveName(ctx context.Context, id int64, name string) (Collection, error) {
	var c Collection
	query := `INSERT INTO collections(name) VALUES(?) RETURNING id,name,created_at`
	args := []any{name}
	if id > 0 {
		query = `UPDATE collections SET name=? WHERE id=? RETURNING id,name,created_at`
		args = append(args, id)
	}
	err := h.DB.QueryRowContext(ctx, query, args...).Scan(&c.ID, &c.Name, &c.CreatedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return c, problem(404, "collection not found")
	}
	return c, err
}

func (h *Handler) delete(ctx context.Context, id int64) error {
	result, err := h.DB.ExecContext(ctx, `DELETE FROM collections WHERE id=?`, id)
	if err != nil {
		return err
	}
	n, err := result.RowsAffected()
	if err == nil && n == 0 {
		return problem(404, "collection not found")
	}
	return err
}

func (h *Handler) membership(ctx context.Context, id int64, t target, add bool) error {
	tx, err := h.DB.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	if _, err := getCollection(ctx, tx, id); err != nil {
		return err
	}
	if add {
		if err := validateTarget(ctx, tx, t); err != nil {
			return err
		}
		// A repeated save must not reorder the item or create another membership.
		_, err = tx.ExecContext(ctx, `INSERT INTO collection_items(collection_id,`+t.column()+`) VALUES(?,?) ON CONFLICT DO NOTHING`, id, t.id)
	} else {
		_, err = tx.ExecContext(ctx, `DELETE FROM collection_items WHERE collection_id=? AND `+t.column()+`=?`, id, t.id)
	}
	if err != nil {
		return err
	}
	return tx.Commit()
}

func (h *Handler) items(ctx context.Context, id int64, limit, offset int, includeNSFW bool) (ItemPage, error) {
	page := ItemPage{Items: []Item{}}
	if _, err := getCollection(ctx, h.DB, id); err != nil {
		return page, err
	}
	// Apply visibility before LIMIT/OFFSET and COUNT, using the same predicates as
	// the library (including NSFW tags on a video's sampled frames).
	from := ` FROM collection_items m
 LEFT JOIN images i ON i.id=m.image_id
 LEFT JOIN videos v ON v.id=m.video_id
 WHERE m.collection_id=?`
	if !includeNSFW {
		from += fmt.Sprintf(` AND ((m.image_id IS NOT NULL AND NOT (%s)) OR (m.video_id IS NOT NULL AND NOT (%s)))`,
			nsfwsql.TagsJSONHasNSFW("i.tags_json", "itag"), nsfwsql.VideoHasNSFW("v.id", "v.tags_json", "ftag", "vtag"))
	}
	if err := h.DB.QueryRowContext(ctx, `SELECT COUNT(*)`+from, id).Scan(&page.Total); err != nil {
		return page, err
	}
	rows, err := h.DB.QueryContext(ctx, `SELECT m.image_id,m.video_id`+from+` ORDER BY m.id DESC LIMIT ? OFFSET ?`, id, limit, offset)
	if err != nil {
		return page, err
	}
	defer rows.Close()
	targets := make([]target, 0, limit)
	for rows.Next() {
		var imageID, videoID sql.NullInt64
		if err := rows.Scan(&imageID, &videoID); err != nil {
			return page, err
		}
		t := target{kind: "image", id: imageID.Int64}
		if videoID.Valid {
			t = target{kind: "video", id: videoID.Int64}
		}
		targets = append(targets, t)
	}
	if err := rows.Err(); err != nil {
		return page, err
	}
	// Release the connection before hydrating; the app may use a one-connection DB.
	if err := rows.Close(); err != nil {
		return page, err
	}
	for _, t := range targets {
		item := Item{MediaType: t.kind}
		if t.kind == "image" {
			value, getErr := images.GetItem(ctx, h.DB, h.ModelID, t.id)
			item.Image, err = &value, getErr
		} else {
			value, getErr := videos.GetItem(ctx, h.DB, h.ModelID, t.id)
			item.Video, err = &value, getErr
		}
		if errors.Is(err, sql.ErrNoRows) {
			continue
		} // media was deleted concurrently
		if err != nil {
			return page, err
		}
		page.Items = append(page.Items, item)
	}
	return page, nil
}
