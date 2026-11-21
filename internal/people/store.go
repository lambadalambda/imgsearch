// Package people groups faces across images and videos.
//
// A face is always detected on an image row; video membership is derived
// through video_frames, so faces never reference videos directly. person_id
// is nullable: detector output can sit in an unassigned pool before
// clustering assigns it. Deleting a person keeps its faces as unassigned;
// deleting media cascades to its faces, never the reverse.
package people

import (
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"errors"
	"fmt"

	"imgsearch/internal/images"
	"imgsearch/internal/nsfwsql"
	"imgsearch/internal/thumbnails"
	"imgsearch/internal/videos"
)

type Person struct {
	ID        int64  `json:"id"`
	UUID      string `json:"uuid"`
	Name      string `json:"name"`
	CreatedAt string `json:"created_at"`
}

type Face struct {
	ID         int64   `json:"id"`
	ImageID    int64   `json:"image_id"`
	PersonID   *int64  `json:"person_id"`
	BoxX       float64 `json:"box_x"`
	BoxY       float64 `json:"box_y"`
	BoxW       float64 `json:"box_w"`
	BoxH       float64 `json:"box_h"`
	Confidence float64 `json:"confidence"`
	CreatedAt  string  `json:"created_at"`
}

// FaceView is a Face joined with its source image for correction UIs.
// (Face.CreatedAt is the detection time; CapturedAt is the photo's
// capture/upload time.)
type FaceView struct {
	Face
	OriginalName  string `json:"original_name"`
	StoragePath   string `json:"storage_path"`
	ThumbnailPath string `json:"thumbnail_path,omitempty"`
	CapturedAt    string `json:"captured_at,omitempty"`
}

type Item struct {
	MediaType string            `json:"media_type"`
	Image     *images.ImageItem `json:"image,omitempty"`
	Video     *videos.VideoItem `json:"video,omitempty"`
}

type ItemPage struct {
	Items     []Item `json:"items"`
	Total     int64  `json:"total"`
	ItemCount int64  `json:"item_count"`
}

type PersonSummary struct {
	Person
	FaceCount int64         `json:"face_count"`
	Covers    []PersonCover `json:"covers"`
}

type PersonCover struct {
	ImageID       int64  `json:"image_id"`
	ThumbnailPath string `json:"thumbnail_path,omitempty"`
}

type target struct {
	kind string
	id   int64
}

func newUUID() (string, error) {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	return hex.EncodeToString(b[:]), nil
}

type rowQuerier interface {
	QueryRowContext(context.Context, string, ...any) *sql.Row
}

func getPerson(ctx context.Context, q rowQuerier, id int64) (Person, error) {
	var p Person
	err := q.QueryRowContext(ctx, `SELECT id,uuid,name,created_at FROM persons WHERE id=?`, id).Scan(&p.ID, &p.UUID, &p.Name, &p.CreatedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return p, problem(404, "person not found")
	}
	return p, err
}

func (h *Handler) list(ctx context.Context) ([]Person, error) {
	rows, err := h.DB.QueryContext(ctx, `SELECT id,uuid,name,created_at FROM persons ORDER BY id DESC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	result := []Person{}
	for rows.Next() {
		var p Person
		if err := rows.Scan(&p.ID, &p.UUID, &p.Name, &p.CreatedAt); err != nil {
			return nil, err
		}
		result = append(result, p)
	}
	return result, rows.Err()
}

func (h *Handler) summaries(ctx context.Context) ([]PersonSummary, error) {
	rows, err := h.DB.QueryContext(ctx, `
SELECT p.id,p.uuid,p.name,p.created_at,COUNT(f.id)
FROM persons p LEFT JOIN face_detections f ON f.person_id=p.id
GROUP BY p.id ORDER BY p.id DESC`)
	if err != nil {
		return nil, err
	}
	result := []PersonSummary{}
	for rows.Next() {
		var s PersonSummary
		if err := rows.Scan(&s.ID, &s.UUID, &s.Name, &s.CreatedAt, &s.FaceCount); err != nil {
			_ = rows.Close()
			return nil, err
		}
		s.Covers = []PersonCover{}
		result = append(result, s)
	}
	if err := rows.Err(); err != nil {
		_ = rows.Close()
		return nil, err
	}
	_ = rows.Close()
	if len(result) == 0 {
		return result, nil
	}
	// One bounded query for newest face-image covers across all persons.
	ids := make([]any, 0, len(result))
	for _, s := range result {
		ids = append(ids, s.ID)
	}
	ph := make([]string, len(ids))
	for i := range ph {
		ph[i] = "?"
	}
	coverRows, err := h.DB.QueryContext(ctx, `
SELECT f.person_id,i.id,i.thumbnail_path,f.id
FROM face_detections f JOIN images i ON i.id=f.image_id
WHERE f.person_id IN (`+joinPlaceholders(len(ids))+`)
ORDER BY f.person_id,f.id DESC`, ids...)
	_ = ph
	if err != nil {
		return nil, err
	}
	defer coverRows.Close()
	byPerson := make(map[int64][]PersonCover, len(result))
	for coverRows.Next() {
		var pid, imageID int64
		var thumb sql.NullString
		var fid int64
		if err := coverRows.Scan(&pid, &imageID, &thumb, &fid); err != nil {
			return nil, err
		}
		if len(byPerson[pid]) >= 4 {
			continue
		}
		cover := PersonCover{ImageID: imageID}
		if thumb.Valid {
			cover.ThumbnailPath = thumbnails.Path(imageID, thumb.String)
		}
		byPerson[pid] = append(byPerson[pid], cover)
	}
	if err := coverRows.Err(); err != nil {
		return nil, err
	}
	for i := range result {
		if covers, ok := byPerson[result[i].ID]; ok {
			result[i].Covers = covers
		}
	}
	return result, nil
}

func joinPlaceholders(n int) string {
	out := ""
	for i := 0; i < n; i++ {
		if i > 0 {
			out += ","
		}
		out += "?"
	}
	return out
}

// saveName creates (id <= 0) or renames a person. An empty name on create
// yields a neutral "Person <id>" default; the backend always keeps a UUID.
func (h *Handler) saveName(ctx context.Context, id int64, name string) (Person, error) {
	if id > 0 {
		var p Person
		err := h.DB.QueryRowContext(ctx, `UPDATE persons SET name=? WHERE id=? RETURNING id,uuid,name,created_at`, name, id).Scan(&p.ID, &p.UUID, &p.Name, &p.CreatedAt)
		if errors.Is(err, sql.ErrNoRows) {
			return p, problem(404, "person not found")
		}
		return p, err
	}
	uuid, err := newUUID()
	if err != nil {
		return Person{}, err
	}
	tx, err := h.DB.BeginTx(ctx, nil)
	if err != nil {
		return Person{}, err
	}
	defer tx.Rollback()
	var p Person
	if err := tx.QueryRowContext(ctx, `INSERT INTO persons(uuid,name) VALUES(?,'tmp') RETURNING id,uuid,name,created_at`, uuid).Scan(&p.ID, &p.UUID, &p.Name, &p.CreatedAt); err != nil {
		return Person{}, err
	}
	if name == "" {
		name = fmt.Sprintf("Person %d", p.ID)
	}
	if err := tx.QueryRowContext(ctx, `UPDATE persons SET name=? WHERE id=? RETURNING id,uuid,name,created_at`, name, p.ID).Scan(&p.ID, &p.UUID, &p.Name, &p.CreatedAt); err != nil {
		return Person{}, err
	}
	return p, tx.Commit()
}

func (h *Handler) delete(ctx context.Context, id int64) error {
	result, err := h.DB.ExecContext(ctx, `DELETE FROM persons WHERE id=?`, id)
	if err != nil {
		return err
	}
	n, err := result.RowsAffected()
	if err == nil && n == 0 {
		return problem(404, "person not found")
	}
	return err
}

// personVisibleSQL reuses the library NSFW predicates over the ranked media.
func personVisibleSQL(includeNSFW bool) string {
	if includeNSFW {
		return "1"
	}
	return fmt.Sprintf(`((r.mtype = 'image' AND NOT (%s)) OR (r.mtype = 'video' AND NOT (%s)))`,
		nsfwsql.TagsJSONHasNSFW("i.tags_json", "pitag"), nsfwsql.VideoHasNSFW("v.id", "v.tags_json", "pftag", "pvtag"))
}

// rankedMedia is the shared dedupe CTE: one row per distinct photo/video,
// newest face first. Frame faces surface as their video, like collections.
const rankedMediaCTE = `
WITH fac AS (
  SELECT f.id AS fid, f.image_id, vf.video_id
  FROM face_detections f
  JOIN images i ON i.id = f.image_id
  LEFT JOIN video_frames vf ON vf.image_id = f.image_id
  WHERE f.person_id = ?
),
ranked AS (
  SELECT
    CASE WHEN video_id IS NOT NULL THEN 'video' ELSE 'image' END AS mtype,
    COALESCE(video_id, image_id) AS mid,
    MAX(fid) AS last_fid,
    MAX(image_id) AS rep_image,
    MAX(video_id) AS rep_video
  FROM fac GROUP BY mtype, mid
)
`

func (h *Handler) items(ctx context.Context, id int64, limit, offset int, includeNSFW bool) (ItemPage, error) {
	page := ItemPage{Items: []Item{}}
	if _, err := getPerson(ctx, h.DB, id); err != nil {
		return page, err
	}
	visible := personVisibleSQL(includeNSFW)
	from := rankedMediaCTE + `SELECT COUNT(*),COALESCE(SUM(CASE WHEN ` + visible + ` THEN 1 ELSE 0 END),0)
FROM ranked r LEFT JOIN images i ON i.id=r.rep_image LEFT JOIN videos v ON v.id=r.rep_video`
	if err := h.DB.QueryRowContext(ctx, from, id).Scan(&page.ItemCount, &page.Total); err != nil {
		return page, err
	}
	rows, err := h.DB.QueryContext(ctx, rankedMediaCTE+`
SELECT r.mtype,r.mid FROM ranked r
LEFT JOIN images i ON i.id=r.rep_image
LEFT JOIN videos v ON v.id=r.rep_video
WHERE `+visible+` ORDER BY r.last_fid DESC LIMIT ? OFFSET ?`, id, limit, offset)
	if err != nil {
		return page, err
	}
	defer rows.Close()
	targets := make([]target, 0, limit)
	for rows.Next() {
		var t target
		if err := rows.Scan(&t.kind, &t.id); err != nil {
			return page, err
		}
		targets = append(targets, t)
	}
	if err := rows.Err(); err != nil {
		return page, err
	}
	if err := rows.Close(); err != nil {
		return page, err
	}
	for _, t := range targets {
		item := Item{MediaType: t.kind}
		var err error
		if t.kind == "image" {
			var value images.ImageItem
			value, err = images.GetItem(ctx, h.DB, h.ModelID, t.id)
			item.Image = &value
		} else {
			var value videos.VideoItem
			value, err = videos.GetItem(ctx, h.DB, h.ModelID, t.id)
			item.Video = &value
		}
		if errors.Is(err, sql.ErrNoRows) {
			continue // media was deleted concurrently
		}
		if err != nil {
			return page, err
		}
		page.Items = append(page.Items, item)
	}
	return page, nil
}

func (h *Handler) listFaces(ctx context.Context, personID *int64, unassignedOnly bool, limit, offset int) ([]FaceView, error) {
	query := `
SELECT f.id,f.image_id,f.person_id,f.box_x,f.box_y,f.box_w,f.box_h,f.confidence,f.created_at,
  i.original_name,i.storage_path,i.thumbnail_path,COALESCE(i.captured_at,i.created_at,'')
FROM face_detections f JOIN images i ON i.id=f.image_id`
	var args []any
	switch {
	case unassignedOnly:
		query += ` WHERE f.person_id IS NULL`
	case personID != nil:
		query += ` WHERE f.person_id = ?`
		args = append(args, *personID)
	}
	query += ` ORDER BY f.id DESC LIMIT ? OFFSET ?`
	args = append(args, limit, offset)
	rows, err := h.DB.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	result := []FaceView{}
	for rows.Next() {
		var v FaceView
		var thumb sql.NullString
		if err := rows.Scan(&v.ID, &v.ImageID, &v.PersonID, &v.BoxX, &v.BoxY, &v.BoxW, &v.BoxH, &v.Confidence, &v.CreatedAt,
			&v.OriginalName, &v.StoragePath, &thumb, &v.CapturedAt); err != nil {
			return nil, err
		}
		if thumb.Valid {
			v.ThumbnailPath = thumbnails.Path(v.ImageID, thumb.String)
		}
		result = append(result, v)
	}
	return result, rows.Err()
}

func (h *Handler) reassignFace(ctx context.Context, faceID int64, personID *int64) (Face, *int64, error) {
	var f Face
	var from *int64
	tx, err := h.DB.BeginTx(ctx, nil)
	if err != nil {
		return f, nil, err
	}
	defer tx.Rollback()
	if err := tx.QueryRowContext(ctx, `SELECT id,image_id,person_id,box_x,box_y,box_w,box_h,confidence,created_at FROM face_detections WHERE id=?`, faceID).
		Scan(&f.ID, &f.ImageID, &f.PersonID, &f.BoxX, &f.BoxY, &f.BoxW, &f.BoxH, &f.Confidence, &f.CreatedAt); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return f, nil, problem(404, "face not found")
		}
		return f, nil, err
	}
	from = f.PersonID
	if personID != nil {
		if _, err := getPerson(ctx, tx, *personID); err != nil {
			return f, nil, err
		}
	}
	if _, err := tx.ExecContext(ctx, `UPDATE face_detections SET person_id=? WHERE id=?`, personID, faceID); err != nil {
		return f, nil, err
	}
	f.PersonID = personID
	deleted, err := deleteIfEmpty(ctx, tx, from)
	if err != nil {
		return f, nil, err
	}
	return f, deleted, tx.Commit()
}

// deleteIfEmpty removes a person left with no faces (e.g. its last face was
// moved away) and reports its id. Manually created empty people are only
// removed by an explicit DELETE: this runs on face-removal operations.
func deleteIfEmpty(ctx context.Context, tx *sql.Tx, personID *int64) (*int64, error) {
	if personID == nil {
		return nil, nil
	}
	var remaining int64
	if err := tx.QueryRowContext(ctx, `SELECT COUNT(*) FROM face_detections WHERE person_id=?`, *personID).Scan(&remaining); err != nil {
		return nil, err
	}
	if remaining > 0 {
		return nil, nil
	}
	if _, err := tx.ExecContext(ctx, `DELETE FROM persons WHERE id=?`, *personID); err != nil {
		return nil, err
	}
	return personID, nil
}

func (h *Handler) merge(ctx context.Context, fromIDs []int64, intoID int64) error {
	seen := map[int64]bool{}
	deduped := make([]int64, 0, len(fromIDs))
	for _, id := range fromIDs {
		if id == intoID {
			return problem(400, "cannot merge a person into itself")
		}
		if !seen[id] {
			seen[id] = true
			deduped = append(deduped, id)
		}
	}
	if len(deduped) == 0 {
		return problem(400, "from_ids must not be empty")
	}
	tx, err := h.DB.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	if _, err := getPerson(ctx, tx, intoID); err != nil {
		return err
	}
	for _, id := range deduped {
		if _, err := getPerson(ctx, tx, id); err != nil {
			return err
		}
	}
	for _, id := range deduped {
		if _, err := tx.ExecContext(ctx, `UPDATE face_detections SET person_id=? WHERE person_id=?`, intoID, id); err != nil {
			return err
		}
		if _, err := tx.ExecContext(ctx, `DELETE FROM persons WHERE id=?`, id); err != nil {
			return err
		}
	}
	return tx.Commit()
}

func (h *Handler) split(ctx context.Context, id int64, faceIDs []int64, name string) (Person, *int64, error) {
	if len(faceIDs) == 0 {
		return Person{}, nil, problem(400, "face_ids must not be empty")
	}
	seen := map[int64]bool{}
	for _, fid := range faceIDs {
		if seen[fid] {
			return Person{}, nil, problem(400, "duplicate face_ids")
		}
		seen[fid] = true
	}
	tx, err := h.DB.BeginTx(ctx, nil)
	if err != nil {
		return Person{}, nil, err
	}
	defer tx.Rollback()
	if _, err := getPerson(ctx, tx, id); err != nil {
		return Person{}, nil, err
	}
	for _, fid := range faceIDs {
		var owner sql.NullInt64
		if err := tx.QueryRowContext(ctx, `SELECT person_id FROM face_detections WHERE id=?`, fid).Scan(&owner); err != nil {
			if errors.Is(err, sql.ErrNoRows) {
				return Person{}, nil, problem(404, "face not found")
			}
			return Person{}, nil, err
		}
		if !owner.Valid || owner.Int64 != id {
			return Person{}, nil, problem(409, "face does not belong to this person")
		}
	}
	uuid, err := newUUID()
	if err != nil {
		return Person{}, nil, err
	}
	var p Person
	if err := tx.QueryRowContext(ctx, `INSERT INTO persons(uuid,name) VALUES(?,'tmp') RETURNING id,uuid,name,created_at`, uuid).Scan(&p.ID, &p.UUID, &p.Name, &p.CreatedAt); err != nil {
		return Person{}, nil, err
	}
	if name == "" {
		name = fmt.Sprintf("Person %d", p.ID)
	}
	if err := tx.QueryRowContext(ctx, `UPDATE persons SET name=? WHERE id=? RETURNING id,uuid,name,created_at`, name, p.ID).Scan(&p.ID, &p.UUID, &p.Name, &p.CreatedAt); err != nil {
		return Person{}, nil, err
	}
	for _, fid := range faceIDs {
		if _, err := tx.ExecContext(ctx, `UPDATE face_detections SET person_id=? WHERE id=?`, p.ID, fid); err != nil {
			return Person{}, nil, err
		}
	}
	deleted, err := deleteIfEmpty(ctx, tx, &id)
	if err != nil {
		return Person{}, nil, err
	}
	return p, deleted, tx.Commit()
}
