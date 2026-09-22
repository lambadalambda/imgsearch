package images

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strings"

	"imgsearch/internal/httputil"
	"imgsearch/internal/tagutil"
)

// Manual cleanup is deliberately separate from unattended import. It only
// accepts a complete, currently visible dHash group and never unlinks files.
type consolidationRequest struct {
	KeeperID     int64   `json:"keeper_id"`
	GroupIDs     []int64 `json:"group_ids"`
	Distance     int     `json:"distance"`
	IncludeNSFW  bool    `json:"include_nsfw"`
	Confirm      bool    `json:"confirm"`
	PreviewToken string  `json:"preview_token"`
}

type consolidationPreview struct {
	KeeperID            int64    `json:"keeper_id"`
	RemovedIDs          []int64  `json:"removed_ids"`
	AffectedCollections int      `json:"affected_collections"`
	FilePolicy          string   `json:"file_policy"`
	Conflicts           []string `json:"conflicts"`
	PreviewToken        string   `json:"preview_token"`
}

type duplicateMetadata struct {
	ID                                          int64
	Title, UserTags, RemovedTags, AnnotatorTags string
	// Include immutable content and visible identity in the freshness check too.
	Name, Digest  string
	Hash          int64
	Width, Height int
}

type duplicateMembership struct{ ID, CollectionID, ImageID int64 }
type mergedDuplicateMetadata struct {
	title                    string
	user, removed, annotator []string
}

var errDuplicateSelection = errors.New("possible-match group changed or is invalid; reload and review the complete group")
var errDuplicatePreview = errors.New("preview changed or metadata conflicts remain; review again before consolidating")

func (h *Handler) consolidateDuplicates(w http.ResponseWriter, r *http.Request) {
	var req consolidationRequest
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 32<<10))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&req); err != nil {
		httputil.WriteJSONError(w, http.StatusBadRequest, "invalid consolidation request")
		return
	}
	if err := dec.Decode(new(any)); err != io.EOF {
		httputil.WriteJSONError(w, http.StatusBadRequest, "unexpected trailing data")
		return
	}
	slices.Sort(req.GroupIDs)
	if len(req.GroupIDs) < 2 || len(req.GroupIDs) > 500 || req.Distance < 0 || req.Distance > MaxDuplicateDistance || !slices.Contains(req.GroupIDs, req.KeeperID) {
		httputil.WriteJSONError(w, http.StatusBadRequest, "select a keeper in a group of 2–500 images and a distance of 0–7")
		return
	}
	for i, id := range req.GroupIDs {
		if id <= 0 || (i > 0 && id == req.GroupIDs[i-1]) {
			httputil.WriteJSONError(w, http.StatusBadRequest, "group IDs must be distinct positive integers")
			return
		}
	}
	preview, err := consolidateDuplicateGroup(r.Context(), h.DB, req)
	if err != nil {
		if errors.Is(err, errDuplicateSelection) || errors.Is(err, errDuplicatePreview) {
			httputil.WriteJSONError(w, http.StatusConflict, err.Error())
			return
		}
		httputil.WriteJSONError(w, http.StatusInternalServerError, "consolidation failed; no library changes were saved")
		return
	}
	httputil.WriteJSON(w, http.StatusOK, preview)
}

func consolidateDuplicateGroup(ctx context.Context, db *sql.DB, req consolidationRequest) (consolidationPreview, error) {
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return consolidationPreview{}, err
	}
	defer tx.Rollback()
	if req.Confirm {
		// Reserve SQLite's writer before reading. Other imports/metadata changes
		// cannot interleave between this validation and commit (including WAL).
		if _, err = tx.ExecContext(ctx, `UPDATE images SET id=id WHERE id=?`, req.KeeperID); err != nil {
			return consolidationPreview{}, err
		}
	}
	hashed, _, err := loadDuplicateHashes(ctx, tx, req.IncludeNSFW)
	if err != nil {
		return consolidationPreview{}, err
	}
	valid := false
	for _, group := range groupByHash(hashed, req.Distance) {
		if slices.Equal(group, req.GroupIDs) {
			valid = true
			break
		}
	}
	if !valid {
		return consolidationPreview{}, errDuplicateSelection
	}

	metadata := make([]duplicateMetadata, 0, len(req.GroupIDs))
	memberships := []duplicateMembership{}
	for _, id := range req.GroupIDs {
		m := duplicateMetadata{ID: id}
		if err = tx.QueryRowContext(ctx, `SELECT user_title,user_tags_json,removed_tags_json,annotator_tags_json,original_name,sha256,phash,width,height FROM images WHERE id=?`, id).Scan(&m.Title, &m.UserTags, &m.RemovedTags, &m.AnnotatorTags, &m.Name, &m.Digest, &m.Hash, &m.Width, &m.Height); err != nil {
			return consolidationPreview{}, err
		}
		metadata = append(metadata, m)
		rows, err := tx.QueryContext(ctx, `SELECT id,collection_id,image_id FROM collection_items WHERE image_id=? ORDER BY id`, id)
		if err != nil {
			return consolidationPreview{}, err
		}
		for rows.Next() {
			var membership duplicateMembership
			if err = rows.Scan(&membership.ID, &membership.CollectionID, &membership.ImageID); err != nil {
				rows.Close()
				return consolidationPreview{}, err
			}
			memberships = append(memberships, membership)
		}
		err = rows.Err()
		rows.Close()
		if err != nil {
			return consolidationPreview{}, err
		}
	}
	merged, conflicts := mergeDuplicateMetadata(metadata, req.KeeperID)
	preview := consolidationPreview{KeeperID: req.KeeperID, RemovedIDs: []int64{}, FilePolicy: "retained", Conflicts: conflicts}
	affected := map[int64]bool{}
	for _, id := range req.GroupIDs {
		if id != req.KeeperID {
			preview.RemovedIDs = append(preview.RemovedIDs, id)
		}
	}
	for _, m := range memberships {
		if m.ImageID != req.KeeperID {
			affected[m.CollectionID] = true
		}
	}
	preview.AffectedCollections = len(affected)
	// Stateless optimistic preview: a freshness checksum, not an authorization
	// token. HTTP auth is unchanged. All inputs are revalidated on confirmation.
	snapshot, _ := json.Marshal(struct {
		Keeper      int64
		Distance    int
		NSFW        bool
		Metadata    []duplicateMetadata
		Memberships []duplicateMembership
	}{req.KeeperID, req.Distance, req.IncludeNSFW, metadata, memberships})
	preview.PreviewToken = fmt.Sprintf("%x", sha256.Sum256(snapshot))
	if !req.Confirm {
		return preview, nil
	}
	if len(conflicts) > 0 || req.PreviewToken != preview.PreviewToken {
		return consolidationPreview{}, errDuplicatePreview
	}

	if _, err = tx.ExecContext(ctx, `UPDATE images SET user_title=?, title=CASE WHEN ?='' THEN title ELSE ? END, user_tags_json=?, removed_tags_json=?, tags_json=? WHERE id=?`, merged.title, merged.title, merged.title, tagutil.EncodeJSON(merged.user), tagutil.EncodeJSON(merged.removed), tagutil.EncodeJSON(tagutil.Merge(merged.annotator, merged.user, merged.removed)), req.KeeperID); err != nil {
		return consolidationPreview{}, err
	}
	// Prefer the keeper's existing membership; otherwise retain the earliest
	// membership ID (collection order). Remove overlaps before retargeting.
	slices.SortFunc(memberships, func(a, b duplicateMembership) int {
		if a.ID < b.ID {
			return -1
		}
		if a.ID > b.ID {
			return 1
		}
		return 0
	})
	kept := map[int64]bool{}
	for _, m := range memberships {
		if m.ImageID == req.KeeperID {
			kept[m.CollectionID] = true
		}
	}
	for _, m := range memberships {
		if m.ImageID == req.KeeperID {
			continue
		}
		if kept[m.CollectionID] {
			_, err = tx.ExecContext(ctx, `DELETE FROM collection_items WHERE id=?`, m.ID)
		} else {
			_, err = tx.ExecContext(ctx, `UPDATE collection_items SET image_id=? WHERE id=?`, req.KeeperID, m.ID)
			kept[m.CollectionID] = true
		}
		if err != nil {
			return consolidationPreview{}, err
		}
	}
	for _, id := range preview.RemovedIDs {
		for _, table := range []string{"image_embeddings", "index_jobs", "images"} {
			column := "image_id"
			if table == "images" {
				column = "id"
			}
			if _, err = tx.ExecContext(ctx, `DELETE FROM `+table+` WHERE `+column+`=?`, id); err != nil {
				return consolidationPreview{}, err
			}
		}
	}
	if err = tx.Commit(); err != nil {
		return consolidationPreview{}, err
	}
	// Keep originals and thumbnails: digest paths may be reused by concurrent
	// uploads. Retention avoids unlink races and permits out-of-band recovery.
	return preview, nil
}

// Stable ascending image-ID order determines tag spelling/order. Distinct
// nonempty manual titles and tag add/remove disagreements block cleanup;
// nothing silently wins. Generated annotations remain those of the keeper.
func mergeDuplicateMetadata(items []duplicateMetadata, keeper int64) (mergedDuplicateMetadata, []string) {
	merged := mergedDuplicateMetadata{}
	conflicts := []string{}
	titles := []string{}
	for _, m := range items {
		title := strings.TrimSpace(m.Title)
		if title != "" && !slices.Contains(titles, title) {
			titles = append(titles, title)
		}
		user, e1 := tagutil.DecodeJSON(m.UserTags)
		removed, e2 := tagutil.DecodeJSON(m.RemovedTags)
		annotator, e3 := tagutil.DecodeJSON(m.AnnotatorTags)
		if e1 != nil || e2 != nil || e3 != nil {
			conflicts = append(conflicts, fmt.Sprintf("Image %d has invalid tag metadata; repair it before cleanup.", m.ID))
			continue
		}
		merged.user = append(merged.user, user...)
		merged.removed = append(merged.removed, removed...)
		if m.ID == keeper {
			merged.annotator = annotator
		}
	}
	if len(titles) > 1 {
		conflicts = append(conflicts, "Conflicting manual titles: "+strings.Join(titles, " / ")+". Edit these images to agree before cleanup.")
	} else if len(titles) == 1 {
		merged.title = titles[0]
	}
	merged.user = tagutil.Normalize(merged.user)
	merged.removed = tagutil.Normalize(merged.removed)
	for _, user := range merged.user {
		for _, removed := range merged.removed {
			if strings.EqualFold(user, removed) {
				conflicts = append(conflicts, fmt.Sprintf("Tag %q is both manually added and removed. Edit these images to agree before cleanup.", user))
				break
			}
		}
	}
	return merged, conflicts
}
