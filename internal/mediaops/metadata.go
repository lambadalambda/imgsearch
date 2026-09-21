package mediaops

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"unicode/utf8"

	"imgsearch/internal/tagutil"
)

// Table names the media table a metadata operation targets. Only the two
// constants are accepted: the value is interpolated into SQL.
type Table string

const (
	TableImages Table = "images"
	TableVideos Table = "videos"
)

const (
	MaxTitleLength = 300
	MaxTagLength   = 64
	MaxTags        = 100
)

// ErrInvalidPatch reports a malformed or out-of-range metadata patch.
var ErrInvalidPatch = errors.New("invalid metadata patch")

type execQuerier interface {
	ExecContext(ctx context.Context, query string, args ...any) (sql.Result, error)
	QueryRowContext(ctx context.Context, query string, args ...any) *sql.Row
}

func (t Table) validate() error {
	switch t {
	case TableImages, TableVideos:
		return nil
	default:
		return fmt.Errorf("mediaops: unknown table %q", string(t))
	}
}

// ApplyAnnotatorTags records a fresh annotator tag list and rewrites the
// served tags_json as annotator minus the user's removals plus the user's
// additions, so re-annotation never clobbers manual edits.
func ApplyAnnotatorTags(ctx context.Context, q execQuerier, table Table, id int64, annotator []string) ([]string, error) {
	if err := table.validate(); err != nil {
		return nil, err
	}
	var userJSON, removedJSON string
	if err := q.QueryRowContext(ctx, fmt.Sprintf(`SELECT COALESCE(user_tags_json, '[]'), COALESCE(removed_tags_json, '[]') FROM %s WHERE id = ?`, table), id).Scan(&userJSON, &removedJSON); err != nil {
		return nil, err
	}
	user, err := tagutil.DecodeJSON(userJSON)
	if err != nil {
		return nil, fmt.Errorf("decode user tags: %w", err)
	}
	removed, err := tagutil.DecodeJSON(removedJSON)
	if err != nil {
		return nil, fmt.Errorf("decode removed tags: %w", err)
	}
	served := tagutil.Merge(annotator, user, removed)
	if _, err := q.ExecContext(ctx, fmt.Sprintf(`UPDATE %s SET annotator_tags_json = ?, tags_json = ? WHERE id = ?`, table),
		tagutil.EncodeJSON(tagutil.Normalize(annotator)), tagutil.EncodeJSON(served), id); err != nil {
		return nil, fmt.Errorf("update %s tags: %w", table, err)
	}
	return served, nil
}

// SetServedTags stores a user-chosen full tag list, deriving the user's
// additions and removals against the annotator's tags. It returns the
// normalized list that is now served.
func SetServedTags(ctx context.Context, q execQuerier, table Table, id int64, served []string) ([]string, error) {
	if err := table.validate(); err != nil {
		return nil, err
	}
	var annotatorJSON string
	if err := q.QueryRowContext(ctx, fmt.Sprintf(`SELECT COALESCE(annotator_tags_json, '[]') FROM %s WHERE id = ?`, table), id).Scan(&annotatorJSON); err != nil {
		return nil, err
	}
	annotator, err := tagutil.DecodeJSON(annotatorJSON)
	if err != nil {
		return nil, fmt.Errorf("decode annotator tags: %w", err)
	}
	served = tagutil.Normalize(served)
	user, removed := tagutil.Diff(annotator, served)
	if _, err := q.ExecContext(ctx, fmt.Sprintf(`UPDATE %s SET tags_json = ?, user_tags_json = ?, removed_tags_json = ? WHERE id = ?`, table),
		tagutil.EncodeJSON(served), tagutil.EncodeJSON(user), tagutil.EncodeJSON(removed), id); err != nil {
		return nil, fmt.Errorf("update %s tags: %w", table, err)
	}
	return served, nil
}

// SetUserTitle stores a manual title. A non-empty title replaces the served
// title and survives re-annotation; an empty one clears the override and
// keeps the current title until the next annotation.
func SetUserTitle(ctx context.Context, q execQuerier, table Table, id int64, title string) error {
	if err := table.validate(); err != nil {
		return err
	}
	title = strings.TrimSpace(title)
	res, err := q.ExecContext(ctx, fmt.Sprintf(`UPDATE %s SET user_title = ?, title = CASE WHEN ? = '' THEN title ELSE ? END WHERE id = ?`, table), title, title, title, id)
	if err != nil {
		return fmt.Errorf("update %s title: %w", table, err)
	}
	rows, err := res.RowsAffected()
	if err != nil {
		return err
	}
	if rows == 0 {
		return sql.ErrNoRows
	}
	return nil
}

// AnnotatorTitleSQL is the assignment the worker uses so a manual title is
// not overwritten by a fresh annotation. It consumes one argument.
const AnnotatorTitleSQL = "title = CASE WHEN COALESCE(user_title, '') = '' THEN ? ELSE user_title END"

// MetadataPatch is the body of PATCH /api/{images,videos}/{id}. Absent
// fields are left unchanged; tags replace the whole served list.
type MetadataPatch struct {
	Title *string   `json:"title"`
	Tags  *[]string `json:"tags"`
}

// DecodeMetadataPatch parses and validates a PATCH body.
func DecodeMetadataPatch(r io.Reader) (MetadataPatch, error) {
	var patch MetadataPatch
	dec := json.NewDecoder(io.LimitReader(r, 64<<10))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&patch); err != nil {
		return MetadataPatch{}, fmt.Errorf("%w: %v", ErrInvalidPatch, err)
	}
	if patch.Title == nil && patch.Tags == nil {
		return MetadataPatch{}, fmt.Errorf("%w: nothing to update", ErrInvalidPatch)
	}
	if patch.Title != nil {
		trimmed := strings.TrimSpace(*patch.Title)
		if utf8.RuneCountInString(trimmed) > MaxTitleLength {
			return MetadataPatch{}, fmt.Errorf("%w: title longer than %d characters", ErrInvalidPatch, MaxTitleLength)
		}
		patch.Title = &trimmed
	}
	if patch.Tags != nil {
		normalized := tagutil.Normalize(*patch.Tags)
		if len(normalized) > MaxTags {
			return MetadataPatch{}, fmt.Errorf("%w: more than %d tags", ErrInvalidPatch, MaxTags)
		}
		for _, tag := range normalized {
			if utf8.RuneCountInString(tag) > MaxTagLength {
				return MetadataPatch{}, fmt.Errorf("%w: tag %q longer than %d characters", ErrInvalidPatch, tag, MaxTagLength)
			}
		}
		patch.Tags = &normalized
	}
	return patch, nil
}

// ApplyMetadataPatch writes a validated patch. It returns sql.ErrNoRows when
// the item does not exist.
func ApplyMetadataPatch(ctx context.Context, q execQuerier, table Table, id int64, patch MetadataPatch) error {
	if patch.Title != nil {
		if err := SetUserTitle(ctx, q, table, id, *patch.Title); err != nil {
			return err
		}
	}
	if patch.Tags != nil {
		if _, err := SetServedTags(ctx, q, table, id, *patch.Tags); err != nil {
			return err
		}
	}
	return nil
}
