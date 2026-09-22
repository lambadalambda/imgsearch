package db

import (
	"context"
	"database/sql"
	"fmt"
)

// TopTags returns the most used tags across standalone images and videos
// (each video counts once regardless of frame count), most frequent first.
// Tags used fewer than minCount times are skipped so one-off inventions do
// not get reinforced.
func TopTags(ctx context.Context, db *sql.DB, limit int, minCount int) ([]string, error) {
	if limit <= 0 {
		limit = 100
	}
	if minCount < 1 {
		minCount = 1
	}
	rows, err := db.QueryContext(ctx, `
WITH unit_tags AS (
  SELECT DISTINCT
         CASE WHEN vf.video_id IS NULL THEN 'image:' || i.id ELSE 'video:' || vf.video_id END AS media_unit,
         lower(trim(j.value)) AS tag
  FROM images i
  JOIN json_each(COALESCE(i.tags_json, '[]')) j
    ON trim(COALESCE(j.value, '')) <> ''
  LEFT JOIN video_frames vf ON vf.image_id = i.id
  UNION
  SELECT DISTINCT 'video:' || v.id, lower(trim(j.value))
  FROM videos v
  JOIN json_each(COALESCE(v.tags_json, '[]')) j
    ON trim(COALESCE(j.value, '')) <> ''
)
SELECT tag
FROM unit_tags
WHERE tag <> 'nsfw'
GROUP BY tag
HAVING COUNT(*) >= ?
ORDER BY COUNT(*) DESC, tag ASC
LIMIT ?`, minCount, limit)
	if err != nil {
		return nil, fmt.Errorf("query top tags: %w", err)
	}
	defer func() { _ = rows.Close() }()
	var tags []string
	for rows.Next() {
		var tag string
		if err := rows.Scan(&tag); err != nil {
			return nil, fmt.Errorf("scan top tag: %w", err)
		}
		tags = append(tags, tag)
	}
	return tags, rows.Err()
}
