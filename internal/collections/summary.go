package collections

import (
	"context"
	"database/sql"

	"imgsearch/internal/thumbnails"
)

// CollectionSummary is opt-in; picker and CRUD responses remain Collection.
type CollectionSummary struct {
	Collection
	ItemCount    int64             `json:"item_count"`
	VisibleCount int64             `json:"visible_count"`
	Covers       []CollectionCover `json:"covers"`
}

type CollectionCover struct {
	MediaType     string `json:"media_type"`
	MediaID       int64  `json:"media_id"`
	ThumbnailPath string `json:"thumbnail_path,omitempty"`
}

func (h *Handler) summaries(ctx context.Context, includeNSFW bool) ([]CollectionSummary, error) {
	// One set-based statement gives counts and covers a consistent snapshot.
	// Rank only visible memberships, then cap before looking up previews. No
	// jobs, embeddings, original paths or full media hydration are needed.
	rows, err := h.DB.QueryContext(ctx, `
WITH memberships AS (
 SELECT m.id,m.collection_id,m.image_id,m.video_id,`+membershipVisibleSQL(includeNSFW)+` AS visible
 FROM collection_items m
 LEFT JOIN images i ON i.id=m.image_id
 LEFT JOIN videos v ON v.id=m.video_id
), counts AS (
 SELECT collection_id,COUNT(*) AS item_count,SUM(visible) AS visible_count
 FROM memberships GROUP BY collection_id
), ranked AS (
 SELECT id,collection_id,image_id,video_id,
        ROW_NUMBER() OVER (PARTITION BY collection_id ORDER BY id DESC) AS rn
 FROM memberships WHERE visible
), covers AS (
 SELECT * FROM ranked WHERE rn<=4
), first_frames AS (
 SELECT vf.video_id,MIN(vf.frame_index) AS frame_index
 FROM video_frames vf
 JOIN (SELECT DISTINCT video_id FROM covers WHERE video_id IS NOT NULL) cv ON cv.video_id=vf.video_id
 GROUP BY vf.video_id
)
SELECT c.id,c.name,c.created_at,COALESCE(n.item_count,0),COALESCE(n.visible_count,0),
       cover.image_id,cover.video_id,preview.id,preview.thumbnail_path
FROM collections c
LEFT JOIN counts n ON n.collection_id=c.id
LEFT JOIN covers cover ON cover.collection_id=c.id
LEFT JOIN first_frames f ON f.video_id=cover.video_id
LEFT JOIN video_frames vf ON vf.video_id=f.video_id AND vf.frame_index=f.frame_index
LEFT JOIN images preview ON preview.id=COALESCE(cover.image_id,vf.image_id)
ORDER BY c.id DESC,cover.id DESC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	result := []CollectionSummary{}
	for rows.Next() {
		var summary CollectionSummary
		var imageID, videoID, previewID sql.NullInt64
		var thumbnail sql.NullString
		if err := rows.Scan(&summary.ID, &summary.Name, &summary.CreatedAt, &summary.ItemCount, &summary.VisibleCount,
			&imageID, &videoID, &previewID, &thumbnail); err != nil {
			return nil, err
		}
		if len(result) == 0 || result[len(result)-1].ID != summary.ID {
			summary.Covers = []CollectionCover{}
			result = append(result, summary)
		}
		if !imageID.Valid && !videoID.Valid {
			continue // Empty or fully hidden collection: covers must be [].
		}
		cover := CollectionCover{MediaType: "image", MediaID: imageID.Int64}
		if videoID.Valid {
			cover.MediaType, cover.MediaID = "video", videoID.Int64
		}
		if previewID.Valid {
			cover.ThumbnailPath = thumbnails.Path(previewID.Int64, thumbnail.String)
		} // A video with no frames keeps its identity but has no thumbnail path.
		last := &result[len(result)-1]
		last.Covers = append(last.Covers, cover)
	}
	return result, rows.Err()
}
