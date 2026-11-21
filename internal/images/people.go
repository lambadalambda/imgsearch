package images

import (
	"context"
	"database/sql"
	"fmt"
	"strings"
)

// PersonRef identifies a grouped person on a media item. It lives here
// (rather than in people) so images, videos, and search can share it:
// people already depends on images and videos for item hydration.
type PersonRef struct {
	ID   int64  `json:"id"`
	Name string `json:"name"`
}

// PeopleForImages returns distinct grouped persons per image, in first-face
// order. Images without faces map to an empty (non-nil) slice.
func PeopleForImages(ctx context.Context, db *sql.DB, imageIDs []int64) (map[int64][]PersonRef, error) {
	out := make(map[int64][]PersonRef, len(imageIDs))
	for _, id := range imageIDs {
		out[id] = []PersonRef{}
	}
	if len(imageIDs) == 0 {
		return out, nil
	}
	ph := make([]string, len(imageIDs))
	args := make([]any, len(imageIDs))
	for i, id := range imageIDs {
		ph[i] = "?"
		args[i] = id
	}
	rows, err := db.QueryContext(ctx, `
SELECT f.image_id, p.id, p.name
FROM face_detections f
JOIN persons p ON p.id = f.person_id
WHERE f.image_id IN (`+strings.Join(ph, ",")+`)
ORDER BY f.image_id, f.id`, args...)
	if err != nil {
		return nil, fmt.Errorf("query image people: %w", err)
	}
	defer rows.Close()
	seen := map[int64]map[int64]bool{}
	for rows.Next() {
		var imageID, personID int64
		var name string
		if err := rows.Scan(&imageID, &personID, &name); err != nil {
			return nil, fmt.Errorf("scan image person: %w", err)
		}
		if seen[imageID] == nil {
			seen[imageID] = map[int64]bool{}
		}
		if seen[imageID][personID] {
			continue
		}
		seen[imageID][personID] = true
		out[imageID] = append(out[imageID], PersonRef{ID: personID, Name: name})
	}
	return out, rows.Err()
}

// PeopleForVideos returns distinct grouped persons per video, aggregated
// over sampled frames in first-face order.
func PeopleForVideos(ctx context.Context, db *sql.DB, videoIDs []int64) (map[int64][]PersonRef, error) {
	out := make(map[int64][]PersonRef, len(videoIDs))
	for _, id := range videoIDs {
		out[id] = []PersonRef{}
	}
	if len(videoIDs) == 0 {
		return out, nil
	}
	ph := make([]string, len(videoIDs))
	args := make([]any, len(videoIDs))
	for i, id := range videoIDs {
		ph[i] = "?"
		args[i] = id
	}
	rows, err := db.QueryContext(ctx, `
SELECT vf.video_id, p.id, p.name
FROM face_detections f
JOIN video_frames vf ON vf.image_id = f.image_id
JOIN persons p ON p.id = f.person_id
WHERE vf.video_id IN (`+strings.Join(ph, ",")+`)
GROUP BY vf.video_id, p.id, p.name
ORDER BY vf.video_id, MIN(f.id)`, args...)
	if err != nil {
		return nil, fmt.Errorf("query video people: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		var videoID, personID int64
		var name string
		if err := rows.Scan(&videoID, &personID, &name); err != nil {
			return nil, fmt.Errorf("scan video person: %w", err)
		}
		out[videoID] = append(out[videoID], PersonRef{ID: personID, Name: name})
	}
	return out, rows.Err()
}
