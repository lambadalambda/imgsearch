package db

import (
	"context"
	"strings"
	"testing"
)

func TestTopTagsCountsMediaUnitsAndSkipsRareAndNSFW(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()
	if err := RunMigrations(ctx, db); err != nil {
		t.Fatal(err)
	}
	if _, err := db.ExecContext(ctx, `
INSERT INTO images(id, sha256, original_name, storage_path, mime_type, width, height, tags_json) VALUES
  (1, 'a', 'a', 'images/a', 'image/jpeg', 1, 1, '["Cat","indoor","nsfw"]'),
  (2, 'b', 'b', 'images/b', 'image/jpeg', 1, 1, '["cat","outdoor"]'),
  (3, 'f1', 'f1', 'images/f1', 'image/jpeg', 1, 1, '["stage","cat"]'),
  (4, 'f2', 'f2', 'images/f2', 'image/jpeg', 1, 1, '["stage"]'),
  (5, 'c', 'c', 'images/c', 'image/jpeg', 1, 1, '["once"]');
INSERT INTO videos(id, sha256, original_name, storage_path, mime_type, duration_ms, width, height, frame_count, tags_json)
  VALUES (1, 'v', 'v', 'videos/v', 'video/mp4', 1, 1, 1, 2, '["concert","cat"]');
INSERT INTO video_frames(video_id, image_id, frame_index, timestamp_ms) VALUES (1, 3, 0, 0), (1, 4, 1, 1);
`); err != nil {
		t.Fatalf("seed: %v", err)
	}
	tags, err := TopTags(ctx, db, 10, 2)
	if err != nil {
		t.Fatal(err)
	}
	// cat: image 1, image 2, video 1 (frames + video tags collapse) = 3; stage: video 1 only = 1 (dropped); once: 1 (dropped).
	if strings.Join(tags, ",") != "cat" {
		t.Fatalf("top tags: %v", tags)
	}
	tags, err = TopTags(ctx, db, 3, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(tags) != 3 || tags[0] != "cat" {
		t.Fatalf("limited top tags: %v", tags)
	}
	for _, tag := range tags {
		if tag == "nsfw" {
			t.Fatal("nsfw must be excluded")
		}
	}
}
