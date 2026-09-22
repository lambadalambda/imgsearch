package worker

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"slices"
	"testing"

	"imgsearch/internal/embedder"
)

type directVideoAnnotator struct{ *fakeAnnotator }

func (*directVideoAnnotator) VideoMode() string { return "direct_frames" }

type acquiredVideoAnnotator struct {
	*fakeAnnotator
	pinned             embedder.VideoJobAnnotator
	acquired, released bool
}

func (a *acquiredVideoAnnotator) AcquireVideoAnnotator(context.Context) (embedder.VideoJobAnnotator, func(), error) {
	a.acquired = true
	return a.pinned, func() { a.released = true }, nil
}

func TestDirectVideoJobKeepsAllStoredSamplesAndFrameCaches(t *testing.T) {
	for _, cached := range []bool{false, true} {
		for _, failed := range []bool{false, true} {
			t.Run(fmt.Sprintf("cached=%t/failed=%t", cached, failed), func(t *testing.T) {
				q, conn := setupQueueTest(t)
				conn.SetMaxOpenConns(1)
				_, err := conn.Exec(`
DELETE FROM index_jobs;
INSERT INTO videos(id,sha256,original_name,storage_path,mime_type,duration_ms,width,height,frame_count,transcript_text,description,tags_json,reannotate_requested)
VALUES(9,'vid9','stage show.mp4','videos/vid9','video/mp4',10000,1280,720,10,'hello audience','previous description','["nsfw"]',1);
INSERT INTO index_jobs(id,kind,video_id,model_id,state) VALUES(30,'annotate_video',9,1,'pending');
UPDATE images SET description='',tags_json='[]',annotation_updated_at=NULL WHERE id=1;
UPDATE videos SET title='My title',user_title='My title',user_tags_json='["holiday"]',removed_tags_json='["blurry"]' WHERE id=9;
`)
				if err != nil {
					t.Fatal(err)
				}
				if cached {
					if _, err = conn.Exec(`UPDATE images SET description='cached frame',tags_json='["nsfw","stage"]',annotation_updated_at='2025-01-01' WHERE id=1`); err != nil {
						t.Fatal(err)
					}
				}
				// Ten stored samples, deliberately reverse index/time order and all
				// referencing the same image. Do not cap at the new-upload default.
				for i := 0; i < 10; i++ {
					if _, err = conn.Exec(`INSERT INTO video_frames(video_id,image_id,frame_index,timestamp_ms) VALUES(9,1,?,?)`, 9-i, i*1000); err != nil {
						t.Fatal(err)
					}
				}
				pinned := &directVideoAnnotator{&fakeAnnotator{err: errors.New("must not annotate frames"), videoAnnotation: embedder.VideoAnnotation{Description: "new video", Title: "Generated title", Tags: []string{"nsfw", "stage", "blurry"}, IsNSFW: true}}}
				if failed {
					pinned.videoErr = errors.New("provider rejects multiple images")
				}
				a := &acquiredVideoAnnotator{fakeAnnotator: &fakeAnnotator{err: errors.New("unpinned backend"), videoErr: errors.New("unpinned backend")}, pinned: pinned}
				q.Annotator = a
				processed, err := q.ProcessOne(context.Background(), "worker")
				if err != nil {
					t.Fatal(err)
				}
				if !processed || !a.acquired || !a.released {
					t.Fatalf("processed=%v acquired=%v released=%v", processed, a.acquired, a.released)
				}
				if pinned.videoCalls != 1 || pinned.videoFrameCalls != 0 || a.videoCalls != 0 {
					t.Fatalf("calls pinned=%d frames=%d unpinned=%d", pinned.videoCalls, pinned.videoFrameCalls, a.videoCalls)
				}
				input := pinned.lastVideoInput
				if len(input.Frames) != 10 || input.ImageMaxSideMultiplier != 2 || input.OriginalName != "stage show.mp4" || input.DurationMS != 10000 || input.TranscriptText != "hello audience" {
					t.Fatalf("input=%+v", input)
				}
				for i, frame := range input.Frames {
					if frame.FrameIndex != 9-i || frame.TimestampMS != int64(i*1000) || frame.Path != filepath.Join(q.DataDir, "images", "abc") {
						t.Errorf("frame %d=%+v", i, frame)
					}
				}
				var desc, tags, updated string
				if err = conn.QueryRow(`SELECT description,tags_json,COALESCE(annotation_updated_at,'') FROM images WHERE id=1`).Scan(&desc, &tags, &updated); err != nil {
					t.Fatal(err)
				}
				if cached {
					if desc != "cached frame" || tags != `["nsfw","stage"]` || updated != "2025-01-01" {
						t.Fatalf("cache changed: %q %q %q", desc, tags, updated)
					}
				} else if desc != "" || tags != "[]" || updated != "" {
					t.Fatalf("wrote frame annotation: %q %q %q", desc, tags, updated)
				}
				var requests, count int
				if err = conn.QueryRow(`SELECT reannotate_requested,frame_count FROM videos WHERE id=9`).Scan(&requests, &count); err != nil {
					t.Fatal(err)
				}
				if count != 10 || (requests == 1) != failed {
					t.Fatalf("requested=%d count=%d", requests, count)
				}
				if failed {
					if err = conn.QueryRow(`SELECT description,tags_json FROM videos WHERE id=9`).Scan(&desc, &tags); err != nil {
						t.Fatal(err)
					}
					if desc != "previous description" || tags != `["nsfw"]` {
						t.Fatalf("lost prior annotation on failure: %q %q", desc, tags)
					}
				}
				if !failed {
					var title string
					if err = conn.QueryRow(`SELECT title,tags_json FROM videos WHERE id=9`).Scan(&title, &tags); err != nil {
						t.Fatal(err)
					}
					var merged []string
					if err = json.Unmarshal([]byte(tags), &merged); err != nil {
						t.Fatal(err)
					}
					if title != "My title" || !slices.Contains(merged, "holiday") || !slices.Contains(merged, "nsfw") || slices.Contains(merged, "blurry") {
						t.Fatalf("lost user edit merge: %q %v", title, merged)
					}
				}

				if len(q.Index.(*fakeIndex).upserts) != 0 {
					t.Fatal("annotation touched embeddings")
				}
			})
		}
	}
}
