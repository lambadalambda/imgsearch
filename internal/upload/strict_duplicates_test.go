package upload

import (
	"bytes"
	"context"
	"crypto/sha256"
	"database/sql"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	dbpkg "imgsearch/internal/db"
	"imgsearch/internal/phash"
)

// Scaled copies have identical dHashes but different file hashes.
func strictCopyPNG(t *testing.T, scale int) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 9*scale, 8*scale))
	for y := 0; y < 8*scale; y++ {
		for x := 0; x < 9*scale; x++ {
			v := uint8(((x/scale)*37 + (y/scale)*71) % 256)
			img.Set(x, y, color.RGBA{v, v, v, 255})
		}
	}
	var out bytes.Buffer
	if err := png.Encode(&out, img); err != nil {
		t.Fatal(err)
	}
	return out.Bytes()
}

func TestStrictImportKeepsLargerExistingImage(t *testing.T) {
	svc, db := setupService(t)
	svc.DeduplicateStrict = true
	ctx := context.Background()
	keeper, err := svc.Store(ctx, "large.png", bytes.NewReader(strictCopyPNG(t, 4)))
	if err != nil {
		t.Fatal(err)
	}
	got, err := svc.Store(ctx, "small.png", bytes.NewReader(strictCopyPNG(t, 2)))
	if err != nil {
		t.Fatal(err)
	}
	if !got.Duplicate || got.ImageID != keeper.ImageID || got.SHA256 != keeper.SHA256 || got.StoragePath != keeper.StoragePath {
		t.Fatalf("got %+v, want duplicate of %+v", got, keeper)
	}
	var count int
	if err := db.QueryRow(`SELECT COUNT(*) FROM images`).Scan(&count); err != nil || count != 1 {
		t.Fatalf("image count=%d, err=%v", count, err)
	}
	if err := db.QueryRow(`SELECT COUNT(*) FROM index_jobs`).Scan(&count); err != nil || count != 2 {
		t.Fatalf("job count=%d, err=%v", count, err)
	}
}

func TestStrictImportDisabledKeepsBothCopies(t *testing.T) {
	svc, db := setupService(t)
	for _, scale := range []int{4, 2} {
		got, err := svc.Store(context.Background(), "copy.png", bytes.NewReader(strictCopyPNG(t, scale)))
		if err != nil || got.Duplicate {
			t.Fatalf("store: %+v, %v", got, err)
		}
	}
	var count int
	if err := db.QueryRow(`SELECT COUNT(*) FROM images`).Scan(&count); err != nil || count != 2 {
		t.Fatalf("image count=%d, err=%v", count, err)
	}
}

func TestStrictImportUpgradePreservesMetadataAndCollections(t *testing.T) {
	svc, db := setupService(t)
	db.SetMaxOpenConns(1)
	if _, err := db.Exec(`PRAGMA foreign_keys=ON`); err != nil {
		t.Fatal(err)
	}
	svc.DeduplicateStrict = true
	ctx := context.Background()
	old, err := svc.Store(ctx, "small.png", bytes.NewReader(strictCopyPNG(t, 2)))
	if err != nil {
		t.Fatal(err)
	}
	_, err = db.Exec(`
UPDATE images SET title='Edited', user_title='Edited', annotator_title='Original', summary='Summary', description='Description', tags_json='["nsfw","user"]', annotator_tags_json='["removed"]', user_tags_json='["nsfw","user"]', removed_tags_json='["removed"]', reannotate_requested=1, captured_at='2001-02-03 00:00:00', created_at='2002-01-01 00:00:00', thumbnail_path='images/old-thumb.webp', annotation_updated_at='2003-01-01 00:00:00' WHERE id=?;
INSERT INTO collections(id,name) VALUES(1,'Favorites'),(2,'Other');
INSERT INTO collection_items(id,collection_id,image_id) VALUES(11,1,?),(12,2,?);
INSERT INTO embedding_models(id,name,version,dimensions,metric,normalized) VALUES(2,'other','v1',4,'cosine',1);
INSERT INTO image_embeddings(image_id,model_id,dim,vector_blob) VALUES(?,1,4,X'00000000'),(?,2,4,X'00000000');
UPDATE index_jobs SET state='leased', lease_owner='old-worker', leased_until=datetime('now','+1 hour') WHERE image_id=?;
`, old.ImageID, old.ImageID, old.ImageID, old.ImageID, old.ImageID, old.ImageID)
	if err != nil {
		t.Fatal(err)
	}
	got, err := svc.Store(ctx, "large.png", bytes.NewReader(strictCopyPNG(t, 4)))
	if err != nil {
		t.Fatal(err)
	}
	if !got.Duplicate || got.ReplacedImageID != old.ImageID || got.ImageID == old.ImageID {
		t.Fatalf("upgrade: %+v, old %+v", got, old)
	}
	var title, userTitle, annotatorTitle, summary, description, tags, annotatorTags, userTags, removedTags, captured, created, annotated string
	var thumb sql.NullString
	var requested, width int
	if err := db.QueryRow(`SELECT title,user_title,annotator_title,summary,description,tags_json,annotator_tags_json,user_tags_json,removed_tags_json,captured_at,created_at,annotation_updated_at,thumbnail_path,reannotate_requested,width FROM images WHERE id=?`, got.ImageID).Scan(&title, &userTitle, &annotatorTitle, &summary, &description, &tags, &annotatorTags, &userTags, &removedTags, &captured, &created, &annotated, &thumb, &requested, &width); err != nil {
		t.Fatal(err)
	}
	if title != "Edited" || userTitle != "Edited" || annotatorTitle != "Original" || summary != "Summary" || description != "Description" || tags != `["nsfw","user"]` || annotatorTags != `["removed"]` || userTags != tags || removedTags != annotatorTags || captured != "2001-02-03 00:00:00" || created != "2002-01-01 00:00:00" || annotated != "2003-01-01 00:00:00" || thumb.Valid || requested != 1 || width != 36 {
		t.Fatalf("metadata not preserved: title=%s tags=%s captured=%s created=%s thumb=%v width=%d", title, tags, captured, created, thumb, width)
	}
	for query, want := range map[string]int{
		`SELECT COUNT(*) FROM images`: 1,
		`SELECT COUNT(*) FROM collection_items WHERE image_id=` + strconv.FormatInt(got.ImageID, 10) + ` AND id IN (11,12)`: 2,
		`SELECT COUNT(*) FROM image_embeddings`:                 0,
		`SELECT COUNT(*) FROM index_jobs WHERE state='leased'`:  0,
		`SELECT COUNT(*) FROM index_jobs WHERE state='pending'`: 2,
	} {
		var n int
		if err := db.QueryRow(query).Scan(&n); err != nil || n != want {
			t.Fatalf("%s: got %d want %d (%v)", query, n, want, err)
		}
	}
	// An old worker cannot attach an embedding to the replacement ID.
	if _, err := db.Exec(`INSERT INTO image_embeddings(image_id,model_id,dim,vector_blob) VALUES(?,1,4,X'00')`, old.ImageID); err == nil {
		t.Fatal("stale embedding accepted")
	}
	for _, path := range []string{old.StoragePath, got.StoragePath} {
		if _, err := os.Stat(filepath.Join(svc.DataDir, path)); err != nil {
			t.Fatal(err)
		}
	}
}

func TestStrictImportThresholdAndExclusions(t *testing.T) {
	for _, tc := range []struct {
		name      string
		hashSQL   string
		extraSQL  string
		duplicate bool
	}{
		{"distance two", "?", "", true},
		{"distance three", "?", "", false},
		{"unhashable", "-1", "", false},
		{"uncomputed", "NULL", "", false},
		{"unknown size", "?", "UPDATE images SET width=0", false},
		{"video frame", "?", `INSERT INTO videos(id,sha256,original_name,storage_path,mime_type,duration_ms,width,height,frame_count) VALUES(1,'video','video','videos/video','video/mp4',100,10,10,1); INSERT INTO video_frames(video_id,image_id,frame_index,timestamp_ms) SELECT 1,id,0,0 FROM images`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			svc, db := setupService(t)
			svc.DeduplicateStrict = true
			first, err := svc.Store(context.Background(), "large.png", bytes.NewReader(strictCopyPNG(t, 4)))
			if err != nil {
				t.Fatal(err)
			}
			h, err := phash.Compute(bytes.NewReader(strictCopyPNG(t, 2)))
			if err != nil {
				t.Fatal(err)
			}
			var args []any
			if tc.hashSQL == "?" {
				mask := uint64(3)
				if tc.name == "distance three" {
					mask = 7
				}
				args = []any{phash.ToInt64(h ^ mask)}
			}
			if _, err := db.Exec(`UPDATE images SET phash=`+tc.hashSQL, args...); err != nil {
				t.Fatal(err)
			}
			if tc.extraSQL != "" {
				if _, err := db.Exec(tc.extraSQL); err != nil {
					t.Fatal(err)
				}
			}
			got, err := svc.Store(context.Background(), "small.png", bytes.NewReader(strictCopyPNG(t, 2)))
			if err != nil {
				t.Fatal(err)
			}
			if got.Duplicate != tc.duplicate || (tc.duplicate && got.ImageID != first.ImageID) {
				t.Fatalf("result %+v, want duplicate=%v", got, tc.duplicate)
			}
		})
	}
}

func TestStrictImportFailedPublishRollsBack(t *testing.T) {
	svc, db := setupService(t)
	svc.DeduplicateStrict = true
	old, err := svc.Store(context.Background(), "small.png", bytes.NewReader(strictCopyPNG(t, 2)))
	if err != nil {
		t.Fatal(err)
	}
	content := strictCopyPNG(t, 4)
	digest := fmt.Sprintf("%x", sha256.Sum256(content))
	blocked := filepath.Join(svc.DataDir, "images", digest)
	if err := os.MkdirAll(filepath.Join(blocked, "block"), 0755); err != nil {
		t.Fatal(err)
	}
	if _, err := svc.Store(context.Background(), "large.png", bytes.NewReader(content)); err == nil {
		t.Fatal("expected publish failure")
	}
	var id int64
	if err := db.QueryRow(`SELECT id FROM images`).Scan(&id); err != nil || id != old.ImageID {
		t.Fatalf("lost old image: %d, %v", id, err)
	}
	var n int
	if err := db.QueryRow(`SELECT COUNT(*) FROM index_jobs`).Scan(&n); err != nil || n != 2 {
		t.Fatalf("jobs=%d err=%v", n, err)
	}
}

func TestStrictImportSelectsDirectKeeperOnly(t *testing.T) {
	svc, db := setupService(t)
	ctx := context.Background()
	// Seed a transitive chain with dedup disabled: incoming -> middle -> far.
	middle, err := svc.Store(ctx, "middle.png", bytes.NewReader(strictCopyPNG(t, 3)))
	if err != nil {
		t.Fatal(err)
	}
	far, err := svc.Store(ctx, "far.png", bytes.NewReader(strictCopyPNG(t, 4)))
	if err != nil {
		t.Fatal(err)
	}
	h, err := phash.Compute(bytes.NewReader(strictCopyPNG(t, 2)))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`UPDATE images SET phash=? WHERE id=?; UPDATE images SET phash=? WHERE id=?`, phash.ToInt64(h^3), middle.ImageID, phash.ToInt64(h^15), far.ImageID); err != nil {
		t.Fatal(err)
	}
	svc.DeduplicateStrict = true
	got, err := svc.Store(ctx, "incoming.png", bytes.NewReader(strictCopyPNG(t, 2)))
	if err != nil {
		t.Fatal(err)
	}
	if !got.Duplicate || got.ImageID != middle.ImageID {
		t.Fatalf("not direct keeper: %+v", got)
	}
	var count int
	if err := db.QueryRow(`SELECT COUNT(*) FROM images`).Scan(&count); err != nil || count != 2 {
		t.Fatalf("swept existing group: %d %v", count, err)
	}
	// An exact re-upload must never trigger a retroactive merge, even if hashes match.
	if _, err := db.Exec(`UPDATE images SET phash=?`, phash.ToInt64(h)); err != nil {
		t.Fatal(err)
	}
	got, err = svc.Store(ctx, "middle-again.png", bytes.NewReader(strictCopyPNG(t, 3)))
	if err != nil {
		t.Fatal(err)
	}
	if got.ImageID != middle.ImageID || got.ReplacedImageID != 0 {
		t.Fatalf("exact reupload replaced entry: %+v", got)
	}
}

func TestStrictImportTieKeepsOldestExisting(t *testing.T) {
	svc, db := setupService(t)
	ctx := context.Background()
	first, err := svc.Store(ctx, "first.png", bytes.NewReader(strictCopyPNG(t, 3)))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := svc.Store(ctx, "second.png", bytes.NewReader(strictCopyPNG(t, 4))); err != nil {
		t.Fatal(err)
	}
	// Same pixel count as incoming, distinct file contents and matching hashes.
	if _, err := db.Exec(`UPDATE images SET width=18,height=16`); err != nil {
		t.Fatal(err)
	}
	svc.DeduplicateStrict = true
	got, err := svc.Store(ctx, "incoming.png", bytes.NewReader(strictCopyPNG(t, 2)))
	if err != nil {
		t.Fatal(err)
	}
	if !got.Duplicate || got.ImageID != first.ImageID || got.ReplacedImageID != 0 {
		t.Fatalf("tie: %+v", got)
	}
}

func TestStrictImportConcurrentServicesKeepLargest(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	dsn := filepath.Join(dir, "test.sqlite") + "?_foreign_keys=on&_busy_timeout=10000&_journal_mode=WAL"
	var services []*Service
	var contents [][]byte
	for i := 0; i < 6; i++ {
		conn, err := sql.Open("sqlite3", dsn)
		if err != nil {
			t.Fatal(err)
		}
		conn.SetMaxOpenConns(1)
		t.Cleanup(func() { _ = conn.Close() })
		if i == 0 {
			if err := dbpkg.RunMigrations(ctx, conn); err != nil {
				t.Fatal(err)
			}
			if _, err := conn.Exec(`INSERT INTO embedding_models(id,name,version,dimensions,metric,normalized) VALUES(1,'test','v1',4,'cosine',1)`); err != nil {
				t.Fatal(err)
			}
		}
		services = append(services, &Service{DB: conn, DataDir: dir, ModelID: 1, DeduplicateStrict: true})
		contents = append(contents, strictCopyPNG(t, i+1))
	}
	start := make(chan struct{})
	results := make(chan error, len(services))
	for i, svc := range services {
		go func() {
			<-start
			_, err := svc.Store(ctx, "concurrent.png", bytes.NewReader(contents[i]))
			results <- err
		}()
	}
	close(start)
	for range services {
		if err := <-results; err != nil {
			t.Fatal(err)
		}
	}
	var count, width int
	if err := services[0].DB.QueryRow(`SELECT COUNT(*),MAX(width) FROM images`).Scan(&count, &width); err != nil || count != 1 || width != 54 {
		t.Fatalf("count=%d width=%d err=%v", count, width, err)
	}
}

func TestStrictImportCommitFailureRestoresOldEntry(t *testing.T) {
	svc, db := setupService(t)
	db.SetMaxOpenConns(1)
	if _, err := db.Exec(`PRAGMA foreign_keys=ON`); err != nil {
		t.Fatal(err)
	}
	svc.DeduplicateStrict = true
	old, err := svc.Store(context.Background(), "small.png", bytes.NewReader(strictCopyPNG(t, 2)))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`
INSERT INTO collections(id,name) VALUES(1,'Saved');
INSERT INTO collection_items(collection_id,image_id) VALUES(1,?);
CREATE TRIGGER fail_commit AFTER INSERT ON images WHEN NEW.width=36 BEGIN
 INSERT INTO collection_items(collection_id,image_id) VALUES(999,NEW.id);
END;
PRAGMA defer_foreign_keys=ON;`, old.ImageID); err != nil {
		t.Fatal(err)
	}
	content := strictCopyPNG(t, 4)
	if _, err := svc.Store(context.Background(), "large.png", bytes.NewReader(content)); err == nil || !strings.Contains(err.Error(), "commit tx") {
		t.Fatalf("wanted deferred commit failure, got %v", err)
	}
	var id int64
	if err := db.QueryRow(`SELECT image_id FROM collection_items WHERE collection_id=1`).Scan(&id); err != nil || id != old.ImageID {
		t.Fatalf("membership not restored: %d %v", id, err)
	}
	if err := db.QueryRow(`SELECT id FROM images`).Scan(&id); err != nil || id != old.ImageID {
		t.Fatalf("old entry not restored: %d %v", id, err)
	}
	// Never unlink a published digest on commit failure: a later import may reuse it.
	for _, path := range []string{old.StoragePath, fmt.Sprintf("images/%x", sha256.Sum256(content))} {
		if _, err := os.Stat(filepath.Join(svc.DataDir, path)); err != nil {
			t.Fatal(err)
		}
	}
}
