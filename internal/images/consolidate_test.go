package images

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func seedConsolidation(t *testing.T) *sql.DB {
	t.Helper()
	db := setupImagesDB(t)
	_, err := db.Exec(`
 INSERT INTO embedding_models(id,name,version,dimensions,metric,normalized) VALUES(1,'test','v1',1,'cosine',1);
 INSERT INTO image_embeddings(image_id,model_id,dim,vector_blob) VALUES(1,1,1,X'00000000'),(2,1,1,X'00000000'),(3,1,1,X'00000000');
 UPDATE images SET phash=42, annotator_title='Generated', title='Generated', annotator_tags_json='["auto","blur"]', tags_json='["auto","blur"]';
 UPDATE images SET user_title='Trip', title='Trip', user_tags_json='["Holiday"]', removed_tags_json='["blur"]' WHERE id=2;
 UPDATE images SET user_tags_json='["holiday","Friends"]' WHERE id=3;
 INSERT INTO collections(id,name) VALUES(1,'A'),(2,'B'),(3,'C');
 INSERT INTO collection_items(collection_id,image_id) VALUES(1,1),(1,2),(2,2),(2,3),(3,3);
 `)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`PRAGMA foreign_keys=ON`); err != nil {
		t.Fatal(err)
	}
	return db
}

func duplicatePost(t *testing.T, h http.Handler, body any) (int, map[string]any) {
	t.Helper()
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, httptest.NewRequest(http.MethodPost, "/api/duplicates", bytes.NewReader(raw)))
	var result map[string]any
	if err := json.Unmarshal(rr.Body.Bytes(), &result); err != nil {
		t.Fatalf("decode %d: %s", rr.Code, rr.Body.String())
	}
	return rr.Code, result
}

func testConsolidationRequest() map[string]any {
	// Intentionally keep the smallest and supply IDs in a different order.
	return map[string]any{"keeper_id": 1, "group_ids": []int{3, 1, 2}, "distance": 0, "include_nsfw": true}
}

func TestDuplicateConsolidationPreviewAndAtomicUnion(t *testing.T) {
	db := seedConsolidation(t)
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "images"), 0700); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"a", "b", "c"} {
		if err := os.WriteFile(filepath.Join(dir, "images", name), []byte(name), 0600); err != nil {
			t.Fatal(err)
		}
	}
	h := NewDuplicatesHandler(&Handler{DB: db, ModelID: 1, DataDir: dir})
	req := testConsolidationRequest()
	status, preview := duplicatePost(t, h, req)
	if status != 200 {
		t.Fatalf("preview %d: %v", status, preview)
	}
	if preview["keeper_id"] != float64(1) || preview["affected_collections"] != float64(3) || preview["file_policy"] != "retained" || len(preview["removed_ids"].([]any)) != 2 || len(preview["conflicts"].([]any)) != 0 {
		t.Fatalf("preview: %v", preview)
	}
	assertScalar(t, db, `SELECT count(*) FROM images`, 3)
	assertScalar(t, db, `SELECT user_title FROM images WHERE id=1`, "")
	req["confirm"] = true
	req["preview_token"] = preview["preview_token"]
	status, result := duplicatePost(t, h, req)
	if status != 200 {
		t.Fatalf("confirm %d: %v", status, result)
	}
	assertScalar(t, db, `SELECT count(*) FROM images`, 1)
	assertScalar(t, db, `SELECT count(*) FROM collection_items WHERE image_id=1`, 3)
	assertScalar(t, db, `SELECT count(*) FROM collection_items`, 3)
	assertScalar(t, db, `SELECT user_title FROM images WHERE id=1`, "Trip")
	assertScalar(t, db, `SELECT title FROM images WHERE id=1`, "Trip")
	assertScalar(t, db, `SELECT user_tags_json FROM images WHERE id=1`, `["Holiday","Friends"]`)
	assertScalar(t, db, `SELECT removed_tags_json FROM images WHERE id=1`, `["blur"]`)
	assertScalar(t, db, `SELECT tags_json FROM images WHERE id=1`, `["auto","Holiday","Friends"]`)
	assertScalar(t, db, `SELECT count(*) FROM index_jobs WHERE image_id!=1`, 0)
	assertScalar(t, db, `SELECT count(*) FROM image_embeddings`, 1)
	assertScalar(t, db, `SELECT id FROM collection_items WHERE collection_id=1`, 1)
	assertScalar(t, db, `SELECT id FROM collection_items WHERE collection_id=2`, 3)
	assertScalar(t, db, `SELECT id FROM collection_items WHERE collection_id=3`, 5)
	for _, name := range []string{"a", "b", "c"} {
		if _, err := os.Stat(filepath.Join(dir, "images", name)); err != nil {
			t.Fatalf("file removed: %v", err)
		}
	}
	if status, _ := duplicatePost(t, h, req); status != 409 {
		t.Fatalf("replay status=%d", status)
	}
}

func assertScalar(t *testing.T, db *sql.DB, query string, want any) {
	t.Helper()
	var got any
	if err := db.QueryRow(query).Scan(&got); err != nil {
		t.Fatal(err)
	}
	// SQLite integers scan to int64.
	if n, ok := want.(int); ok {
		want = int64(n)
	}
	if got != want {
		t.Fatalf("%s: got=%v want=%v", query, got, want)
	}
}

func TestDuplicateConsolidationRollsBackMembershipsMetadataAndDeletes(t *testing.T) {
	for _, trigger := range []string{
		`CREATE TRIGGER fail_transfer BEFORE UPDATE OF image_id ON collection_items BEGIN SELECT RAISE(ABORT,'transfer failed'); END`,
		`CREATE TRIGGER fail_delete BEFORE DELETE ON images WHEN OLD.id=3 BEGIN SELECT RAISE(ABORT,'delete failed'); END`,
	} {
		t.Run(trigger, func(t *testing.T) {
			db := seedConsolidation(t)
			h := NewDuplicatesHandler(&Handler{DB: db, ModelID: 1})
			req := testConsolidationRequest()
			status, preview := duplicatePost(t, h, req)
			if status != 200 {
				t.Fatalf("preview: %d %v", status, preview)
			}
			if _, err := db.Exec(trigger); err != nil {
				t.Fatal(err)
			}
			req["confirm"] = true
			req["preview_token"] = preview["preview_token"]
			if status, _ := duplicatePost(t, h, req); status != 500 {
				t.Fatalf("status=%d", status)
			}
			assertScalar(t, db, `SELECT count(*) FROM images`, 3)
			assertScalar(t, db, `SELECT count(*) FROM collection_items`, 5)
			assertScalar(t, db, `SELECT count(*) FROM collection_items WHERE image_id=1`, 1)
			assertScalar(t, db, `SELECT user_title FROM images WHERE id=1`, "")
			assertScalar(t, db, `SELECT user_tags_json FROM images WHERE id=1`, `[]`)
			assertScalar(t, db, `SELECT tags_json FROM images WHERE id=1`, `["auto","blur"]`)
			assertScalar(t, db, `SELECT count(*) FROM index_jobs`, 3)
			assertScalar(t, db, `SELECT count(*) FROM image_embeddings`, 3)
			assertScalar(t, db, `SELECT removed_tags_json FROM images WHERE id=1`, `[]`)
			assertScalar(t, db, `SELECT group_concat(id || ':' || collection_id || ':' || image_id) FROM (SELECT * FROM collection_items ORDER BY id)`, "1:1:1,2:1:2,3:2:2,4:2:3,5:3:3")
		})
	}
}

func TestDuplicateConsolidationConflictsAndStalePreview(t *testing.T) {
	for _, test := range []struct{ name, change, conflict string }{
		{"titles", `UPDATE images SET user_title='Other' WHERE id=3`, "title"},
		{"tag polarity", `UPDATE images SET removed_tags_json='["HOLIDAY"]' WHERE id=1`, "Holiday"},
		{"invalid metadata", `UPDATE images SET user_tags_json='oops' WHERE id=3`, "metadata"},
	} {
		t.Run(test.name, func(t *testing.T) {
			db := seedConsolidation(t)
			if _, err := db.Exec(test.change); err != nil {
				t.Fatal(err)
			}
			h := NewDuplicatesHandler(&Handler{DB: db, ModelID: 1})
			req := testConsolidationRequest()
			status, preview := duplicatePost(t, h, req)
			if status != 200 || !strings.Contains(strings.ToLower(fmtJSON(preview["conflicts"])), strings.ToLower(test.conflict)) {
				t.Fatalf("conflict %d: %v", status, preview)
			}
			req["confirm"] = true
			req["preview_token"] = preview["preview_token"]
			if status, _ := duplicatePost(t, h, req); status != 409 {
				t.Fatalf("confirm status=%d", status)
			}
			assertScalar(t, db, `SELECT count(*) FROM images`, 3)
			assertScalar(t, db, `SELECT count(*) FROM collection_items`, 5)
		})
	}
	for _, change := range []string{`UPDATE images SET user_title='Changed' WHERE id=2`, `DELETE FROM collection_items WHERE collection_id=3`, `UPDATE images SET phash=99 WHERE id=3`} {
		t.Run(change, func(t *testing.T) {
			db := seedConsolidation(t)
			h := NewDuplicatesHandler(&Handler{DB: db, ModelID: 1})
			req := testConsolidationRequest()
			status, preview := duplicatePost(t, h, req)
			if status != 200 {
				t.Fatalf("preview %d", status)
			}
			if _, err := db.Exec(change); err != nil {
				t.Fatal(err)
			}
			req["confirm"] = true
			req["preview_token"] = preview["preview_token"]
			if status, _ := duplicatePost(t, h, req); status != 409 {
				t.Fatalf("stale status=%d", status)
			}
			assertScalar(t, db, `SELECT count(*) FROM images`, 3)
		})
	}
}
func fmtJSON(v any) string { b, _ := json.Marshal(v); return string(b) }

func TestDuplicateConsolidationRejectsInvalidSelection(t *testing.T) {
	for _, test := range []struct {
		name     string
		ids      any
		keeper   int
		distance int
		seed     string
		status   int
	}{
		{"duplicate IDs", []int{1, 2, 2}, 1, 0, "", 400},
		{"missing ID", []int{1, 2, 99}, 1, 0, "", 409},
		{"partial group", []int{1, 2}, 1, 0, "", 409},
		{"keeper outside group", []int{1, 2, 3}, 4, 0, "", 400},
		{"negative ID", []int{-1, 2, 3}, 2, 0, "", 400},
		{"string/path ID", []string{"../2", "1"}, 1, 0, "", 400},
		{"too loose", []int{1, 2, 3}, 1, 8, "", 400},
		{"nonmatch", []int{1, 2, 3}, 1, 0, `UPDATE images SET phash=99 WHERE id=3`, 409},
		{"unhashed", []int{1, 2, 3}, 1, 0, `UPDATE images SET phash=NULL WHERE id=3`, 409},
		{"frame", []int{1, 2, 3}, 1, 0, `INSERT INTO videos(id,sha256,original_name,storage_path,mime_type,width,height,duration_ms,frame_count) VALUES(1,'v','v','v','video/mp4',10,10,1000,1); INSERT INTO video_frames(video_id,image_id,timestamp_ms,frame_index) VALUES(1,3,0,0)`, 409},
	} {
		t.Run(test.name, func(t *testing.T) {
			db := seedConsolidation(t)
			if test.seed != "" {
				if _, err := db.Exec(test.seed); err != nil {
					t.Fatal(err)
				}
			}
			h := NewDuplicatesHandler(&Handler{DB: db, ModelID: 1})
			status, result := duplicatePost(t, h, map[string]any{"group_ids": test.ids, "keeper_id": test.keeper, "distance": test.distance, "include_nsfw": true})
			if status != test.status {
				t.Fatalf("status %d want %d: %v", status, test.status, result)
			}
			assertScalar(t, db, `SELECT count(*) FROM images`, 3)
		})
	}
}

func TestDuplicateConsolidationRequiresPreviewAndRejectsUnknownFields(t *testing.T) {
	db := seedConsolidation(t)
	h := NewDuplicatesHandler(&Handler{DB: db, ModelID: 1})
	req := testConsolidationRequest()
	req["confirm"] = true
	if status, _ := duplicatePost(t, h, req); status != 409 {
		t.Fatalf("no preview status=%d", status)
	}
	req = testConsolidationRequest()
	req["storage_path"] = "../../secret"
	if status, _ := duplicatePost(t, h, req); status != 400 {
		t.Fatalf("unknown field status=%d", status)
	}
	assertScalar(t, db, `SELECT count(*) FROM images`, 3)
}

func TestDuplicateFileSizesStayWithinDataDirectory(t *testing.T) {
	db := seedConsolidation(t)
	dir := t.TempDir()
	outside := filepath.Join(t.TempDir(), "private")
	if err := os.WriteFile(outside, []byte("private"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "image"), []byte("safe"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(dir, "escape")); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`UPDATE images SET storage_path=CASE id WHEN 1 THEN 'image' WHEN 2 THEN 'escape' ELSE ? END`, outside); err != nil {
		t.Fatal(err)
	}
	rr := httptest.NewRecorder()
	NewDuplicatesHandler(&Handler{DB: db, ModelID: 1, DataDir: dir}).ServeHTTP(rr, httptest.NewRequest("GET", "/api/duplicates?distance=0", nil))
	var response struct {
		Groups []struct {
			Items []struct {
				ID   int64  `json:"image_id"`
				Size *int64 `json:"file_size_bytes"`
			}
		}
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if len(response.Groups) != 1 {
		t.Fatalf("groups: %s", rr.Body.String())
	}
	for _, item := range response.Groups[0].Items {
		if item.ID == 1 && (item.Size == nil || *item.Size != 4) {
			t.Fatalf("safe size=%v", item.Size)
		}
		if item.ID != 1 && item.Size != nil {
			t.Fatalf("escaped data directory: %+v", item)
		}
	}
}

func TestDuplicateConsolidationValidatesVisibleTransitiveGroup(t *testing.T) {
	db := seedConsolidation(t)
	if _, err := db.Exec(`UPDATE images SET phash=CASE id WHEN 1 THEN 42 WHEN 2 THEN 43 ELSE 41 END;
 INSERT INTO images(id,sha256,original_name,storage_path,mime_type,width,height,phash) VALUES(4,'d','hidden.jpg','images/d','image/jpeg',10,10,42);
 UPDATE images SET tags_json='["nsfw"]' WHERE id=4`); err != nil {
		t.Fatal(err)
	}
	h := NewDuplicatesHandler(&Handler{DB: db, ModelID: 1})
	req := testConsolidationRequest()
	req["distance"] = 1
	req["include_nsfw"] = false
	status, preview := duplicatePost(t, h, req)
	if status != 200 {
		t.Fatalf("transitive preview %d: %v", status, preview)
	}
	req["confirm"] = true
	req["preview_token"] = preview["preview_token"]
	if status, result := duplicatePost(t, h, req); status != 200 {
		t.Fatalf("transitive confirm %d: %v", status, result)
	}
	assertScalar(t, db, `SELECT count(*) FROM images`, 2)
	assertScalar(t, db, `SELECT original_name FROM images WHERE id=4`, "hidden.jpg")
}
