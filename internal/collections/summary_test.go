package collections

import (
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"testing"

	"imgsearch/internal/thumbnails"
)

// Decode independently of the production types so these tests also catch wire
// regressions (including absent zero counts and omitted/extra cover fields).
func summaryList(t *testing.T, h http.Handler, query string) []map[string]any {
	t.Helper()
	response := decode[map[string]json.RawMessage](t, request(t, h, "GET", "/api/collections?summary=true"+query, "", 200))
	if len(response) != 1 {
		t.Fatalf("unexpected summary envelope: %v", response)
	}
	var list []map[string]any
	if err := json.Unmarshal(response["collections"], &list); err != nil || list == nil {
		t.Fatalf("collections must be an array: %s (%v)", response["collections"], err)
	}
	return list
}

func cover(kind string, id int64, path string) map[string]any {
	c := map[string]any{"media_type": kind, "media_id": float64(id)}
	if path != "" {
		c["thumbnail_path"] = path
	}
	return c
}

func assertSummary(t *testing.T, got map[string]any, c Collection, all, visible int, covers ...map[string]any) {
	t.Helper()
	wantCovers := make([]any, 0, len(covers))
	for _, c := range covers {
		wantCovers = append(wantCovers, c)
	}
	want := map[string]any{
		"id": float64(c.ID), "name": c.Name, "created_at": c.CreatedAt,
		"item_count": float64(all), "visible_count": float64(visible), "covers": wantCovers,
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("summary mismatch\ngot:  %#v\nwant: %#v", got, want)
	}
}

func TestCollectionSummariesCountsCoversAndOrdering(t *testing.T) {
	db, h := setup(t) // Deliberately one connection: nested queries would deadlock.
	if list := summaryList(t, h, ""); len(list) != 0 {
		t.Fatalf("empty database: %v", list)
	}
	for _, q := range []string{
		`UPDATE images SET thumbnail_path='derivatives/one.webp' WHERE id=1`,
		`INSERT INTO images(id,sha256,original_name,storage_path,mime_type,width,height) VALUES
 (4,'four','four.jpg','original/four.jpg','image/jpeg',10,20),
 (5,'five','five.jpg','original/five.jpg','image/jpeg',10,20),
 (6,'early','early.jpg','original/early.jpg','image/jpeg',10,20)`,
		// Earliest frame_index is not the lowest image ID or timestamp.
		`UPDATE video_frames SET frame_index=8 WHERE video_id=1`,
		`INSERT INTO video_frames(video_id,image_id,frame_index,timestamp_ms) VALUES(1,6,2,999)`,
	} {
		if _, err := db.Exec(q); err != nil {
			t.Fatal(err)
		}
	}
	populated := create(t, h, "Populated")
	empty := create(t, h, "Empty")
	hidden := create(t, h, "Hidden")
	for _, target := range []string{"image/4", "image/5", "image/1", "video/1", "video/2", "image/2", "image/4"} {
		request(t, h, "PUT", fmt.Sprintf("/api/collections/%d/items/%s", populated.ID, target), "", 204)
	}
	request(t, h, "PUT", fmt.Sprintf("/api/collections/%d/items/image/2", hidden.ID), "", 204)

	for _, query := range []string{"", "&include_nsfw=false", "&include_nsfw=invalid"} {
		list := summaryList(t, h, query)
		if len(list) != 3 {
			t.Fatalf("collections: %v", list)
		}
		assertSummary(t, list[0], hidden, 1, 0)
		assertSummary(t, list[1], empty, 0, 0)
		// Hidden newest membership cannot consume a cover slot. Image/video 1
		// stay distinct; saving image 4 again cannot move it ahead of image 5.
		assertSummary(t, list[2], populated, 6, 5,
			cover("video", 2, ""), cover("video", 1, thumbnails.Path(6, "")),
			cover("image", 1, "derivatives/one.webp"), cover("image", 5, thumbnails.Path(5, "")))
	}
	list := summaryList(t, h, "&include_nsfw=true")
	assertSummary(t, list[0], hidden, 1, 1, cover("image", 2, thumbnails.Path(2, "")))
	assertSummary(t, list[1], empty, 0, 0)
	assertSummary(t, list[2], populated, 6, 6,
		cover("image", 2, thumbnails.Path(2, "")), cover("video", 2, ""),
		cover("video", 1, thumbnails.Path(6, "")), cover("image", 1, "derivatives/one.webp"))

	// Stored frame derivatives are honored, not original frame/video paths.
	if _, err := db.Exec(`UPDATE images SET thumbnail_path='derivatives/early.webp' WHERE id=6`); err != nil {
		t.Fatal(err)
	}
	list = summaryList(t, h, "")
	assertSummary(t, list[2], populated, 6, 5,
		cover("video", 2, ""), cover("video", 1, "derivatives/early.webp"),
		cover("image", 1, "derivatives/one.webp"), cover("image", 5, thumbnails.Path(5, "")))
}

func TestCollectionSummariesWholeVideoPrivacy(t *testing.T) {
	for _, tc := range []struct{ name, update string }{
		{"video own tags", `UPDATE videos SET tags_json='[" NSFW "]' WHERE id=1`},
		{"preview tags", `UPDATE images SET tags_json='["NsFw"]' WHERE id=3`},
		{"non-preview tags", `UPDATE images SET tags_json='[" nsfw "]' WHERE id=4`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			db, h := setup(t)
			for _, q := range []string{
				`INSERT INTO images(id,sha256,original_name,storage_path,mime_type,width,height) VALUES
 (4,'later','later.jpg','private/later.jpg','image/jpeg',10,20)`,
				`INSERT INTO video_frames(video_id,image_id,frame_index,timestamp_ms) VALUES(1,4,9,900)`,
				tc.update,
			} {
				if _, err := db.Exec(q); err != nil {
					t.Fatal(err)
				}
			}
			c := create(t, h, "Private video")
			request(t, h, "PUT", "/api/collections/1/items/video/1", "", 204)
			assertSummary(t, summaryList(t, h, "")[0], c, 1, 0)
			assertSummary(t, summaryList(t, h, "&include_nsfw=true")[0], c, 1, 1,
				cover("video", 1, thumbnails.Path(3, "")))
			// The detail visibility predicate must agree with summaries.
			page := decode[map[string]any](t, request(t, h, "GET", "/api/collections/1/items", "", 200))
			if page["item_count"] != float64(1) || page["total"] != float64(0) || len(page["items"].([]any)) != 0 {
				t.Fatalf("hidden detail: %v", page)
			}
		})
	}
}

func TestCollectionDetailUnfilteredCount(t *testing.T) {
	_, h := setup(t)
	create(t, h, "Mixed")
	create(t, h, "Empty")
	for _, target := range []string{"image/1", "image/2", "video/2"} {
		request(t, h, "PUT", "/api/collections/1/items/"+target, "", 204)
	}
	for _, tc := range []struct {
		path                string
		all, visible, items int
	}{
		{"/api/collections/1/items?limit=1&offset=1", 3, 2, 1},
		{"/api/collections/1/items?include_nsfw=true&limit=1", 3, 3, 1},
		{"/api/collections/1/items?offset=20", 3, 2, 0},
		{"/api/collections/2/items", 0, 0, 0},
	} {
		t.Run(tc.path, func(t *testing.T) {
			page := decode[map[string]any](t, request(t, h, "GET", tc.path, "", 200))
			if len(page) != 3 || page["item_count"] != float64(tc.all) || page["total"] != float64(tc.visible) || len(page["items"].([]any)) != tc.items {
				t.Fatalf("detail counts: %v", page)
			}
		})
	}
}

func TestCollectionPlainResponsesUnchanged(t *testing.T) {
	_, h := setup(t)
	assertPlain := func(got map[string]any) {
		t.Helper()
		if len(got) != 3 || got["id"] == nil || got["name"] == nil || got["created_at"] == nil {
			t.Fatalf("plain collection shape changed: %v", got)
		}
	}
	assertPlain(decode[map[string]any](t, request(t, h, "POST", "/api/collections?summary=true", `{"name":"Plain"}`, 201)))
	request(t, h, "PUT", "/api/collections/1/items/image/1", "", 204)
	for _, query := range []string{"", "?include_nsfw=true", "?summary=false", "?summary=invalid", "?media_type=image&media_id=1", "?media_type=image&media_id=1&summary=true"} {
		list := decode[struct {
			Collections []map[string]any `json:"collections"`
		}](t, request(t, h, "GET", "/api/collections"+query, "", 200))
		if len(list.Collections) != 1 {
			t.Fatalf("plain list: %v", list)
		}
		assertPlain(list.Collections[0])
	}
	assertPlain(decode[map[string]any](t, request(t, h, "GET", "/api/collections/1?summary=true", "", 200)))
	assertPlain(decode[map[string]any](t, request(t, h, "PATCH", "/api/collections/1?summary=true", `{"name":"Renamed"}`, 200)))
	for _, route := range []struct{ method, path string }{
		{"PUT", "/api/collections/1/items/image/1"},
		{"DELETE", "/api/collections/1/items/image/1"},
		{"DELETE", "/api/collections/1"},
	} {
		if w := request(t, h, route.method, route.path, "", 204); w.Body.Len() != 0 {
			t.Fatalf("membership/delete response changed: %s", w.Body.String())
		}
	}
}
