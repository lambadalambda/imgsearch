package collections

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	_ "github.com/mattn/go-sqlite3"
	dbpkg "imgsearch/internal/db"
	"imgsearch/internal/httputil"
)

func setup(t *testing.T) (*sql.DB, http.Handler) {
	t.Helper()
	db, err := sql.Open("sqlite3", ":memory:?_foreign_keys=on")
	if err != nil {
		t.Fatal(err)
	}
	db.SetMaxOpenConns(1)
	t.Cleanup(func() { db.Close() })
	if err := dbpkg.RunMigrations(context.Background(), db); err != nil {
		t.Fatal(err)
	}
	for _, query := range []string{
		`INSERT INTO images(id,sha256,original_name,storage_path,mime_type,width,height,tags_json) VALUES
 (1,'one','one.jpg','images/one.jpg','image/jpeg',100,200,'[]'),
 (2,'two','two.jpg','images/two.jpg','image/jpeg',100,200,'[" NSFW "]'),
 (3,'frame','frame.jpg','images/frame.jpg','image/jpeg',100,200,'[]')`,
		`INSERT INTO videos(id,sha256,original_name,storage_path,mime_type,width,height,duration_ms,frame_count) VALUES
 (1,'video','movie.mp4','videos/movie.mp4','video/mp4',100,200,1000,1),
 (2,'video2','second.mp4','videos/second.mp4','video/mp4',100,200,1000,0)`,
		`INSERT INTO video_frames(video_id,image_id,frame_index,timestamp_ms) VALUES(1,3,0,0)`,
	} {
		if _, err := db.Exec(query); err != nil {
			t.Fatal(err)
		}
	}
	return db, NewHandler(&Handler{DB: db, ModelID: 1})
}

func request(t *testing.T, h http.Handler, method, path, body string, want int) *httptest.ResponseRecorder {
	t.Helper()
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest(method, path, strings.NewReader(body)))
	if w.Code != want {
		t.Fatalf("%s %s: status %d want %d: %s", method, path, w.Code, want, w.Body.String())
	}
	return w
}

func decode[T any](t *testing.T, w *httptest.ResponseRecorder) T {
	t.Helper()
	var v T
	if err := json.Unmarshal(w.Body.Bytes(), &v); err != nil {
		t.Fatal(err)
	}
	return v
}

func create(t *testing.T, h http.Handler, name string) Collection {
	t.Helper()
	b, _ := json.Marshal(map[string]string{"name": name})
	return decode[Collection](t, request(t, h, "POST", "/api/collections", string(b), 201))
}

func TestCollectionCRUD(t *testing.T) {
	db, h := setup(t)
	empty := request(t, h, "GET", "/api/collections", "", 200)
	if strings.TrimSpace(empty.Body.String()) != `{"collections":[]}` {
		t.Fatal(empty.Body.String())
	}
	first := create(t, h, "  Inspiration  ")
	if first.Name != "Inspiration" || first.ID <= 0 || first.CreatedAt == "" {
		t.Fatalf("%+v", first)
	}
	second := create(t, h, "Inspiration") // identity is ID, not name
	list := decode[struct {
		Collections []Collection `json:"collections"`
	}](t, request(t, h, "GET", "/api/collections", "", 200))
	if len(list.Collections) != 2 || list.Collections[0].ID != second.ID {
		t.Fatalf("%+v", list)
	}
	path := fmt.Sprintf("/api/collections/%d", first.ID)
	renamed := decode[Collection](t, request(t, h, "PATCH", path, `{"name":"旅行"}`, 200))
	if renamed.Name != "旅行" || renamed.CreatedAt != first.CreatedAt {
		t.Fatalf("%+v", renamed)
	}
	request(t, h, "GET", path, "", 200)
	request(t, h, "PUT", path+"/items/image/1", "", 204)
	request(t, h, "DELETE", path, "", 204)
	request(t, h, "GET", path, "", 404)
	request(t, h, "DELETE", path, "", 404)
	var media, memberships int
	db.QueryRow(`SELECT COUNT(*) FROM images`).Scan(&media)
	db.QueryRow(`SELECT COUNT(*) FROM collection_items`).Scan(&memberships)
	if media != 3 || memberships != 0 {
		t.Fatalf("media=%d memberships=%d", media, memberships)
	}
}

func TestValidationAndRouting(t *testing.T) {
	_, h := setup(t)
	create(t, h, "Valid")
	for _, body := range []string{`{}`, `null`, `{"name":null}`, `{"name":1}`, `{"name":"  "}`, `{"name":"ok","extra":true}`, `{"name":"ok"} {}`, `{"name":`, `{"name":"` + strings.Repeat("x", 101) + `"}`, `{"name":"a\u0000b"}`} {
		for _, route := range []struct{ method, path string }{{"POST", "/api/collections"}, {"PATCH", "/api/collections/1"}} {
			request(t, h, route.method, route.path, body, 400)
		}
	}
	create(t, h, strings.Repeat("猫", 100))
	for _, tc := range []struct {
		method, path string
		code         int
	}{
		{"GET", "/api/collections/0", 400}, {"GET", "/api/collections/no", 400},
		{"GET", "/api/collections/999", 404}, {"GET", "/api/collections/1/unknown", 404},
		{"GET", "/api/collections/", 404}, {"GET", "/api/collections/1/items/image/1/extra", 404},
		{"PUT", "/api/collections/1/items/audio/1", 400}, {"PUT", "/api/collections/1/items/image/0", 400},
		{"PUT", "/api/collections/1/items/image/999", 404}, {"PUT", "/api/collections/999/items/image/1", 404},
		{"GET", "/api/collections/999/items", 404}, {"PUT", "/api/collections/1/items/image/3", 409},
		{"GET", "/api/collections?media_type=image", 400}, {"GET", "/api/collections?media_id=1", 400},
		{"GET", "/api/collections?media_type=video&media_id=-1", 400},
		{"GET", "/api/collections?media_type=image&media_id=999", 404},
		{"GET", "/api/collections?media_type=image&media_id=3", 409},
	} {
		t.Run(tc.method+tc.path, func(t *testing.T) { request(t, h, tc.method, tc.path, "", tc.code) })
	}
	for _, path := range []string{"/api/collections", "/api/collections/1", "/api/collections/1/items", "/api/collections/1/items/image/1"} {
		w := request(t, h, "OPTIONS", path, "", 405)
		if w.Header().Get("Allow") == "" {
			t.Fatal("missing Allow")
		}
	}
	request(t, NewHandler(&Handler{}), "GET", "/api/collections", "", 503)
}

func TestMembershipIdentityOrderAndFiltering(t *testing.T) {
	db, h := setup(t)
	create(t, h, "First")
	create(t, h, "Second")
	for _, target := range []string{"image/1", "image/2", "video/1", "video/2", "image/1"} {
		request(t, h, "PUT", "/api/collections/1/items/"+target, "", 204)
	}
	request(t, h, "PUT", "/api/collections/2/items/image/1", "", 204)
	list := func(query string) ItemPage {
		return decode[ItemPage](t, request(t, h, "GET", "/api/collections/1/items"+query, "", 200))
	}
	page := list("?include_nsfw=true")
	if page.Total != 4 || len(page.Items) != 4 || page.Items[0].Video.VideoID != 2 || page.Items[1].Video.VideoID != 1 || page.Items[2].Image.ImageID != 2 || page.Items[3].Image.ImageID != 1 {
		t.Fatalf("bad mixed ordering: %+v", page)
	}
	if page.Items[1].Video.ImageID != 3 || page.Items[1].MediaType != "video" || page.Items[3].Image.ThumbnailPath == "" {
		t.Fatal("media hydration lost existing metadata")
	}
	page = list("?limit=1&offset=2")
	if page.Total != 3 || len(page.Items) != 1 || page.Items[0].Image.ImageID != 1 {
		t.Fatalf("NSFW must filter before pagination: %+v", page)
	}
	if _, err := db.Exec(`UPDATE images SET tags_json='["nsfw"]' WHERE id=3`); err != nil {
		t.Fatal(err)
	}
	page = list("")
	if page.Total != 2 || len(page.Items) != 2 {
		t.Fatalf("frame NSFW not respected: %+v", page)
	}
	if _, err := db.Exec(`UPDATE videos SET tags_json='["nsfw"]' WHERE id=2`); err != nil {
		t.Fatal(err)
	}
	page = list("")
	if page.Total != 1 || page.Items[0].Image.ImageID != 1 {
		t.Fatal("video NSFW not respected")
	}
	page = list("?limit=garbage&offset=-5&include_nsfw=true")
	if len(page.Items) != 4 {
		t.Fatal("query fallback")
	}
	page = list("?offset=20")
	if page.Items == nil || len(page.Items) != 0 {
		t.Fatal("empty items must be []")
	}
	memberships := decode[struct {
		Collections []Collection `json:"collections"`
	}](t, request(t, h, "GET", "/api/collections?media_type=image&media_id=1", "", 200))
	if len(memberships.Collections) != 2 {
		t.Fatalf("%+v", memberships)
	}
	memberships = decode[struct {
		Collections []Collection `json:"collections"`
	}](t, request(t, h, "GET", "/api/collections?media_type=video&media_id=1", "", 200))
	if len(memberships.Collections) != 1 {
		t.Fatal("image/video IDs conflated")
	}
	request(t, h, "DELETE", "/api/collections/1/items/image/1", "", 204)
	request(t, h, "DELETE", "/api/collections/1/items/image/1", "", 204)
	request(t, h, "DELETE", "/api/collections/1/items/image/999", "", 204)
	var count int
	if err := db.QueryRow(`SELECT COUNT(*) FROM images WHERE id=1`).Scan(&count); err != nil || count != 1 {
		t.Fatal("unsave deleted media", err)
	}
}

func TestMembershipConstraintsAndCascade(t *testing.T) {
	db, h := setup(t)
	create(t, h, "Saved")
	for _, target := range []string{"image/1", "video/2"} {
		request(t, h, "PUT", "/api/collections/1/items/"+target, "", 204)
	}
	for _, q := range []string{
		`INSERT INTO collection_items(collection_id,image_id) VALUES(1,1)`,
		`INSERT INTO collection_items(collection_id,video_id) VALUES(1,2)`,
		`INSERT INTO collection_items(collection_id,image_id,video_id) VALUES(1,2,2)`,
		`INSERT INTO collection_items(collection_id) VALUES(1)`,
		`INSERT INTO collection_items(collection_id,image_id) VALUES(1,999)`,
		`INSERT INTO collection_items(collection_id,image_id) VALUES(999,2)`,
	} {
		if _, err := db.Exec(q); err == nil {
			t.Fatalf("constraint allowed %s", q)
		}
	}
	for _, q := range []string{`DELETE FROM images WHERE id=1`, `DELETE FROM videos WHERE id=2`} {
		if _, err := db.Exec(q); err != nil {
			t.Fatal(err)
		}
	}
	page := decode[ItemPage](t, request(t, h, "GET", "/api/collections/1/items", "", 200))
	if page.Total != 0 || len(page.Items) != 0 {
		t.Fatal("media deletion left memberships")
	}
}

func TestCollectionsRequireExistingAPIAuthentication(t *testing.T) {
	_, h := setup(t)
	secured := httputil.NewAPIAuthMiddleware("test-key")(h)
	request(t, secured, "GET", "/api/collections", "", 401)
	request(t, secured, "POST", "/api/collections", `{"name":"Private"}`, 401)
}
