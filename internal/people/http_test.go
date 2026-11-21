package people

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

// addFace inserts a face row directly (the worker will own this path later).
func addFace(t *testing.T, db *sql.DB, imageID int64, personID any) int64 {
	t.Helper()
	res, err := db.Exec(`INSERT INTO face_detections(image_id,person_id,box_x,box_y,box_w,box_h,confidence) VALUES(?,?,0.1,0.1,0.3,0.3,0.9)`, imageID, personID)
	if err != nil {
		t.Fatal(err)
	}
	id, err := res.LastInsertId()
	if err != nil {
		t.Fatal(err)
	}
	return id
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

func create(t *testing.T, h http.Handler, name string) Person {
	t.Helper()
	b, _ := json.Marshal(map[string]string{"name": name})
	return decode[Person](t, request(t, h, "POST", "/api/people", string(b), 201))
}

func TestPersonCRUD(t *testing.T) {
	db, h := setup(t)
	empty := request(t, h, "GET", "/api/people", "", 200)
	if strings.TrimSpace(empty.Body.String()) != `{"people":[]}` {
		t.Fatal(empty.Body.String())
	}
	first := create(t, h, "  Aunt Jessie  ")
	if first.Name != "Aunt Jessie" || first.ID <= 0 || first.UUID == "" || first.CreatedAt == "" {
		t.Fatalf("%+v", first)
	}
	second := create(t, h, "Aunt Jessie") // identity is ID, not name
	list := decode[struct {
		People []Person `json:"people"`
	}](t, request(t, h, "GET", "/api/people", "", 200))
	if len(list.People) != 2 || list.People[0].ID != second.ID {
		t.Fatalf("%+v", list)
	}
	// Auto-naming: empty body yields a neutral Person N name plus a UUID.
	auto := decode[Person](t, request(t, h, "POST", "/api/people", `{}`, 201))
	if !strings.HasPrefix(auto.Name, "Person ") || auto.UUID == "" || auto.UUID == first.UUID {
		t.Fatalf("%+v", auto)
	}
	path := fmt.Sprintf("/api/people/%d", first.ID)
	renamed := decode[Person](t, request(t, h, "PATCH", path, `{"name":"旅行"}`, 200))
	if renamed.Name != "旅行" || renamed.CreatedAt != first.CreatedAt || renamed.UUID != first.UUID {
		t.Fatalf("%+v", renamed)
	}
	request(t, h, "GET", path, "", 200)
	request(t, h, "DELETE", path, "", 204)
	request(t, h, "GET", path, "", 404)
	request(t, h, "DELETE", path, "", 404)
	var persons, images int
	db.QueryRow(`SELECT COUNT(*) FROM persons`).Scan(&persons)
	db.QueryRow(`SELECT COUNT(*) FROM images`).Scan(&images)
	if persons != 2 || images != 3 {
		t.Fatalf("persons=%d images=%d", persons, images)
	}
}

func TestValidationAndRouting(t *testing.T) {
	_, h := setup(t)
	create(t, h, "Valid")
	for _, body := range []string{`null`, `{"name":null}`, `{"name":1}`, `{"name":"  "}`, `{"name":"ok","extra":true}`, `{"name":"ok"} {}`, `{"name":`, `{"name":"` + strings.Repeat("x", 101) + `"}`, `{"name":"a\u0000b"}`} {
		for _, route := range []struct{ method, path string }{{"POST", "/api/people"}, {"PATCH", "/api/people/1"}} {
			request(t, h, route.method, route.path, body, 400)
		}
	}
	create(t, h, strings.Repeat("猫", 100))
	for _, tc := range []struct {
		method, path, body string
		code               int
	}{
		{"GET", "/api/people/0", "", 400}, {"GET", "/api/people/no", "", 400},
		{"GET", "/api/people/999", "", 404}, {"GET", "/api/people/1/unknown", "", 404},
		{"GET", "/api/people/", "", 404}, {"GET", "/api/people/1/faces/1/extra", "", 404},
		{"POST", "/api/people/merge", `{"from_ids":[1],"into_id":1}`, 400}, // from == into
		{"POST", "/api/people/merge", `{"from_ids":[],"into_id":1}`, 400},
		{"POST", "/api/people/merge", `{"from_ids":[999],"into_id":1}`, 404},
		{"POST", "/api/people/1/split", `{"face_ids":[]}`, 400},
		{"POST", "/api/people/999/split", `{"face_ids":[1]}`, 404},
		{"POST", "/api/faces/999/reassign", `{"person_id":1}`, 404},
		{"POST", "/api/faces/1/reassign", `{"person_id":999}`, 404},
	} {
		t.Run(tc.method+tc.path+tc.body, func(t *testing.T) { request(t, h, tc.method, tc.path, tc.body, tc.code) })
	}
	for _, path := range []string{"/api/people", "/api/people/1", "/api/people/1/items", "/api/faces"} {
		w := request(t, h, "OPTIONS", path, "", 405)
		if w.Header().Get("Allow") == "" {
			t.Fatal("missing Allow")
		}
	}
	request(t, NewHandler(&Handler{}), "GET", "/api/people", "", 503)
}

func TestPersonSummary(t *testing.T) {
	db, h := setup(t)
	empty := decode[struct {
		People []PersonSummary `json:"people"`
	}](t, request(t, h, "GET", "/api/people?summary=true", "", 200))
	if len(empty.People) != 0 {
		t.Fatal(empty)
	}
	aunt := create(t, h, "Aunt Jessie")
	create(t, h, "Empty")
	addFace(t, db, 1, aunt.ID)
	addFace(t, db, 3, aunt.ID)
	list := decode[struct {
		People []PersonSummary `json:"people"`
	}](t, request(t, h, "GET", "/api/people?summary=true", "", 200))
	if len(list.People) != 2 {
		t.Fatalf("%+v", list)
	}
	var auntSummary, emptySummary *PersonSummary
	for i := range list.People {
		switch list.People[i].Name {
		case "Aunt Jessie":
			auntSummary = &list.People[i]
		case "Empty":
			emptySummary = &list.People[i]
		}
	}
	if auntSummary == nil || emptySummary == nil {
		t.Fatalf("%+v", list)
	}
	if auntSummary.FaceCount != 2 || len(auntSummary.Covers) != 2 {
		t.Fatalf("%+v", auntSummary)
	}
	if emptySummary.FaceCount != 0 || len(emptySummary.Covers) != 0 {
		t.Fatalf("%+v", emptySummary)
	}
}

func TestPersonItemsDedupeAndFiltering(t *testing.T) {
	db, h := setup(t)
	aunt := create(t, h, "Aunt Jessie")
	stranger := create(t, h, "Stranger")
	// Two faces on image 1 collapse to one item; frame image 3 surfaces as its video.
	addFace(t, db, 1, aunt.ID)
	addFace(t, db, 1, aunt.ID)
	addFace(t, db, 3, aunt.ID)
	addFace(t, db, 2, aunt.ID) // NSFW standalone image
	addFace(t, db, 2, stranger.ID)

	list := func(query string) ItemPage {
		return decode[ItemPage](t, request(t, h, "GET", fmt.Sprintf("/api/people/%d/items%s", aunt.ID, query), "", 200))
	}
	page := list("?include_nsfw=true")
	if page.ItemCount != 3 || len(page.Items) != 3 {
		t.Fatalf("want 3 saved media (image 1, image 2, video 1), got %+v", page)
	}
	if page.Items[0].MediaType != "image" || page.Items[0].Image.ImageID != 2 {
		t.Fatalf("newest face first (image 2), got %+v", page.Items[0])
	}
	if page.Items[1].MediaType != "video" || page.Items[1].Video.VideoID != 1 {
		t.Fatalf("then video 1, got %+v", page.Items[1])
	}
	if page.Items[2].MediaType != "image" || page.Items[2].Image.ImageID != 1 {
		t.Fatalf("then image 1, got %+v", page.Items[2])
	}
	// Default hides NSFW: image 2 gone, video 1 stays (its own tags are clean).
	page = list("")
	if page.Total != 2 || len(page.Items) != 2 {
		t.Fatalf("NSFW must filter before pagination: %+v", page)
	}
	// NSFW frame tag hides the whole video, like collections.
	if _, err := db.Exec(`UPDATE images SET tags_json='["nsfw"]' WHERE id=3`); err != nil {
		t.Fatal(err)
	}
	page = list("")
	if page.Total != 1 || page.Items[0].Image.ImageID != 1 {
		t.Fatalf("frame NSFW not respected: %+v", page)
	}
	page = list("?limit=garbage&offset=-5&include_nsfw=true")
	if len(page.Items) != 3 {
		t.Fatal("query fallback")
	}
	page = list("?offset=20&include_nsfw=true")
	if page.Items == nil || len(page.Items) != 0 {
		t.Fatal("empty items must be []")
	}
	request(t, h, "GET", "/api/people/999/items", "", 404)
	_ = stranger
}

func TestMergeSplitReassign(t *testing.T) {
	db, h := setup(t)
	a := create(t, h, "A")
	b := create(t, h, "B")
	f1 := addFace(t, db, 1, a.ID)
	f2 := addFace(t, db, 2, a.ID)
	f3 := addFace(t, db, 3, nil) // unassigned pool

	// Reassign unassigned face onto B.
	request(t, h, "POST", fmt.Sprintf("/api/faces/%d/reassign", f3), fmt.Sprintf(`{"person_id":%d}`, b.ID), 200)
	// Unassign it again.
	request(t, h, "POST", fmt.Sprintf("/api/faces/%d/reassign", f3), `{"person_id":null}`, 200)
	var n int
	db.QueryRow(`SELECT COUNT(*) FROM face_detections WHERE person_id IS NULL`).Scan(&n)
	if n != 1 {
		t.Fatalf("unassign failed, nulls=%d", n)
	}
	// B emptied itself above, so it was auto-deleted; recreate it with a face.
	b = create(t, h, "B")
	f4 := addFace(t, db, 2, b.ID)
	_ = f4

	// Split f2 off A into a new person.
	splitBody := decode[struct {
		Person          Person `json:"person"`
		DeletedSourceID *int64 `json:"deleted_source_id"`
	}](t, request(t, h, "POST", fmt.Sprintf("/api/people/%d/split", a.ID), fmt.Sprintf(`{"face_ids":[%d],"name":"Split"}`, f2), 201))
	split := splitBody.Person
	if splitBody.DeletedSourceID != nil {
		t.Fatalf("source still has faces, must not be deleted: %v", splitBody.DeletedSourceID)
	}
	db.QueryRow(`SELECT person_id FROM face_detections WHERE id=?`, f2).Scan(&n)
	if int64(n) != split.ID {
		t.Fatalf("split did not move face: person=%d want %d", n, split.ID)
	}
	// Splitting a face that is not owned fails.
	request(t, h, "POST", fmt.Sprintf("/api/people/%d/split", a.ID), fmt.Sprintf(`{"face_ids":[%d]}`, f2), 409)

	// Merge B and Split into A.
	request(t, h, "POST", "/api/people/merge", fmt.Sprintf(`{"from_ids":[%d,%d],"into_id":%d}`, b.ID, split.ID, a.ID), 204)
	db.QueryRow(`SELECT COUNT(*) FROM persons`).Scan(&n)
	if n != 1 {
		t.Fatalf("merge must delete sources, persons=%d", n)
	}
	db.QueryRow(`SELECT COUNT(*) FROM face_detections WHERE person_id=?`, a.ID).Scan(&n)
	if n != 3 { // f1 + f2 + f4 (f3 was unassigned)
		t.Fatalf("merge must move faces, faces=%d", n)
	}
	_ = f1
}

func TestListFacesFiltersAndPhotoMetadata(t *testing.T) {
	db, h := setup(t)
	a := create(t, h, "A")
	f1 := addFace(t, db, 1, a.ID)
	f2 := addFace(t, db, 3, nil) // unassigned frame face (video 1)

	inPerson := decode[struct {
		Faces []FaceView `json:"faces"`
	}](t, request(t, h, "GET", fmt.Sprintf("/api/faces?person_id=%d", a.ID), "", 200))
	if len(inPerson.Faces) != 1 || inPerson.Faces[0].ID != f1 {
		t.Fatalf("%+v", inPerson)
	}
	face := inPerson.Faces[0]
	if face.OriginalName != "one.jpg" || face.StoragePath != "images/one.jpg" || face.ImageID != 1 {
		t.Fatalf("missing photo metadata: %+v", face)
	}
	if face.BoxW <= 0 || face.BoxH <= 0 || face.Confidence <= 0 {
		t.Fatalf("missing geometry: %+v", face)
	}

	unassigned := decode[struct {
		Faces []FaceView `json:"faces"`
	}](t, request(t, h, "GET", "/api/faces?unassigned=true", "", 200))
	if len(unassigned.Faces) != 1 || unassigned.Faces[0].ID != f2 {
		t.Fatalf("%+v", unassigned)
	}

	all := decode[struct {
		Faces []FaceView `json:"faces"`
	}](t, request(t, h, "GET", "/api/faces", "", 200))
	if len(all.Faces) != 2 {
		t.Fatalf("%+v", all)
	}
	request(t, h, "GET", "/api/faces?person_id=0", "", 400)
}

func TestEmptyPersonDeletedWhenLastFaceMovesAway(t *testing.T) {
	db, h := setup(t)
	a := create(t, h, "A")
	b := create(t, h, "B")
	f1 := addFace(t, db, 1, a.ID)
	f2 := addFace(t, db, 2, a.ID)

	// Moving one of two faces away keeps the source.
	moved := decode[struct {
		Face            Face   `json:"face"`
		DeletedPersonID *int64 `json:"deleted_person_id"`
	}](t, request(t, h, "POST", fmt.Sprintf("/api/faces/%d/reassign", f1), fmt.Sprintf(`{"person_id":%d}`, b.ID), 200))
	if moved.Face.PersonID == nil || *moved.Face.PersonID != b.ID || moved.DeletedPersonID != nil {
		t.Fatalf("%+v", moved)
	}

	// Moving the last face away deletes the emptied person.
	moved = decode[struct {
		Face            Face   `json:"face"`
		DeletedPersonID *int64 `json:"deleted_person_id"`
	}](t, request(t, h, "POST", fmt.Sprintf("/api/faces/%d/reassign", f2), fmt.Sprintf(`{"person_id":%d}`, b.ID), 200))
	if moved.DeletedPersonID == nil || *moved.DeletedPersonID != a.ID {
		t.Fatalf("%+v", moved)
	}
	request(t, h, "GET", fmt.Sprintf("/api/people/%d", a.ID), "", 404)
	var persons, faces int
	db.QueryRow(`SELECT COUNT(*) FROM persons`).Scan(&persons)
	db.QueryRow(`SELECT COUNT(*) FROM face_detections WHERE person_id=?`, b.ID).Scan(&faces)
	if persons != 1 || faces != 2 {
		t.Fatalf("persons=%d faces on B=%d", persons, faces)
	}

	// Splitting every face off deletes the emptied source.
	c := create(t, h, "C")
	f3 := addFace(t, db, 1, c.ID)
	splitBody := decode[struct {
		Person          Person `json:"person"`
		DeletedSourceID *int64 `json:"deleted_source_id"`
	}](t, request(t, h, "POST", fmt.Sprintf("/api/people/%d/split", c.ID), fmt.Sprintf(`{"face_ids":[%d]}`, f3), 201))
	if splitBody.DeletedSourceID == nil || *splitBody.DeletedSourceID != c.ID {
		t.Fatalf("%+v", splitBody)
	}
	request(t, h, "GET", fmt.Sprintf("/api/people/%d", c.ID), "", 404)
	db.QueryRow(`SELECT COUNT(*) FROM persons`).Scan(&persons)
	if persons != 2 {
		t.Fatalf("persons=%d, want B + split target", persons)
	}
}

func TestDeletePersonUnassignsFaces(t *testing.T) {
	db, h := setup(t)
	a := create(t, h, "A")
	addFace(t, db, 1, a.ID)
	request(t, h, "DELETE", fmt.Sprintf("/api/people/%d", a.ID), "", 204)
	var n, nulls int
	db.QueryRow(`SELECT COUNT(*) FROM face_detections`).Scan(&n)
	db.QueryRow(`SELECT COUNT(*) FROM face_detections WHERE person_id IS NULL`).Scan(&nulls)
	if n != 1 || nulls != 1 {
		t.Fatalf("delete must keep faces unassigned: n=%d nulls=%d", n, nulls)
	}
}

func TestDeleteImageCascadesFaces(t *testing.T) {
	db, h := setup(t)
	a := create(t, h, "A")
	addFace(t, db, 1, a.ID)
	if _, err := db.Exec(`DELETE FROM images WHERE id=1`); err != nil {
		t.Fatal(err)
	}
	var n int
	db.QueryRow(`SELECT COUNT(*) FROM face_detections`).Scan(&n)
	if n != 0 {
		t.Fatalf("image deletion left faces: %d", n)
	}
}

func TestPeopleRequireAuth(t *testing.T) {
	_, h := setup(t)
	secured := httputil.NewAPIAuthMiddleware("test-key")(h)
	request(t, secured, "GET", "/api/people", "", 401)
	request(t, secured, "POST", "/api/people", `{"name":"Private"}`, 401)
}
