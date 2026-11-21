package images

import (
	"context"
	"database/sql"
	"testing"

	_ "github.com/mattn/go-sqlite3"
	dbpkg "imgsearch/internal/db"
)

func setupPeople(t *testing.T) *sql.DB {
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
 (2,'two','two.jpg','images/two.jpg','image/jpeg',100,200,'[]'),
 (3,'frame','frame.jpg','images/frame.jpg','image/jpeg',100,200,'[]')`,
		`INSERT INTO videos(id,sha256,original_name,storage_path,mime_type,width,height,duration_ms,frame_count) VALUES
 (1,'video','movie.mp4','videos/movie.mp4','video/mp4',100,200,1000,1)`,
		`INSERT INTO video_frames(video_id,image_id,frame_index,timestamp_ms) VALUES(1,3,0,0)`,
		`INSERT INTO persons(id,uuid,name) VALUES(1,'u1','Aunt Jessie'),(2,'u2','Bob')`,
		`INSERT INTO face_detections(image_id,person_id,box_x,box_y,box_w,box_h,confidence) VALUES
 (1,1,0.1,0.1,0.2,0.2,0.9),
 (1,1,0.5,0.5,0.2,0.2,0.8),
 (1,2,0.7,0.1,0.1,0.1,0.7),
 (3,2,0.2,0.2,0.2,0.2,0.9)`,
	} {
		if _, err := db.Exec(query); err != nil {
			t.Fatal(err)
		}
	}
	return db
}

func TestPeopleForImagesDedupesAndOrders(t *testing.T) {
	db := setupPeople(t)
	got, err := PeopleForImages(context.Background(), db, []int64{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	if len(got[1]) != 2 || got[1][0] != (PersonRef{ID: 1, Name: "Aunt Jessie"}) || got[1][1] != (PersonRef{ID: 2, Name: "Bob"}) {
		t.Fatalf("image 1: %+v", got[1])
	}
	if len(got[2]) != 0 {
		t.Fatalf("image 2 must have no people: %+v", got[2])
	}
	if len(got[3]) != 1 || got[3][0].ID != 2 {
		t.Fatalf("frame image 3: %+v", got[3])
	}
	empty, err := PeopleForImages(context.Background(), db, nil)
	if err != nil || len(empty) != 0 {
		t.Fatalf("empty ids: %+v %v", empty, err)
	}
}

func TestPeopleForVideosAggregatesFrames(t *testing.T) {
	db := setupPeople(t)
	got, err := PeopleForVideos(context.Background(), db, []int64{1, 999})
	if err != nil {
		t.Fatal(err)
	}
	if len(got[1]) != 1 || got[1][0] != (PersonRef{ID: 2, Name: "Bob"}) {
		t.Fatalf("video 1: %+v", got)
	}
	if len(got[999]) != 0 {
		t.Fatalf("unknown video: %+v", got)
	}
}
