package people

import (
	"context"
	"database/sql"
	"testing"

	_ "github.com/mattn/go-sqlite3"
	dbpkg "imgsearch/internal/db"
)

func setupDetect(t *testing.T) *sql.DB {
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
	if _, err := db.Exec(`INSERT INTO images(id,sha256,original_name,storage_path,mime_type,width,height,tags_json) VALUES
 (1,'one','one.jpg','images/one.jpg','image/jpeg',100,200,'[]'),
 (2,'two','two.jpg','images/two.jpg','image/jpeg',100,200,'[]')`); err != nil {
		t.Fatal(err)
	}
	return db
}

// unit returns a unit vector along one axis so same-axis pairs match.
func unit(dim, axis int) []float32 {
	v := make([]float32, dim)
	v[axis] = 1
	return v
}

func faceCount(t *testing.T, db *sql.DB, personID int64) int {
	t.Helper()
	var n int
	if err := db.QueryRow(`SELECT COUNT(*) FROM face_detections WHERE person_id=?`, personID).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}

func TestAssignDetectionsCreatesAndMatchesPersons(t *testing.T) {
	db := setupDetect(t)
	ctx := context.Background()

	if err := AssignDetections(ctx, db, 1, []Detection{
		{BoxX: 0.1, BoxY: 0.1, BoxW: 0.2, BoxH: 0.2, Confidence: 0.9, Embedding: unit(4, 0)},
		{BoxX: 0.5, BoxY: 0.5, BoxW: 0.2, BoxH: 0.2, Confidence: 0.9, Embedding: unit(4, 1)},
	}, 0.5); err != nil {
		t.Fatal(err)
	}
	var persons int
	if err := db.QueryRow(`SELECT COUNT(*) FROM persons`).Scan(&persons); err != nil {
		t.Fatal(err)
	}
	if persons != 2 {
		t.Fatalf("two distinct faces must create two persons, got %d", persons)
	}

	// A near-identical embedding joins the existing person; a new axis splits.
	if err := AssignDetections(ctx, db, 2, []Detection{
		{Confidence: 0.9, Embedding: unit(4, 0)},
		{Confidence: 0.9, Embedding: unit(4, 2)},
	}, 0.5); err != nil {
		t.Fatal(err)
	}
	if err := db.QueryRow(`SELECT COUNT(*) FROM persons`).Scan(&persons); err != nil {
		t.Fatal(err)
	}
	if persons != 3 {
		t.Fatalf("one match + one new person, got %d persons", persons)
	}
	var matchedID int64
	if err := db.QueryRow(`SELECT person_id FROM face_detections WHERE image_id=2 AND box_x=0 AND box_y=0`).Scan(&matchedID); err != nil {
		t.Fatal(err)
	}
	var firstID int64
	if err := db.QueryRow(`SELECT person_id FROM face_detections WHERE image_id=1 AND box_x=0.1`).Scan(&firstID); err != nil {
		t.Fatal(err)
	}
	if matchedID != firstID {
		t.Fatalf("same identity must reuse person %d, got %d", firstID, matchedID)
	}

	// Re-running the same image replaces its faces instead of duplicating.
	if err := AssignDetections(ctx, db, 1, []Detection{
		{Confidence: 0.9, Embedding: unit(4, 0)},
	}, 0.5); err != nil {
		t.Fatal(err)
	}
	var faces int
	if err := db.QueryRow(`SELECT COUNT(*) FROM face_detections WHERE image_id=1`).Scan(&faces); err != nil {
		t.Fatal(err)
	}
	if faces != 1 {
		t.Fatalf("re-run must replace faces, got %d rows", faces)
	}
}

func TestAssignDetectionsRejectsBadInput(t *testing.T) {
	db := setupDetect(t)
	ctx := context.Background()
	for _, tc := range []struct {
		name string
		dets []Detection
	}{
		{"empty embedding", []Detection{{Confidence: 0.9}}},
		{"zero vector", []Detection{{Confidence: 0.9, Embedding: make([]float32, 4)}}},
	} {
		if err := AssignDetections(ctx, db, 1, tc.dets, 0.5); err == nil {
			t.Fatalf("%s must fail", tc.name)
		}
	}
	if err := AssignDetections(ctx, db, 999, []Detection{{Confidence: 0.9, Embedding: unit(4, 0)}}, 0.5); err == nil {
		t.Fatal("unknown image must fail")
	}
	// Threshold above 1 matches nothing: every face becomes its own person.
	if err := AssignDetections(ctx, db, 1, []Detection{
		{Confidence: 0.9, Embedding: unit(4, 0)},
		{Confidence: 0.9, Embedding: unit(4, 0)},
	}, 1.5); err != nil {
		t.Fatal(err)
	}
	var persons int
	if err := db.QueryRow(`SELECT COUNT(*) FROM persons`).Scan(&persons); err != nil {
		t.Fatal(err)
	}
	if persons != 2 {
		t.Fatalf("impossible threshold must split, got %d persons", persons)
	}
}

func TestCosineSimilarity(t *testing.T) {
	if got := cosineSimilarity(unit(4, 0), unit(4, 0)); got != 1 {
		t.Fatalf("identical = 1, got %v", got)
	}
	if got := cosineSimilarity(unit(4, 0), unit(4, 1)); got != 0 {
		t.Fatalf("orthogonal = 0, got %v", got)
	}
}
