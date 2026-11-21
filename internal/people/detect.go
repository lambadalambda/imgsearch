package people

import (
	"context"
	"database/sql"
	"fmt"
	"math"

	"imgsearch/internal/vectorindex"
)

// Detection is one face found in an image file. Box coordinates are relative
// (0..1). Embedding is a normalized identity vector (e.g. ArcFace 512-d);
// empty means the detector found a face it could not embed.
type Detection struct {
	BoxX       float64
	BoxY       float64
	BoxW       float64
	BoxH       float64
	Confidence float64
	Embedding  []float32
}

// Detector finds faces in a stored image file. Implementations stay behind
// this interface so the worker never depends on a concrete model runtime.
type Detector interface {
	DetectFaces(ctx context.Context, imagePath string) ([]Detection, error)
}

// DefaultMatchThreshold is the minimum cosine similarity for two faces to
// count as the same person. ArcFace-class embeddings typically score above
// 0.4 for same identities; 0.5 favors splitting over wrong merges.
const DefaultMatchThreshold = 0.5

func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func normalize(v []float32) []float32 {
	var n float64
	for _, x := range v {
		n += float64(x) * float64(x)
	}
	if n == 0 {
		return v
	}
	s := float32(math.Sqrt(n))
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = x / s
	}
	return out
}

// AssignDetections stores faces for one image and clusters them greedily:
// each detection joins the person whose embedding centroid is most similar
// above threshold, otherwise it founds a new neutral person. Detections are
// matched in order, so later faces in the same image see earlier ones.
// Re-running for an image replaces its faces, so crashed jobs can safely
// retry without duplicating rows.
func AssignDetections(ctx context.Context, db *sql.DB, imageID int64, dets []Detection, threshold float64) error {
	if db == nil {
		return fmt.Errorf("db is nil")
	}
	var exists bool
	if err := db.QueryRowContext(ctx, `SELECT EXISTS(SELECT 1 FROM images WHERE id=?)`, imageID).Scan(&exists); err != nil {
		return err
	}
	if !exists {
		return problem(404, "image not found")
	}
	for i, d := range dets {
		if len(d.Embedding) == 0 {
			return fmt.Errorf("detection %d has no embedding", i)
		}
		if cosineSimilarity(d.Embedding, d.Embedding) == 0 {
			return fmt.Errorf("detection %d has a zero embedding", i)
		}
	}
	if len(dets) == 0 {
		return nil
	}

	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()

	// Idempotent re-runs: a retried job replaces this image's faces.
	if _, err := tx.ExecContext(ctx, `DELETE FROM face_detections WHERE image_id=?`, imageID); err != nil {
		return err
	}

	type centroid struct {
		personID int64
		vec      []float32
	}
	loadCentroids := func() ([]centroid, error) {
		rows, err := tx.QueryContext(ctx, `
SELECT person_id, embedding_blob FROM face_detections
WHERE person_id IS NOT NULL AND embedding_blob IS NOT NULL`)
		if err != nil {
			return nil, err
		}
		defer rows.Close()
		sums := map[int64][]float64{}
		counts := map[int64]int{}
		dim := 0
		for rows.Next() {
			var pid int64
			var blob []byte
			if err := rows.Scan(&pid, &blob); err != nil {
				return nil, err
			}
			vec := vectorindex.BlobToFloats(blob)
			if dim == 0 {
				dim = len(vec)
			}
			if len(vec) != dim {
				continue
			}
			if sums[pid] == nil {
				sums[pid] = make([]float64, dim)
			}
			for i, x := range vec {
				sums[pid][i] += float64(x)
			}
			counts[pid]++
		}
		if err := rows.Err(); err != nil {
			return nil, err
		}
		out := make([]centroid, 0, len(sums))
		for pid, sum := range sums {
			c := make([]float32, dim)
			for i, s := range sum {
				c[i] = float32(s / float64(counts[pid]))
			}
			out = append(out, centroid{personID: pid, vec: normalize(c)})
		}
		return out, nil
	}

	for _, d := range dets {
		vec := normalize(d.Embedding)
		centroids, err := loadCentroids()
		if err != nil {
			return err
		}
		bestID := int64(0)
		best := threshold
		for _, c := range centroids {
			if len(c.vec) != len(vec) {
				continue
			}
			if sim := cosineSimilarity(vec, c.vec); sim > best {
				best, bestID = sim, c.personID
			}
		}
		var pid *int64
		if bestID != 0 {
			pid = &bestID
		} else {
			uuid, err := newUUID()
			if err != nil {
				return err
			}
			var p Person
			if err := tx.QueryRowContext(ctx, `INSERT INTO persons(uuid,name) VALUES(?,'tmp') RETURNING id,uuid,name,created_at`, uuid).
				Scan(&p.ID, &p.UUID, &p.Name, &p.CreatedAt); err != nil {
				return err
			}
			name := fmt.Sprintf("Person %d", p.ID)
			if err := tx.QueryRowContext(ctx, `UPDATE persons SET name=? WHERE id=? RETURNING id,uuid,name,created_at`, name, p.ID).
				Scan(&p.ID, &p.UUID, &p.Name, &p.CreatedAt); err != nil {
				return err
			}
			pid = &p.ID
		}
		if _, err := tx.ExecContext(ctx, `
INSERT INTO face_detections(image_id,person_id,box_x,box_y,box_w,box_h,embedding_blob,confidence)
VALUES(?,?,?,?,?,?,?,?)`,
			imageID, pid, d.BoxX, d.BoxY, d.BoxW, d.BoxH, vectorindex.FloatsToBlob(vec), d.Confidence); err != nil {
			return err
		}
	}
	return tx.Commit()
}
