package bruteforce

import (
	"context"
	"database/sql"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sort"

	"imgsearch/internal/vectorindex"
)

type Index struct {
	DB *sql.DB
}

func NewIndex(db *sql.DB) *Index {
	return &Index{DB: db}
}

func (i *Index) Upsert(ctx context.Context, imageID int64, modelID int64, vec []float32) error {
	if i == nil || i.DB == nil {
		return fmt.Errorf("index db is nil")
	}
	if len(vec) == 0 {
		return fmt.Errorf("vector must not be empty")
	}

	_, err := i.DB.ExecContext(ctx, `
INSERT INTO image_embeddings(image_id, model_id, dim, vector_blob)
VALUES (?, ?, ?, ?)
ON CONFLICT(image_id, model_id)
DO UPDATE SET
  dim = excluded.dim,
  vector_blob = excluded.vector_blob,
  updated_at = datetime('now')
`, imageID, modelID, len(vec), floatsToBlob(vec))
	if err != nil {
		return fmt.Errorf("upsert embedding vector: %w", err)
	}
	return nil
}

func (i *Index) Delete(ctx context.Context, imageID int64, modelID int64) error {
	if i == nil || i.DB == nil {
		return fmt.Errorf("index db is nil")
	}

	_, err := i.DB.ExecContext(ctx, `DELETE FROM image_embeddings WHERE image_id = ? AND model_id = ?`, imageID, modelID)
	if err != nil {
		return fmt.Errorf("delete embedding vector: %w", err)
	}
	return nil
}

func (i *Index) Search(ctx context.Context, modelID int64, query []float32, limit int) ([]vectorindex.SearchHit, error) {
	if i == nil || i.DB == nil {
		return nil, fmt.Errorf("index db is nil")
	}
	if len(query) == 0 {
		return nil, fmt.Errorf("query vector must not be empty")
	}
	if limit <= 0 {
		limit = 20
	}

	rows, err := i.DB.QueryContext(ctx, `
SELECT image_id, vector_blob
FROM image_embeddings
WHERE model_id = ?
`, modelID)
	if err != nil {
		return nil, fmt.Errorf("query embedding vectors: %w", err)
	}
	defer func() { _ = rows.Close() }()

	hits := make([]vectorindex.SearchHit, 0, limit)
	for rows.Next() {
		var imageID int64
		var blob []byte
		if err := rows.Scan(&imageID, &blob); err != nil {
			return nil, fmt.Errorf("scan embedding vector: %w", err)
		}

		vec := blobToFloats(blob)
		if len(vec) == 0 {
			return nil, fmt.Errorf("decode embedding vector for image %d", imageID)
		}

		similarity := cosine(query, vec)
		hits = append(hits, vectorindex.SearchHit{
			ImageID:  imageID,
			ModelID:  modelID,
			Distance: 1 - similarity,
		})
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate embedding vectors: %w", err)
	}

	sort.Slice(hits, func(a, b int) bool {
		if hits[a].Distance == hits[b].Distance {
			return hits[a].ImageID < hits[b].ImageID
		}
		return hits[a].Distance < hits[b].Distance
	})

	if len(hits) > limit {
		hits = hits[:limit]
	}
	return hits, nil
}

func (i *Index) SearchByImageID(ctx context.Context, modelID int64, imageID int64, limit int) ([]vectorindex.SearchHit, error) {
	if i == nil || i.DB == nil {
		return nil, fmt.Errorf("index db is nil")
	}

	var blob []byte
	if err := i.DB.QueryRowContext(ctx, `
SELECT vector_blob
FROM image_embeddings
WHERE image_id = ? AND model_id = ?
`, imageID, modelID).Scan(&blob); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, vectorindex.ErrNotFound
		}
		return nil, fmt.Errorf("load query image vector: %w", err)
	}

	query := blobToFloats(blob)
	if len(query) == 0 {
		return nil, fmt.Errorf("decode query image vector")
	}

	hits, err := i.Search(ctx, modelID, query, limit+1)
	if err != nil {
		return nil, err
	}

	filtered := make([]vectorindex.SearchHit, 0, len(hits))
	for _, hit := range hits {
		if hit.ImageID == imageID {
			continue
		}
		filtered = append(filtered, hit)
		if limit > 0 && len(filtered) >= limit {
			break
		}
	}
	return filtered, nil
}

func cosine(a, b []float32) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	if n == 0 {
		return 0
	}

	var dot float64
	var na float64
	var nb float64
	for i := 0; i < n; i++ {
		av := float64(a[i])
		bv := float64(b[i])
		dot += av * bv
		na += av * av
		nb += bv * bv
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func floatsToBlob(values []float32) []byte {
	blob := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(blob[i*4:], math.Float32bits(v))
	}
	return blob
}

func blobToFloats(blob []byte) []float32 {
	if len(blob)%4 != 0 {
		return nil
	}
	out := make([]float32, len(blob)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(blob[i*4:]))
	}
	return out
}
