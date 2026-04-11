package sqlitevector

import (
	"context"
	"database/sql"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sync"

	"imgsearch/internal/vectorindex"
)

type Index struct {
	DB          *sql.DB
	Distance    string
	initialized map[int64]bool
	mu          sync.Mutex
}

func NewIndex(db *sql.DB) *Index {
	return &Index{DB: db, Distance: "COSINE", initialized: map[int64]bool{}}
}

func (i *Index) Upsert(ctx context.Context, imageID int64, modelID int64, vec []float32) error {
	if err := i.ensureInitialized(ctx, modelID, len(vec)); err != nil {
		return err
	}
	_, err := i.DB.ExecContext(ctx, `
INSERT INTO image_embeddings(image_id, model_id, dim, vector_blob)
VALUES (?, ?, ?, ?)
ON CONFLICT(image_id, model_id)
DO UPDATE SET
  vector_blob = excluded.vector_blob,
  dim = excluded.dim,
  updated_at = datetime('now')
`, imageID, modelID, len(vec), floatsToBlob(vec))
	if err != nil {
		return fmt.Errorf("update embedding vector for index: %w", err)
	}
	return nil
}

func (i *Index) Delete(ctx context.Context, imageID int64, modelID int64) error {
	_, err := i.DB.ExecContext(ctx, `
DELETE FROM image_embeddings WHERE image_id = ? AND model_id = ?
`, imageID, modelID)
	if err != nil {
		return fmt.Errorf("delete indexed vector: %w", err)
	}
	return nil
}

func (i *Index) Search(ctx context.Context, modelID int64, query []float32, limit int) ([]vectorindex.SearchHit, error) {
	if limit <= 0 {
		limit = 20
	}
	if err := i.ensureInitialized(ctx, modelID, len(query)); err != nil {
		return nil, err
	}

	rows, err := i.DB.QueryContext(ctx, `
SELECT ie.image_id, v.distance
FROM image_embeddings AS ie
JOIN vector_full_scan('image_embeddings', 'vector_blob', ?, ?) AS v
  ON ie.rowid = v.rowid
WHERE ie.model_id = ?
ORDER BY v.distance ASC
LIMIT ?
`, floatsToBlob(query), limit, modelID, limit)
	if err != nil {
		return nil, fmt.Errorf("vector search query: %w", err)
	}
	defer func() { _ = rows.Close() }()

	hits := make([]vectorindex.SearchHit, 0, limit)
	for rows.Next() {
		var h vectorindex.SearchHit
		h.ModelID = modelID
		if err := rows.Scan(&h.ImageID, &h.Distance); err != nil {
			return nil, fmt.Errorf("scan vector hit: %w", err)
		}
		hits = append(hits, h)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate vector hits: %w", err)
	}
	return hits, nil
}

func (i *Index) SearchByImageID(ctx context.Context, modelID int64, imageID int64, limit int) ([]vectorindex.SearchHit, error) {
	var blob []byte
	if err := i.DB.QueryRowContext(ctx, `
SELECT vector_blob FROM image_embeddings WHERE image_id = ? AND model_id = ?
`, imageID, modelID).Scan(&blob); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, vectorindex.ErrNotFound
		}
		return nil, fmt.Errorf("load image query vector: %w", err)
	}

	vec := blobToFloats(blob)
	hits, err := i.Search(ctx, modelID, vec, limit+1)
	if err != nil {
		return nil, err
	}

	out := make([]vectorindex.SearchHit, 0, len(hits))
	for _, hit := range hits {
		if hit.ImageID == imageID {
			continue
		}
		out = append(out, hit)
		if len(out) == limit {
			break
		}
	}
	return out, nil
}

func (i *Index) ensureInitialized(ctx context.Context, modelID int64, dim int) error {
	i.mu.Lock()
	defer i.mu.Unlock()

	if i.initialized[modelID] {
		return nil
	}
	if dim <= 0 {
		return fmt.Errorf("vector dimension must be positive")
	}

	_, err := i.DB.ExecContext(ctx, `
SELECT vector_init('image_embeddings', 'vector_blob', ?)
`, fmt.Sprintf("type=FLOAT32,dimension=%d,distance=%s", dim, i.Distance))
	if err != nil {
		return fmt.Errorf("vector_init failed: %w", err)
	}
	i.initialized[modelID] = true
	return nil
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
