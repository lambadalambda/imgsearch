package sqlitevector

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"
	"sync"

	"imgsearch/internal/vectorindex"
)

var errQuantizationUnavailable = errors.New("sqlite-vector quantization unavailable")

type quantizationSnapshot struct {
	totalCount      int64
	latestUpdatedAt string
}

type embeddingSnapshot struct {
	modelCount int64
	quantizationSnapshot
}

type Index struct {
	DB                    *sql.DB
	Distance              string
	initialized           map[int64]bool
	quantized             map[int64]quantizationSnapshot
	quantizationAvailable bool
	mu                    sync.Mutex
}

func NewIndex(db *sql.DB) *Index {
	return &Index{
		DB:                    db,
		Distance:              "COSINE",
		initialized:           map[int64]bool{},
		quantized:             map[int64]quantizationSnapshot{},
		quantizationAvailable: true,
	}
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
`, imageID, modelID, len(vec), vectorindex.FloatsToBlob(vec))
	if err != nil {
		return fmt.Errorf("update embedding vector for index: %w", err)
	}
	i.mu.Lock()
	delete(i.quantized, modelID)
	i.mu.Unlock()
	return nil
}

func (i *Index) Delete(ctx context.Context, imageID int64, modelID int64) error {
	_, err := i.DB.ExecContext(ctx, `
DELETE FROM image_embeddings WHERE image_id = ? AND model_id = ?
`, imageID, modelID)
	if err != nil {
		return fmt.Errorf("delete indexed vector: %w", err)
	}
	i.mu.Lock()
	delete(i.quantized, modelID)
	i.mu.Unlock()
	return nil
}

func (i *Index) Search(ctx context.Context, modelID int64, query []float32, limit int) ([]vectorindex.SearchHit, error) {
	if limit <= 0 {
		limit = 20
	}
	if err := i.ensureInitialized(ctx, modelID, len(query)); err != nil {
		return nil, err
	}

	snapshot, err := i.embeddingSnapshot(ctx, modelID)
	if err != nil {
		return nil, err
	}
	if snapshot.modelCount <= 0 {
		return []vectorindex.SearchHit{}, nil
	}

	quantizedK := quantizedK(limit, snapshot.modelCount)
	fullScanK := snapshot.modelCount
	if snapshot.totalCount > fullScanK {
		fullScanK = snapshot.totalCount
	}

	if err := i.ensureQuantized(ctx, modelID, snapshot.quantizationSnapshot); err == nil {
		hits, qerr := i.searchQuantized(ctx, modelID, query, limit, quantizedK)
		if qerr == nil {
			vectorindex.SetSearchDebug(ctx, vectorindex.SearchDebug{Backend: "sqlite-vector", Strategy: "quantize_scan", Quantized: true})
			return hits, nil
		}
		if !isQuantizationUnsupportedErr(qerr) {
			return nil, qerr
		}
		i.mu.Lock()
		i.quantizationAvailable = false
		i.mu.Unlock()
	}

	return i.searchFullScan(ctx, modelID, query, limit, fullScanK)
}

func quantizedK(limit int, modelCount int64) int64 {
	if modelCount <= 0 {
		return 0
	}
	k := int64(limit)
	if k <= 0 {
		k = 20
	}
	if k > modelCount {
		k = modelCount
	}
	if k <= 0 {
		return 1
	}
	return k
}

func (i *Index) searchQuantized(ctx context.Context, modelID int64, query []float32, limit int, scanK int64) ([]vectorindex.SearchHit, error) {
	rows, err := i.DB.QueryContext(ctx, `
SELECT ie.image_id, v.distance
FROM image_embeddings AS ie
JOIN vector_quantize_scan('image_embeddings', 'vector_blob', ?, ?) AS v
  ON ie.rowid = v.rowid
WHERE ie.model_id = ?
ORDER BY v.distance ASC
LIMIT ?
`, vectorindex.FloatsToBlob(query), scanK, modelID, limit)
	if err != nil {
		if isQuantizationUnsupportedErr(err) {
			return nil, err
		}
		return nil, fmt.Errorf("vector quantized search query: %w", err)
	}
	defer func() { _ = rows.Close() }()

	return collectHits(rows, modelID, limit)
}

func (i *Index) searchFullScan(ctx context.Context, modelID int64, query []float32, limit int, scanK int64) ([]vectorindex.SearchHit, error) {
	rows, err := i.DB.QueryContext(ctx, `
SELECT ie.image_id, v.distance
FROM image_embeddings AS ie
JOIN vector_full_scan('image_embeddings', 'vector_blob', ?, ?) AS v
  ON ie.rowid = v.rowid
WHERE ie.model_id = ?
ORDER BY v.distance ASC
LIMIT ?
`, vectorindex.FloatsToBlob(query), scanK, modelID, limit)
	if err != nil {
		return nil, fmt.Errorf("vector full scan query: %w", err)
	}
	defer func() { _ = rows.Close() }()

	vectorindex.SetSearchDebug(ctx, vectorindex.SearchDebug{Backend: "sqlite-vector", Strategy: "full_scan", Quantized: false})
	return collectHits(rows, modelID, limit)
}

func collectHits(rows *sql.Rows, modelID int64, limit int) ([]vectorindex.SearchHit, error) {

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

func (i *Index) countEmbeddings(ctx context.Context, modelID int64) (int64, error) {
	var total int64
	if err := i.DB.QueryRowContext(ctx, `SELECT COUNT(*) FROM image_embeddings WHERE model_id = ?`, modelID).Scan(&total); err != nil {
		return 0, fmt.Errorf("count embeddings: %w", err)
	}
	return total, nil
}

func (i *Index) embeddingSnapshot(ctx context.Context, modelID int64) (embeddingSnapshot, error) {
	var snapshot embeddingSnapshot
	if err := i.DB.QueryRowContext(ctx, `
SELECT
  (SELECT COUNT(*) FROM image_embeddings WHERE model_id = ?) AS model_count,
  COUNT(*) AS total_count,
  COALESCE(MAX(updated_at), '') AS latest_updated_at
FROM image_embeddings
`, modelID).Scan(&snapshot.modelCount, &snapshot.totalCount, &snapshot.latestUpdatedAt); err != nil {
		return embeddingSnapshot{}, fmt.Errorf("embedding snapshot: %w", err)
	}
	return snapshot, nil
}

func (i *Index) ensureQuantized(ctx context.Context, modelID int64, snapshot quantizationSnapshot) error {
	i.mu.Lock()
	if !i.quantizationAvailable {
		i.mu.Unlock()
		return errQuantizationUnavailable
	}
	if !i.needsQuantizationRefresh(modelID, snapshot) {
		i.mu.Unlock()
		return nil
	}
	i.mu.Unlock()

	if _, err := i.DB.ExecContext(ctx, `SELECT vector_quantize('image_embeddings', 'vector_blob')`); err != nil {
		if isQuantizationUnsupportedErr(err) {
			i.mu.Lock()
			i.quantizationAvailable = false
			i.mu.Unlock()
			return errQuantizationUnavailable
		}
		return fmt.Errorf("vector quantize: %w", err)
	}
	if _, err := i.DB.ExecContext(ctx, `SELECT vector_quantize_preload('image_embeddings', 'vector_blob')`); err != nil {
		if isQuantizationUnsupportedErr(err) {
			i.mu.Lock()
			i.quantizationAvailable = false
			i.mu.Unlock()
			return errQuantizationUnavailable
		}
		return fmt.Errorf("vector quantize preload: %w", err)
	}

	i.mu.Lock()
	i.quantized[modelID] = snapshot
	i.mu.Unlock()
	return nil
}

func (i *Index) needsQuantizationRefresh(modelID int64, snapshot quantizationSnapshot) bool {
	known, ok := i.quantized[modelID]
	if !ok {
		return true
	}
	if known.totalCount != snapshot.totalCount {
		return true
	}
	return known.latestUpdatedAt != snapshot.latestUpdatedAt
}

func isQuantizationUnsupportedErr(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	if strings.Contains(msg, "no such function") {
		return true
	}
	if strings.Contains(msg, "no such table") {
		return true
	}
	return strings.Contains(msg, "vector_quantize") || strings.Contains(msg, "vector_quantize_scan")
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

	vec := vectorindex.BlobToFloats(blob)
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
