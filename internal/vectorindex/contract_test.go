package vectorindex

import (
	"context"
	"sort"
	"testing"
)

type fakeIndex struct {
	byModel map[int64]map[int64][]float32
}

func newFakeIndex() *fakeIndex {
	return &fakeIndex{byModel: make(map[int64]map[int64][]float32)}
}

func (f *fakeIndex) Upsert(_ context.Context, imageID int64, modelID int64, vec []float32) error {
	m, ok := f.byModel[modelID]
	if !ok {
		m = make(map[int64][]float32)
		f.byModel[modelID] = m
	}
	cp := make([]float32, len(vec))
	copy(cp, vec)
	m[imageID] = cp
	return nil
}

func (f *fakeIndex) Delete(_ context.Context, imageID int64, modelID int64) error {
	if m, ok := f.byModel[modelID]; ok {
		delete(m, imageID)
	}
	return nil
}

func (f *fakeIndex) Search(_ context.Context, modelID int64, query []float32, limit int) ([]SearchHit, error) {
	m := f.byModel[modelID]
	hits := make([]SearchHit, 0, len(m))
	for imageID, vec := range m {
		hits = append(hits, SearchHit{
			ImageID:  imageID,
			ModelID:  modelID,
			Distance: l2(query, vec),
		})
	}
	sort.Slice(hits, func(i, j int) bool { return hits[i].Distance < hits[j].Distance })
	if limit > 0 && len(hits) > limit {
		hits = hits[:limit]
	}
	return hits, nil
}

func (f *fakeIndex) SearchByImageID(ctx context.Context, modelID int64, imageID int64, limit int) ([]SearchHit, error) {
	query := f.byModel[modelID][imageID]
	hits, err := f.Search(ctx, modelID, query, 0)
	if err != nil {
		return nil, err
	}

	filtered := hits[:0]
	for _, h := range hits {
		if h.ImageID == imageID {
			continue
		}
		filtered = append(filtered, h)
	}
	if limit > 0 && len(filtered) > limit {
		filtered = filtered[:limit]
	}
	return filtered, nil
}

func l2(a, b []float32) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var sum float64
	for i := 0; i < n; i++ {
		d := float64(a[i] - b[i])
		sum += d * d
	}
	return sum
}

func TestVectorIndexContractUpsertSearchDelete(t *testing.T) {
	ctx := context.Background()
	var idx VectorIndex = newFakeIndex()

	if err := idx.Upsert(ctx, 1, 10, []float32{1, 0}); err != nil {
		t.Fatalf("upsert image 1: %v", err)
	}
	if err := idx.Upsert(ctx, 2, 10, []float32{0, 1}); err != nil {
		t.Fatalf("upsert image 2: %v", err)
	}

	hits, err := idx.Search(ctx, 10, []float32{1, 0}, 2)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(hits) != 2 {
		t.Fatalf("expected 2 hits, got %d", len(hits))
	}
	if hits[0].ImageID != 1 {
		t.Fatalf("expected closest image 1, got %d", hits[0].ImageID)
	}

	if err := idx.Delete(ctx, 1, 10); err != nil {
		t.Fatalf("delete image 1: %v", err)
	}

	hits, err = idx.Search(ctx, 10, []float32{1, 0}, 2)
	if err != nil {
		t.Fatalf("search after delete: %v", err)
	}
	if len(hits) != 1 || hits[0].ImageID != 2 {
		t.Fatalf("unexpected hits after delete: %+v", hits)
	}
}

func TestVectorIndexContractSearchByImageIDExcludesSelf(t *testing.T) {
	ctx := context.Background()
	var idx VectorIndex = newFakeIndex()

	_ = idx.Upsert(ctx, 1, 10, []float32{1, 0})
	_ = idx.Upsert(ctx, 2, 10, []float32{0.9, 0.1})
	_ = idx.Upsert(ctx, 3, 10, []float32{0, 1})

	hits, err := idx.SearchByImageID(ctx, 10, 1, 2)
	if err != nil {
		t.Fatalf("search by image id: %v", err)
	}
	if len(hits) != 2 {
		t.Fatalf("expected 2 hits, got %d", len(hits))
	}
	if hits[0].ImageID == 1 || hits[1].ImageID == 1 {
		t.Fatalf("self match should be excluded: %+v", hits)
	}
	if hits[0].ImageID != 2 {
		t.Fatalf("expected closest neighbor to be 2, got %d", hits[0].ImageID)
	}
}
