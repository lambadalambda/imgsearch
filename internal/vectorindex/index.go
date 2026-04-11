package vectorindex

import (
	"context"
	"errors"
)

var ErrNotFound = errors.New("vector not found")

type SearchHit struct {
	ImageID  int64
	ModelID  int64
	Distance float64
}

type VectorIndex interface {
	Upsert(ctx context.Context, imageID int64, modelID int64, vec []float32) error
	Delete(ctx context.Context, imageID int64, modelID int64) error
	Search(ctx context.Context, modelID int64, query []float32, limit int) ([]SearchHit, error)
	SearchByImageID(ctx context.Context, modelID int64, imageID int64, limit int) ([]SearchHit, error)
}
