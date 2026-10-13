// Package retrievalbench scores embedding models on captioned image sets.
package retrievalbench

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"math"
)

// Item is one image with its reference captions.
type Item struct {
	Image    string
	Captions []string
}

// Result holds Recall@K for each direction, keyed by K.
type Result struct {
	TextToImage map[int]float64
	ImageToText map[int]float64
}

// ParseFlickrCSV reads the Karpathy-split CSV layout: a `filename` column and
// a `raw` column holding a JSON array of captions.
func ParseFlickrCSV(r io.Reader) ([]Item, error) {
	rows, err := csv.NewReader(r).ReadAll()
	if err != nil {
		return nil, fmt.Errorf("read csv: %w", err)
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("empty csv")
	}
	filenameCol, rawCol := -1, -1
	for i, name := range rows[0] {
		switch name {
		case "filename":
			filenameCol = i
		case "raw":
			rawCol = i
		}
	}
	if filenameCol < 0 || rawCol < 0 {
		return nil, fmt.Errorf("csv needs filename and raw columns, got %v", rows[0])
	}

	items := make([]Item, 0, len(rows)-1)
	for line, row := range rows[1:] {
		var captions []string
		if err := json.Unmarshal([]byte(row[rawCol]), &captions); err != nil {
			return nil, fmt.Errorf("row %d captions: %w", line+2, err)
		}
		if len(captions) == 0 {
			return nil, fmt.Errorf("row %d (%s) has no captions", line+2, row[filenameCol])
		}
		items = append(items, Item{Image: row[filenameCol], Captions: captions})
	}
	return items, nil
}

// Evaluate computes Recall@K in both directions. Vectors must be
// L2-normalized; captionImage maps each caption to its image index. Ties
// count against the target.
func Evaluate(images, captions [][]float32, captionImage []int, ks []int) Result {
	scores := make([][]float64, len(captions))
	for c, cv := range captions {
		scores[c] = make([]float64, len(images))
		for i, iv := range images {
			scores[c][i] = dot(cv, iv)
		}
	}

	t2iRanks := make([]int, len(captions))
	for c, target := range captionImage {
		t2iRanks[c] = countAtLeast(len(images), func(i int) bool {
			return i != target && scores[c][i] >= scores[c][target]
		})
	}

	i2tRanks := make([]int, len(images))
	for i := range images {
		best := math.Inf(-1)
		for c, owner := range captionImage {
			if owner == i && scores[c][i] > best {
				best = scores[c][i]
			}
		}
		i2tRanks[i] = countAtLeast(len(captions), func(c int) bool {
			return captionImage[c] != i && scores[c][i] >= best
		})
	}

	return Result{TextToImage: recallAt(t2iRanks, ks), ImageToText: recallAt(i2tRanks, ks)}
}

// TruncateNormalize keeps the first dims values (Matryoshka truncation) and
// re-normalizes. dims <= 0 or >= len(vec) keeps every value.
func TruncateNormalize(vec []float32, dims int) []float32 {
	if dims <= 0 || dims > len(vec) {
		dims = len(vec)
	}
	out := make([]float32, dims)
	copy(out, vec[:dims])
	norm := math.Sqrt(dot(out, out))
	if norm == 0 {
		return out
	}
	for i := range out {
		out[i] = float32(float64(out[i]) / norm)
	}
	return out
}

func dot(a, b []float32) float64 {
	var sum float64
	for i := range a {
		sum += float64(a[i]) * float64(b[i])
	}
	return sum
}

func countAtLeast(n int, match func(int) bool) int {
	count := 0
	for i := 0; i < n; i++ {
		if match(i) {
			count++
		}
	}
	return count
}

func recallAt(ranks []int, ks []int) map[int]float64 {
	out := make(map[int]float64, len(ks))
	if len(ranks) == 0 {
		return out
	}
	for _, k := range ks {
		hits := 0
		for _, rank := range ranks {
			if rank < k {
				hits++
			}
		}
		out[k] = float64(hits) / float64(len(ranks))
	}
	return out
}
