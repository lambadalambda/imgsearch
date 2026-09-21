// Package phash computes a 64-bit difference hash (dHash) of an image so
// resized, re-encoded, or lightly cropped copies land within a small
// Hamming distance of each other.
package phash

import (
	"image"
	"image/color"
	"io"
	"math/bits"

	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"

	_ "golang.org/x/image/webp"
)

// Unhashable marks a row whose file could not be decoded (for example AVIF,
// which has no pure-Go decoder). It is stored so backfills do not retry.
const Unhashable int64 = -1

const (
	gridWidth  = 9
	gridHeight = 8
)

// Compute decodes an image (JPEG, PNG, GIF, WEBP) and returns its dHash.
func Compute(r io.Reader) (uint64, error) {
	img, _, err := image.Decode(r)
	if err != nil {
		return 0, err
	}
	return FromImage(img), nil
}

// FromImage returns the dHash of an already decoded image: the picture is
// reduced to a 9x8 grayscale grid by box averaging, and each bit records
// whether a pixel is brighter than its right-hand neighbour.
func FromImage(img image.Image) uint64 {
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	if width == 0 || height == 0 {
		return 0
	}
	var grid [gridHeight][gridWidth]float64
	for gy := 0; gy < gridHeight; gy++ {
		y0 := bounds.Min.Y + gy*height/gridHeight
		y1 := bounds.Min.Y + (gy+1)*height/gridHeight
		if y1 <= y0 {
			y1 = y0 + 1
		}
		for gx := 0; gx < gridWidth; gx++ {
			x0 := bounds.Min.X + gx*width/gridWidth
			x1 := bounds.Min.X + (gx+1)*width/gridWidth
			if x1 <= x0 {
				x1 = x0 + 1
			}
			var sum float64
			var n float64
			for y := y0; y < y1 && y < bounds.Max.Y; y++ {
				for x := x0; x < x1 && x < bounds.Max.X; x++ {
					sum += luma(img.At(x, y))
					n++
				}
			}
			if n > 0 {
				grid[gy][gx] = sum / n
			}
		}
	}
	var hash uint64
	for gy := 0; gy < gridHeight; gy++ {
		for gx := 0; gx < gridWidth-1; gx++ {
			hash <<= 1
			if grid[gy][gx] > grid[gy][gx+1] {
				hash |= 1
			}
		}
	}
	return hash
}

func luma(c color.Color) float64 {
	r, g, b, _ := c.RGBA()
	return 0.299*float64(r>>8) + 0.587*float64(g>>8) + 0.114*float64(b>>8)
}

// Distance is the Hamming distance between two hashes (0 = identical).
func Distance(a, b uint64) int {
	return bits.OnesCount64(a ^ b)
}

// ToInt64 and FromInt64 convert for storage in an INTEGER column.
func ToInt64(h uint64) int64   { return int64(h) }
func FromInt64(v int64) uint64 { return uint64(v) }
