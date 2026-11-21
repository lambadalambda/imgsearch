package faces

import (
	"bytes"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"math"

	"golang.org/x/image/draw"
	"golang.org/x/image/webp"
)

// DecodeImage decodes JPEG, PNG, GIF (first frame), and WebP from memory.
// AVIF has no pure-Go decoder here; callers surface ErrUnsupportedFormat so
// the worker can skip the file instead of failing loudly.
func DecodeImage(content []byte) (image.Image, error) {
	if len(bytes.TrimSpace(content)) == 0 {
		return nil, fmt.Errorf("empty image content")
	}
	if img, err := webp.Decode(bytes.NewReader(content)); err == nil {
		return img, nil
	}
	img, format, err := image.Decode(bytes.NewReader(content))
	if err != nil {
		if isAVIF(content) {
			return nil, ErrUnsupportedFormat
		}
		return nil, fmt.Errorf("decode image (format %q): %w", format, err)
	}
	return img, nil
}

func isAVIF(content []byte) bool {
	if len(content) < 12 {
		return false
	}
	return bytes.Equal(content[4:8], []byte("ftyp")) &&
		(bytes.Equal(content[8:12], []byte("avif")) || bytes.Equal(content[8:12], []byte("avis")))
}

// Letterbox scales img to fit inside size x size, pastes it at the top-left
// of a black canvas, and returns the canvas plus the scale factor
// (canvas coordinates divide by scale to recover original coordinates).
func Letterbox(img image.Image, size int) (*image.RGBA, float64) {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	if w <= 0 || h <= 0 {
		return image.NewRGBA(image.Rect(0, 0, size, size)), 1
	}
	scale := float64(size) / math.Max(float64(w), float64(h))
	nw, nh := int(math.Round(float64(w)*scale)), int(math.Round(float64(h)*scale))
	if nw < 1 {
		nw = 1
	}
	if nh < 1 {
		nh = 1
	}
	canvas := image.NewRGBA(image.Rect(0, 0, size, size))
	scaled := image.NewRGBA(image.Rect(0, 0, nw, nh))
	draw.ApproxBiLinear.Scale(scaled, scaled.Bounds(), img, b, draw.Over, nil)
	draw.Draw(canvas, image.Rect(0, 0, nw, nh), scaled, image.Point{}, draw.Src)
	return canvas, float64(nh) / float64(h)
}

// RGBPixels returns row-major normalized RGB triples in 0..1 for a canvas.
func RGBPixels(canvas *image.RGBA) []float32 {
	b := canvas.Bounds()
	w, h := b.Dx(), b.Dy()
	out := make([]float32, 0, w*h*3)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, bl, _ := canvas.At(b.Min.X+x, b.Min.Y+y).RGBA()
			out = append(out, float32(r>>8)/255, float32(g>>8)/255, float32(bl>>8)/255)
		}
	}
	return out
}

// DetBlob builds the SCRFD input blob: NCHW RGB floats of (v - 127.5) / 128.
func DetBlob(canvas *image.RGBA) []float32 {
	b := canvas.Bounds()
	w, h := b.Dx(), b.Dy()
	chw := make([]float32, 0, 3*w*h)
	for c := 0; c < 3; c++ {
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				r, g, bl, _ := canvas.At(b.Min.X+x, b.Min.Y+y).RGBA()
				var v uint32
				switch c {
				case 0:
					v = r >> 8
				case 1:
					v = g >> 8
				default:
					v = bl >> 8
				}
				chw = append(chw, (float32(v)-127.5)/128)
			}
		}
	}
	return chw
}
