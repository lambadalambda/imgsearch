//go:build cgo

package imageutil

import (
	"bytes"
	"image"
	"image/color"
	"image/gif"
	"os"
	"path/filepath"
	"testing"
)

func TestThumbnailWebPUsesFirstAnimationFrame(t *testing.T) {
	palette := color.Palette{color.RGBA{255, 0, 0, 255}, color.RGBA{0, 0, 255, 255}}
	first := image.NewPaletted(image.Rect(0, 0, 40, 20), palette)
	second := image.NewPaletted(first.Rect, palette)
	for i := range second.Pix {
		second.Pix[i] = 1
	}
	var input bytes.Buffer
	if err := gif.EncodeAll(&input, &gif.GIF{Image: []*image.Paletted{first, second}, Delay: []int{10, 10}}); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	source := filepath.Join(dir, "animated.gif")
	output := filepath.Join(dir, "thumb.webp")
	if err := os.WriteFile(source, input.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	if err := ThumbnailWebP(source, output, 640); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	img, format, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	if format != "webp" || img.Bounds().Dx() != 40 || img.Bounds().Dy() != 20 {
		t.Fatalf("format=%s bounds=%v", format, img.Bounds())
	}
	r, _, b, _ := img.At(20, 10).RGBA()
	if r <= b {
		t.Fatal("thumbnail did not use the red first frame")
	}
}
