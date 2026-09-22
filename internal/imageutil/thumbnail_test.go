//go:build cgo

package imageutil

import (
	"bytes"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"testing"

	_ "golang.org/x/image/webp"
	"imgsearch/internal/exif"
)

func TestThumbnailWebP(t *testing.T) {
	for _, tc := range []struct {
		name                                              string
		width, height, orientation, wantWidth, wantHeight int
	}{
		{"landscape", 1280, 800, 1, 640, 400},
		{"portrait", 800, 1280, 1, 400, 640},
		{"small", 80, 50, 1, 80, 50},
		{"rotated", 1280, 800, 6, 400, 640},
	} {
		t.Run(tc.name, func(t *testing.T) {
			img := image.NewRGBA(image.Rect(0, 0, tc.width, tc.height))
			for y := 0; y < tc.height; y++ {
				for x := 0; x < tc.width; x++ {
					img.Set(x, y, color.RGBA{uint8(x), uint8(y), 127, 255})
				}
			}
			var input bytes.Buffer
			if err := jpeg.Encode(&input, img, nil); err != nil {
				t.Fatal(err)
			}
			original := exif.InsertAPP1(input.Bytes(), exif.BuildAPP1(tc.orientation, "2024:05:06 07:08:09"))
			dir := t.TempDir()
			source := filepath.Join(dir, "original")
			output := filepath.Join(dir, "thumb.webp")
			if err := os.WriteFile(source, original, 0600); err != nil {
				t.Fatal(err)
			}
			if err := ThumbnailWebP(source, output, 640); err != nil {
				t.Fatal(err)
			}
			data, err := os.ReadFile(output)
			if err != nil {
				t.Fatal(err)
			}
			cfg, format, err := image.DecodeConfig(bytes.NewReader(data))
			if err != nil {
				t.Fatal(err)
			}
			if format != "webp" || cfg.Width != tc.wantWidth || cfg.Height != tc.wantHeight {
				t.Fatalf("got %s %dx%d, want webp %dx%d", format, cfg.Width, cfg.Height, tc.wantWidth, tc.wantHeight)
			}
			unchanged, _ := os.ReadFile(source)
			if !bytes.Equal(original, unchanged) {
				t.Fatal("original changed")
			}
			if tc.name == "landscape" && len(data) >= len(original) {
				t.Fatalf("thumbnail %d bytes not smaller than original %d", len(data), len(original))
			}
			t.Logf("original=%d bytes thumbnail=%d bytes", len(original), len(data))
		})
	}
}

func TestThumbnailWebPSupportsFormatsAndTransparency(t *testing.T) {
	dir := t.TempDir()
	img := image.NewNRGBA(image.Rect(0, 0, 32, 16))
	img.SetNRGBA(16, 8, color.NRGBA{255, 0, 0, 255})
	var b bytes.Buffer
	if err := png.Encode(&b, img); err != nil {
		t.Fatal(err)
	}
	source := filepath.Join(dir, "alpha.png")
	if err := os.WriteFile(source, b.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	paths := []string{source, "../../fixtures/images/cat_2.webp", "../../fixtures/images/dog_2.avif"}
	for _, src := range paths {
		output := filepath.Join(dir, "out.webp")
		if err := ThumbnailWebP(src, output, 640); err != nil {
			t.Fatal(err)
		}
		data, _ := os.ReadFile(output)
		decoded, format, err := image.Decode(bytes.NewReader(data))
		if err != nil || format != "webp" {
			t.Fatalf("decode: %s %v", format, err)
		}
		if decoded.Bounds().Dx() > 640 || decoded.Bounds().Dy() > 640 {
			t.Fatal("unbounded thumbnail")
		}
		if src == source {
			_, _, _, alpha := decoded.At(0, 0).RGBA()
			if alpha != 0 {
				t.Fatal("lost transparency")
			}
		}
	}
	if err := ThumbnailWebP(source, filepath.Join(dir, "bad.webp"), 0); err == nil {
		t.Fatal("accepted invalid size")
	}
	if err := ThumbnailWebP(filepath.Join(dir, "missing"), filepath.Join(dir, "bad.webp"), 640); err == nil {
		t.Fatal("accepted missing source")
	}
}
