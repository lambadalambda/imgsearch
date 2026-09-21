package phash

import (
	"bytes"
	"image"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func fixture(t *testing.T, name string) []byte {
	t.Helper()
	_, thisFile, _, _ := runtime.Caller(0)
	data, err := os.ReadFile(filepath.Join(filepath.Dir(thisFile), "..", "..", "fixtures", "images", name))
	if err != nil {
		t.Fatalf("read fixture %s: %v", name, err)
	}
	return data
}

func hashOf(t *testing.T, name string) uint64 {
	t.Helper()
	h, err := Compute(bytes.NewReader(fixture(t, name)))
	if err != nil {
		t.Fatalf("compute %s: %v", name, err)
	}
	return h
}

func TestSameImageAcrossFormatsIsClose(t *testing.T) {
	jpeg := hashOf(t, "cat_2.jpg")
	webp := hashOf(t, "cat_2.webp")
	if d := Distance(jpeg, webp); d > 6 {
		t.Fatalf("cat_2 jpeg vs webp distance %d, want <= 6", d)
	}
	dogJPEG := hashOf(t, "dog_2.jpg")
	if d := Distance(jpeg, dogJPEG); d < 16 {
		t.Fatalf("cat vs dog distance %d, want >= 16", d)
	}
	if d := Distance(hashOf(t, "cat_1.jpg"), jpeg); d < 10 {
		t.Fatalf("two different cat photos distance %d, want >= 10", d)
	}
}

func TestResizedCopyIsClose(t *testing.T) {
	src := hashOf(t, "woman.jpg")
	img, _, err := image.Decode(bytes.NewReader(fixture(t, "woman.jpg")))
	if err != nil {
		t.Fatal(err)
	}
	// Nearest-neighbour downscale to a third of the size.
	b := img.Bounds()
	small := image.NewRGBA(image.Rect(0, 0, b.Dx()/3, b.Dy()/3))
	for y := 0; y < small.Bounds().Dy(); y++ {
		for x := 0; x < small.Bounds().Dx(); x++ {
			small.Set(x, y, img.At(b.Min.X+x*3, b.Min.Y+y*3))
		}
	}
	if d := Distance(src, FromImage(small)); d > 6 {
		t.Fatalf("resized copy distance %d, want <= 6", d)
	}
}

func TestUnsupportedAndDegenerateInput(t *testing.T) {
	if _, err := Compute(bytes.NewReader(fixture(t, "dog_2.avif"))); err == nil {
		t.Fatal("expected AVIF to be undecodable without a decoder")
	}
	if got := FromImage(image.NewGray(image.Rect(0, 0, 0, 0))); got != 0 {
		t.Fatalf("empty image hash: %d", got)
	}
	flat := image.NewGray(image.Rect(0, 0, 32, 32))
	for i := range flat.Pix {
		flat.Pix[i] = 128
	}
	if got := FromImage(flat); got != 0 {
		t.Fatalf("uniform image hash: %d (no gradient expected)", got)
	}
	if Distance(0, ^uint64(0)) != 64 || Distance(5, 5) != 0 {
		t.Fatal("distance arithmetic")
	}
	if FromInt64(ToInt64(^uint64(0))) != ^uint64(0) {
		t.Fatal("int64 round trip")
	}
}
