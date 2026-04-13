//go:build cgo

package llamacppnative

import (
	"image"
	_ "image/jpeg"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func testFixtureImagePath(t *testing.T, name string) string {
	t.Helper()

	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}

	path := filepath.Join(filepath.Dir(thisFile), "..", "..", "..", "fixtures", "images", name)
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("stat fixture %q: %v", path, err)
	}

	return path
}

func TestPreprocessImageForEmbeddingWithVipsgenSupportsWebPAndAVIF(t *testing.T) {
	t.Parallel()

	fixtures := []string{"cat_2.webp", "dog_2.avif"}
	for _, fixture := range fixtures {
		fixture := fixture
		t.Run(fixture, func(t *testing.T) {
			t.Parallel()

			sourcePath := testFixtureImagePath(t, fixture)
			outPath, cleanup, err := preprocessImageForEmbeddingWithVipsgen(sourcePath, 512)
			if err != nil {
				t.Fatalf("preprocess %s: %v", fixture, err)
			}
			if outPath == "" {
				t.Fatal("expected output path")
			}
			if cleanup == nil {
				t.Fatal("expected cleanup function")
			}
			defer cleanup()

			f, err := os.Open(outPath)
			if err != nil {
				t.Fatalf("open output file: %v", err)
			}
			defer func() { _ = f.Close() }()

			cfg, format, err := image.DecodeConfig(f)
			if err != nil {
				t.Fatalf("decode output image config: %v", err)
			}
			if format != "jpeg" {
				t.Fatalf("expected jpeg output format, got %q", format)
			}
			if cfg.Width <= 0 || cfg.Height <= 0 {
				t.Fatalf("unexpected output dimensions: %dx%d", cfg.Width, cfg.Height)
			}
			if cfg.Width > 512 || cfg.Height > 512 {
				t.Fatalf("output exceeds max side: %dx%d", cfg.Width, cfg.Height)
			}
		})
	}
}
