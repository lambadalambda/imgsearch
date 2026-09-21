//go:build cgo

package llamacppnative

import (
	"bytes"
	"context"
	"image"
	"testing"
)

func TestPrepareImageJPEGResizesAndEncodesJPEG(t *testing.T) {
	data, mime, err := PrepareImageJPEG(context.Background(), testFixtureImagePath(t, "cat_2.webp"), 256)
	if err != nil {
		t.Fatalf("prepare: %v", err)
	}
	if mime != "image/jpeg" {
		t.Fatalf("mime: %q", mime)
	}
	cfg, format, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil || format != "jpeg" {
		t.Fatalf("expected decodable jpeg, got format=%q err=%v", format, err)
	}
	if cfg.Width > 256 || cfg.Height > 256 {
		t.Fatalf("expected max side 256, got %dx%d", cfg.Width, cfg.Height)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := PrepareImageJPEG(ctx, testFixtureImagePath(t, "cat_1.jpg"), 256); err == nil {
		t.Fatal("expected canceled context error")
	}
}
