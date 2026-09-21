package main

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"imgsearch/internal/settings"
)

func TestAnnotationSettingsFromConfigSeedsNativeVariant(t *testing.T) {
	cfg := runtimeConfig{AnnotatorVariant: " 26B "}
	got := annotationSettingsFromConfig(cfg)
	if got.Backend != settings.BackendNative || got.NativeVariant != settings.NativeVariant26B {
		t.Fatalf("unexpected seeded settings: %+v", got)
	}
	if err := got.Validate(); err != nil {
		t.Fatalf("seeded settings should validate: %v", err)
	}
	empty := annotationSettingsFromConfig(runtimeConfig{})
	if empty.NativeVariant != settings.NativeVariantE4B {
		t.Fatalf("empty variant should default to e4b, got %q", empty.NativeVariant)
	}
}

func TestOpenAIConfigFromSettingsMapsFields(t *testing.T) {
	s := settings.AnnotationSettings{
		Backend: settings.BackendOpenAI,
		OpenAI:  settings.OpenAISettings{BaseURL: "http://x/v1", APIKey: "k", Model: "m", TimeoutSeconds: 45, Concurrency: 3},
	}
	cfg := openAIConfigFromSettings(s, 768, nil)
	if cfg.BaseURL != "http://x/v1" || cfg.APIKey != "k" || cfg.Model != "m" || cfg.Concurrency != 3 || cfg.ImageMaxSide != 768 {
		t.Fatalf("unexpected config: %+v", cfg)
	}
	if cfg.Timeout.Seconds() != 45 {
		t.Fatalf("timeout: %v", cfg.Timeout)
	}
	if cfg.PrepareImage == nil {
		t.Fatal("expected image preparer")
	}
}

func TestPrepareImageForRemoteSurfacesDecodeFailures(t *testing.T) {
	path := filepath.Join(t.TempDir(), "blob.bin")
	if err := os.WriteFile(path, []byte("not an image"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := prepareImageForRemote(context.Background(), path, 256); err == nil {
		t.Fatal("expected decode failure to surface instead of a raw upload")
	}
}
