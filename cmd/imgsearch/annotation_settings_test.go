package main

import (
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
