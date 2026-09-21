package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
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

func TestAnnotationStatusHelpers(t *testing.T) {
	pinned := nativeAnnotationStatus("26b", true, "/models/custom/my-model.gguf")
	if pinned.Model != "my-model.gguf" || pinned.Backend != settings.BackendNative || pinned.Detail == "" {
		t.Fatalf("unexpected pinned status: %+v", pinned)
	}
	variant := nativeAnnotationStatus("26b", false, "")
	if variant.Model != "26b" || variant.Detail != "" {
		t.Fatalf("unexpected variant status: %+v", variant)
	}
	remote := remoteAnnotationStatus(settings.AnnotationSettings{OpenAI: settings.OpenAISettings{BaseURL: "http://x", Model: "m"}})
	if remote.Backend != settings.BackendOpenAI || remote.Model != "m" || remote.Detail != "http://x" {
		t.Fatalf("unexpected remote status: %+v", remote)
	}
}

func TestAnnotationStatusFuncWithoutResolverFollowsSettings(t *testing.T) {
	conn := openResolverDB(t)
	status := annotationStatusFunc(nil, conn, settings.DefaultAnnotation(), false, "")
	got, err := status(context.Background())
	if err != nil || got.Backend != settings.BackendNative || got.Model != settings.NativeVariantE4B || got.SettingsVersion != 0 {
		t.Fatalf("unexpected default status: %+v err=%v", got, err)
	}
	if _, err := settings.SaveAnnotation(context.Background(), conn, remoteSettings("llava")); err != nil {
		t.Fatal(err)
	}
	got, err = status(context.Background())
	if err != nil || got.Backend != settings.BackendOpenAI || got.Model != "llava" || got.SettingsVersion != 1 || got.Source != settings.ActiveSourceSettings {
		t.Fatalf("unexpected remote status: %+v err=%v", got, err)
	}
}

func TestNativeAnnotatorLoaderFactoryUsesPinnedPathsWithoutDownloading(t *testing.T) {
	loader := nativeAnnotatorLoaderFactory(llamaCPPNativeAnnotatorOptions{ContextSize: 1, BatchSize: 1}, true, "/nonexistent/model.gguf", "/nonexistent/mmproj.gguf")("26b")
	_, err := loader(context.Background())
	if err == nil || !strings.Contains(err.Error(), "model path") {
		t.Fatalf("expected pinned path validation error, got %v", err)
	}
}
