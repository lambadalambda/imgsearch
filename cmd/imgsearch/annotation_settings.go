package main

import (
	"context"
	"errors"
	"time"

	"imgsearch/internal/embedder/llamacppnative"
	"imgsearch/internal/embedder/openaicompat"
	"imgsearch/internal/settings"
)

// annotationSettingsFromConfig seeds the settings page defaults from the
// startup flags so an unsaved settings row mirrors the flag-driven runtime.
func annotationSettingsFromConfig(cfg runtimeConfig) settings.AnnotationSettings {
	s := settings.DefaultAnnotation()
	s.NativeVariant = cfg.AnnotatorVariant
	return s.Normalized()
}

// openAIConfigFromSettings maps persisted remote settings onto the client
// config, reusing the native annotator's image size and libvips resizing.
func openAIConfigFromSettings(s settings.AnnotationSettings, imageMaxSide int, logf func(string, ...any)) openaicompat.Config {
	return openaicompat.Config{
		BaseURL:      s.OpenAI.BaseURL,
		APIKey:       s.OpenAI.APIKey,
		Model:        s.OpenAI.Model,
		Timeout:      time.Duration(s.OpenAI.TimeoutSeconds) * time.Second,
		Concurrency:  s.OpenAI.Concurrency,
		ImageMaxSide: imageMaxSide,
		PrepareImage: prepareImageForRemote,
		Logf:         logf,
	}
}

// prepareImageForRemote resizes through libvips when cgo is available and
// otherwise sends the stored bytes unchanged. Decode or resize failures are
// returned, not papered over with a raw upload.
func prepareImageForRemote(ctx context.Context, path string, maxSide int) ([]byte, string, error) {
	data, mime, err := llamacppnative.PrepareImageJPEG(ctx, path, maxSide)
	if errors.Is(err, llamacppnative.ErrImagePrepareUnavailable) {
		return openaicompat.ReadImageFile(ctx, path, maxSide)
	}
	if err != nil {
		return nil, "", err
	}
	return data, mime, nil
}

// testAnnotationConnection backs POST /api/settings/annotation/test.
func testAnnotationConnection(imageMaxSide int) func(context.Context, settings.AnnotationSettings) error {
	return func(ctx context.Context, s settings.AnnotationSettings) error {
		ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
		defer cancel()
		return openaicompat.TestConnection(ctx, openAIConfigFromSettings(s, imageMaxSide, nil))
	}
}
