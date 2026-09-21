package main

import (
	"context"
	"database/sql"
	"errors"
	"path/filepath"
	"time"

	"imgsearch/internal/embedder"
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

// nativeAnnotatorLoaderFactory returns switchboard loaders that resolve the
// requested variant's GGUF files lazily (downloading if needed) unless
// explicit paths were pinned by flags.
func nativeAnnotatorLoaderFactory(base llamaCPPNativeAnnotatorOptions, pinned bool, pinnedModelPath string, pinnedMMProjPath string) func(variant string) llamaAnnotatorLoader {
	return func(variant string) llamaAnnotatorLoader {
		return func(ctx context.Context) (embedder.ImageAnnotator, error) {
			opts := base
			opts.ModelPath, opts.VisionModelPath = pinnedModelPath, pinnedMMProjPath
			if !pinned {
				modelPath, mmprojPath, err := ensureDefaultLlamaNativeAnnotatorAssetsForVariant(ctx, variant, "", "")
				if err != nil {
					return nil, err
				}
				opts.ModelPath, opts.VisionModelPath = modelPath, mmprojPath
			}
			return newLlamaCPPNativeAnnotator(opts)
		}
	}
}

func nativeAnnotationStatus(variant string, pinned bool, pinnedModelPath string) settings.ActiveAnnotation {
	if pinned {
		return settings.ActiveAnnotation{Backend: settings.BackendNative, Model: filepath.Base(pinnedModelPath), Detail: "custom model paths from flags"}
	}
	return settings.ActiveAnnotation{Backend: settings.BackendNative, Model: variant}
}

func remoteAnnotationStatus(s settings.AnnotationSettings) settings.ActiveAnnotation {
	return settings.ActiveAnnotation{Backend: settings.BackendOpenAI, Model: s.OpenAI.Model, Detail: s.OpenAI.BaseURL}
}

// annotationStatusFunc reports the live backend when this process runs the
// worker, and otherwise what the persisted settings select, so the API-only
// process still shows something accurate.
func annotationStatusFunc(resolver *annotatorResolver, db *sql.DB, defaults settings.AnnotationSettings, pinned bool, pinnedModelPath string) func(context.Context) (settings.ActiveAnnotation, error) {
	if resolver != nil {
		return resolver.Status
	}
	return func(ctx context.Context) (settings.ActiveAnnotation, error) {
		s, err := settings.LoadAnnotationOrDefault(ctx, db, defaults)
		if err != nil {
			return settings.ActiveAnnotation{}, err
		}
		version, err := settings.Version(ctx, db)
		if err != nil {
			return settings.ActiveAnnotation{}, err
		}
		var status settings.ActiveAnnotation
		if s.Backend == settings.BackendOpenAI {
			status = remoteAnnotationStatus(s)
		} else {
			status = nativeAnnotationStatus(s.NativeVariant, pinned, pinnedModelPath)
		}
		status.SettingsVersion = version
		status.Source = settings.ActiveSourceSettings
		return status, nil
	}
}
