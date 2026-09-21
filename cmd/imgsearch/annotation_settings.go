package main

import (
	"imgsearch/internal/settings"
)

// annotationSettingsFromConfig seeds the settings page defaults from the
// startup flags so an unsaved settings row mirrors the flag-driven runtime.
func annotationSettingsFromConfig(cfg runtimeConfig) settings.AnnotationSettings {
	s := settings.DefaultAnnotation()
	s.NativeVariant = cfg.AnnotatorVariant
	return s.Normalized()
}
