// Package settings stores user-editable runtime settings in SQLite so the
// API and worker processes share them, and serves them over /api/settings.
package settings

import (
	"fmt"
	"net/url"
	"strings"
)

const (
	BackendNative = "native"
	BackendOpenAI = "openai"

	NativeVariantE4B = "e4b"
	NativeVariant26B = "26b"

	DefaultOpenAITimeoutSeconds = 120
	DefaultOpenAIConcurrency    = 2
	MaxOpenAITimeoutSeconds     = 3600
	MaxOpenAIConcurrency        = 64
)

// OpenAISettings configures a remote OpenAI-compatible chat-completions
// server used for annotations.
type OpenAISettings struct {
	BaseURL        string `json:"base_url"`
	APIKey         string `json:"api_key,omitempty"`
	Model          string `json:"model"`
	TimeoutSeconds int    `json:"timeout_seconds"`
	Concurrency    int    `json:"concurrency"`
}

// AnnotationSettings selects where descriptions, titles, and tags are produced.
type AnnotationSettings struct {
	Backend       string         `json:"backend"`
	NativeVariant string         `json:"native_variant"`
	OpenAI        OpenAISettings `json:"openai"`
}

// OpenAIView is OpenAISettings with the key replaced by a presence flag.
type OpenAIView struct {
	BaseURL        string `json:"base_url"`
	APIKeySet      bool   `json:"api_key_set"`
	Model          string `json:"model"`
	TimeoutSeconds int    `json:"timeout_seconds"`
	Concurrency    int    `json:"concurrency"`
}

// AnnotationView is the API-safe shape of AnnotationSettings.
type AnnotationView struct {
	Backend       string     `json:"backend"`
	NativeVariant string     `json:"native_variant"`
	OpenAI        OpenAIView `json:"openai"`
}

// DefaultAnnotation is the built-in native e4b configuration.
func DefaultAnnotation() AnnotationSettings {
	return AnnotationSettings{
		Backend:       BackendNative,
		NativeVariant: NativeVariantE4B,
		OpenAI: OpenAISettings{
			TimeoutSeconds: DefaultOpenAITimeoutSeconds,
			Concurrency:    DefaultOpenAIConcurrency,
		},
	}
}

// Normalized trims and lowercases enum fields, strips a trailing slash from
// the base URL, and fills zero-valued numeric fields with defaults.
func (s AnnotationSettings) Normalized() AnnotationSettings {
	s.Backend = strings.ToLower(strings.TrimSpace(s.Backend))
	s.NativeVariant = strings.ToLower(strings.TrimSpace(s.NativeVariant))
	if s.NativeVariant == "" {
		s.NativeVariant = NativeVariantE4B
	}
	s.OpenAI.BaseURL = strings.TrimRight(strings.TrimSpace(s.OpenAI.BaseURL), "/")
	s.OpenAI.APIKey = strings.TrimSpace(s.OpenAI.APIKey)
	s.OpenAI.Model = strings.TrimSpace(s.OpenAI.Model)
	if s.OpenAI.TimeoutSeconds == 0 {
		s.OpenAI.TimeoutSeconds = DefaultOpenAITimeoutSeconds
	}
	if s.OpenAI.Concurrency == 0 {
		s.OpenAI.Concurrency = DefaultOpenAIConcurrency
	}
	return s
}

// Validate reports the first problem with s. Call on Normalized values.
func (s AnnotationSettings) Validate() error {
	switch s.Backend {
	case BackendNative:
		switch s.NativeVariant {
		case NativeVariantE4B, NativeVariant26B:
			return nil
		default:
			return fmt.Errorf("native_variant must be %q or %q", NativeVariantE4B, NativeVariant26B)
		}
	case BackendOpenAI:
		return s.OpenAI.validate()
	default:
		return fmt.Errorf("backend must be %q or %q", BackendNative, BackendOpenAI)
	}
}

func (o OpenAISettings) validate() error {
	if o.BaseURL == "" {
		return fmt.Errorf("openai.base_url is required")
	}
	u, err := url.Parse(o.BaseURL)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return fmt.Errorf("openai.base_url must be an http(s) URL")
	}
	if o.Model == "" {
		return fmt.Errorf("openai.model is required")
	}
	if o.TimeoutSeconds < 0 || o.TimeoutSeconds > MaxOpenAITimeoutSeconds {
		return fmt.Errorf("openai.timeout_seconds must be between 0 and %d", MaxOpenAITimeoutSeconds)
	}
	if o.Concurrency < 1 || o.Concurrency > MaxOpenAIConcurrency {
		return fmt.Errorf("openai.concurrency must be between 1 and %d", MaxOpenAIConcurrency)
	}
	return nil
}

// View masks the API key for responses.
func (s AnnotationSettings) View() AnnotationView {
	return AnnotationView{
		Backend:       s.Backend,
		NativeVariant: s.NativeVariant,
		OpenAI: OpenAIView{
			BaseURL:        s.OpenAI.BaseURL,
			APIKeySet:      s.OpenAI.APIKey != "",
			Model:          s.OpenAI.Model,
			TimeoutSeconds: s.OpenAI.TimeoutSeconds,
			Concurrency:    s.OpenAI.Concurrency,
		},
	}
}
