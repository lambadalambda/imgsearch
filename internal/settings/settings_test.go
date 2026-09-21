package settings

import (
	"strings"
	"testing"
)

func TestDefaultAnnotationIsValidNative(t *testing.T) {
	d := DefaultAnnotation()
	if d.Backend != BackendNative || d.NativeVariant != NativeVariantE4B {
		t.Fatalf("unexpected defaults: %+v", d)
	}
	if err := d.Validate(); err != nil {
		t.Fatalf("defaults should validate: %v", err)
	}
}

func TestNormalizedTrimsAndFillsDefaults(t *testing.T) {
	s := AnnotationSettings{
		Backend:       " OpenAI ",
		NativeVariant: "",
		OpenAI: OpenAISettings{
			BaseURL: " http://localhost:8081/v1/ ",
			Model:   " gemma ",
		},
	}.Normalized()
	if s.Backend != BackendOpenAI {
		t.Fatalf("backend: %q", s.Backend)
	}
	if s.NativeVariant != NativeVariantE4B {
		t.Fatalf("variant should default: %q", s.NativeVariant)
	}
	if s.OpenAI.BaseURL != "http://localhost:8081/v1" {
		t.Fatalf("base url should be trimmed without trailing slash: %q", s.OpenAI.BaseURL)
	}
	if s.OpenAI.Model != "gemma" {
		t.Fatalf("model: %q", s.OpenAI.Model)
	}
	if s.OpenAI.TimeoutSeconds != DefaultOpenAITimeoutSeconds || s.OpenAI.Concurrency != DefaultOpenAIConcurrency {
		t.Fatalf("timeout/concurrency should default: %+v", s.OpenAI)
	}
}

func TestValidateRejectsBadValues(t *testing.T) {
	cases := []struct {
		name string
		in   AnnotationSettings
		want string
	}{
		{"bad backend", AnnotationSettings{Backend: "cloud"}, "backend"},
		{"bad variant", AnnotationSettings{Backend: BackendNative, NativeVariant: "99b"}, "native_variant"},
		{"openai missing model", AnnotationSettings{Backend: BackendOpenAI, OpenAI: OpenAISettings{BaseURL: "http://x"}}, "model"},
		{"openai missing url", AnnotationSettings{Backend: BackendOpenAI, OpenAI: OpenAISettings{Model: "m"}}, "base_url"},
		{"openai bad scheme", AnnotationSettings{Backend: BackendOpenAI, OpenAI: OpenAISettings{BaseURL: "ftp://x", Model: "m"}}, "base_url"},
		{"openai negative timeout", AnnotationSettings{Backend: BackendOpenAI, OpenAI: OpenAISettings{BaseURL: "http://x", Model: "m", TimeoutSeconds: -1}}, "timeout"},
		{"openai zero concurrency", AnnotationSettings{Backend: BackendOpenAI, OpenAI: OpenAISettings{BaseURL: "http://x", Model: "m", TimeoutSeconds: 1, Concurrency: 0}}, "concurrency"},
		{"openai huge timeout", AnnotationSettings{Backend: BackendOpenAI, OpenAI: OpenAISettings{BaseURL: "http://x", Model: "m", TimeoutSeconds: 99999, Concurrency: 1}}, "timeout"},
		{"openai huge concurrency", AnnotationSettings{Backend: BackendOpenAI, OpenAI: OpenAISettings{BaseURL: "http://x", Model: "m", TimeoutSeconds: 1, Concurrency: 500}}, "concurrency"},
	}
	for _, tc := range cases {
		err := tc.in.Validate()
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Fatalf("%s: expected error mentioning %q, got %v", tc.name, tc.want, err)
		}
	}
	// Native backend ignores incomplete openai fields.
	ok := AnnotationSettings{Backend: BackendNative, NativeVariant: NativeVariant26B, OpenAI: OpenAISettings{BaseURL: "nope"}}
	if err := ok.Validate(); err != nil {
		t.Fatalf("native backend should not validate openai fields: %v", err)
	}
	// Local servers may run without a key.
	local := AnnotationSettings{Backend: BackendOpenAI, OpenAI: OpenAISettings{BaseURL: "http://127.0.0.1:11434/v1", Model: "llava"}}.Normalized()
	if err := local.Validate(); err != nil {
		t.Fatalf("keyless openai settings should validate: %v", err)
	}
}

func TestViewMasksAPIKey(t *testing.T) {
	v := AnnotationSettings{Backend: BackendOpenAI, OpenAI: OpenAISettings{BaseURL: "http://x", Model: "m", APIKey: "secret"}}.View()
	if !v.OpenAI.APIKeySet {
		t.Fatal("expected api_key_set true")
	}
	if strings.Contains(strings.ToLower(marshalForTest(t, v)), "secret") {
		t.Fatal("view must not contain the api key")
	}
}
