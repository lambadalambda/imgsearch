package settings

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func newTestHandler(t *testing.T, h *Handler) http.Handler {
	t.Helper()
	if h.DB == nil {
		h.DB = openTestDB(t)
	}
	if h.Defaults == (AnnotationSettings{}) {
		h.Defaults = DefaultAnnotation()
	}
	return NewHandler(h)
}

func do(t *testing.T, h http.Handler, method, path, body string) (*httptest.ResponseRecorder, map[string]any) {
	t.Helper()
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	var out map[string]any
	if rec.Body.Len() > 0 {
		if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
			t.Fatalf("decode body %q: %v", rec.Body.String(), err)
		}
	}
	return rec, out
}

func TestGetSettingsReturnsDefaultsWhenUnsaved(t *testing.T) {
	defaults := DefaultAnnotation()
	defaults.NativeVariant = NativeVariant26B
	h := newTestHandler(t, &Handler{Defaults: defaults})

	rec, out := do(t, h, http.MethodGet, "/api/settings", "")
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d body=%s", rec.Code, rec.Body.String())
	}
	if out["version"].(float64) != 0 {
		t.Fatalf("expected version 0, got %v", out["version"])
	}
	ann := out["annotation"].(map[string]any)
	if ann["backend"] != BackendNative || ann["native_variant"] != NativeVariant26B {
		t.Fatalf("expected flag-seeded defaults, got %v", ann)
	}
	openai := ann["openai"].(map[string]any)
	if _, has := openai["api_key"]; has {
		t.Fatal("api_key must never be returned")
	}
	if openai["api_key_set"] != false {
		t.Fatalf("expected api_key_set false, got %v", openai["api_key_set"])
	}
}

func TestPutSettingsValidatesAndSaves(t *testing.T) {
	h := newTestHandler(t, &Handler{})

	rec, out := do(t, h, http.MethodPut, "/api/settings", `{"annotation":{"backend":"openai","openai":{"base_url":"http://x","model":""}}}`)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(out["error"].(string), "model") {
		t.Fatalf("expected validation message about model, got %v", out["error"])
	}

	rec, out = do(t, h, http.MethodPut, "/api/settings", `{"annotation":{"backend":"openai","openai":{"base_url":"http://127.0.0.1:8081/v1/","api_key":"secret","model":"gemma"}}}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", rec.Code, rec.Body.String())
	}
	if out["version"].(float64) != 1 {
		t.Fatalf("expected version 1, got %v", out["version"])
	}
	openai := out["annotation"].(map[string]any)["openai"].(map[string]any)
	if openai["api_key_set"] != true || openai["base_url"] != "http://127.0.0.1:8081/v1" {
		t.Fatalf("unexpected saved view: %v", openai)
	}
	if strings.Contains(rec.Body.String(), "secret") {
		t.Fatal("response leaked api key")
	}
}

func TestPutSettingsKeepsExistingKeyWhenBlankAndClearsOnRequest(t *testing.T) {
	conn := openTestDB(t)
	h := newTestHandler(t, &Handler{DB: conn})
	do(t, h, http.MethodPut, "/api/settings", `{"annotation":{"backend":"openai","openai":{"base_url":"http://x","api_key":"secret","model":"m"}}}`)

	rec, _ := do(t, h, http.MethodPut, "/api/settings", `{"annotation":{"backend":"openai","openai":{"base_url":"http://x","model":"m2"}}}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", rec.Code, rec.Body.String())
	}
	stored, err := LoadAnnotation(context.Background(), conn)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if stored.OpenAI.APIKey != "secret" || stored.OpenAI.Model != "m2" {
		t.Fatalf("blank api_key should keep existing key while updating other fields: %+v", stored.OpenAI)
	}

	rec, out := do(t, h, http.MethodPut, "/api/settings", `{"annotation":{"backend":"openai","openai":{"base_url":"http://x","model":"m2"}},"clear_api_key":true}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	if out["annotation"].(map[string]any)["openai"].(map[string]any)["api_key_set"] != false {
		t.Fatal("expected api_key_set false after clear")
	}
}

func TestPutSettingsRejectsMalformedBody(t *testing.T) {
	h := newTestHandler(t, &Handler{})
	rec, _ := do(t, h, http.MethodPut, "/api/settings", `{"annotation":`)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", rec.Code)
	}
	rec, _ = do(t, h, http.MethodDelete, "/api/settings", "")
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405, got %d", rec.Code)
	}
}

func TestTestConnectionUsesInjectedTester(t *testing.T) {
	var seen AnnotationSettings
	h := newTestHandler(t, &Handler{
		TestConnection: func(_ context.Context, s AnnotationSettings) error {
			seen = s
			if s.OpenAI.Model == "bad" {
				return errors.New("model not found")
			}
			return nil
		},
	})

	rec, out := do(t, h, http.MethodPost, "/api/settings/annotation/test", `{"annotation":{"backend":"openai","openai":{"base_url":"http://x/","api_key":"k","model":"good"}}}`)
	if rec.Code != http.StatusOK || out["ok"] != true {
		t.Fatalf("expected ok, got %d %v", rec.Code, out)
	}
	if seen.OpenAI.BaseURL != "http://x" || seen.OpenAI.APIKey != "k" {
		t.Fatalf("tester should receive normalized settings with key: %+v", seen)
	}

	rec, out = do(t, h, http.MethodPost, "/api/settings/annotation/test", `{"annotation":{"backend":"openai","openai":{"base_url":"http://x","model":"bad"}}}`)
	if rec.Code != http.StatusOK || out["ok"] != false || !strings.Contains(out["error"].(string), "model not found") {
		t.Fatalf("expected 200 with ok=false, got %d %v", rec.Code, out)
	}

	rec, _ = do(t, h, http.MethodPost, "/api/settings/annotation/test", `{"annotation":{"backend":"native"}}`)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("native backend has nothing to test; expected 400, got %d", rec.Code)
	}
}

func TestTestConnectionFallsBackToStoredKeyWhenBlank(t *testing.T) {
	var seen AnnotationSettings
	h := newTestHandler(t, &Handler{TestConnection: func(_ context.Context, s AnnotationSettings) error { seen = s; return nil }})
	do(t, h, http.MethodPut, "/api/settings", `{"annotation":{"backend":"openai","openai":{"base_url":"http://x","api_key":"stored","model":"m"}}}`)
	do(t, h, http.MethodPost, "/api/settings/annotation/test", `{"annotation":{"backend":"openai","openai":{"base_url":"http://x","model":"m"}}}`)
	if seen.OpenAI.APIKey != "stored" {
		t.Fatalf("expected stored key to be used for test, got %q", seen.OpenAI.APIKey)
	}
}

func TestTestConnectionUnavailableWithoutTester(t *testing.T) {
	h := newTestHandler(t, &Handler{})
	rec, _ := do(t, h, http.MethodPost, "/api/settings/annotation/test", `{"annotation":{"backend":"openai","openai":{"base_url":"http://x","model":"m"}}}`)
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503, got %d", rec.Code)
	}
}

func TestPutKeepsStoredRemoteBlockWhenSwitchingToNative(t *testing.T) {
	conn := openTestDB(t)
	h := newTestHandler(t, &Handler{DB: conn})
	do(t, h, http.MethodPut, "/api/settings", `{"annotation":{"backend":"openai","openai":{"base_url":"http://x","api_key":"secret","model":"m"}}}`)
	rec, out := do(t, h, http.MethodPut, "/api/settings", `{"annotation":{"backend":"native","native_variant":"26b"}}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", rec.Code, rec.Body.String())
	}
	openai := out["annotation"].(map[string]any)["openai"].(map[string]any)
	if openai["base_url"] != "http://x" || openai["model"] != "m" || openai["api_key_set"] != true {
		t.Fatalf("expected stored remote block carried over, got %v", openai)
	}
	stored, err := LoadAnnotation(context.Background(), conn)
	if err != nil || stored.Backend != BackendNative || stored.OpenAI.APIKey != "secret" {
		t.Fatalf("unexpected stored: %+v err=%v", stored, err)
	}
}

func TestUnknownSettingsSubpathReturnsJSON404(t *testing.T) {
	h := newTestHandler(t, &Handler{})
	rec, out := do(t, h, http.MethodGet, "/api/settings/nope", "")
	if rec.Code != http.StatusNotFound || out["error"] == nil {
		t.Fatalf("expected JSON 404, got %d %s", rec.Code, rec.Body.String())
	}
}

func TestPutRejectsOversizedBody(t *testing.T) {
	h := newTestHandler(t, &Handler{})
	rec, _ := do(t, h, http.MethodPut, "/api/settings", `{"annotation":{"backend":"native","native_variant":"`+strings.Repeat("x", maxSettingsBodyBytes)+`"}}`)
	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("expected 413, got %d", rec.Code)
	}
}

func TestGetSettingsReportsLockAndActiveStatus(t *testing.T) {
	h := newTestHandler(t, &Handler{
		NativeVariantLocked: true,
		Status: func(context.Context) (ActiveAnnotation, error) {
			return ActiveAnnotation{Backend: BackendNative, Model: "custom.gguf", SettingsVersion: 0}, nil
		},
	})
	rec, out := do(t, h, http.MethodGet, "/api/settings", "")
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d", rec.Code)
	}
	if out["native_variant_locked"] != true || out["annotations_disabled"] != false {
		t.Fatalf("expected lock flags, got %v", out)
	}
	active := out["active"].(map[string]any)
	if active["backend"] != BackendNative || active["model"] != "custom.gguf" {
		t.Fatalf("unexpected active: %v", active)
	}

	failing := newTestHandler(t, &Handler{Status: func(context.Context) (ActiveAnnotation, error) {
		return ActiveAnnotation{}, errors.New("model load failed")
	}})
	rec, out = do(t, failing, http.MethodGet, "/api/settings", "")
	if rec.Code != http.StatusOK || out["active_error"] != "model load failed" || out["active"] != nil {
		t.Fatalf("expected active_error without active, got %d %v", rec.Code, out)
	}
}

func TestListModelsUsesInjectedListerAndToleratesBlankModel(t *testing.T) {
	var seen AnnotationSettings
	h := newTestHandler(t, &Handler{
		ListModels: func(_ context.Context, s AnnotationSettings) ([]string, error) {
			seen = s
			if s.OpenAI.BaseURL == "http://down" {
				return nil, errors.New("connection refused")
			}
			return []string{"vision-a", "vision-b"}, nil
		},
	})
	rec, out := do(t, h, http.MethodPost, "/api/settings/annotation/models", `{"annotation":{"backend":"openai","openai":{"base_url":"http://x/","api_key":"k"}}}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", rec.Code, rec.Body.String())
	}
	if models := out["models"].([]any); len(models) != 2 || models[0] != "vision-a" {
		t.Fatalf("unexpected models: %v", out["models"])
	}
	if seen.OpenAI.BaseURL != "http://x" || seen.OpenAI.APIKey != "k" {
		t.Fatalf("lister should receive normalized settings, got %+v", seen)
	}
	rec, out = do(t, h, http.MethodPost, "/api/settings/annotation/models", `{"annotation":{"backend":"openai","openai":{"base_url":"http://down"}}}`)
	if rec.Code != http.StatusOK || len(out["models"].([]any)) != 0 || !strings.Contains(out["error"].(string), "connection refused") {
		t.Fatalf("expected 200 with empty models and error, got %d %v", rec.Code, out)
	}
	rec, _ = do(t, h, http.MethodPost, "/api/settings/annotation/models", `{"annotation":{"backend":"native"}}`)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for native, got %d", rec.Code)
	}
	none := newTestHandler(t, &Handler{})
	rec, _ = do(t, none, http.MethodPost, "/api/settings/annotation/models", `{"annotation":{"backend":"openai","openai":{"base_url":"http://x"}}}`)
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503 without lister, got %d", rec.Code)
	}
}
