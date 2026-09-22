package settings

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"strings"
	"testing"
)

func tokenView(t *testing.T, out map[string]any) map[string]any {
	t.Helper()
	tokens, ok := out["annotation"].(map[string]any)["output_tokens"].(map[string]any)
	if !ok {
		t.Fatalf("missing output_tokens: %v", out)
	}
	return tokens
}

func defaultTokenView() map[string]any {
	return map[string]any{"image": float64(1024), "image_retry": float64(512), "video_frame": float64(320), "video_frame_retry": float64(256), "video": float64(1200), "video_retry": float64(640)}
}

func TestOutputTokensDefaultsAndLegacyPersistence(t *testing.T) {
	conn := openTestDB(t)
	h := newTestHandler(t, &Handler{DB: conn})
	_, out := do(t, h, http.MethodGet, "/api/settings", "")
	if got := tokenView(t, out); !reflect.DeepEqual(got, defaultTokenView()) {
		t.Fatalf("defaults: %v", got)
	}
	for _, raw := range []string{`{"backend":"native","native_variant":"26b"}`, `{"backend":"native","output_tokens":{"image":2048}}`} {
		if _, err := conn.Exec(`INSERT OR REPLACE INTO settings(key,value_json,updated_at) VALUES('annotation',?,datetime('now'))`, raw); err != nil {
			t.Fatal(err)
		}
		stored, err := LoadAnnotation(context.Background(), conn)
		if err != nil {
			t.Fatal(err)
		}
		var view map[string]any
		if err := json.Unmarshal([]byte(marshalForTest(t, stored.View())), &view); err != nil {
			t.Fatal(err)
		}
		want := defaultTokenView()
		if strings.Contains(raw, "2048") {
			want["image"] = float64(2048)
		}
		if !reflect.DeepEqual(view["output_tokens"], want) {
			t.Fatalf("legacy %s: got %v want %v", raw, view, want)
		}
	}
}

func TestOutputTokensAPIRoundTripAndOmissions(t *testing.T) {
	conn := openTestDB(t)
	h := newTestHandler(t, &Handler{DB: conn})
	want := map[string]any{"image": float64(2048), "image_retry": float64(768), "video_frame": float64(400), "video_frame_retry": float64(300), "video": float64(2400), "video_retry": float64(900)}
	body := `{"annotation":{"backend":"native","output_tokens":` + marshalForTest(t, want) + `}}`
	rec, out := do(t, h, http.MethodPut, "/api/settings", body)
	if rec.Code != 200 {
		t.Fatalf("save: %s", rec.Body.String())
	}
	if !reflect.DeepEqual(tokenView(t, out), want) {
		t.Fatalf("save: %v", out)
	}
	// A fresh handler proves the database, not handler state, owns the values.
	h = newTestHandler(t, &Handler{DB: conn})
	for _, body := range []string{`{"annotation":{"backend":"native"}}`, `{"annotation":{"backend":"openai","openai":{"base_url":"http://x","model":"m"},"output_tokens":{}}}`, `{"annotation":{"backend":"native","output_tokens":{"image":64,"video_retry":32768}}}`} {
		if body == `{"annotation":{"backend":"native","output_tokens":{"image":64,"video_retry":32768}}}` {
			want["image"] = float64(64)
			want["video_retry"] = float64(32768)
		}
		rec, out = do(t, h, http.MethodPut, "/api/settings", body)
		if rec.Code != 200 {
			t.Fatalf("update %s: %s", body, rec.Body.String())
		}
		if !reflect.DeepEqual(tokenView(t, out), want) {
			t.Fatalf("update: got %v want %v", tokenView(t, out), want)
		}
		_, out = do(t, h, http.MethodGet, "/api/settings", "")
		if !reflect.DeepEqual(tokenView(t, out), want) {
			t.Fatalf("get: %v", out)
		}
	}
}

func TestOutputTokensAPIRejectsInvalidWithoutSaving(t *testing.T) {
	conn := openTestDB(t)
	h := newTestHandler(t, &Handler{DB: conn})
	for field := range defaultTokenView() {
		for _, value := range []string{"0", "-1", "63", "32769", "1.5", `"1024"`, "true", "999999999999999999999999"} {
			for _, backend := range []string{"native", "openai"} {
				t.Run(field+"/"+value+"/"+backend, func(t *testing.T) {
					body := fmt.Sprintf(`{"annotation":{"backend":%q,"openai":{"base_url":"http://x","model":"m"},"output_tokens":{%q:%s}}}`, backend, field, value)
					rec, _ := do(t, h, http.MethodPut, "/api/settings", body)
					if rec.Code != 400 {
						t.Fatalf("want 400: %d %s", rec.Code, rec.Body.String())
					}
					if version, err := Version(context.Background(), conn); err != nil || version != 0 {
						t.Fatalf("invalid update saved: %d %v", version, err)
					}
				})
			}
		}
	}
}

func TestOutputTokensStoreRejectsInvalidBudgets(t *testing.T) {
	conn := openTestDB(t)
	for field := range defaultTokenView() {
		for _, value := range []int{-1, 63, 32769} {
			s := DefaultAnnotation()
			if err := json.Unmarshal([]byte(fmt.Sprintf(`{"output_tokens":{%q:%d}}`, field, value)), &s); err != nil {
				t.Fatal(err)
			}
			if _, err := SaveAnnotation(context.Background(), conn, s); err == nil {
				t.Fatalf("accepted %s=%d", field, value)
			}
		}
	}
	if version, err := Version(context.Background(), conn); err != nil || version != 0 {
		t.Fatalf("invalid save changed version: %d %v", version, err)
	}
}

func TestOutputTokensRemoteProbesPreserveOmittedValuesAndValidate(t *testing.T) {
	conn := openTestDB(t)
	saved := DefaultAnnotation()
	saved.OutputTokens.Image = 2300
	if _, err := SaveAnnotation(context.Background(), conn, saved); err != nil {
		t.Fatal(err)
	}
	var seen AnnotationSettings
	h := newTestHandler(t, &Handler{DB: conn,
		TestConnection: func(_ context.Context, s AnnotationSettings) error { seen = s; return nil },
		ListModels:     func(_ context.Context, s AnnotationSettings) ([]string, error) { seen = s; return nil, nil },
	})
	for _, path := range []string{"/api/settings/annotation/test", "/api/settings/annotation/models"} {
		rec, _ := do(t, h, http.MethodPost, path, `{"annotation":{"backend":"openai","openai":{"base_url":"http://x","model":"m"}}}`)
		if rec.Code != 200 || seen.OutputTokens != saved.OutputTokens {
			t.Fatalf("probe lost saved budgets: status=%d settings=%+v", rec.Code, seen)
		}
		rec, _ = do(t, h, http.MethodPost, path, `{"annotation":{"backend":"openai","openai":{"base_url":"http://x","model":"m"},"output_tokens":{"image":0}}}`)
		if rec.Code != 400 {
			t.Fatalf("probe accepted invalid budget: %d", rec.Code)
		}
	}
	if version, err := Version(context.Background(), conn); err != nil || version != 1 {
		t.Fatalf("probe changed settings: %d %v", version, err)
	}
}
