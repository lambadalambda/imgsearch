package settings

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"testing"
)

func TestVideoSettingsDefaultsAndLegacyFlags(t *testing.T) {
	conn := openTestDB(t)
	defaults := DefaultAnnotation()
	if err := json.Unmarshal([]byte(`{"video_frame_count":10}`), &defaults); err != nil {
		t.Fatal(err)
	}
	h := newTestHandler(t, &Handler{DB: conn, Defaults: defaults})
	for _, legacy := range []bool{false, true} {
		if legacy {
			if _, err := conn.Exec(`INSERT INTO settings(key,value_json) VALUES('annotation','{"backend":"native","native_variant":"26b"}')`); err != nil {
				t.Fatal(err)
			}
		}
		_, out := do(t, h, http.MethodGet, "/api/settings", "")
		a := out["annotation"].(map[string]any)
		if a["video_frame_count"] != float64(10) || a["video_mode"] != "frame_descriptions" {
			t.Fatalf("legacy=%v: %v", legacy, a)
		}
	}
}

func TestVideoSettingsRoundTripAndOmission(t *testing.T) {
	conn := openTestDB(t)
	h := newTestHandler(t, &Handler{DB: conn})
	for _, body := range []string{
		`{"annotation":{"backend":"openai","video_mode":"direct_frames","video_frame_count":10,"openai":{"base_url":"http://remote/v1","model":"vision","api_key":"private"}}}`,
		`{"annotation":{"backend":"native"}}`,
		`{"annotation":{"backend":"openai"}}`,
	} {
		rec, out := do(t, h, http.MethodPut, "/api/settings", body)
		if rec.Code != 200 {
			t.Fatalf("%d: %s", rec.Code, rec.Body.String())
		}
		a := out["annotation"].(map[string]any)
		if a["video_mode"] != "direct_frames" || a["video_frame_count"] != float64(10) {
			t.Fatalf("lost video settings: %v", a)
		}
		h = newTestHandler(t, &Handler{DB: conn})
		_, out = do(t, h, http.MethodGet, "/api/settings", "")
		if out["annotation"].(map[string]any)["video_mode"] != "direct_frames" {
			t.Fatal("mode not persisted")
		}
	}
}

func TestVideoSettingsRejectInvalidWithoutSaving(t *testing.T) {
	conn := openTestDB(t)
	h := newTestHandler(t, &Handler{DB: conn})
	for _, field := range []string{`"video_frame_count":0`, `"video_frame_count":-1`, `"video_frame_count":33`, `"video_frame_count":1.5`, `"video_frame_count":"5"`, `"video_mode":"typo"`} {
		rec, _ := do(t, h, http.MethodPut, "/api/settings", fmt.Sprintf(`{"annotation":{"backend":"native",%s}}`, field))
		if rec.Code != 400 {
			t.Errorf("%s: want400 got%d", field, rec.Code)
		}
	}
	if version, err := Version(context.Background(), conn); err != nil || version != 0 {
		t.Fatalf("invalid settings saved: %d %v", version, err)
	}
	for _, n := range []int{1, 32} {
		rec, _ := do(t, h, http.MethodPut, "/api/settings", fmt.Sprintf(`{"annotation":{"backend":"native","video_frame_count":%d}}`, n))
		if rec.Code != 200 {
			t.Fatalf("boundary%d: %s", n, rec.Body.String())
		}
	}
}

func TestSavingLegacyVideoSettingsFreezesFlagCountWithoutRequeue(t *testing.T) {
	conn := openTestDB(t)
	_, err := conn.Exec(`
 INSERT INTO settings(key,value_json) VALUES('annotation','{"backend":"native"}');
 INSERT INTO embedding_models(id,name,version,dimensions,metric,normalized) VALUES(1,'test','1',2,'cosine',1);
 INSERT INTO images(id,sha256,original_name,storage_path,mime_type,width,height) VALUES(1,'frame','frame.jpg','images/frame','image/jpeg',10,10);
 INSERT INTO videos(id,sha256,original_name,storage_path,mime_type,duration_ms,width,height,frame_count) VALUES(1,'video','video.mp4','videos/video','video/mp4',1000,10,10,1);
 INSERT INTO video_frames(video_id,image_id,frame_index,timestamp_ms) VALUES(1,1,0,500);
 INSERT INTO index_jobs(kind,video_id,model_id,state) VALUES('annotate_video',1,1,'done');`)
	if err != nil {
		t.Fatal(err)
	}
	defaults := DefaultAnnotation()
	defaults.VideoFrameCount = 10
	h := newTestHandler(t, &Handler{DB: conn, Defaults: defaults})
	rec, _ := do(t, h, http.MethodPut, "/api/settings", `{"annotation":{"backend":"native","video_mode":"direct_frames"}}`)
	if rec.Code != 200 {
		t.Fatalf("save: %s", rec.Body.String())
	}
	defaults.VideoFrameCount = 3
	h = newTestHandler(t, &Handler{DB: conn, Defaults: defaults})
	_, out := do(t, h, http.MethodGet, "/api/settings", "")
	if out["annotation"].(map[string]any)["video_frame_count"] != float64(10) {
		t.Fatal("later flag changed saved sampling count")
	}
	var frames, jobs, stored int
	if err := conn.QueryRow(`SELECT COUNT(*) FROM video_frames WHERE video_id=1 AND image_id=1 AND timestamp_ms=500`).Scan(&frames); err != nil {
		t.Fatal(err)
	}
	if err := conn.QueryRow(`SELECT COUNT(*) FROM index_jobs WHERE kind='annotate_video' AND state='done'`).Scan(&jobs); err != nil {
		t.Fatal(err)
	}
	if err := conn.QueryRow(`SELECT frame_count FROM videos WHERE id=1`).Scan(&stored); err != nil {
		t.Fatal(err)
	}
	if frames != 1 || jobs != 1 || stored != 1 {
		t.Fatalf("settings changed existing video: frames%d jobs%d stored%d", frames, jobs, stored)
	}
	var totalJobs int
	if err := conn.QueryRow(`SELECT COUNT(*) FROM index_jobs`).Scan(&totalJobs); err != nil {
		t.Fatal(err)
	}
	if totalJobs != 1 {
		t.Fatalf("settings inserted jobs: %d", totalJobs)
	}
}
