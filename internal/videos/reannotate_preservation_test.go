package videos

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"imgsearch/internal/nsfwsql"
)

func TestReannotatePreservesVideoOnlyNSFWAndUserEditsUntilSuccess(t *testing.T) {
	conn := setupVideosDB(t)
	_, err := conn.Exec(`UPDATE videos SET title='My title',user_title='My title',summary='Old summary',description='Old description',tags_json='["nsfw","holiday"]',user_tags_json='["holiday"]',removed_tags_json='["blurry"]',annotation_updated_at='2025-01-01' WHERE id=1;
 UPDATE images SET tags_json='[]'`)
	if err != nil {
		t.Fatal(err)
	}
	if err = Reannotate(context.Background(), conn, 1, 1); err != nil {
		t.Fatal(err)
	}
	for _, state := range []string{"pending", "failed"} {
		if _, err = conn.Exec(`UPDATE index_jobs SET state=? WHERE kind='annotate_video' AND video_id=1`, state); err != nil {
			t.Fatal(err)
		}
		var title, summary, description, tags, userTitle, userTags, removedTags, updated string
		var nsfw, requested int
		err = conn.QueryRow(`SELECT title,summary,description,tags_json,user_title,user_tags_json,removed_tags_json,annotation_updated_at,reannotate_requested,`+nsfwsql.VideoHasNSFW("v.id", "v.tags_json", "ft", "vt")+` FROM videos v WHERE id=1`).Scan(&title, &summary, &description, &tags, &userTitle, &userTags, &removedTags, &updated, &requested, &nsfw)
		if err != nil {
			t.Fatal(err)
		}
		if title != "My title" || userTitle != title || summary != "Old summary" || description != "Old description" || tags != `["nsfw","holiday"]` || userTags != `["holiday"]` || removedTags != `["blurry"]` || updated != "2025-01-01" || requested != 1 || nsfw != 1 {
			t.Fatalf("state=%s lost annotations: %q %q %q %q %q %q %q %q requested=%d nsfw=%d", state, title, summary, description, tags, userTitle, userTags, removedTags, updated, requested, nsfw)
		}
		h := NewHandler(&Handler{DB: conn, ModelID: 1})
		rr := httptest.NewRecorder()
		h.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/api/videos?nsfw=hide", nil))
		if rr.Code != http.StatusOK {
			t.Fatalf("list status=%d body=%s", rr.Code, rr.Body.String())
		}
		var list ListResponse
		if err = json.Unmarshal(rr.Body.Bytes(), &list); err != nil {
			t.Fatal(err)
		}
		for _, video := range list.Videos {
			if video.VideoID == 1 {
				t.Fatalf("NSFW video exposed while %s", state)
			}
		}
	}
}
