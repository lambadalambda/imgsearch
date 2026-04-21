package nsfwsql

import "testing"

func TestTagsJSONHasNSFWExact(t *testing.T) {
	got := TagsJSONHasNSFW("i.tags_json", "tag")
	want := `EXISTS (
	SELECT 1
	FROM json_each(COALESCE(i.tags_json, '[]')) tag
	WHERE lower(trim(COALESCE(tag.value, ''))) = 'nsfw'
)`
	if got != want {
		t.Fatalf("unexpected tags nsfw fragment:\nwant:\n%s\n\ngot:\n%s", want, got)
	}
}

func TestVideoFramesHaveNSFWExact(t *testing.T) {
	got := VideoFramesHaveNSFW("v.id", "frame_tag")
	want := `EXISTS (
	SELECT 1
	FROM video_frames vf_nsfw
	JOIN images i_nsfw ON i_nsfw.id = vf_nsfw.image_id
	JOIN json_each(COALESCE(i_nsfw.tags_json, '[]')) frame_tag
	  ON lower(trim(COALESCE(frame_tag.value, ''))) = 'nsfw'
	WHERE vf_nsfw.video_id = v.id
)`
	if got != want {
		t.Fatalf("unexpected frame nsfw fragment:\nwant:\n%s\n\ngot:\n%s", want, got)
	}
}

func TestVideoHasNSFWExact(t *testing.T) {
	got := VideoHasNSFW("v.id", "v.tags_json", "frame_tag", "video_tag")
	want := `(` + VideoFramesHaveNSFW("v.id", "frame_tag") + ` OR ` + TagsJSONHasNSFW("v.tags_json", "video_tag") + `)`
	if got != want {
		t.Fatalf("unexpected video nsfw fragment:\nwant:\n%s\n\ngot:\n%s", want, got)
	}
}
