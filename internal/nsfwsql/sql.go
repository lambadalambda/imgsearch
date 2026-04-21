package nsfwsql

import "fmt"

// TagsJSONHasNSFW returns a SQL EXISTS expression that reports whether the
// provided tags JSON expression contains the tag "nsfw".
//
// SECURITY: tagsJSONExpr and iteratorAlias are interpolated directly into SQL.
// Callers must pass only trusted SQL expressions and aliases defined in code,
// never user-controlled input.
func TagsJSONHasNSFW(tagsJSONExpr string, iteratorAlias string) string {
	return fmt.Sprintf(`EXISTS (
	SELECT 1
	FROM json_each(COALESCE(%s, '[]')) %s
	WHERE lower(trim(COALESCE(%s.value, ''))) = 'nsfw'
)`, tagsJSONExpr, iteratorAlias, iteratorAlias)
}

// VideoFramesHaveNSFW returns a SQL EXISTS expression that reports whether any
// frame image in the referenced video has an "nsfw" tag.
//
// SECURITY: videoIDExpr and frameTagAlias are interpolated directly into SQL.
// Callers must pass only trusted SQL expressions and aliases defined in code,
// never user-controlled input.
func VideoFramesHaveNSFW(videoIDExpr string, frameTagAlias string) string {
	return fmt.Sprintf(`EXISTS (
	SELECT 1
	FROM video_frames vf_nsfw
	JOIN images i_nsfw ON i_nsfw.id = vf_nsfw.image_id
	JOIN json_each(COALESCE(i_nsfw.tags_json, '[]')) %s
	  ON lower(trim(COALESCE(%s.value, ''))) = 'nsfw'
	WHERE vf_nsfw.video_id = %s
)`, frameTagAlias, frameTagAlias, videoIDExpr)
}

// VideoHasNSFW returns a SQL expression that reports whether the referenced
// video is NSFW either by its own tags or by any of its frame image tags.
//
// SECURITY: all arguments are interpolated directly into SQL. Callers must
// pass only trusted SQL expressions and aliases defined in code, never
// user-controlled input.
func VideoHasNSFW(videoIDExpr string, videoTagsExpr string, frameTagAlias string, videoTagAlias string) string {
	return fmt.Sprintf(`(%s OR %s)`, VideoFramesHaveNSFW(videoIDExpr, frameTagAlias), TagsJSONHasNSFW(videoTagsExpr, videoTagAlias))
}
