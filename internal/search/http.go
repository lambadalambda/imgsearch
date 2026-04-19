package search

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"imgsearch/internal/embedder"
	"imgsearch/internal/httputil"
	"imgsearch/internal/vectorindex"
)

type Handler struct {
	DB       *sql.DB
	ModelID  int64
	DataDir  string
	Embedder embedder.Embedder
	Index    vectorindex.VectorIndex
}

type SearchResult struct {
	ImageID          int64    `json:"image_id"`
	MediaType        string   `json:"media_type,omitempty"`
	VideoID          int64    `json:"video_id,omitempty"`
	PreviewPath      string   `json:"preview_path,omitempty"`
	MatchTimestampMS int64    `json:"match_timestamp_ms,omitempty"`
	TranscriptText   string   `json:"transcript_text,omitempty"`
	SearchSource     string   `json:"search_source,omitempty"`
	MimeType         string   `json:"mime_type,omitempty"`
	Distance         float64  `json:"distance"`
	OriginalName     string   `json:"original_name"`
	StoragePath      string   `json:"storage_path"`
	Description      string   `json:"description,omitempty"`
	Tags             []string `json:"tags,omitempty"`
}

type SearchResponse struct {
	Results []SearchResult       `json:"results"`
	Total   int64                `json:"total,omitempty"`
	Debug   *SearchDebugResponse `json:"debug,omitempty"`
}

type SearchDebugResponse struct {
	DurationMS    int64  `json:"duration_ms"`
	IndexBackend  string `json:"index_backend,omitempty"`
	IndexStrategy string `json:"index_strategy,omitempty"`
	Quantization  string `json:"quantization,omitempty"`
}

type TagCount struct {
	Tag   string `json:"tag"`
	Count int64  `json:"count"`
}

type TagCloudResponse struct {
	Tags []TagCount `json:"tags"`
}

const maxNegativePromptChars = 500

var errNegativePromptNearZero = errors.New("negative prompt too similar to query")

func NewHandler(h *Handler) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/search/text", h.handleTextSearch)
	mux.HandleFunc("/api/search/similar", h.handleSimilarSearch)
	mux.HandleFunc("/api/search/tags", h.handleTagSearch)
	mux.HandleFunc("/api/search/tag-cloud", h.handleTagCloud)
	return mux
}

func (h *Handler) handleTextSearch(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	if r.Method != http.MethodGet {
		httputil.WriteJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	q := strings.TrimSpace(r.URL.Query().Get("q"))
	if q == "" {
		httputil.WriteJSONError(w, http.StatusBadRequest, "missing query")
		return
	}
	neg := strings.TrimSpace(r.URL.Query().Get("neg"))
	if len(neg) > maxNegativePromptChars {
		httputil.WriteJSONError(w, http.StatusBadRequest, fmt.Sprintf("negative prompt too long (max %d characters)", maxNegativePromptChars))
		return
	}
	tagFilters := parseExplicitTagFilters(r)
	tagMode := strings.ToLower(strings.TrimSpace(r.URL.Query().Get("tag_mode")))
	tagMatchAll := tagMode != "any"
	includeNSFW := parseIncludeNSFW(r)

	vec, err := h.embedQueryVector(r.Context(), q, neg)
	if err != nil {
		if errors.Is(err, errNegativeEmbedding) {
			httputil.WriteJSONError(w, http.StatusInternalServerError, "negative embedding failed")
			return
		}
		httputil.WriteJSONError(w, http.StatusInternalServerError, "embedding failed")
		return
	}

	limit := parseLimit(r, 20)
	searchLimit := limit
	if len(tagFilters) > 0 || !includeNSFW {
		searchLimit = limit * 8
		if searchLimit > 200 {
			searchLimit = 200
		}
	}
	indexDebug := vectorindex.SearchDebug{}
	searchCtx := vectorindex.WithSearchDebug(r.Context(), &indexDebug)
	hits, err := h.Index.Search(searchCtx, h.ModelID, vec, searchLimit)
	if err != nil {
		httputil.WriteJSONError(w, http.StatusInternalServerError, "search failed")
		return
	}

	frameResults, err := h.enrich(r.Context(), hits, includeNSFW)
	if err != nil {
		httputil.WriteJSONError(w, http.StatusInternalServerError, "result enrich failed")
		return
	}
	transcriptResults, err := h.searchTranscriptEmbeddings(r.Context(), vec, searchLimit, includeNSFW)
	if err != nil {
		httputil.WriteJSONError(w, http.StatusInternalServerError, "transcript search failed")
		return
	}
	results := mergeResults(frameResults, transcriptResults, searchLimit)
	if len(tagFilters) > 0 {
		results = filterResultsByTags(results, tagFilters, tagMatchAll)
	}
	if len(results) > limit {
		results = results[:limit]
	}

	httputil.WriteJSON(w, http.StatusOK, SearchResponse{Results: results, Total: int64(len(results)), Debug: buildSearchDebugResponse(start, indexDebug)})
}

var errNegativeEmbedding = errors.New("negative embedding failed")

func (h *Handler) embedQueryVector(ctx context.Context, query string, negative string) ([]float32, error) {
	if negative == "" {
		return h.Embedder.EmbedText(ctx, query)
	}

	queryVec, err := h.Embedder.EmbedText(ctx, query)
	if err != nil {
		return nil, err
	}
	negativeVec, err := h.Embedder.EmbedText(ctx, negative)
	if err != nil {
		return nil, errNegativeEmbedding
	}

	combined, err := combineQueryWithNegative(queryVec, negativeVec)
	if err != nil {
		if errors.Is(err, errNegativePromptNearZero) {
			return queryVec, nil
		}
		return nil, err
	}

	return combined, nil
}

func combineQueryWithNegative(query []float32, negative []float32) ([]float32, error) {
	if len(query) != len(negative) {
		return nil, fmt.Errorf("query and negative dimensions differ: %d vs %d", len(query), len(negative))
	}
	if len(query) == 0 {
		return nil, fmt.Errorf("query embedding is empty")
	}

	out := make([]float32, len(query))
	var squareNorm float64
	for i := range query {
		v := float64(query[i]) - float64(negative[i])
		out[i] = float32(v)
		squareNorm += v * v
	}

	if squareNorm < 1e-12 {
		return nil, errNegativePromptNearZero
	}

	invNorm := float32(1 / math.Sqrt(squareNorm))
	for i := range out {
		out[i] *= invNorm
	}

	return out, nil
}

func (h *Handler) handleSimilarSearch(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	if r.Method != http.MethodGet {
		httputil.WriteJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	imageIDStr := r.URL.Query().Get("image_id")
	if imageIDStr == "" {
		httputil.WriteJSONError(w, http.StatusBadRequest, "missing image_id")
		return
	}
	imageID, err := strconv.ParseInt(imageIDStr, 10, 64)
	if err != nil || imageID <= 0 {
		httputil.WriteJSONError(w, http.StatusBadRequest, "invalid image_id")
		return
	}
	includeNSFW := parseIncludeNSFW(r)

	limit := parseLimit(r, 20)
	searchLimit := limit
	if !includeNSFW {
		searchLimit = limit * 8
		if searchLimit > 200 {
			searchLimit = 200
		}
	}
	indexDebug := vectorindex.SearchDebug{}
	searchCtx := vectorindex.WithSearchDebug(r.Context(), &indexDebug)
	hits, err := h.Index.SearchByImageID(searchCtx, h.ModelID, imageID, searchLimit)
	if err != nil {
		if errors.Is(err, vectorindex.ErrNotFound) {
			httputil.WriteJSONError(w, http.StatusNotFound, "image not indexed")
			return
		}
		httputil.WriteJSONError(w, http.StatusInternalServerError, "search failed")
		return
	}

	results, err := h.enrich(r.Context(), hits, includeNSFW)
	if err != nil {
		httputil.WriteJSONError(w, http.StatusInternalServerError, "result enrich failed")
		return
	}
	if len(results) > limit {
		results = results[:limit]
	}

	httputil.WriteJSON(w, http.StatusOK, SearchResponse{Results: results, Total: int64(len(results)), Debug: buildSearchDebugResponse(start, indexDebug)})
}

func buildSearchDebugResponse(start time.Time, info vectorindex.SearchDebug) *SearchDebugResponse {
	durationMS := time.Since(start).Milliseconds()
	if durationMS < 0 {
		durationMS = 0
	}
	resp := &SearchDebugResponse{DurationMS: durationMS}
	if strings.TrimSpace(info.Backend) != "" {
		resp.IndexBackend = info.Backend
	}
	if strings.TrimSpace(info.Strategy) != "" {
		resp.IndexStrategy = info.Strategy
	}
	switch {
	case info.Quantized:
		resp.Quantization = "on"
	case strings.TrimSpace(info.Backend) != "":
		resp.Quantization = "off"
	default:
		resp.Quantization = "unknown"
	}
	return resp
}

func (h *Handler) handleTagSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httputil.WriteJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	tags := parseTagFilters(r)
	if len(tags) == 0 {
		httputil.WriteJSONError(w, http.StatusBadRequest, "missing tag")
		return
	}

	limit := parseLimit(r, 24)
	offset := parseOffset(r)
	mode := strings.ToLower(strings.TrimSpace(r.URL.Query().Get("mode")))
	matchAll := mode == "all"
	includeNSFW := parseIncludeNSFW(r)

	results, total, err := h.searchByTags(r.Context(), tags, limit, offset, matchAll, includeNSFW)
	if err != nil {
		httputil.WriteJSONError(w, http.StatusInternalServerError, "tag search failed")
		return
	}

	httputil.WriteJSON(w, http.StatusOK, SearchResponse{Results: results, Total: total})
}

func (h *Handler) handleTagCloud(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httputil.WriteJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	limit := parseLimit(r, 60)
	if limit > 120 {
		limit = 120
	}
	minCount := parseMinCount(r, 1)
	prefix := strings.ToLower(strings.TrimSpace(r.URL.Query().Get("q")))
	includeNSFW := parseIncludeNSFW(r)

	tags, err := h.tagCloud(r.Context(), limit, minCount, prefix, includeNSFW)
	if err != nil {
		httputil.WriteJSONError(w, http.StatusInternalServerError, "tag cloud failed")
		return
	}

	httputil.WriteJSON(w, http.StatusOK, TagCloudResponse{Tags: tags})
}

func (h *Handler) enrich(ctx context.Context, hits []vectorindex.SearchHit, includeNSFW bool) ([]SearchResult, error) {
	if len(hits) == 0 {
		return []SearchResult{}, nil
	}

	ids := make([]int64, 0, len(hits))
	seen := make(map[int64]struct{}, len(hits))
	for _, hit := range hits {
		if _, ok := seen[hit.ImageID]; ok {
			continue
		}
		seen[hit.ImageID] = struct{}{}
		ids = append(ids, hit.ImageID)
	}

	placeholders := make([]string, len(ids))
	args := make([]any, 0, len(ids)+1)
	for i, id := range ids {
		placeholders[i] = "?"
		args = append(args, id)
	}
	args = append(args, boolToInt(includeNSFW))

	rows, err := h.DB.QueryContext(ctx, fmt.Sprintf(`
	SELECT i.id,
	       i.original_name,
	       i.storage_path,
	       i.mime_type,
	       COALESCE(i.description, ''),
	       COALESCE(i.tags_json, '[]'),
	       vf.video_id,
	       v.original_name,
	       v.storage_path,
	       v.mime_type,
	       COALESCE(v.description, ''),
	       COALESCE(v.tags_json, '[]'),
	       COALESCE(v.transcript_text, ''),
	       vf.timestamp_ms
	FROM images i
	LEFT JOIN video_frames vf ON vf.image_id = i.id
	LEFT JOIN videos v ON v.id = vf.video_id
	WHERE i.id IN (%s)
	  AND (
	    ? = 1
	    OR (
	      (
	        vf.video_id IS NULL
	        AND NOT EXISTS (
	          SELECT 1
	          FROM json_each(COALESCE(i.tags_json, '[]')) tag
	          WHERE lower(trim(COALESCE(tag.value, ''))) = 'nsfw'
	        )
	      )
	      OR (
	        vf.video_id IS NOT NULL
	        AND NOT EXISTS (
	          SELECT 1
	          FROM video_frames vf_nsfw
	          JOIN images i_nsfw ON i_nsfw.id = vf_nsfw.image_id
	          JOIN json_each(COALESCE(i_nsfw.tags_json, '[]')) tag_nsfw
	            ON lower(trim(COALESCE(tag_nsfw.value, ''))) = 'nsfw'
	          WHERE vf_nsfw.video_id = vf.video_id
	        )
	        AND NOT EXISTS (
	          SELECT 1
	          FROM json_each(COALESCE(v.tags_json, '[]')) video_tag_nsfw
	          WHERE lower(trim(COALESCE(video_tag_nsfw.value, ''))) = 'nsfw'
	        )
	      )
	    )
	  )
	`, strings.Join(placeholders, ",")), args...)
	if err != nil {
		return nil, fmt.Errorf("load images for enrich: %w", err)
	}
	defer func() { _ = rows.Close() }()

	type mediaRow struct {
		imageID             int64
		originalName        string
		storagePath         string
		mimeType            string
		description         string
		tags                []string
		videoID             sql.NullInt64
		videoOriginalName   sql.NullString
		videoStoragePath    sql.NullString
		videoMimeType       sql.NullString
		videoDescription    sql.NullString
		videoTags           []string
		videoTranscriptText sql.NullString
		timestampMS         sql.NullInt64
	}
	byID := make(map[int64][]mediaRow, len(ids))
	for rows.Next() {
		var row mediaRow
		var tagsJSON string
		var videoTagsJSON string
		if err := rows.Scan(&row.imageID, &row.originalName, &row.storagePath, &row.mimeType, &row.description, &tagsJSON, &row.videoID, &row.videoOriginalName, &row.videoStoragePath, &row.videoMimeType, &row.videoDescription, &videoTagsJSON, &row.videoTranscriptText, &row.timestampMS); err != nil {
			return nil, fmt.Errorf("scan enriched image row: %w", err)
		}
		tags, err := decodeTagsJSON(tagsJSON)
		if err != nil {
			return nil, fmt.Errorf("decode image %d tags: %w", row.imageID, err)
		}
		row.tags = tags
		if row.videoID.Valid {
			videoTags, err := decodeTagsJSON(videoTagsJSON)
			if err != nil {
				return nil, fmt.Errorf("decode video %d tags: %w", row.videoID.Int64, err)
			}
			row.videoTags = videoTags
		}
		byID[row.imageID] = append(byID[row.imageID], row)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate enriched image rows: %w", err)
	}

	seenVideos := make(map[int64]struct{}, len(hits))
	results := make([]SearchResult, 0, len(hits))
	for _, hit := range hits {
		rowsForImage, ok := byID[hit.ImageID]
		if !ok || len(rowsForImage) == 0 {
			continue
		}
		for _, row := range rowsForImage {
			if row.videoID.Valid {
				if _, seen := seenVideos[row.videoID.Int64]; seen {
					continue
				}
				seenVideos[row.videoID.Int64] = struct{}{}
			}
			result := SearchResult{
				ImageID:      hit.ImageID,
				MediaType:    "image",
				MimeType:     row.mimeType,
				Distance:     hit.Distance,
				OriginalName: row.originalName,
				StoragePath:  filepath.ToSlash(row.storagePath),
				Description:  row.description,
				Tags:         row.tags,
			}
			if row.videoID.Valid {
				result.MediaType = "video"
				result.VideoID = row.videoID.Int64
				result.OriginalName = row.videoOriginalName.String
				result.StoragePath = filepath.ToSlash(row.videoStoragePath.String)
				result.MimeType = row.videoMimeType.String
				if desc := strings.TrimSpace(row.videoDescription.String); desc != "" {
					result.Description = desc
				}
				if len(row.videoTags) > 0 {
					result.Tags = row.videoTags
				}
				result.TranscriptText = row.videoTranscriptText.String
				result.PreviewPath = filepath.ToSlash(row.storagePath)
				result.MatchTimestampMS = row.timestampMS.Int64
			}
			results = append(results, result)
			if !row.videoID.Valid {
				continue
			}
		}
	}
	return results, nil
}

func (h *Handler) searchByTags(ctx context.Context, tags []string, limit int, offset int, matchAll bool, includeNSFW bool) ([]SearchResult, int64, error) {
	if len(tags) == 0 {
		return []SearchResult{}, 0, nil
	}

	placeholders := make([]string, len(tags))
	args := make([]any, 0, len(tags)+4)
	for i, tag := range tags {
		placeholders[i] = "?"
		args = append(args, tag)
	}
	requiredCount := 0
	if matchAll {
		requiredCount = len(tags)
	}
	includeNSFWInt := boolToInt(includeNSFW)

	countArgs := append([]any{}, args...)
	countArgs = append(countArgs, requiredCount, requiredCount, includeNSFWInt)

	var total int64
	if err := h.DB.QueryRowContext(ctx, fmt.Sprintf(`
WITH requested_tags(tag) AS (
  SELECT %s
), image_tag_matches AS (
  SELECT i.id AS image_id,
         COUNT(DISTINCT lower(trim(j.value))) AS matched_count
  FROM images i
  JOIN json_each(COALESCE(i.tags_json, '[]')) j
    ON trim(COALESCE(j.value, '')) <> ''
  JOIN requested_tags rt
    ON lower(trim(j.value)) = rt.tag
  GROUP BY i.id
  HAVING (? = 0 OR COUNT(DISTINCT lower(trim(j.value))) = ?)
), media_units AS (
  SELECT
    CASE WHEN vf.video_id IS NULL THEN 'image' ELSE 'video' END AS media_type,
    COALESCE(vf.video_id, i.id) AS unit_id,
    CASE
      WHEN vf.video_id IS NULL THEN CASE
        WHEN EXISTS (
          SELECT 1
          FROM json_each(COALESCE(i.tags_json, '[]')) nsfw_tag
          WHERE lower(trim(COALESCE(nsfw_tag.value, ''))) = 'nsfw'
        ) THEN 1 ELSE 0 END
      ELSE CASE
        WHEN EXISTS (
          SELECT 1
          FROM video_frames vf_nsfw
          JOIN images i_nsfw ON i_nsfw.id = vf_nsfw.image_id
          JOIN json_each(COALESCE(i_nsfw.tags_json, '[]')) nsfw_tag
            ON lower(trim(COALESCE(nsfw_tag.value, ''))) = 'nsfw'
          WHERE vf_nsfw.video_id = vf.video_id
        ) OR EXISTS (
          SELECT 1
          FROM json_each(COALESCE(v.tags_json, '[]')) nsfw_tag_video
          WHERE lower(trim(COALESCE(nsfw_tag_video.value, ''))) = 'nsfw'
        ) THEN 1 ELSE 0 END
    END AS is_nsfw,
    ROW_NUMBER() OVER (
      PARTITION BY CASE WHEN vf.video_id IS NULL THEN 'image:' || i.id ELSE 'video:' || vf.video_id END
      ORDER BY itm.matched_count DESC, i.id DESC
    ) AS rn
  FROM image_tag_matches itm
  JOIN images i ON i.id = itm.image_id
  LEFT JOIN video_frames vf ON vf.image_id = i.id
  LEFT JOIN videos v ON v.id = vf.video_id
)
SELECT COUNT(*)
FROM media_units
WHERE rn = 1
  AND (? = 1 OR is_nsfw = 0)
`, strings.Join(placeholders, " UNION ALL SELECT ")), countArgs...).Scan(&total); err != nil {
		return nil, 0, fmt.Errorf("count tag matches: %w", err)
	}
	if total == 0 {
		return []SearchResult{}, 0, nil
	}

	args = append(args, requiredCount, requiredCount, includeNSFWInt, limit, offset)

	rows, err := h.DB.QueryContext(ctx, fmt.Sprintf(`
WITH requested_tags(tag) AS (
  SELECT %s
), image_tag_matches AS (
  SELECT i.id AS image_id,
         COUNT(DISTINCT lower(trim(j.value))) AS matched_count
  FROM images i
  JOIN json_each(COALESCE(i.tags_json, '[]')) j
    ON trim(COALESCE(j.value, '')) <> ''
  JOIN requested_tags rt
    ON lower(trim(j.value)) = rt.tag
  GROUP BY i.id
  HAVING (? = 0 OR COUNT(DISTINCT lower(trim(j.value))) = ?)
), media_units AS (
  SELECT
    CASE WHEN vf.video_id IS NULL THEN 'image' ELSE 'video' END AS media_type,
    COALESCE(vf.video_id, i.id) AS unit_id,
    i.id AS image_id,
    i.original_name AS image_original_name,
    i.storage_path AS image_storage_path,
    i.mime_type AS image_mime_type,
    COALESCE(i.description, '') AS image_description,
    COALESCE(i.tags_json, '[]') AS image_tags_json,
    vf.video_id AS video_id,
	    v.original_name AS video_original_name,
	    v.storage_path AS video_storage_path,
	    v.mime_type AS video_mime_type,
	    COALESCE(v.description, '') AS video_description,
	    COALESCE(v.tags_json, '[]') AS video_tags_json,
	    COALESCE(v.transcript_text, '') AS video_transcript_text,
	    vf.timestamp_ms AS timestamp_ms,
	    itm.matched_count AS matched_count,
    CASE
      WHEN vf.video_id IS NULL THEN CASE
        WHEN EXISTS (
          SELECT 1
          FROM json_each(COALESCE(i.tags_json, '[]')) nsfw_tag
          WHERE lower(trim(COALESCE(nsfw_tag.value, ''))) = 'nsfw'
        ) THEN 1 ELSE 0 END
      ELSE CASE
        WHEN EXISTS (
          SELECT 1
          FROM video_frames vf_nsfw
          JOIN images i_nsfw ON i_nsfw.id = vf_nsfw.image_id
          JOIN json_each(COALESCE(i_nsfw.tags_json, '[]')) nsfw_tag
            ON lower(trim(COALESCE(nsfw_tag.value, ''))) = 'nsfw'
          WHERE vf_nsfw.video_id = vf.video_id
        ) OR EXISTS (
          SELECT 1
          FROM json_each(COALESCE(v.tags_json, '[]')) nsfw_tag_video
          WHERE lower(trim(COALESCE(nsfw_tag_video.value, ''))) = 'nsfw'
        ) THEN 1 ELSE 0 END
    END AS is_nsfw,
    ROW_NUMBER() OVER (
      PARTITION BY CASE WHEN vf.video_id IS NULL THEN 'image:' || i.id ELSE 'video:' || vf.video_id END
      ORDER BY itm.matched_count DESC, i.id DESC
    ) AS rn
  FROM image_tag_matches itm
  JOIN images i ON i.id = itm.image_id
  LEFT JOIN video_frames vf ON vf.image_id = i.id
  LEFT JOIN videos v ON v.id = vf.video_id
)
SELECT media_type,
       unit_id,
       image_id,
       image_original_name,
       image_storage_path,
       image_mime_type,
       image_description,
       image_tags_json,
       video_id,
       video_original_name,
       video_storage_path,
       video_mime_type,
       video_description,
       video_tags_json,
       video_transcript_text,
       timestamp_ms,
       matched_count
FROM media_units
WHERE rn = 1
  AND (? = 1 OR is_nsfw = 0)
ORDER BY matched_count DESC, unit_id DESC
LIMIT ?
OFFSET ?
`, strings.Join(placeholders, " UNION ALL SELECT ")), args...)
	if err != nil {
		return nil, 0, fmt.Errorf("query tag matches: %w", err)
	}
	defer func() { _ = rows.Close() }()

	results := make([]SearchResult, 0, limit)
	for rows.Next() {
		var result SearchResult
		var mediaType string
		var unitID int64
		var imageStoragePath string
		var imageTagsJSON string
		var videoID sql.NullInt64
		var videoOriginalName sql.NullString
		var videoStoragePath sql.NullString
		var videoMimeType sql.NullString
		var videoDescription sql.NullString
		var videoTagsJSON sql.NullString
		var videoTranscriptText sql.NullString
		var timestampMS sql.NullInt64
		var matchedCount int64
		if err := rows.Scan(
			&mediaType,
			&unitID,
			&result.ImageID,
			&result.OriginalName,
			&imageStoragePath,
			&result.MimeType,
			&result.Description,
			&imageTagsJSON,
			&videoID,
			&videoOriginalName,
			&videoStoragePath,
			&videoMimeType,
			&videoDescription,
			&videoTagsJSON,
			&videoTranscriptText,
			&timestampMS,
			&matchedCount,
		); err != nil {
			return nil, 0, fmt.Errorf("scan tag match row: %w", err)
		}
		decodedTags, err := decodeTagsJSON(imageTagsJSON)
		if err != nil {
			return nil, 0, fmt.Errorf("decode image %d tags: %w", result.ImageID, err)
		}
		result.Tags = decodedTags
		result.SearchSource = "tag"
		result.MediaType = mediaType
		result.StoragePath = filepath.ToSlash(imageStoragePath)
		if mediaType == "video" && videoID.Valid {
			videoTags, err := decodeTagsJSON(videoTagsJSON.String)
			if err != nil {
				return nil, 0, fmt.Errorf("decode video %d tags: %w", videoID.Int64, err)
			}
			result.VideoID = videoID.Int64
			result.OriginalName = videoOriginalName.String
			result.StoragePath = filepath.ToSlash(videoStoragePath.String)
			result.MimeType = videoMimeType.String
			if desc := strings.TrimSpace(videoDescription.String); desc != "" {
				result.Description = desc
			}
			if len(videoTags) > 0 {
				result.Tags = videoTags
			}
			result.TranscriptText = videoTranscriptText.String
			result.PreviewPath = filepath.ToSlash(imageStoragePath)
			if timestampMS.Valid {
				result.MatchTimestampMS = timestampMS.Int64
			}
		}
		if mediaType == "image" {
			result.ImageID = unitID
		}
		results = append(results, result)
	}
	if err := rows.Err(); err != nil {
		return nil, 0, fmt.Errorf("iterate tag match rows: %w", err)
	}

	return results, total, nil
}

func (h *Handler) tagCloud(ctx context.Context, limit int, minCount int, prefix string, includeNSFW bool) ([]TagCount, error) {
	includeNSFWInt := boolToInt(includeNSFW)
	rows, err := h.DB.QueryContext(ctx, `
WITH unit_tags AS (
  SELECT DISTINCT
         CASE WHEN vf.video_id IS NULL THEN 'image:' || i.id ELSE 'video:' || vf.video_id END AS media_unit,
         lower(trim(j.value)) AS tag,
         CASE
           WHEN vf.video_id IS NULL THEN CASE
             WHEN EXISTS (
               SELECT 1
               FROM json_each(COALESCE(i.tags_json, '[]')) nsfw_tag
               WHERE lower(trim(COALESCE(nsfw_tag.value, ''))) = 'nsfw'
             ) THEN 1 ELSE 0 END
           ELSE CASE
             WHEN EXISTS (
               SELECT 1
               FROM video_frames vf_nsfw
               JOIN images i_nsfw ON i_nsfw.id = vf_nsfw.image_id
               JOIN json_each(COALESCE(i_nsfw.tags_json, '[]')) nsfw_tag
                 ON lower(trim(COALESCE(nsfw_tag.value, ''))) = 'nsfw'
               WHERE vf_nsfw.video_id = vf.video_id
              ) OR EXISTS (
                SELECT 1
                FROM json_each(COALESCE(v.tags_json, '[]')) nsfw_tag_video
                WHERE lower(trim(COALESCE(nsfw_tag_video.value, ''))) = 'nsfw'
              ) THEN 1 ELSE 0 END
          END AS is_nsfw
  FROM images i
  JOIN json_each(COALESCE(i.tags_json, '[]')) j
    ON trim(COALESCE(j.value, '')) <> ''
  LEFT JOIN video_frames vf ON vf.image_id = i.id
  LEFT JOIN videos v ON v.id = vf.video_id
)
SELECT tag,
       COUNT(*) AS count
FROM unit_tags
WHERE (? = 1 OR is_nsfw = 0)
  AND (? = '' OR tag LIKE ? || '%')
GROUP BY tag
HAVING COUNT(*) >= ?
ORDER BY count DESC, tag ASC
LIMIT ?
`, includeNSFWInt, prefix, prefix, minCount, limit)
	if err != nil {
		return nil, fmt.Errorf("query tag cloud: %w", err)
	}
	defer func() { _ = rows.Close() }()

	tags := make([]TagCount, 0, limit)
	for rows.Next() {
		var item TagCount
		if err := rows.Scan(&item.Tag, &item.Count); err != nil {
			return nil, fmt.Errorf("scan tag cloud row: %w", err)
		}
		tags = append(tags, item)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate tag cloud rows: %w", err)
	}
	return tags, nil
}

func (h *Handler) searchTranscriptEmbeddings(ctx context.Context, query []float32, limit int, includeNSFW bool) ([]SearchResult, error) {
	includeNSFWInt := boolToInt(includeNSFW)
	rows, err := h.DB.QueryContext(ctx, `
WITH preview_frames AS (
  SELECT vf.video_id,
         vf.image_id,
         i.storage_path,
         ROW_NUMBER() OVER (PARTITION BY vf.video_id ORDER BY vf.frame_index ASC) AS rn
  FROM video_frames vf
  JOIN images i ON i.id = vf.image_id
)
SELECT v.id,
       v.original_name,
       v.storage_path,
       v.mime_type,
       COALESCE(v.description, ''),
       COALESCE(v.tags_json, '[]'),
       COALESCE(v.transcript_text, ''),
       COALESCE(p.image_id, 0),
       COALESCE(p.storage_path, ''),
       vte.vector_blob
FROM video_transcript_embeddings vte
JOIN videos v ON v.id = vte.video_id
LEFT JOIN preview_frames p ON p.video_id = v.id AND p.rn = 1
WHERE vte.model_id = ?
  AND (
    ? = 1
    OR (
      NOT EXISTS (
        SELECT 1
        FROM video_frames vf_nsfw
        JOIN images i_nsfw ON i_nsfw.id = vf_nsfw.image_id
        JOIN json_each(COALESCE(i_nsfw.tags_json, '[]')) nsfw_tag
          ON lower(trim(COALESCE(nsfw_tag.value, ''))) = 'nsfw'
        WHERE vf_nsfw.video_id = v.id
      )
      AND NOT EXISTS (
        SELECT 1
        FROM json_each(COALESCE(v.tags_json, '[]')) nsfw_tag_video
        WHERE lower(trim(COALESCE(nsfw_tag_video.value, ''))) = 'nsfw'
      )
    )
  )
`, h.ModelID, includeNSFWInt)
	if err != nil {
		return nil, fmt.Errorf("query transcript embeddings: %w", err)
	}
	defer func() { _ = rows.Close() }()

	results := make([]SearchResult, 0, limit)
	for rows.Next() {
		var result SearchResult
		var blob []byte
		var tagsJSON string
		if err := rows.Scan(&result.VideoID, &result.OriginalName, &result.StoragePath, &result.MimeType, &result.Description, &tagsJSON, &result.TranscriptText, &result.ImageID, &result.PreviewPath, &blob); err != nil {
			return nil, fmt.Errorf("scan transcript embedding row: %w", err)
		}
		tags, err := decodeTagsJSON(tagsJSON)
		if err != nil {
			return nil, fmt.Errorf("decode transcript video %d tags: %w", result.VideoID, err)
		}
		result.Tags = tags
		vec := vectorindex.BlobToFloats(blob)
		if len(vec) == 0 {
			continue
		}
		result.MediaType = "video"
		result.Distance = 1 - cosine(query, vec)
		result.StoragePath = filepath.ToSlash(result.StoragePath)
		result.PreviewPath = filepath.ToSlash(result.PreviewPath)
		results = append(results, result)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate transcript embedding rows: %w", err)
	}
	sort.Slice(results, func(i, j int) bool {
		if results[i].Distance == results[j].Distance {
			return results[i].VideoID < results[j].VideoID
		}
		return results[i].Distance < results[j].Distance
	})
	if len(results) > limit {
		results = results[:limit]
	}
	return results, nil
}

func mergeResults(frameResults []SearchResult, transcriptResults []SearchResult, limit int) []SearchResult {
	merged := make(map[string]SearchResult, len(frameResults)+len(transcriptResults))
	mergeOne := func(result SearchResult) {
		key := fmt.Sprintf("image:%d", result.ImageID)
		if result.MediaType == "video" && result.VideoID > 0 {
			key = fmt.Sprintf("video:%d", result.VideoID)
		}
		existing, ok := merged[key]
		if !ok || result.Distance < existing.Distance {
			merged[key] = result
			return
		}
		if existing.TranscriptText == "" && result.TranscriptText != "" {
			existing.TranscriptText = result.TranscriptText
			merged[key] = existing
		}
	}
	for _, result := range frameResults {
		mergeOne(result)
	}
	for _, result := range transcriptResults {
		mergeOne(result)
	}
	out := make([]SearchResult, 0, len(merged))
	for _, result := range merged {
		out = append(out, result)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Distance == out[j].Distance {
			if out[i].MediaType == out[j].MediaType {
				if out[i].MediaType == "video" {
					return out[i].VideoID < out[j].VideoID
				}
				return out[i].ImageID < out[j].ImageID
			}
			return out[i].MediaType < out[j].MediaType
		}
		return out[i].Distance < out[j].Distance
	})
	if limit > 0 && len(out) > limit {
		out = out[:limit]
	}
	return out
}

func cosine(a, b []float32) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	if n == 0 {
		return 0
	}
	var dot, na, nb float64
	for i := 0; i < n; i++ {
		av := float64(a[i])
		bv := float64(b[i])
		dot += av * bv
		na += av * av
		nb += bv * bv
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func decodeTagsJSON(raw string) ([]string, error) {
	if raw == "" {
		return nil, nil
	}
	var tags []string
	if err := json.Unmarshal([]byte(raw), &tags); err != nil {
		return nil, err
	}
	return tags, nil
}

func parseLimit(r *http.Request, fallback int) int {
	v := r.URL.Query().Get("limit")
	if v == "" {
		return fallback
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 || n > 200 {
		return fallback
	}
	return n
}

func parseOffset(r *http.Request) int {
	v := r.URL.Query().Get("offset")
	if v == "" {
		return 0
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < 0 || n > 1000000 {
		return 0
	}
	return n
}

func parseMinCount(r *http.Request, fallback int) int {
	v := r.URL.Query().Get("min_count")
	if v == "" {
		return fallback
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 || n > 1000 {
		return fallback
	}
	return n
}

func parseIncludeNSFW(r *http.Request) bool {
	v := strings.TrimSpace(r.URL.Query().Get("include_nsfw"))
	if v == "" {
		return false
	}
	parsed, err := strconv.ParseBool(v)
	if err != nil {
		return false
	}
	return parsed
}

func boolToInt(v bool) int {
	if v {
		return 1
	}
	return 0
}

func parseTagFilters(r *http.Request) []string {
	parts := make([]string, 0, 8)
	parts = append(parts, r.URL.Query()["tag"]...)
	if v := strings.TrimSpace(r.URL.Query().Get("tags")); v != "" {
		parts = append(parts, strings.Split(v, ",")...)
	}
	if len(parts) == 0 {
		if v := strings.TrimSpace(r.URL.Query().Get("q")); v != "" {
			parts = append(parts, strings.Split(v, ",")...)
		}
	}
	return normalizeTagFilters(parts)
}

func parseExplicitTagFilters(r *http.Request) []string {
	parts := make([]string, 0, 8)
	parts = append(parts, r.URL.Query()["tag"]...)
	if v := strings.TrimSpace(r.URL.Query().Get("tags")); v != "" {
		parts = append(parts, strings.Split(v, ",")...)
	}
	return normalizeTagFilters(parts)
}

func normalizeTagFilters(raw []string) []string {
	seen := make(map[string]struct{}, len(raw))
	out := make([]string, 0, len(raw))
	for _, part := range raw {
		tag := strings.ToLower(strings.TrimSpace(part))
		if tag == "" {
			continue
		}
		if _, ok := seen[tag]; ok {
			continue
		}
		seen[tag] = struct{}{}
		out = append(out, tag)
	}
	return out
}

func filterResultsByTags(results []SearchResult, required []string, matchAll bool) []SearchResult {
	normalizedRequired := normalizeTagFilters(required)
	if len(normalizedRequired) == 0 {
		return results
	}

	out := make([]SearchResult, 0, len(results))
	for _, result := range results {
		tags := normalizeTagFilters(result.Tags)
		if len(tags) == 0 {
			continue
		}
		tagSet := make(map[string]struct{}, len(tags))
		for _, tag := range tags {
			tagSet[tag] = struct{}{}
		}

		if matchAll {
			allPresent := true
			for _, requiredTag := range normalizedRequired {
				if _, ok := tagSet[requiredTag]; !ok {
					allPresent = false
					break
				}
			}
			if allPresent {
				out = append(out, result)
			}
			continue
		}

		for _, requiredTag := range normalizedRequired {
			if _, ok := tagSet[requiredTag]; ok {
				out = append(out, result)
				break
			}
		}
	}

	return out
}
