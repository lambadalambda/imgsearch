export type MediaType = "image" | "video";

export interface ImageRecord {
  image_id: number;
  original_name: string;
  storage_path: string;
  mime_type: string;
  width: number;
  height: number;
  index_state: string;
  created_at?: string;
  description?: string;
  tags?: string[];
}

export interface VideoRecord extends ImageRecord {
  video_id: number;
  preview_path?: string;
  preview_width?: number;
  preview_height?: number;
  duration_ms?: number;
  frame_count?: number;
}

export interface SearchResult {
  image_id: number;
  media_type: MediaType;
  video_id?: number;
  preview_path?: string;
  match_timestamp_ms?: number;
  mime_type: string;
  duration_ms?: number;
  width: number;
  height: number;
  frame_count?: number;
  distance: number;
  original_name: string;
  storage_path: string;
  description?: string;
  tags?: string[];
  is_anchor?: boolean;
  /** Tag-based searches set this to "tag" so the UI can suppress similarity badges. */
  search_source?: "tag" | "embedding" | string;
}

export interface SearchDebug {
  duration_ms?: number;
  index_backend?: string;
  index_strategy?: string;
  quantization?: string;
}

export interface ImagesPage {
  images: ImageRecord[];
  total: number;
}

export interface VideosPage {
  videos: VideoRecord[];
  total: number;
}

export interface SearchResponse {
  results: SearchResult[];
  total: number;
  debug?: SearchDebug;
}

export interface TagCloudEntry {
  tag: string;
  count: number;
}

export interface TagCloudResponse {
  tags: TagCloudEntry[];
}

export interface QueueStats {
  total: number;
  tracked: number;
  missing: number;
  annotations_missing: number;
  runnable: number;
  pending: number;
  leased: number;
  done: number;
  failed: number;
  oldest_runnable_age_seconds: number;
}

export interface JobKindStats {
  tracked: number;
  runnable: number;
  pending: number;
  leased: number;
  done: number;
  failed: number;
  oldest_runnable_age_seconds: number;
}

export interface FailureItem {
  job_id: number;
  kind: string;
  media_type: MediaType;
  image_id?: number;
  video_id?: number;
  original_name: string;
  attempts: number;
  last_error: string;
  updated_at: string;
}

export interface StatsResponse {
  images_total: number;
  standalone_images_total: number;
  video_frame_images_total?: number;
  videos_total: number;
  queue?: QueueStats;
  image_annotation_expected?: number;
  image_annotation_missing?: number;
  video_annotation_expected?: number;
  video_annotation_missing?: number;
  video_transcription_expected?: number;
  video_transcription_missing?: number;
  job_kinds?: Record<string, JobKindStats>;
  recent_failures?: FailureItem[];
}

export interface UploadEntry {
  filename: string;
  media_type: MediaType;
  image_id?: number;
  video_id?: number;
  sha256?: string;
  duplicate?: boolean;
  error?: string;
}

export interface UploadBatchResponse {
  uploads: UploadEntry[];
  created: number;
  duplicates: number;
  failed: number;
}

/** A unified pin used by the masonry grid, abstracting over images, videos,
 *  and search results. */
export interface Pin {
  /** Stable id for the DOM key; combines media type + id. */
  key: string;
  imageId: number;
  videoId?: number;
  mediaType: MediaType;
  /** URL to render in the thumbnail. */
  thumbUrl: string;
  /** URL to play / open in a lightbox. */
  mediaUrl: string;
  width: number;
  height: number;
  title: string;
  filename: string;
  tags: string[];
  /** 0-1 similarity (1 - distance) when this pin came from a search result. */
  matchScore?: number;
  /** Raw backend vector distance; lower is closer. Used for Feed reranking. */
  distance?: number;
  /** Match offset within a video, when applicable. */
  matchTimestampMs?: number;
  /** Duration in ms for video pins. */
  durationMs?: number;
  /** ISO-ish backend creation timestamp. Used for client-side mixed-media sorting. */
  createdAt?: string;
  /** Whether this is the anchor of a similar-search. */
  isAnchor?: boolean;
  /** Whether this pin is currently flagged as NSFW (best-effort, derived from tags). */
  isNSFW?: boolean;
  /** Stored media MIME type (e.g. video/mp4). Populated for video pins so the
   *  Feed launcher can call canPlayType() before opening. */
  mimeType?: string;
}
