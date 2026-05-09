import type { ImageRecord, Pin, SearchResult, VideoRecord } from "./types";

/** Build a media URL from a `storage_path` like "images/abc..." or "videos/abc...". */
export function mediaUrl(storagePath: string | undefined | null): string {
  if (!storagePath) return "";
  const trimmed = storagePath.replace(/^\/+/, "");
  return `/media/${trimmed}`;
}

/** Convert backend `distance` (lower = closer) into a 0-1 similarity score. */
export function matchScore(distance: number | undefined): number | undefined {
  if (distance === undefined || distance === null || Number.isNaN(distance)) return undefined;
  return Math.max(0, Math.min(1, 1 - distance));
}

export function formatPercent(score: number | undefined): string {
  if (score === undefined) return "";
  return `${Math.round(score * 100)}%`;
}

export function formatDuration(ms: number | undefined): string {
  if (!ms || ms <= 0) return "";
  const seconds = Math.round(ms / 1000);
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

export function trimTitle(text: string | undefined, max = 140): string {
  if (!text) return "";
  const cleaned = text.replace(/\s+/g, " ").trim();
  if (cleaned.length <= max) return cleaned;
  return cleaned.slice(0, max - 1).trimEnd() + "…";
}

/** Pick a best title for a pin: first sentence of description, else filename. */
export function deriveTitle(record: { description?: string; original_name?: string }): string {
  const desc = record.description?.trim();
  if (desc) {
    const firstSentence = desc.split(/(?<=[.!?])\s+/)[0] ?? desc;
    return trimTitle(firstSentence, 160);
  }
  return record.original_name ?? "Untitled";
}

export function pinFromImage(record: ImageRecord): Pin {
  return {
    key: `image:${record.image_id}`,
    imageId: record.image_id,
    mediaType: "image",
    thumbUrl: mediaUrl(record.storage_path),
    mediaUrl: mediaUrl(record.storage_path),
    width: record.width,
    height: record.height,
    title: deriveTitle(record),
    filename: record.original_name,
    tags: record.tags ?? [],
  };
}

export function pinFromVideo(record: VideoRecord): Pin {
  return {
    key: `video:${record.video_id}`,
    imageId: record.image_id,
    videoId: record.video_id,
    mediaType: "video",
    // Best-effort: the videos endpoint provides preview frames in `storage_path` already
    // when used via the search results. For the gallery videos endpoint, use the hash too.
    thumbUrl: mediaUrl(record.storage_path),
    mediaUrl: mediaUrl(record.storage_path),
    width: record.width,
    height: record.height,
    title: deriveTitle(record),
    filename: record.original_name,
    tags: record.tags ?? [],
    durationMs: record.duration_ms,
  };
}

export function pinFromSearchResult(record: SearchResult): Pin {
  const thumb = mediaUrl(record.preview_path || record.storage_path);
  const media = mediaUrl(record.storage_path);
  return {
    key: `${record.media_type}:${record.video_id ?? record.image_id}`,
    imageId: record.image_id,
    videoId: record.video_id,
    mediaType: record.media_type,
    thumbUrl: thumb,
    mediaUrl: media,
    width: record.width,
    height: record.height,
    title: deriveTitle(record),
    filename: record.original_name,
    tags: record.tags ?? [],
    matchScore: matchScore(record.distance),
    matchTimestampMs: record.match_timestamp_ms,
    durationMs: record.duration_ms,
    isAnchor: record.is_anchor,
  };
}

const TAG_TONES = ["plum", "moss", "gold"] as const;
export type TagTone = (typeof TAG_TONES)[number];

/** Deterministic tone for a tag so the chip color stays stable across renders. */
export function tagTone(tag: string): TagTone | undefined {
  if (!tag) return undefined;
  let hash = 0;
  for (let i = 0; i < tag.length; i += 1) {
    hash = (hash * 31 + tag.charCodeAt(i)) >>> 0;
  }
  // Only colorize a third of tags to keep things calm.
  if (hash % 3 !== 0) return undefined;
  return TAG_TONES[(hash >> 2) % TAG_TONES.length];
}

export function pluralize(count: number, singular: string, plural?: string): string {
  return count === 1 ? singular : (plural ?? `${singular}s`);
}

export function formatCount(value: number): string {
  return value.toLocaleString();
}
