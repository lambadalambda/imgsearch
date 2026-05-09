import type {
  ImagesPage,
  SearchResponse,
  StatsResponse,
  TagCloudResponse,
  UploadBatchResponse,
  VideosPage,
} from "./types";

class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function getJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    credentials: "same-origin",
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!response.ok) {
    let message = response.statusText || "request failed";
    try {
      const payload = await response.json();
      if (payload && typeof payload === "object" && "error" in payload) {
        message = String((payload as { error: unknown }).error);
      }
    } catch {
      /* ignore JSON parse errors */
    }
    throw new ApiError(response.status, message);
  }
  return (await response.json()) as T;
}

export interface ListImagesOptions {
  limit?: number;
  offset?: number;
  includeNSFW?: boolean;
  signal?: AbortSignal;
}

export async function listImages(opts: ListImagesOptions = {}): Promise<ImagesPage> {
  const params = new URLSearchParams();
  params.set("limit", String(opts.limit ?? 24));
  params.set("offset", String(opts.offset ?? 0));
  if (opts.includeNSFW) {
    params.set("include_nsfw", "1");
  }
  return getJSON<ImagesPage>(`/api/images?${params.toString()}`, { signal: opts.signal });
}

export async function listVideos(opts: ListImagesOptions = {}): Promise<VideosPage> {
  const params = new URLSearchParams();
  params.set("limit", String(opts.limit ?? 24));
  params.set("offset", String(opts.offset ?? 0));
  if (opts.includeNSFW) {
    params.set("include_nsfw", "1");
  }
  return getJSON<VideosPage>(`/api/videos?${params.toString()}`, { signal: opts.signal });
}

export interface SearchTextOptions {
  query: string;
  limit?: number;
  offset?: number;
  includeNSFW?: boolean;
  signal?: AbortSignal;
}

export async function searchText(opts: SearchTextOptions): Promise<SearchResponse> {
  const params = new URLSearchParams();
  params.set("q", opts.query);
  params.set("limit", String(opts.limit ?? 48));
  params.set("offset", String(opts.offset ?? 0));
  if (opts.includeNSFW) {
    params.set("include_nsfw", "1");
  }
  return getJSON<SearchResponse>(`/api/search/text?${params.toString()}`, { signal: opts.signal });
}

export interface SimilarSearchOptions {
  imageId: number;
  limit?: number;
  includeNSFW?: boolean;
  signal?: AbortSignal;
}

export async function searchSimilar(opts: SimilarSearchOptions): Promise<SearchResponse> {
  const params = new URLSearchParams();
  params.set("image_id", String(opts.imageId));
  params.set("limit", String(opts.limit ?? 48));
  if (opts.includeNSFW) {
    params.set("include_nsfw", "1");
  }
  return getJSON<SearchResponse>(`/api/search/similar?${params.toString()}`, { signal: opts.signal });
}

export interface TagCloudOptions {
  limit?: number;
  signal?: AbortSignal;
}

export async function listTagCloud(opts: TagCloudOptions = {}): Promise<TagCloudResponse> {
  const params = new URLSearchParams();
  params.set("limit", String(opts.limit ?? 12));
  return getJSON<TagCloudResponse>(`/api/search/tag-cloud?${params.toString()}`, {
    signal: opts.signal,
  });
}

export async function getStats(signal?: AbortSignal): Promise<StatsResponse> {
  return getJSON<StatsResponse>("/api/stats", { signal });
}

export interface SearchTagsOptions {
  tags: string[];
  mode?: "any" | "all";
  limit?: number;
  offset?: number;
  includeNSFW?: boolean;
  signal?: AbortSignal;
}

export async function searchTags(opts: SearchTagsOptions): Promise<SearchResponse> {
  const params = new URLSearchParams();
  for (const tag of opts.tags) {
    params.append("tag", tag);
  }
  params.set("tag_mode", opts.mode ?? "any");
  params.set("limit", String(opts.limit ?? 48));
  params.set("offset", String(opts.offset ?? 0));
  if (opts.includeNSFW) {
    params.set("include_nsfw", "1");
  }
  return getJSON<SearchResponse>(`/api/search/tags?${params.toString()}`, { signal: opts.signal });
}

export type MediaKind = "image" | "video";

async function postJSON(url: string): Promise<unknown> {
  const response = await fetch(url, { method: "POST", credentials: "same-origin" });
  if (!response.ok) {
    let message = response.statusText || "request failed";
    try {
      const payload = await response.json();
      if (payload && typeof payload === "object" && "error" in payload) {
        message = String((payload as { error: unknown }).error);
      }
    } catch {
      /* ignore */
    }
    throw new ApiError(response.status, message);
  }
  try {
    return await response.json();
  } catch {
    return null;
  }
}

export async function toggleNSFW(kind: MediaKind, id: number): Promise<void> {
  const path = kind === "video" ? "videos" : "images";
  await postJSON(`/api/${path}/${id}/toggle-nsfw`);
}

export async function reannotate(kind: MediaKind, id: number): Promise<void> {
  const path = kind === "video" ? "videos" : "images";
  await postJSON(`/api/${path}/${id}/reannotate`);
}

export async function deleteMedia(kind: MediaKind, id: number): Promise<void> {
  const path = kind === "video" ? "videos" : "images";
  const response = await fetch(`/api/${path}/${id}`, { method: "DELETE", credentials: "same-origin" });
  if (!response.ok) {
    let message = response.statusText || "delete failed";
    try {
      const payload = await response.json();
      if (payload && typeof payload === "object" && "error" in payload) {
        message = String((payload as { error: unknown }).error);
      }
    } catch {
      /* ignore */
    }
    throw new ApiError(response.status, message);
  }
}

/** Backend caps from internal/upload/http.go. Keep in sync. */
export const UPLOAD_MAX_FILES = 32;
export const UPLOAD_MAX_BYTES = 64 * 1024 * 1024; // 64 MiB request body

export const UPLOAD_ACCEPT =
  "image/png,image/jpeg,image/webp,image/avif,video/mp4,video/quicktime,video/webm,video/x-matroska,.mp4,.mov,.webm,.mkv";

export interface UploadOptions {
  signal?: AbortSignal;
}

/**
 * POST one multipart batch to /api/upload (field name "file"). The backend
 * returns the same JSON envelope for 200/201/207 success and for 400 when
 * every file failed; in either case we surface the per-file results so the
 * UI can render row-level states. Network/auth errors throw ApiError.
 */
export async function uploadFiles(
  files: File[],
  opts: UploadOptions = {},
): Promise<UploadBatchResponse> {
  const form = new FormData();
  for (const file of files) {
    form.append("file", file, file.name);
  }
  const response = await fetch("/api/upload", {
    method: "POST",
    credentials: "same-origin",
    body: form,
    signal: opts.signal,
  });
  let payload: unknown = null;
  try {
    payload = await response.json();
  } catch {
    /* ignore */
  }
  if (
    payload &&
    typeof payload === "object" &&
    Array.isArray((payload as { uploads?: unknown }).uploads)
  ) {
    return payload as UploadBatchResponse;
  }
  let message = response.statusText || "upload failed";
  if (payload && typeof payload === "object" && "error" in payload) {
    message = String((payload as { error: unknown }).error);
  }
  throw new ApiError(response.status, message);
}

export { ApiError };
