import type {
  ImagesPage,
  SearchResponse,
  StatsResponse,
  TagCloudResponse,
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

export async function toggleNSFW(kind: "image" | "video", id: number): Promise<void> {
  const path = kind === "video" ? "videos" : "images";
  await fetch(`/api/${path}/${id}/toggle-nsfw`, {
    method: "POST",
    credentials: "same-origin",
  });
}

export { ApiError };
