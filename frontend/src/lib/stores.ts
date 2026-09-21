import { writable, derived, get } from "svelte/store";
import type { Pin, StatsResponse } from "./types";
import { parseSearch, searchFor, type AppMode } from "./urlState";

export type { AppMode, ViewMode } from "./urlState";
export type LibrarySort = "random" | "newest";
export type LibraryMedia = "all" | "images" | "videos";

function readURL(): AppMode {
  if (typeof window === "undefined") return { mode: "library" };
  return parseSearch(window.location.search);
}

function writeURL(state: AppMode, replace: boolean): void {
  if (typeof window === "undefined") return;
  const targetSearch = searchFor(state);
  // Re-asserting the current view (e.g. re-submitting the same search) must
  // not pile up duplicate history entries.
  if (window.location.search === targetSearch) return;
  const url = targetSearch || window.location.pathname;
  if (replace) {
    window.history.replaceState(null, "", url);
  } else {
    window.history.pushState(null, "", url);
  }
}

export const mode = writable<AppMode>(readURL());

// In-app navigation pushes history entries so the browser Back button walks
// previous views instead of leaving the site. The initial subscription run
// only normalizes the URL in place, and popstate-driven updates must not
// write again (the browser already moved the history pointer).
let restoringFromHistory = false;
let urlInitialized = false;

mode.subscribe((value) => {
  if (restoringFromHistory) return;
  writeURL(value, !urlInitialized);
  urlInitialized = true;
});

if (typeof window !== "undefined") {
  window.addEventListener("popstate", () => {
    restoringFromHistory = true;
    try {
      mode.set(readURL());
    } finally {
      restoringFromHistory = false;
    }
  });
}

export function setLibrary(): void {
  mode.set({ mode: "library" });
}

export function setQuery(query: string): void {
  const trimmed = query.trim();
  if (!trimmed) {
    setLibrary();
    return;
  }
  mode.set({ mode: "search", query: trimmed });
}

export function setSimilar(imageId: number): void {
  mode.set({ mode: "similar", similarTo: imageId });
}

export function setTagSearch(tags: string[], tagMode: "any" | "all" = "any"): void {
  const cleaned = tags.map((t) => t.trim()).filter(Boolean);
  if (cleaned.length === 0) {
    setLibrary();
    return;
  }
  mode.set({ mode: "tag", tags: cleaned, tagMode });
}

export function setStats(): void {
  mode.set({ mode: "stats" });
}

export function setSettings(): void {
  mode.set({ mode: "settings" });
}

/** Device-local view preference persisted in localStorage. Invalid or
 *  unreadable stored values (private mode, manual edits) fall back to the
 *  default. */
function persistedStore<T>(key: string, fallback: T, parse: (raw: string) => T | undefined) {
  const storageKey = `imgsearch.${key}`;
  let initial = fallback;
  if (typeof window !== "undefined") {
    try {
      const raw = window.localStorage.getItem(storageKey);
      if (raw !== null) {
        const parsed = parse(raw);
        if (parsed !== undefined) initial = parsed;
      }
    } catch {
      /* storage unavailable */
    }
  }
  const store = writable<T>(initial);
  store.subscribe((value) => {
    if (typeof window === "undefined") return;
    try {
      window.localStorage.setItem(storageKey, String(value));
    } catch {
      /* storage unavailable */
    }
  });
  return store;
}

export const includeNSFW = persistedStore<boolean>("includeNSFW", false, (raw) =>
  raw === "true" ? true : raw === "false" ? false : undefined,
);

export const librarySort = persistedStore<LibrarySort>("librarySort", "random", (raw) =>
  raw === "random" || raw === "newest" ? raw : undefined,
);

export const libraryMedia = persistedStore<LibraryMedia>("libraryMedia", "all", (raw) =>
  raw === "all" || raw === "images" || raw === "videos" ? raw : undefined,
);

export const lightboxPin = writable<Pin | null>(null);

export interface StatsSnapshot extends StatsResponse {
  images: number;
  videos: number;
}

export const stats = writable<StatsSnapshot | null>(null);

export const topTags = writable<Array<{ tag: string; count: number }>>([]);

/** Snapshot of the current pins shown in the masonry. */
export const pins = writable<Pin[]>([]);

export interface ResultsMeta {
  total: number;
  durationMs?: number;
  loading: boolean;
  error?: string;
}

export const resultsMeta = writable<ResultsMeta>({ total: 0, loading: false });

export const headline = derived(mode, ($mode) => {
  if ($mode.mode === "search" && $mode.query) {
    return $mode.query;
  }
  if ($mode.mode === "similar") {
    return "Similar in your library";
  }
  if ($mode.mode === "tag" && $mode.tags?.length) {
    const joiner = $mode.tagMode === "all" ? " + " : " · ";
    return $mode.tags.join(joiner);
  }
  if ($mode.mode === "stats") {
    return "Statistics";
  }
  if ($mode.mode === "settings") {
    return "Settings";
  }
  return "Library";
});

/** Drives "load more" pagination by re-running the data effect when bumped. */
export const pageBump = writable<number>(0);

export function bumpPage(): void {
  pageBump.update((n) => n + 1);
}

/** Drives a fresh (replacing) data fetch — bumped after successful uploads
 *  so newly indexed media surfaces in the library without a page reload. */
export const dataEpoch = writable<number>(0);

export function bumpDataEpoch(): void {
  dataEpoch.update((n) => n + 1);
}

/** Whether the in-app Upload modal is open. */
export const uploadOpen = writable<boolean>(false);

export function openUpload(): void {
  uploadOpen.set(true);
}

export function closeUpload(): void {
  uploadOpen.set(false);
}

/** Seed pin for the similar-video Feed overlay. null = closed. */
export const feedSeed = writable<Pin | null>(null);

export function openFeed(seed: Pin): void {
  if (seed.mediaType !== "video" || !seed.videoId) return;
  feedSeed.set(seed);
}

export function closeFeed(): void {
  feedSeed.set(null);
}

export function getMode(): AppMode {
  return get(mode);
}
