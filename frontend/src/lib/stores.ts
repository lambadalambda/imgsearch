import { writable, derived, get } from "svelte/store";
import type { Pin } from "./types";

export type ViewMode = "library" | "search" | "similar" | "tag";

export interface AppMode {
  mode: ViewMode;
  query?: string;
  similarTo?: number;
  tags?: string[];
  tagMode?: "any" | "all";
}

function readURL(): AppMode {
  if (typeof window === "undefined") return { mode: "library" };
  const params = new URLSearchParams(window.location.search);
  const q = params.get("q");
  const similar = params.get("similar");
  const tags = params.getAll("tag").filter(Boolean);
  if (tags.length > 0) {
    const tagMode = params.get("tag_mode") === "all" ? "all" : "any";
    return { mode: "tag", tags, tagMode };
  }
  if (similar && Number.isFinite(Number(similar))) {
    return { mode: "similar", similarTo: Number(similar) };
  }
  if (q) {
    return { mode: "search", query: q };
  }
  return { mode: "library" };
}

function writeURL(state: AppMode): void {
  if (typeof window === "undefined") return;
  const params = new URLSearchParams();
  if (state.mode === "search" && state.query) {
    params.set("q", state.query);
  } else if (state.mode === "similar" && state.similarTo !== undefined) {
    params.set("similar", String(state.similarTo));
  } else if (state.mode === "tag" && state.tags?.length) {
    for (const tag of state.tags) {
      params.append("tag", tag);
    }
    if (state.tagMode === "all") {
      params.set("tag_mode", "all");
    }
  }
  const newSearch = params.toString();
  const url = newSearch ? `?${newSearch}` : window.location.pathname;
  window.history.replaceState(null, "", url);
}

export const mode = writable<AppMode>(readURL());

mode.subscribe((value) => writeURL(value));

if (typeof window !== "undefined") {
  window.addEventListener("popstate", () => {
    mode.set(readURL());
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

export const includeNSFW = writable<boolean>(false);

export const lightboxPin = writable<Pin | null>(null);

export const stats = writable<{ images: number; videos: number } | null>(null);

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
  return "Library";
});

/** Drives "load more" pagination by re-running the data effect when bumped. */
export const pageBump = writable<number>(0);

export function bumpPage(): void {
  pageBump.update((n) => n + 1);
}

export function getMode(): AppMode {
  return get(mode);
}
