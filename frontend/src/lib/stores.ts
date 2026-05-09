import { writable, derived, get } from "svelte/store";
import type { Pin } from "./types";

export type ViewMode = "library" | "search" | "similar";

export interface AppMode {
  mode: ViewMode;
  query?: string;
  similarTo?: number;
}

function readURL(): AppMode {
  if (typeof window === "undefined") return { mode: "library" };
  const params = new URLSearchParams(window.location.search);
  const q = params.get("q");
  const similar = params.get("similar");
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
  return "Library";
});

export function getMode(): AppMode {
  return get(mode);
}
