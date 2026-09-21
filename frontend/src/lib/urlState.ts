export type ViewMode = "library" | "search" | "similar" | "tag" | "stats" | "settings" | "duplicates" | "byimage";

export interface AppMode {
  mode: ViewMode;
  query?: string;
  similarTo?: number;
  tags?: string[];
  tagMode?: "any" | "all";
}

/** Parse a location search string (with or without the leading "?") into a view. */
export function parseSearch(search: string): AppMode {
  const params = new URLSearchParams(search);
  if (params.get("view") === "stats") {
    return { mode: "stats" };
  }
  if (params.get("view") === "settings") {
    return { mode: "settings" };
  }
  if (params.get("view") === "duplicates") {
    return { mode: "duplicates" };
  }
  // "byimage" is deliberately not restored from the URL: the query image
  // lives only in memory, so a reload lands on the library.
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

/** The search string for a view: "?..." or "" for the library. */
export function searchFor(state: AppMode): string {
  const params = new URLSearchParams();
  if (state.mode === "stats") {
    params.set("view", "stats");
  } else if (state.mode === "settings") {
    params.set("view", "settings");
  } else if (state.mode === "duplicates") {
    params.set("view", "duplicates");
  } else if (state.mode === "byimage") {
    params.set("view", "byimage");
  } else if (state.mode === "search" && state.query) {
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
  const search = params.toString();
  return search ? `?${search}` : "";
}
