import type { ViewMode } from "./urlState";
import type { LibraryMedia, LibrarySort } from "./stores";

export function hasSearchChrome(mode: ViewMode): boolean {
  return ["library", "search", "tag", "similar", "byimage"].includes(mode);
}

export function filterSummary(
  mode: ViewMode,
  media: LibraryMedia,
  sort: LibrarySort,
  includeNSFW: boolean,
): string {
  const visibility = `NSFW ${includeNSFW ? "shown" : "hidden"}`;
  if (mode === "library") {
    const types = {
      all: "All media",
      images: "Images only",
      videos: "Videos only",
    };
    const orders = {
      random: "Random",
      newest: "Recently added",
      captured: "Date taken",
    };
    return `${types[media]} · ${orders[sort]} · ${visibility}`;
  }
  if (mode === "duplicates") return `Images only · ${visibility}`;
  if (mode === "collection" || mode === "collections")
    return `All saved media · ${visibility}`;
  const ranked = ["search", "similar", "byimage"].includes(mode);
  return `All media · ${ranked ? "Relevance · " : ""}${visibility}`;
}

/** The existing relative similarity scale, not calibrated confidence. */
export function relativeMatchLabel(score: number | undefined): string {
  return score === undefined ? "" : `Match ${Math.round(score * 100)}/100`;
}

/** Ranked APIs report the size of a capped batch, not an exhaustive hit count. */
export function resultsCountLabel(
  mode: ViewMode,
  visible: number,
  total: number,
): string {
  const shown = `Visible results: ${visible.toLocaleString()}`;
  if (["search", "similar", "byimage"].includes(mode)) {
    return `${shown} · Top results returned (not exhaustive)`;
  }
  return `${shown} · ${total.toLocaleString()} ${total === 1 ? "result" : "results"} in this scope`;
}
