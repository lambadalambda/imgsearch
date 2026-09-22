import { describe, expect, it } from "vitest";
import {
  filterSummary,
  hasSearchChrome,
  relativeMatchLabel,
  resultsCountLabel,
} from "./searchPresentation";

describe("search scope presentation", () => {
  it("keeps utility pages and collection detail free of global search chrome", () => {
    for (const mode of [
      "settings",
      "stats",
      "duplicates",
      "collections",
      "collection",
    ] as const) {
      expect(hasSearchChrome(mode)).toBe(false);
    }
    for (const mode of [
      "library",
      "search",
      "tag",
      "similar",
      "byimage",
    ] as const) {
      expect(hasSearchChrome(mode)).toBe(true);
    }
  });

  it("never implies library type/sort preferences apply to search or collections", () => {
    expect(filterSummary("library", "videos", "captured", false)).toBe(
      "Videos only · Date taken · NSFW hidden",
    );
    expect(filterSummary("search", "videos", "captured", true)).toBe(
      "All media · Relevance · NSFW shown",
    );
    expect(filterSummary("duplicates", "videos", "newest", false)).toBe(
      "Images only · NSFW hidden",
    );
    expect(filterSummary("tag", "images", "random", false)).toBe(
      "All media · NSFW hidden",
    );
    expect(filterSummary("collection", "videos", "captured", false)).toBe(
      "All saved media · NSFW hidden",
    );
  });

  it("does not present capped ranked batches as exhaustive totals", () => {
    for (const mode of ["search", "similar", "byimage"] as const) {
      expect(resultsCountLabel(mode, 48, 48)).toBe(
        "Visible results: 48 · Top results returned (not exhaustive)",
      );
    }
    for (const mode of ["library", "tag", "collection"] as const) {
      expect(resultsCountLabel(mode, 48, 150)).toBe(
        "Visible results: 48 · 150 results in this scope",
      );
      expect(resultsCountLabel(mode, 1, 1)).toBe(
        "Visible results: 1 · 1 result in this scope",
      );
    }
  });

  it("labels relative match scores without a probability percentage", () => {
    expect(relativeMatchLabel(0.42)).toBe("Match 42/100");
    expect(relativeMatchLabel(0)).toBe("Match 0/100");
    expect(relativeMatchLabel(undefined)).toBe("");
  });
});
