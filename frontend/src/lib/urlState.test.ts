import { describe, expect, it } from "vitest";
import { parseSearch, searchFor, type AppMode } from "./urlState";

describe("parseSearch", () => {
  it("maps each query form to a view", () => {
    expect(parseSearch("")).toEqual({ mode: "library" });
    expect(parseSearch("?view=stats")).toEqual({ mode: "stats" });
    expect(parseSearch("?view=settings")).toEqual({ mode: "settings" });
    expect(parseSearch("?q=warm+portrait")).toEqual({ mode: "search", query: "warm portrait" });
    expect(parseSearch("?similar=42")).toEqual({ mode: "similar", similarTo: 42 });
    expect(parseSearch("?tag=cat&tag=indoor&tag_mode=all")).toEqual({ mode: "tag", tags: ["cat", "indoor"], tagMode: "all" });
    expect(parseSearch("?tag=cat")).toEqual({ mode: "tag", tags: ["cat"], tagMode: "any" });
  });
  it("prefers tags over similar over query, and ignores a non-numeric similar id", () => {
    expect(parseSearch("?q=x&similar=7&tag=cat").mode).toBe("tag");
    expect(parseSearch("?q=x&similar=7").mode).toBe("similar");
    expect(parseSearch("?similar=abc&q=x")).toEqual({ mode: "search", query: "x" });
    expect(parseSearch("?similar=abc")).toEqual({ mode: "library" });
  });
});

describe("searchFor", () => {
  it("round-trips every view through the URL", () => {
    const views: AppMode[] = [
      { mode: "library" },
      { mode: "stats" },
      { mode: "settings" },
      { mode: "search", query: "a b & c" },
      { mode: "similar", similarTo: 9 },
      { mode: "tag", tags: ["cat", "sun set"], tagMode: "any" },
      { mode: "tag", tags: ["cat"], tagMode: "all" },
    ];
    for (const view of views) {
      expect(parseSearch(searchFor(view))).toEqual(view);
    }
  });
  it("falls back to the library for incomplete views", () => {
    expect(searchFor({ mode: "search" })).toBe("");
    expect(searchFor({ mode: "tag", tags: [] })).toBe("");
    expect(searchFor({ mode: "library" })).toBe("");
  });
});
