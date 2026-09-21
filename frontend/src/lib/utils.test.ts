import { describe, expect, it } from "vitest";
import type { Pin } from "./types";
import { appendUniquePins, deriveTitle, formatDuration, matchScore, mediaUrl, tagTone, trimTitle } from "./utils";

function pin(key: string): Pin {
  return { key, imageId: 1, mediaType: "image", thumbUrl: "", mediaUrl: "", title: key, tags: [] } as unknown as Pin;
}

describe("matchScore", () => {
  it("inverts distance and clamps to 0..1", () => {
    expect(matchScore(0.25)).toBe(0.75);
    expect(matchScore(-1)).toBe(1);
    expect(matchScore(2)).toBe(0);
    expect(matchScore(undefined)).toBeUndefined();
    expect(matchScore(Number.NaN)).toBeUndefined();
  });
});

describe("deriveTitle", () => {
  it("prefers the explicit title, then the first sentence, then the filename", () => {
    expect(deriveTitle({ title: "  Sunset  ", description: "A long text." })).toBe("Sunset");
    expect(deriveTitle({ full_description: "A tabby cat. It sits on a mat.", original_name: "x.jpg" })).toBe("A tabby cat.");
    expect(deriveTitle({ description: "Only one sentence without a period", original_name: "x.jpg" })).toBe(
      "Only one sentence without a period",
    );
    expect(deriveTitle({ original_name: "x.jpg" })).toBe("x.jpg");
    expect(deriveTitle({})).toBe("Untitled");
  });
  it("clamps long titles with an ellipsis", () => {
    const long = "word ".repeat(60);
    const title = deriveTitle({ title: long });
    expect(title.length).toBeLessThanOrEqual(160);
    expect(title.endsWith("…")).toBe(true);
  });
});

describe("trimTitle / formatDuration / mediaUrl", () => {
  it("collapses whitespace and trims", () => {
    expect(trimTitle("  a \n b  ")).toBe("a b");
    expect(trimTitle("abcdef", 4)).toBe("abc…");
  });
  it("formats durations as m:ss", () => {
    expect(formatDuration(65_400)).toBe("1:05");
    expect(formatDuration(0)).toBe("");
    expect(formatDuration(undefined)).toBe("");
  });
  it("builds media urls without doubled slashes", () => {
    expect(mediaUrl("images/abc")).toBe("/media/images/abc");
    expect(mediaUrl("/videos/abc")).toBe("/media/videos/abc");
    expect(mediaUrl(undefined)).toBe("");
  });
});

describe("tagTone", () => {
  it("is deterministic and only colours some tags", () => {
    const tags = Array.from({ length: 200 }, (_, i) => `tag-${i}`);
    const tones = tags.map(tagTone);
    expect(tags.map(tagTone)).toEqual(tones);
    const coloured = tones.filter(Boolean);
    expect(coloured.length).toBeGreaterThan(0);
    expect(coloured.length).toBeLessThan(tags.length);
    for (const tone of coloured) expect(["plum", "moss", "gold"]).toContain(tone);
    expect(tagTone("")).toBeUndefined();
  });
});

describe("appendUniquePins", () => {
  it("drops pins whose key is already present and keeps order", () => {
    const existing = [pin("a"), pin("b")];
    const out = appendUniquePins(existing, [pin("b"), pin("c"), pin("c")]);
    expect(out.map((p) => p.key)).toEqual(["a", "b", "c"]);
  });
  it("returns the same array when nothing new arrives", () => {
    const existing = [pin("a")];
    expect(appendUniquePins(existing, [pin("a")])).toBe(existing);
  });
});
