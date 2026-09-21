import { describe, expect, it } from "vitest";
import { combineLibraryPins, newLibrarySeed, randomKey } from "./library";
import type { Pin } from "./types";

function pin(key: string, createdAt?: string): Pin {
  return { key, imageId: 1, mediaType: key.startsWith("video") ? "video" : "image", thumbUrl: "", mediaUrl: "", title: key, tags: [], createdAt } as unknown as Pin;
}

describe("newLibrarySeed / randomKey", () => {
  it("produces 31-bit seeds and deterministic keys", () => {
    for (let i = 0; i < 20; i += 1) {
      const seed = newLibrarySeed();
      expect(seed).toBeGreaterThanOrEqual(0);
      expect(seed).toBeLessThan(0x80000000);
    }
    expect(randomKey("image:1", 7)).toBe(randomKey("image:1", 7));
    expect(randomKey("image:1", 7)).not.toBe(randomKey("image:1", 8));
    expect(randomKey("image:1", 7)).not.toBe(randomKey("image:2", 7));
  });
});

describe("combineLibraryPins", () => {
  it("sorts newest first with a stable key tie-break", () => {
    const out = combineLibraryPins(
      [pin("image:1", "2026-01-01T00:00:00Z"), pin("image:2", "2026-03-01T00:00:00Z")],
      [pin("video:1", "2026-02-01T00:00:00Z"), pin("video:2", "2026-03-01T00:00:00Z")],
      "newest",
      0,
    );
    expect(out.map((p) => p.key)).toEqual(["video:2", "image:2", "video:1", "image:1"]);
  });
  it("interleaves images and videos deterministically for a seed and keeps every pin", () => {
    const images = [pin("image:1"), pin("image:2"), pin("image:3")];
    const videos = [pin("video:1")];
    const a = combineLibraryPins(images, videos, "random", 123);
    const b = combineLibraryPins(images, videos, "random", 123);
    expect(a).toEqual(b);
    expect(a.map((p) => p.key).sort()).toEqual(["image:1", "image:2", "image:3", "video:1"]);
    expect(a.slice(0, 2).map((p) => p.mediaType).sort()).toEqual(["image", "video"]);
    const leads = new Set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((seed) => combineLibraryPins(images, videos, "random", seed)[0].mediaType));
    expect(leads.size).toBe(2);
  });
});
