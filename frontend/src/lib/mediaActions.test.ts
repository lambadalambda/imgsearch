import { afterEach, describe, expect, it, vi } from "vitest";
import { get } from "svelte/store";
import { createMediaActions } from "./mediaActions";
import { collectionEpoch, lightboxPin, pins } from "./stores";
import { pinFromImage, pinFromVideo } from "./utils";

const image = { image_id: 123, original_name: "cat.jpg", storage_path: "cat.jpg", mime_type: "image/jpeg", width: 100, height: 100, index_state: "done" };
afterEach(() => {
  vi.unstubAllGlobals();
  pins.set([]);
  lightboxPin.set(null);
});

describe("collection invalidation after media deletion", () => {
  it.each([
    pinFromImage(image),
    pinFromVideo({ ...image, image_id: 5000, video_id: 200, mime_type: "video/mp4" }),
  ])("refreshes collection counts/pagination after deleting $key", async (pin) => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(null, { status: 204 })));
    pins.set([pin]);
    lightboxPin.set(pin);
    const epoch = get(collectionEpoch);
    expect(await createMediaActions().remove(pin)).toBe(true);
    expect(get(collectionEpoch)).toBe(epoch + 1);
    expect(get(pins)).toEqual([]);
    expect(get(lightboxPin)).toBeNull();
  });

  it("does not invalidate or remove a pin when media deletion fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ error: "delete failed" }), { status: 503 })));
    const pin = pinFromImage(image);
    pins.set([pin]);
    const epoch = get(collectionEpoch);
    expect(await createMediaActions().remove(pin)).toBe(false);
    expect(get(collectionEpoch)).toBe(epoch);
    expect(get(pins)).toEqual([pin]);
  });
});
