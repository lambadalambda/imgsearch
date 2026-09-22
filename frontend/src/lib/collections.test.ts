import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "./api";
import { actionTarget } from "./mediaActions";
import { parseSearch, searchFor } from "./urlState";

afterEach(() => vi.unstubAllGlobals());

describe("collection URLs", () => {
  it("restores management and detail on reload", () => {
    expect(parseSearch("?view=collections")).toEqual({ mode: "collections" });
    expect(parseSearch("?view=collection&collection_id=123")).toEqual({ mode: "collection", collectionId: 123 });
    expect(searchFor({ mode: "collection", collectionId: 123 })).toBe("?view=collection&collection_id=123");
    expect(searchFor({ mode: "collections" })).toBe("?view=collections");
  });
  it.each(["", "0", "-1", "abc", "1.5", "9007199254740992"])("rejects invalid collection ID %s", (id) => {
    expect(parseSearch(`?view=collection&collection_id=${id}`)).toEqual({ mode: "collections" });
  });
});

describe("collection API", () => {
  it("lists all or only memberships and paginates mixed media", async () => {
    const fetch = vi.fn().mockImplementation(() => Promise.resolve(new Response(JSON.stringify({ collections: [] }))));
    vi.stubGlobal("fetch", fetch);
    const signal = new AbortController().signal;
    await api.listCollections({ signal });
    expect(fetch).toHaveBeenLastCalledWith("/api/collections", expect.objectContaining({ signal }));
    await api.listCollections({ target: actionTarget({ mediaType: "video", videoId: 200, imageId: 5000 }), signal });
    expect(fetch).toHaveBeenLastCalledWith("/api/collections?media_type=video&media_id=200", expect.objectContaining({ signal }));
    fetch.mockResolvedValue(new Response(JSON.stringify({ items: [], total: 0 })));
    await api.listCollectionItems(123, { offset: 50, includeNSFW: false, signal });
    expect(fetch).toHaveBeenLastCalledWith("/api/collections/123/items?limit=50&offset=50&include_nsfw=false", expect.objectContaining({ signal }));
  });
  it("creates, reads, renames, deletes and handles bodyless membership responses", async () => {
    const collection = { id: 123, name: "Cats", created_at: "2026-01-01" };
    const fetch = vi.fn().mockImplementation(() => Promise.resolve(new Response(JSON.stringify(collection))));
    vi.stubGlobal("fetch", fetch);
    expect(await api.createCollection("Cats")).toEqual(collection);
    expect(fetch).toHaveBeenLastCalledWith("/api/collections", expect.objectContaining({ method: "POST", body: JSON.stringify({ name: "Cats" }) }));
    expect(await api.getCollection(123)).toEqual(collection);
    await api.renameCollection(123, "Kittens");
    expect(fetch).toHaveBeenLastCalledWith("/api/collections/123", expect.objectContaining({ method: "PATCH", body: JSON.stringify({ name: "Kittens" }) }));
    fetch.mockImplementation(() => Promise.resolve(new Response(null, { status: 204 })));
    await api.setCollectionMembership(123, { kind: "video", id: 200 }, true);
    expect(fetch).toHaveBeenLastCalledWith("/api/collections/123/items/video/200", expect.objectContaining({ method: "PUT" }));
    await api.setCollectionMembership(123, { kind: "image", id: 1000 }, false);
    expect(fetch).toHaveBeenLastCalledWith("/api/collections/123/items/image/1000", expect.objectContaining({ method: "DELETE" }));
    await expect(api.deleteCollection(123)).resolves.toBeUndefined();
  });
  it("surfaces backend JSON errors without treating failures as saved", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ error: "derived frames cannot be saved" }), { status: 409 })));
    await expect(api.setCollectionMembership(123, { kind: "image", id: 5000 }, true)).rejects.toThrow("derived frames cannot be saved");
  });
});
