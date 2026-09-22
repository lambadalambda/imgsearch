import { afterEach, expect, it, vi } from "vitest";
import { consolidateDuplicates } from "./api";
import { duplicateFileInfo, resolutionLabel } from "./duplicates";

const item = { image_id: 1, original_name: "one.jpg", storage_path: "images/a", mime_type: "image/jpeg", width: 450, height: 424, index_state: "done" };
afterEach(() => vi.unstubAllGlobals());
it("describes resolution ties without asserting quality or byte equality", () => {
  const other = { ...item, image_id: 2 };
  expect(resolutionLabel(item, [item, other])).toBe("Same resolution");
  expect(resolutionLabel(item, [item, { ...other, width: 100 }])).toBe("Most pixels (not necessarily best quality)");
  expect(duplicateFileInfo({ ...item, file_size_bytes: 2048 })).toBe("JPEG · 2 KiB");
  expect(duplicateFileInfo(item)).toBe("JPEG · Size unavailable");
});
it("distinguishes identical dimensions from equal pixel counts", () => {
  const standard = { ...item, width: 800, height: 600 };
  const wide = { ...item, image_id: 2, width: 1200, height: 400 };
  const identical = { ...standard, image_id: 3 };
  expect(resolutionLabel(standard, [standard, wide])).toBe("Same pixel count");
  expect(resolutionLabel(wide, [standard, wide])).toBe("Same pixel count");
  expect(resolutionLabel(standard, [standard, wide, identical])).toBe("Same resolution");
  expect(resolutionLabel(wide, [standard, wide, identical])).toBe("Same pixel count");
});
it("posts the full group and user-selected keeper with the reviewed token", async () => {
  const response = { keeper_id: 2, removed_ids: [1], affected_collections: 3, file_policy: "retained", conflicts: [], preview_token: "fresh" };
  const fetch = vi.fn().mockImplementation(async () => new Response(JSON.stringify(response)));
  vi.stubGlobal("fetch", fetch);
  const request = { keeper_id: 2, group_ids: [1,2], distance: 0, include_nsfw: false };
  expect(await consolidateDuplicates(request)).toEqual(response);
  const confirm = { ...request, confirm: true, preview_token: "fresh" };
  await consolidateDuplicates(confirm);
  expect(fetch.mock.calls.map(([url, init]) => [url, init.method, JSON.parse(init.body)])).toEqual([
    ["/api/duplicates", "POST", request], ["/api/duplicates", "POST", confirm],
  ]);
});
it("surfaces stale confirmation errors instead of reporting success", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ error: "preview changed" }), { status: 409 })));
  await expect(consolidateDuplicates({ keeper_id: 2, group_ids: [1,2], distance: 0, include_nsfw: true, confirm: true, preview_token: "old" })).rejects.toThrow("preview changed");
});
