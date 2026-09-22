import { describe, expect, it } from "vitest";
import { collectionPresentation } from "./collectionPresentation";

describe("collection counts and guidance", () => {
  it("distinguishes truly empty from fully filtered", () => {
    expect(collectionPresentation(0, 0)).toEqual({ count: "0 saved items", message: "No saved items yet.", action: "browse" });
    expect(collectionPresentation(3, 0)).toEqual({ count: "3 saved items · 3 hidden", message: "All saved items are hidden by the NSFW filter.", action: "show" });
  });
  it("reports partially hidden counts without confusing visible total with saved total", () => {
    expect(collectionPresentation(4, 2)).toEqual({ count: "4 saved items · 2 hidden", message: "2 items hidden by the NSFW filter.", action: "show" });
    expect(collectionPresentation(2, 1).message).toBe("1 item hidden by the NSFW filter.");
  });
  it("keeps singular and unfiltered populated copy succinct", () => {
    expect(collectionPresentation(1, 1)).toEqual({ count: "1 saved item", message: "", action: null });
    expect(collectionPresentation(5, 5)).toEqual({ count: "5 saved items", message: "", action: null });
  });
});
