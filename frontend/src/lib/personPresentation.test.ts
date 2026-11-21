import { describe, expect, it } from "vitest";
import { personFacesLabel, personItemsPresentation } from "./personPresentation";

describe("person presentation", () => {
  it("labels face counts", () => {    expect(personFacesLabel(0)).toBe("0 faces");
    expect(personFacesLabel(1)).toBe("1 face");
    expect(personFacesLabel(12)).toBe("12 faces");
  });
  it("guides empty and NSFW-filtered person views like collections", () => {
    expect(personItemsPresentation(0, 0)).toEqual({
      count: "0 grouped items",
      message: "No grouped photos or videos yet.",
      action: "browse",
    });
    expect(personItemsPresentation(3, 3)).toEqual({ count: "3 grouped items", message: "", action: null });
    expect(personItemsPresentation(3, 0)).toEqual({
      count: "3 grouped items · 3 hidden",
      message: "All grouped items are hidden by the NSFW filter.",
      action: "show",
    });
  });
});

describe("drag-and-drop merge", () => {
  it("only allows merging two different people", async () => {
    const { canMergePeople } = await import("./personPresentation");
    expect(canMergePeople(1, 2)).toBe(true);
    expect(canMergePeople(2, 2)).toBe(false);
    expect(canMergePeople(0, 2)).toBe(false);
    expect(canMergePeople(1, -3)).toBe(false);
  });
});
