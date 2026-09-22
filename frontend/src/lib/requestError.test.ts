import { describe, expect, it } from "vitest";
import { ApiError } from "./api";
import { requestErrorMessage } from "./requestError";

describe("request error presentation", () => {
  it("uses plain language for injected fetch and HTTP failures", () => {
    expect(requestErrorMessage(new TypeError("Failed to fetch"))).toBe("Couldn’t connect to the library. Check your connection, then retry.");
    expect(requestErrorMessage(new ApiError(503, "raw backend error"))).toBe("The library couldn’t complete this request. Try again in a moment.");
    expect(requestErrorMessage(new ApiError(400, "bad request"))).toContain("Check your search");
  });
});
