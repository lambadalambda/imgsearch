import { describe, expect, it } from "vitest";
import {
  FEED_PREFERENCE_MAX_TAGS,
  FEED_TAG_SCORE_MAX,
  FEED_TAG_SCORE_MIN,
  applyFeedFeedback,
  classifyFeedFeedback,
  decayFeedTagScores,
  feedPreferenceTags,
  type FeedMetrics,
} from "./feed";

const base: FeedMetrics = { playbackStarted: true, action: "next", watchMs: 10_000, dwellMs: 12_000, completionRatio: 0.5 };

describe("classifyFeedFeedback", () => {
  it("ignores items whose playback never started", () => {
    expect(classifyFeedFeedback({ ...base, playbackStarted: false, completionRatio: 1 }).klass).toBe("neutral");
  });
  it("treats a watch-through or natural end as positive", () => {
    expect(classifyFeedFeedback({ ...base, completionRatio: 0.75 })).toEqual({ klass: "positive", reason: "watch-through" });
    expect(classifyFeedFeedback({ ...base, action: "ended", completionRatio: 0.1 }).klass).toBe("positive");
  });
  it("treats a quick skip as a soft negative only when every signal is small", () => {
    expect(classifyFeedFeedback({ ...base, watchMs: 500, dwellMs: 900, completionRatio: 0.05 }).klass).toBe("soft-negative");
    expect(classifyFeedFeedback({ ...base, watchMs: 500, dwellMs: 5000, completionRatio: 0.05 }).klass).toBe("neutral");
    expect(classifyFeedFeedback({ ...base, action: "prev", watchMs: 500, dwellMs: 900, completionRatio: 0.05 }).klass).toBe("neutral");
  });
});

describe("applyFeedFeedback", () => {
  it("adds +1 for positive and -0.5 for soft-negative, case-insensitively, within the clamp", () => {
    const scores = new Map<string, number>();
    applyFeedFeedback(scores, ["Cat", "indoor"], "positive");
    applyFeedFeedback(scores, ["cat"], "soft-negative");
    expect(scores.get("cat")).toBeCloseTo(0.5);
    expect(scores.get("indoor")).toBe(1);
    for (let i = 0; i < 20; i += 1) applyFeedFeedback(scores, ["cat"], "positive");
    expect(scores.get("cat")).toBe(FEED_TAG_SCORE_MAX);
    for (let i = 0; i < 40; i += 1) applyFeedFeedback(scores, ["cat"], "soft-negative");
    expect(scores.get("cat")).toBe(FEED_TAG_SCORE_MIN);
  });
  it("leaves the map untouched for neutral feedback", () => {
    const scores = new Map([["cat", 1]]);
    applyFeedFeedback(scores, ["cat", "dog"], "neutral");
    expect([...scores.entries()]).toEqual([["cat", 1]]);
  });
});

describe("decayFeedTagScores", () => {
  it("multiplies scores and drops the ones that fall under the threshold", () => {
    const scores = new Map([["cat", 1], ["dog", 0.05], ["bird", -0.06]]);
    decayFeedTagScores(scores, 0.5, 0.05);
    expect(scores.get("cat")).toBe(0.5);
    expect(scores.has("dog")).toBe(false);
    expect(scores.has("bird")).toBe(false);
  });
});

describe("feedPreferenceTags", () => {
  it("splits by sign at the threshold, sorts by magnitude then name, and caps the count", () => {
    const scores = new Map<string, number>([["b", 2], ["a", 2], ["c", 0.2], ["d", -1], ["e", -0.3]]);
    expect(feedPreferenceTags(scores)).toEqual({ prefer: ["a", "b"], avoid: ["d", "e"] });
    const many = new Map<string, number>();
    for (let i = 0; i < FEED_PREFERENCE_MAX_TAGS + 5; i += 1) many.set(`t${String(i).padStart(2, "0")}`, 1);
    expect(feedPreferenceTags(many).prefer).toHaveLength(FEED_PREFERENCE_MAX_TAGS);
  });
});
