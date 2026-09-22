import { describe, expect, it } from "vitest";
import { healthCards, capabilityCopy } from "./health";
import type { StatsResponse } from "./types";

const snapshot: StatsResponse = {
  images_total: 15, standalone_images_total: 10, video_frame_images_total: 5, videos_total: 2,
  image_annotation_expected: 10, video_annotation_expected: 2, video_transcription_expected: 2,
  job_kinds: { embed_image: { tracked: 12, pending: 3, leased: 2, waiting: 4, processing: 1, runnable: 4, done: 6, failed: 1, oldest_runnable_age_seconds: 5 } },
};
describe("library health", () => {
  it("uses distinct denominators and non-overlapping human states", () => {
    const cards = healthCards(snapshot);
    expect(cards.map(c => c.total)).toEqual([15, 10, 2, 2]);
    expect(cards[0]).toMatchObject({ waiting: 4, processing: 1, complete: 6, failed: 1, missing: 3 });
    expect(cards[0].scope).toContain("sampled video frames");
    expect(cards[1].scope).toContain("Standalone images only");
  });
  it("never infers disabled from empty job counts", () => {
    expect(healthCards(snapshot)[3].capability).toBe("unknown");
    expect(healthCards(snapshot)[3].canRetry).toBe(false); // Older APIs may ignore the kind filter.
    for (const state of ["disabled", "unconfigured", "unknown", "configured"] as const) {
      const card = healthCards({ ...snapshot, capabilities: { worker: "local", transcription: state, annotations: state } })[3];
      expect(card.capability).toBe(state);
      expect(card.canRetry).toBe(state === "configured" || state === "unknown");
    }
    expect(capabilityCopy("unconfigured")).toBe("Not configured");
  });
  it("reports zero processing for absent current-server kinds, not legacy lease counts", () => {
    const cards = healthCards({ ...snapshot, capabilities: { worker: "local", annotations: "configured", transcription: "disabled" } });
    expect(cards[3]).toMatchObject({ waiting: 0, processing: 0, complete: 0, failed: 0 });
    const legacy = { ...snapshot, job_kinds: { embed_image: { ...snapshot.job_kinds!.embed_image, waiting: undefined, processing: undefined } } };
    expect(healthCards(legacy)[0].processing).toBeUndefined();
  });
  it("describes empty transcription without asserting no audio", () => {
    const card = healthCards({ ...snapshot, video_transcription_empty: 2 })[3];
    expect(card.note).toContain("2 completed without recognized speech");
    expect(card.note).toContain("cannot distinguish");
    expect(card.note).not.toContain("No audio");
  });
});
