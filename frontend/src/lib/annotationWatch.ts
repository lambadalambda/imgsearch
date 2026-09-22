import { get } from "svelte/store";
import { fetchMediaStatus, getImage, getVideo } from "./api";
import { lightboxPin, pins } from "./stores";
import type { AnnotationState, Pin } from "./types";
import { deriveFullDescription, deriveSummary, deriveTitle } from "./utils";

/** Default poll interval; tests can shorten it through window.__imgsearchAnnotationPollMs. */
export const ANNOTATION_POLL_MS = 3000;
/** The status endpoint accepts at most 500 ids; keep a margin. */
const MAX_IDS = 400;

function pollInterval(): number {
  const override = (globalThis as { __imgsearchAnnotationPollMs?: number }).__imgsearchAnnotationPollMs;
  return typeof override === "number" && override > 0 ? override : ANNOTATION_POLL_MS;
}

function patchPin(key: string, update: (pin: Pin) => Pin): void {
  pins.update((list) => list.map((p) => (p.key === key ? update(p) : p)));
  lightboxPin.update((p) => (p && p.key === key ? update(p) : p));
}

/**
 * One poll: ask the server about the pins on screen, update their
 * annotation state badges, and reload the text of any pin whose annotation
 * timestamp moved. Exported so tests can drive it directly.
 */
export async function pollAnnotationStatus(signal?: AbortSignal): Promise<void> {
  const current = get(pins);
  if (current.length === 0) return;
  const images: number[] = [];
  const videos: number[] = [];
  for (const pin of current) {
    if (images.length + videos.length >= MAX_IDS) break;
    if (pin.mediaType === "video" && pin.videoId !== undefined) videos.push(pin.videoId);
    else images.push(pin.imageId);
  }
  const status = await fetchMediaStatus({ images, videos, signal });
  const byKey = new Map<string, { state: AnnotationState; updatedAt: string; indexState?: string }>();
  for (const s of status.images) byKey.set(`image:${s.image_id}`, { state: s.annotation_state, updatedAt: s.annotation_updated_at, indexState: s.index_state });
  for (const s of status.videos) byKey.set(`video:${s.video_id}`, { state: s.annotation_state, updatedAt: s.annotation_updated_at });

  const refresh: Pin[] = [];
  for (const pin of current) {
    const next = byKey.get(pin.key);
    if (!next) continue;
    const textChanged = next.updatedAt !== "" && next.updatedAt !== (pin.annotationUpdatedAt ?? "");
    if (next.state !== pin.annotationState || textChanged) {
      patchPin(pin.key, (p) => ({ ...p, annotationState: next.state, annotationUpdatedAt: textChanged ? p.annotationUpdatedAt : next.updatedAt }));
    }
    if (textChanged) refresh.push(pin);
  }

  await Promise.all(
    refresh.map(async (pin) => {
      try {
        const record = pin.mediaType === "video" && pin.videoId !== undefined ? await getVideo(pin.videoId, signal) : await getImage(pin.imageId, signal);
        const tags = record.tags ?? [];
        patchPin(pin.key, (p) => ({
          ...p,
          title: deriveTitle(record),
          summary: deriveSummary(record),
          fullDescription: deriveFullDescription(record),
          tags,
          isNSFW: tags.some((t) => t.toLowerCase() === "nsfw"),
          annotationState: record.annotation_state ?? p.annotationState,
          annotationUpdatedAt: record.annotation_updated_at ?? p.annotationUpdatedAt,
        }));
      } catch {
        /* the next poll retries */
      }
    }),
  );
}

/** Start polling while the tab is visible; returns a stop function. */
export function startAnnotationWatch(): () => void {
  let stopped = false;
  let inFlight = false;
  let controller: AbortController | null = null;
  const tick = async () => {
    if (stopped || inFlight || document.visibilityState !== "visible") return;
    inFlight = true;
    controller = new AbortController();
    try {
      await pollAnnotationStatus(controller.signal);
    } catch {
      /* transient network errors: try again next tick */
    } finally {
      inFlight = false;
    }
  };
  // A timeout chain rather than setInterval, so the interval override is
  // re-read every tick and a slow poll never overlaps the next one.
  let timer = 0;
  const schedule = () => {
    if (stopped) return;
    timer = window.setTimeout(async () => {
      await tick();
      schedule();
    }, pollInterval());
  };
  schedule();
  const onVisible = () => {
    if (document.visibilityState === "visible") void tick();
  };
  document.addEventListener("visibilitychange", onVisible);
  return () => {
    stopped = true;
    window.clearTimeout(timer);
    document.removeEventListener("visibilitychange", onVisible);
    controller?.abort();
  };
}
