import { writable, type Readable } from "svelte/store";
import { ApiError, deleteMedia, reannotate as reannotateApi, toggleNSFW, updateMedia, type MediaMetadataPatch } from "./api";
import { lightboxPin, pins } from "./stores";
import type { Pin } from "./types";
import { deriveTitle } from "./utils";

/** The API target for a pin: videos act on their video id, images on the image id. */
export function actionTarget(pin: Pick<Pin, "mediaType" | "imageId" | "videoId">): {
  kind: "image" | "video";
  id: number;
} {
  const kind = pin.mediaType;
  const id = kind === "video" && pin.videoId !== undefined ? pin.videoId : pin.imageId;
  return { kind, id };
}

/** Write a pin's NSFW flag to every place it is rendered from. */
export function setPinNSFW(key: string, value: boolean): void {
  pins.update((list) => list.map((p) => (p.key === key ? { ...p, isNSFW: value } : p)));
  lightboxPin.update((p) => (p && p.key === key ? { ...p, isNSFW: value } : p));
}

/** Apply an edited title and tag list everywhere the pin is rendered from. */
export function setPinMetadata(key: string, title: string, tags: string[]): void {
  const apply = (p: Pin): Pin => ({ ...p, title, tags, isNSFW: tags.some((t) => t.toLowerCase() === "nsfw") });
  pins.update((list) => list.map((p) => (p.key === key ? apply(p) : p)));
  lightboxPin.update((p) => (p && p.key === key ? apply(p) : p));
}

/** Drop a pin from the results and close the lightbox if it shows that pin. */
export function removePin(key: string): void {
  pins.update((list) => list.filter((p) => p.key !== key));
  lightboxPin.update((p) => (p && p.key === key ? null : p));
}

export interface MediaActions {
  /** True while one of the actions is in flight; actions ignore calls meanwhile. */
  pending: Readable<boolean>;
  /** Message of the last failed action, cleared when the next one starts. */
  error: Readable<string | null>;
  flagNSFW(pin: Pin): Promise<boolean>;
  reannotate(pin: Pin): Promise<boolean>;
  remove(pin: Pin): Promise<boolean>;
  /** Save a manual title and/or full tag list; the stores update from the server's response. */
  edit(pin: Pin, patch: MediaMetadataPatch): Promise<boolean>;
}

/**
 * Per-view media actions (flag NSFW, re-annotate, delete) with optimistic
 * updates written to the shared stores, so a card and the lightbox always
 * agree. Each component creates its own instance to keep pending/error
 * feedback local to that view.
 */
export function createMediaActions(): MediaActions {
  const pending = writable(false);
  const error = writable<string | null>(null);
  let inFlight = false;

  async function run(label: string, fn: () => Promise<unknown>): Promise<boolean> {
    if (inFlight) return false;
    inFlight = true;
    pending.set(true);
    error.set(null);
    try {
      await fn();
      return true;
    } catch (err) {
      error.set(err instanceof ApiError || err instanceof Error ? err.message : `${label} failed`);
      return false;
    } finally {
      inFlight = false;
      pending.set(false);
    }
  }

  return {
    pending,
    error,
    async flagNSFW(pin) {
      const { kind, id } = actionTarget(pin);
      const next = !(pin.isNSFW ?? false);
      setPinNSFW(pin.key, next); // optimistic
      const ok = await run(next ? "flag" : "unflag", () => toggleNSFW(kind, id));
      if (!ok) setPinNSFW(pin.key, !next);
      return ok;
    },
    async reannotate(pin) {
      const { kind, id } = actionTarget(pin);
      return run("re-annotate", () => reannotateApi(kind, id));
    },
    async remove(pin) {
      const { kind, id } = actionTarget(pin);
      const ok = await run("delete", () => deleteMedia(kind, id));
      if (ok) removePin(pin.key);
      return ok;
    },
    async edit(pin, patch) {
      const { kind, id } = actionTarget(pin);
      return run("save", async () => {
        const record = await updateMedia(kind, id, patch);
        setPinMetadata(pin.key, deriveTitle(record), record.tags ?? []);
      });
    },
  };
}
