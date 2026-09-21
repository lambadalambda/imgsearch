import type { Pin } from "./types";

export type LibrarySortOrder = "random" | "newest" | "captured";

const RANDOM_SEED_MAX = 0x80000000;

/** A fresh 31-bit seed for the seeded random library order. */
export function newLibrarySeed(): number {
  if (globalThis.crypto?.getRandomValues) {
    const values = new Uint32Array(1);
    globalThis.crypto.getRandomValues(values);
    return values[0] & 0x7fffffff;
  }
  return Math.floor(Math.random() * RANDOM_SEED_MAX);
}

/** FNV-1a style hash of a key mixed with the session seed. */
export function randomKey(key: string, seed: number): number {
  let hash = seed >>> 0;
  for (let i = 0; i < key.length; i += 1) {
    hash = Math.imul(hash ^ key.charCodeAt(i), 16777619) >>> 0;
  }
  return hash;
}

function parseTime(value: string | undefined): number {
  if (!value) return 0;
  // SQLite "YYYY-MM-DD HH:MM:SS" has no zone; read it as UTC like created_at.
  return Date.parse(value.includes("T") ? value : value.replace(" ", "T") + "Z") || 0;
}

/** Newest first by created_at, then by key for a stable tie-break. */
export function compareRecentPins(left: Pin, right: Pin): number {
  return parseTime(right.createdAt) - parseTime(left.createdAt) || right.key.localeCompare(left.key);
}

/** Latest capture first, falling back to the upload time, then by key. */
export function compareCapturedPins(left: Pin, right: Pin): number {
  const l = parseTime(left.capturedAt) || parseTime(left.createdAt);
  const r = parseTime(right.capturedAt) || parseTime(right.createdAt);
  return r - l || right.key.localeCompare(left.key);
}

/**
 * Merge the image and video pages into one library order. "newest" sorts by
 * creation time; "random" interleaves the two lists, with the seed deciding
 * which media type leads so the mix stays stable across Load More.
 */
export function combineLibraryPins(imagePins: Pin[], videoPins: Pin[], sort: LibrarySortOrder, seed: number): Pin[] {
  if (sort === "newest") {
    return [...imagePins, ...videoPins].sort(compareRecentPins);
  }
  if (sort === "captured") {
    return [...imagePins, ...videoPins].sort(compareCapturedPins);
  }

  const first = randomKey("media:first", seed) % 2 === 0 ? videoPins : imagePins;
  const second = first === videoPins ? imagePins : videoPins;
  const out: Pin[] = [];
  const max = Math.max(first.length, second.length);
  for (let i = 0; i < max; i += 1) {
    if (first[i]) out.push(first[i]);
    if (second[i]) out.push(second[i]);
  }
  return out;
}
