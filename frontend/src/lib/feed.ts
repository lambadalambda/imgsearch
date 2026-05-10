/**
 * Session-local feedback model for the similar-video Feed.
 *
 * Mirrors the legacy implementation in `internal/webui/static/app.js`
 * (classifyFeedFeedback / recordFeedFeedback / decayFeedTagScores /
 * feedPreferenceTags) so the Atelier port produces the same prefer/avoid
 * tag list for `/api/search/similar-videos`. Vector feedback separately sends
 * recent frame IDs for session-local query adaptation; nothing is persisted.
 */

/** Per-item playback metrics gathered while the item was visible. */
export interface FeedMetrics {
  /** Did playback ever start? false → all signals are ignored. */
  playbackStarted: boolean;
  /** Why we're recording: explicit user navigation, or the video ended. */
  action: "next" | "prev" | "ended" | "close" | "manual";
  /** Cumulative milliseconds the video was actually playing (timeupdate-derived). */
  watchMs: number;
  /** Wall-clock milliseconds the item was on screen. */
  dwellMs: number;
  /** currentTime / duration at the moment we leave (clamped 0..1). */
  completionRatio: number;
}

export type FeedClass = "positive" | "soft-negative" | "neutral";

export interface FeedClassification {
  klass: FeedClass;
  reason: string;
}

/** Tag-preference scoring constants (ported verbatim from legacy). */
export const FEED_TAG_SCORE_MIN = -3;
export const FEED_TAG_SCORE_MAX = 5;
export const FEED_TAG_SCORE_DECAY = 0.9;
export const FEED_TAG_SCORE_DROP = 0.05;
export const FEED_PREFERENCE_THRESHOLD = 0.25;
export const FEED_PREFERENCE_MAX_TAGS = 12;
export const FEED_VECTOR_FEEDBACK_MAX_IDS = 8;

/** Lazy-batching constants. */
export const FEED_INITIAL_LIMIT = 4;
export const FEED_BATCH_SIZE = 3;
/** When fewer than this many items remain ahead of the current item, fetch
 *  the next batch in the background. */
export const FEED_FETCH_AHEAD_THRESHOLD = 2;

/**
 * Classify a single item's interaction trace.
 *
 * Order matters — earlier rules win. Identical to the legacy ladder.
 */
export function classifyFeedFeedback(metrics: FeedMetrics): FeedClassification {
  if (!metrics.playbackStarted) {
    return { klass: "neutral", reason: "playback-not-started" };
  }
  if (metrics.completionRatio >= 0.75 || metrics.action === "ended") {
    return { klass: "positive", reason: "watch-through" };
  }
  if (
    metrics.action === "next" &&
    metrics.watchMs < 3000 &&
    metrics.dwellMs < 3000 &&
    metrics.completionRatio < 0.2
  ) {
    return { klass: "soft-negative", reason: "quick-skip" };
  }
  return { klass: "neutral", reason: "ambiguous" };
}

/** Mutates `scores` to reflect the contribution of `tags` from a single
 *  classified interaction. Returns the changed map for fluent style. */
export function applyFeedFeedback(
  scores: Map<string, number>,
  tags: readonly string[],
  klass: FeedClass,
): Map<string, number> {
  const weight = klass === "positive" ? 1 : klass === "soft-negative" ? -0.5 : 0;
  if (weight === 0) return scores;
  for (const tag of tags) {
    const key = tag.toLowerCase();
    const next = clamp(
      (scores.get(key) ?? 0) + weight,
      FEED_TAG_SCORE_MIN,
      FEED_TAG_SCORE_MAX,
    );
    scores.set(key, next);
  }
  return scores;
}

/** Multiplicative decay once per fetch. Drops near-zero tags so the map
 *  doesn't grow unboundedly across long sessions. */
export function decayFeedTagScores(
  scores: Map<string, number>,
  factor = FEED_TAG_SCORE_DECAY,
  drop = FEED_TAG_SCORE_DROP,
): Map<string, number> {
  for (const [tag, score] of scores) {
    const next = score * factor;
    if (Math.abs(next) < drop) {
      scores.delete(tag);
    } else {
      scores.set(tag, next);
    }
  }
  return scores;
}

export interface PreferenceTags {
  prefer: string[];
  avoid: string[];
}

/** Derive the prefer / avoid CSV inputs for /api/search/similar-videos.
 *
 *  Uses |score| ≥ FEED_PREFERENCE_THRESHOLD as the cutoff and keeps the top
 *  FEED_PREFERENCE_MAX_TAGS by magnitude (alphabetic tiebreak). */
export function feedPreferenceTags(scores: Map<string, number>): PreferenceTags {
  const prefer: Array<[string, number]> = [];
  const avoid: Array<[string, number]> = [];
  for (const [tag, score] of scores) {
    if (score >= FEED_PREFERENCE_THRESHOLD) prefer.push([tag, score]);
    else if (score <= -FEED_PREFERENCE_THRESHOLD) avoid.push([tag, score]);
  }
  const cmp = (a: [string, number], b: [string, number]) =>
    Math.abs(b[1]) - Math.abs(a[1]) || a[0].localeCompare(b[0]);
  prefer.sort(cmp);
  avoid.sort(cmp);
  return {
    prefer: prefer.slice(0, FEED_PREFERENCE_MAX_TAGS).map(([t]) => t),
    avoid: avoid.slice(0, FEED_PREFERENCE_MAX_TAGS).map(([t]) => t),
  };
}

function clamp(value: number, min: number, max: number): number {
  if (value < min) return min;
  if (value > max) return max;
  return value;
}
