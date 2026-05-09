<script lang="ts">
  import { closeFeed, feedSeed, includeNSFW } from "../lib/stores";
  import {
    ApiError,
    searchSimilarVideos,
    type SimilarVideoOptions,
  } from "../lib/api";
  import { pinFromSearchResult } from "../lib/utils";
  import type { Pin } from "../lib/types";
  import {
    FEED_BATCH_SIZE,
    FEED_FETCH_AHEAD_THRESHOLD,
    FEED_INITIAL_LIMIT,
    applyFeedFeedback,
    classifyFeedFeedback,
    decayFeedTagScores,
    feedPreferenceTags,
    type FeedMetrics,
  } from "../lib/feed";
  import Icon from "./Icon.svelte";

  /**
   * Similar-video Feed overlay.
   *
   * Behaviour mirrors the legacy implementation in
   * `internal/webui/static/app.js` (see docs/issues/add-mobile-adaptive-
   * similar-video-feed-mode.md and add-dynamic-adaptive-batching-...md):
   *
   *   - Three persistent <video> elements; CSS slot positions (-100%, 0,
   *     +100% translateY) place them as prev / current / next. We never
   *     re-parent DOM nodes — instead we rotate which queue index is
   *     "assigned" to each element, which preserves the playing buffer
   *     for whichever element is becoming the new current.
   *   - Initial fetch: seed + FEED_INITIAL_LIMIT candidates.
   *   - Refill: when fewer than FEED_FETCH_AHEAD_THRESHOLD items remain
   *     ahead of current, kick a single in-flight lookahead of
   *     FEED_BATCH_SIZE items, seeded off queue[0] (not current — the
   *     legacy uses the original seed so the session has a stable centre).
   *   - Per-tag preference scores decay 0.9 each fetch and clamp to the
   *     legacy bounds; prefer/avoid CSV is sent on every batch.
   *   - All state is wiped on close. Feedback never leaves the tab.
   */

  // Tunables ----------------------------------------------------------------
  const SWIPE_DISTANCE_THRESHOLD_PX = 72;
  const SWIPE_DISTANCE_THRESHOLD_VH = 0.16;
  const SWIPE_VELOCITY_THRESHOLD = 0.65; // px/ms
  const TAP_DEAD_ZONE_PX = 8;

  // Reactive state ----------------------------------------------------------
  let queue = $state<Pin[]>([]);
  let currentIndex = $state(0);
  let muted = $state(true);
  let exhausted = $state(false);
  let loading = $state(false); // initial open
  let loadingMore = $state(false);
  let progress = $state(0); // 0..1 for the active video
  let dragOffsetPx = $state(0); // mid-drag visual offset on track

  // The three persistent video element refs.
  const videoEls: Array<HTMLVideoElement | undefined> = [
    undefined,
    undefined,
    undefined,
  ];
  /** assigned[slotIdx] = queueIndex shown by videoEls[slotIdx]. -1 = unused. */
  let assigned = $state<[number, number, number]>([0, 1, -1]);

  // Per-session bookkeeping -------------------------------------------------
  const seenVideoIDs = new Set<number>();
  const rejectedVideoIDs = new Set<number>(); // unplayable / failed candidates
  const tagScores = new Map<string, number>();

  let candidateRequestToken = 0;
  let lookaheadPromise: Promise<void> | null = null;

  // Per-item playback metrics, recorded once per advance.
  let feedbackRecordedIndex = -1;
  let itemStartedAt = 0;
  let watchStartedAt = 0;
  let accumulatedWatchMs = 0;
  let playbackStarted = false;

  // Touch tracking ----------------------------------------------------------
  let trackEl: HTMLDivElement | undefined = $state();
  let touchStartY = 0;
  let touchStartX = 0;
  let touchStartTime = 0;
  let dragging = $state(false);
  let suppressClickUntil = 0;

  // Derivations -------------------------------------------------------------
  const currentItem = $derived(queue[currentIndex]);
  const remainingAhead = $derived(queue.length - currentIndex - 1);
  const seedItem = $derived(queue[0]);
  const isOpen = $derived($feedSeed !== null);

  // ------------------------------------------------------------------------
  // Open / close lifecycle.
  // ------------------------------------------------------------------------

  $effect(() => {
    const seed = $feedSeed;
    if (!seed) return;
    if (queue.length === 0) {
      void start(seed);
    }
  });

  $effect(() => {
    if (isOpen) {
      document.body.classList.add("modal-open");
    } else {
      document.body.classList.remove("modal-open");
    }
    return () => document.body.classList.remove("modal-open");
  });

  async function start(seed: Pin): Promise<void> {
    candidateRequestToken += 1;
    queue = [seed];
    currentIndex = 0;
    assigned = [0, -1, -1];
    seenVideoIDs.clear();
    rejectedVideoIDs.clear();
    tagScores.clear();
    if (seed.videoId) seenVideoIDs.add(seed.videoId);
    feedbackRecordedIndex = -1;
    progress = 0;
    exhausted = false;
    loading = true;
    startItemTimers(0);
    try {
      await fetchNextBatch(FEED_INITIAL_LIMIT);
    } finally {
      loading = false;
    }
    syncAssignments();
    void playCurrent();
  }

  function close(): void {
    candidateRequestToken += 1;
    lookaheadPromise = null;
    closeFeed();
    // Wipe local state so reopen starts fresh.
    queue = [];
    currentIndex = 0;
    assigned = [0, -1, -1];
    seenVideoIDs.clear();
    rejectedVideoIDs.clear();
    tagScores.clear();
    feedbackRecordedIndex = -1;
    accumulatedWatchMs = 0;
    progress = 0;
    exhausted = false;
    loading = false;
    loadingMore = false;
    dragOffsetPx = 0;
    pauseAll();
  }

  // ------------------------------------------------------------------------
  // Slot assignment. We map queue[currentIndex-1 / currentIndex / +1] onto
  // the three video element slots. Whichever element was at "next" before
  // an advance keeps its src and continues playing as it rotates into the
  // "current" CSS slot — that's the preservation-of-playback trick.
  // ------------------------------------------------------------------------

  function syncAssignments(): void {
    const want: number[] = [];
    if (currentIndex - 1 >= 0) want.push(currentIndex - 1);
    want.push(currentIndex);
    if (currentIndex + 1 < queue.length) want.push(currentIndex + 1);
    const next: [number, number, number] = [...assigned];
    // Slots already showing a wanted index keep it.
    const usedSlots = new Set<number>();
    const satisfied = new Set<number>();
    for (let s = 0; s < 3; s += 1) {
      if (want.includes(next[s]) && !satisfied.has(next[s])) {
        usedSlots.add(s);
        satisfied.add(next[s]);
      } else {
        next[s] = -1;
      }
    }
    // Any wanted index that isn't satisfied gets the first free slot.
    for (const idx of want) {
      if (satisfied.has(idx)) continue;
      for (let s = 0; s < 3; s += 1) {
        if (!usedSlots.has(s)) {
          next[s] = idx;
          usedSlots.add(s);
          satisfied.add(idx);
          break;
        }
      }
    }
    assigned = next;
  }

  function slotKindOf(slotIdx: 0 | 1 | 2): "prev" | "current" | "next" | "off" {
    const idx = assigned[slotIdx];
    if (idx === -1) return "off";
    if (idx === currentIndex) return "current";
    if (idx === currentIndex - 1) return "prev";
    if (idx === currentIndex + 1) return "next";
    return "off";
  }

  function slotTransform(slotIdx: 0 | 1 | 2): string {
    const kind = slotKindOf(slotIdx);
    if (kind === "prev") return "translateY(-100%)";
    if (kind === "current") return "translateY(0)";
    if (kind === "next") return "translateY(100%)";
    return "translateY(300%)"; // park off-screen
  }

  function activeVideo(): HTMLVideoElement | undefined {
    for (let s = 0; s < 3; s += 1) {
      if (assigned[s] === currentIndex) return videoEls[s];
    }
    return undefined;
  }

  async function playCurrent(): Promise<void> {
    const el = activeVideo();
    if (!el) return;
    el.muted = muted;
    try {
      await el.play();
    } catch {
      /* autoplay rejection is expected on some browsers; user can tap. */
    }
  }

  function pauseAll(): void {
    for (const el of videoEls) {
      try {
        el?.pause();
      } catch {
        /* ignore */
      }
    }
  }

  // ------------------------------------------------------------------------
  // Navigation: next / prev.
  // ------------------------------------------------------------------------

  async function advance(action: "next" | "ended"): Promise<void> {
    if (loading) return;
    if (currentIndex >= queue.length - 1) {
      // We're at the tail — try to refill before bailing so a single tap
      // on the Next button feels responsive even when the lookahead has
      // not started yet.
      await ensureLookahead();
      if (currentIndex >= queue.length - 1) {
        if (exhausted) {
          recordFeedbackForCurrent(action);
        }
        return;
      }
    }
    recordFeedbackForCurrent(action);
    currentIndex += 1;
    syncAssignments();
    progress = 0;
    startItemTimers(currentIndex);
    void playCurrent();
    void ensureLookahead();
  }

  function retreat(): void {
    if (loading || currentIndex <= 0) return;
    recordFeedbackForCurrent("prev");
    currentIndex -= 1;
    syncAssignments();
    progress = 0;
    startItemTimers(currentIndex);
    void playCurrent();
  }

  function startItemTimers(idx: number): void {
    itemStartedAt = performance.now();
    watchStartedAt = 0;
    accumulatedWatchMs = 0;
    playbackStarted = false;
    if (idx > feedbackRecordedIndex) feedbackRecordedIndex = idx - 1;
  }

  function recordFeedbackForCurrent(action: FeedMetrics["action"]): void {
    if (feedbackRecordedIndex >= currentIndex) return;
    const item = queue[currentIndex];
    if (!item) return;
    const el = activeVideo();
    let completionRatio = 0;
    if (el && el.duration > 0) {
      completionRatio = Math.min(1, Math.max(0, el.currentTime / el.duration));
    }
    const watchMs = accumulatedWatchMs + accumulateSinceWatch();
    const dwellMs = performance.now() - itemStartedAt;
    const metrics: FeedMetrics = {
      playbackStarted,
      action,
      watchMs,
      dwellMs,
      completionRatio,
    };
    const { klass } = classifyFeedFeedback(metrics);
    applyFeedFeedback(tagScores, item.tags ?? [], klass);
    feedbackRecordedIndex = currentIndex;
  }

  function accumulateSinceWatch(): number {
    if (watchStartedAt === 0) return 0;
    const slice = performance.now() - watchStartedAt;
    watchStartedAt = performance.now();
    return slice;
  }

  // ------------------------------------------------------------------------
  // Lookahead loop.
  // ------------------------------------------------------------------------

  async function ensureLookahead(): Promise<void> {
    if (exhausted) return;
    if (lookaheadPromise) return;
    if (remainingAhead > FEED_FETCH_AHEAD_THRESHOLD) return;
    lookaheadPromise = (async () => {
      loadingMore = true;
      try {
        await fetchNextBatch(FEED_BATCH_SIZE);
      } finally {
        loadingMore = false;
        lookaheadPromise = null;
        syncAssignments();
      }
    })();
    await lookaheadPromise;
  }

  async function fetchNextBatch(limit: number): Promise<void> {
    const seed = queue[0];
    if (!seed?.videoId) return;
    decayFeedTagScores(tagScores);
    const { prefer, avoid } = feedPreferenceTags(tagScores);
    const seenIds = collectSeenIds();
    const opts: SimilarVideoOptions = {
      videoId: seed.videoId,
      seedImageId: seed.imageId,
      limit,
      seenIds,
      preferTags: prefer,
      avoidTags: avoid,
      includeNSFW: $includeNSFW,
    };
    const token = ++candidateRequestToken;
    let response;
    try {
      response = await searchSimilarVideos(opts);
    } catch (err) {
      if (err instanceof ApiError && err.message) {
        // Surface only as exhaustion — keep the overlay visible.
      }
      exhausted = true;
      return;
    }
    if (token !== candidateRequestToken) return;
    const incoming = (response.results ?? [])
      .map(pinFromSearchResult)
      .filter((p) => p.mediaType === "video" && p.videoId !== undefined);
    if (incoming.length === 0) {
      // True empty batch → end of session.
      exhausted = true;
      return;
    }
    let appended = 0;
    for (const candidate of incoming) {
      const vid = candidate.videoId!;
      if (seenVideoIDs.has(vid) || rejectedVideoIDs.has(vid)) continue;
      seenVideoIDs.add(vid);
      queue = [...queue, candidate];
      appended += 1;
    }
    if (appended === 0) {
      // Whole batch was filtered (already seen). Don't end the session — the
      // next fetch with the updated `seen` list will pull fresh candidates.
      return;
    }
  }

  function collectSeenIds(): number[] {
    const ids: number[] = [];
    for (const item of queue) {
      if (item.videoId !== undefined) ids.push(item.videoId);
    }
    for (const id of seenVideoIDs) ids.push(id);
    for (const id of rejectedVideoIDs) ids.push(id);
    // Dedupe — server caps at 1000 anyway.
    return Array.from(new Set(ids));
  }

  // ------------------------------------------------------------------------
  // Per-element event handlers — only the active <video> drives progress.
  // ------------------------------------------------------------------------

  function onTimeUpdate(slotIdx: 0 | 1 | 2): void {
    if (slotKindOf(slotIdx) !== "current") return;
    const el = videoEls[slotIdx];
    if (!el || !el.duration) return;
    progress = Math.min(1, Math.max(0, el.currentTime / el.duration));
  }

  function onPlay(slotIdx: 0 | 1 | 2): void {
    if (slotKindOf(slotIdx) !== "current") return;
    playbackStarted = true;
    if (watchStartedAt === 0) watchStartedAt = performance.now();
  }

  function onPause(slotIdx: 0 | 1 | 2): void {
    if (slotKindOf(slotIdx) !== "current") return;
    if (watchStartedAt > 0) {
      accumulatedWatchMs += performance.now() - watchStartedAt;
      watchStartedAt = 0;
    }
  }

  function onEnded(slotIdx: 0 | 1 | 2): void {
    if (slotKindOf(slotIdx) !== "current") return;
    void advance("ended");
  }

  function onError(slotIdx: 0 | 1 | 2): void {
    const idx = assigned[slotIdx];
    const item = queue[idx];
    if (item?.videoId !== undefined) rejectedVideoIDs.add(item.videoId);
    // Don't auto-advance — let the user decide. Future batches exclude this
    // video via collectSeenIds, and the next-button stays available.
  }

  // ------------------------------------------------------------------------
  // Touch / keyboard / button input.
  // ------------------------------------------------------------------------

  function onKey(event: KeyboardEvent): void {
    if (!isOpen) return;
    if (event.key === "Escape") {
      event.preventDefault();
      close();
      return;
    }
    if (event.key === "ArrowDown" || event.key === "PageDown") {
      event.preventDefault();
      void advance("next");
      return;
    }
    if (event.key === "ArrowUp" || event.key === "PageUp") {
      event.preventDefault();
      retreat();
      return;
    }
    if (event.key === " " || event.key === "k") {
      const target = event.target as HTMLElement | null;
      if (target && (target.tagName === "BUTTON" || target.tagName === "INPUT")) return;
      event.preventDefault();
      togglePlay();
    }
  }

  function togglePlay(): void {
    const el = activeVideo();
    if (!el) return;
    if (el.paused) void el.play();
    else el.pause();
  }

  function toggleMute(): void {
    muted = !muted;
    for (const el of videoEls) {
      if (el) el.muted = muted;
    }
  }

  function onTouchStart(event: TouchEvent): void {
    if (event.touches.length !== 1) return;
    const t = event.touches[0];
    touchStartY = t.clientY;
    touchStartX = t.clientX;
    touchStartTime = performance.now();
    dragging = false;
  }

  function onTouchMove(event: TouchEvent): void {
    if (event.touches.length !== 1) return;
    const t = event.touches[0];
    const dy = t.clientY - touchStartY;
    const dx = t.clientX - touchStartX;
    if (!dragging) {
      if (Math.abs(dy) < TAP_DEAD_ZONE_PX || Math.abs(dy) < Math.abs(dx) * 1.15) {
        return;
      }
      dragging = true;
    }
    event.preventDefault();
    const vh = window.innerHeight || 1;
    let clamped = Math.max(-vh, Math.min(vh, dy));
    // Rubber-band when no neighbour exists.
    const slotKindForDir = dy < 0 ? "next" : "prev";
    const hasNeighbour = anySlotIs(slotKindForDir);
    if (!hasNeighbour) clamped = clamped * 0.28;
    dragOffsetPx = clamped;
  }

  function anySlotIs(kind: "prev" | "next"): boolean {
    for (let s = 0; s < 3; s += 1) {
      if (slotKindOf(s as 0 | 1 | 2) === kind) return true;
    }
    return false;
  }

  function onTouchEnd(event: TouchEvent): void {
    if (!dragging) {
      dragOffsetPx = 0;
      return;
    }
    const dy = dragOffsetPx;
    const elapsed = Math.max(1, performance.now() - touchStartTime);
    const velocity = dy / elapsed; // px/ms
    const vh = window.innerHeight || 1;
    const distanceThreshold = Math.max(
      SWIPE_DISTANCE_THRESHOLD_PX,
      vh * SWIPE_DISTANCE_THRESHOLD_VH,
    );
    const goNext =
      dy < -distanceThreshold || velocity < -SWIPE_VELOCITY_THRESHOLD;
    const goPrev =
      dy > distanceThreshold || velocity > SWIPE_VELOCITY_THRESHOLD;
    dragging = false;
    dragOffsetPx = 0;
    suppressClickUntil = performance.now() + 350;
    if (goNext) void advance("next");
    else if (goPrev) retreat();
    event.preventDefault();
  }

  function onCanvasClick(event: MouseEvent): void {
    if (performance.now() < suppressClickUntil) return;
    // Tap toggles play/pause when the click landed on the video canvas
    // (not on overlay chrome).
    const target = event.target as HTMLElement | null;
    if (target && target.closest("[data-feed-chrome]")) return;
    togglePlay();
  }

  // Reactive: keep all slot videos' muted state in sync.
  $effect(() => {
    for (const el of videoEls) {
      if (el) el.muted = muted;
    }
  });

  // Reactive: when assignments change, ensure non-current videos are paused
  // (mobile autoplay rules let them keep buffering paused).
  $effect(() => {
    void assigned;
    for (let s = 0; s < 3; s += 1) {
      const el = videoEls[s];
      if (!el) continue;
      const kind = slotKindOf(s as 0 | 1 | 2);
      if (kind !== "current") {
        try {
          el.pause();
        } catch {
          /* ignore */
        }
      }
    }
  });
</script>

<svelte:window onkeydown={onKey} />

{#if isOpen}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div
    data-feed-overlay
    role="dialog"
    aria-modal="true"
    aria-label="Similar videos"
    tabindex="-1"
    data-feed-current-index={currentIndex}
    data-feed-queue-size={queue.length}
    data-feed-exhausted={exhausted ? "true" : undefined}
    class="fixed inset-0 z-[1400] bg-black text-[#fffdf9] [touch-action:none] [overscroll-behavior:contain] flex flex-col"
    ontouchstart={onTouchStart}
    ontouchmove={onTouchMove}
    ontouchend={onTouchEnd}
    onclick={onCanvasClick}
  >
    <div
      class="relative flex-1 overflow-hidden"
      bind:this={trackEl}
    >
      {#each [0, 1, 2] as slotIdx (slotIdx)}
        {@const idx = assigned[slotIdx]}
        {@const item = queue[idx]}
        {@const kind = slotKindOf(slotIdx as 0 | 1 | 2)}
        <div
          class="absolute inset-0 grid place-items-center bg-black transition-transform duration-300 ease-soft will-change-transform"
          style:transform={`${slotTransform(slotIdx as 0 | 1 | 2)} translateY(${dragOffsetPx}px)`}
          style:transition={dragging ? "none" : undefined}
          data-feed-slot={kind}
        >
          {#if item}
            <!-- User-uploaded media has no captions track; suppress nag -->
            <!-- svelte-ignore a11y_media_has_caption -->
            <video
              bind:this={videoEls[slotIdx]}
              data-feed-video={kind}
              data-feed-current={kind === "current" ? "true" : undefined}
              src={item.mediaUrl}
              poster={item.thumbUrl}
              muted={muted}
              playsinline
              preload={kind === "current" ? "auto" : "metadata"}
              autoplay={kind === "current"}
              class="block max-w-full max-h-full w-auto h-auto object-contain"
              ontimeupdate={() => onTimeUpdate(slotIdx as 0 | 1 | 2)}
              onplay={() => onPlay(slotIdx as 0 | 1 | 2)}
              onpause={() => onPause(slotIdx as 0 | 1 | 2)}
              onended={() => onEnded(slotIdx as 0 | 1 | 2)}
              onerror={() => onError(slotIdx as 0 | 1 | 2)}
            ></video>
          {/if}
        </div>
      {/each}

      <!-- Top bar: title + close. -->
      <div
        data-feed-chrome
        class="absolute top-0 inset-x-0 px-4 pt-[max(12px,env(safe-area-inset-top))] pb-3 flex items-start gap-3 bg-gradient-to-b from-black/60 to-transparent pointer-events-none"
      >
        <div class="flex-1 min-w-0 pointer-events-auto">
          <p class="m-0 text-[12px] uppercase tracking-[0.08em] text-white/60">
            Similar videos
          </p>
          <h2 class="m-0 mt-0.5 text-[15px] font-semibold leading-tight truncate">
            {currentItem?.title ?? seedItem?.title ?? ""}
          </h2>
        </div>
        <button
          type="button"
          data-feed-close
          aria-label="Close feed"
          onclick={(e) => {
            e.stopPropagation();
            close();
          }}
          class="pointer-events-auto grid place-items-center w-10 h-10 rounded-full bg-black/55 hover:bg-black/75 border-0 cursor-pointer text-[#fffdf9]"
        >
          <Icon name="close" class="w-4 h-4" />
        </button>
      </div>

      <!-- Bottom bar: progress + controls -->
      <div
        data-feed-chrome
        class="absolute bottom-0 inset-x-0 px-4 pt-3 pb-[max(16px,env(safe-area-inset-bottom))] flex flex-col gap-3 bg-gradient-to-t from-black/70 to-transparent pointer-events-none"
      >
        <div
          class="w-full h-[3px] rounded-full overflow-hidden bg-white/25 pointer-events-auto"
          aria-hidden="true"
        >
          <span
            data-feed-progress
            class="block h-full bg-white/95 transition-[width] duration-100 ease-linear"
            style:width={`${Math.round(progress * 100)}%`}
          ></span>
        </div>
        <div class="flex items-center gap-2 pointer-events-auto">
          <button
            type="button"
            data-feed-prev
            disabled={currentIndex <= 0 || loading}
            onclick={(e) => {
              e.stopPropagation();
              retreat();
            }}
            aria-label="Previous video"
            class="grid place-items-center w-11 h-11 rounded-full bg-black/55 hover:bg-black/75 disabled:opacity-40 disabled:cursor-not-allowed border-0 cursor-pointer text-[#fffdf9]"
          >
            <span aria-hidden="true" class="rotate-90 inline-block">
              <Icon name="feed" class="w-4 h-4 -rotate-180" />
            </span>
          </button>
          <button
            type="button"
            data-feed-playpause
            onclick={(e) => {
              e.stopPropagation();
              togglePlay();
            }}
            aria-label="Play / pause"
            class="grid place-items-center w-11 h-11 rounded-full bg-black/55 hover:bg-black/75 border-0 cursor-pointer text-[#fffdf9]"
          >
            <Icon name="feed" class="w-4 h-4" />
          </button>
          <button
            type="button"
            data-feed-mute
            onclick={(e) => {
              e.stopPropagation();
              toggleMute();
            }}
            aria-label={muted ? "Unmute" : "Mute"}
            class="px-3 h-11 rounded-full bg-black/55 hover:bg-black/75 border-0 cursor-pointer text-[#fffdf9] text-[13px] font-medium"
          >
            {muted ? "Muted" : "Sound"}
          </button>
          <span class="flex-1 text-right text-[12px] text-white/70 tabular-nums">
            {currentIndex + 1} / {queue.length}{loadingMore ? " · …" : ""}
          </span>
          <button
            type="button"
            data-feed-next
            disabled={loading}
            onclick={(e) => {
              e.stopPropagation();
              void advance("next");
            }}
            aria-label="Next video"
            class="grid place-items-center w-11 h-11 rounded-full bg-black/55 hover:bg-black/75 disabled:opacity-40 disabled:cursor-not-allowed border-0 cursor-pointer text-[#fffdf9]"
          >
            <Icon name="feed" class="w-4 h-4" />
          </button>
        </div>
      </div>

      {#if loading && queue.length <= 1}
        <div
          class="absolute inset-0 grid place-items-center bg-black/80 text-white/80 text-sm pointer-events-none"
        >
          Loading similar videos…
        </div>
      {/if}

      {#if exhausted && currentIndex >= queue.length - 1}
        <div
          data-feed-end
          class="absolute inset-0 grid place-items-center bg-black/85 px-6"
        >
          <div class="max-w-sm text-center flex flex-col items-center gap-3">
            <p class="text-white/80 text-[14px]">
              That's the end of this feed.
            </p>
            <button
              type="button"
              data-feed-end-back
              onclick={(e) => {
                e.stopPropagation();
                close();
              }}
              class="px-5 py-2 bg-[#fffdf9] text-ink rounded-full text-[13.5px] font-semibold border-0 cursor-pointer"
            >
              Back to library
            </button>
          </div>
        </div>
      {/if}
    </div>
  </div>
{/if}
