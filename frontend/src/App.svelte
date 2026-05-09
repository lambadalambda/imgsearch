<script lang="ts">
  import Rail from "./components/Rail.svelte";
  import Header from "./components/Header.svelte";
  import SearchBar from "./components/SearchBar.svelte";
  import QuickRow from "./components/QuickRow.svelte";
  import Masonry from "./components/Masonry.svelte";
  import Lightbox from "./components/Lightbox.svelte";

  import { untrack } from "svelte";
  import { get } from "svelte/store";
  import {
    mode,
    pins,
    resultsMeta,
    includeNSFW,
    stats,
    topTags,
  } from "./lib/stores";
  import { ApiError, getStats, listImages, listTagCloud, searchSimilar, searchText } from "./lib/api";
  import { pinFromImage, pinFromSearchResult } from "./lib/utils";
  import type { Pin } from "./lib/types";

  // Bootstrap: stats + top tags. Don't block rendering on failure.
  void (async () => {
    try {
      const [s, t] = await Promise.all([getStats(), listTagCloud({ limit: 12 })]);
      stats.set({ images: s.standalone_images_total ?? s.images_total, videos: s.videos_total });
      topTags.set(t.tags);
    } catch (err) {
      console.warn("bootstrap failed", err);
    }
  })();

  let currentRequestToken = 0;

  $effect(() => {
    const state = $mode;
    const includeNsfw = $includeNSFW;
    const token = ++currentRequestToken;

    // Use untracked reads/writes for resultsMeta so the effect only re-runs
    // when mode or includeNSFW actually change.
    untrack(() => {
      const previous = get(resultsMeta);
      resultsMeta.set({ total: previous.total, loading: true });
    });

    const start = performance.now();
    const ac = new AbortController();
    void (async () => {
      try {
        let nextPins: Pin[] = [];
        let total = 0;

        if (state.mode === "search" && state.query) {
          const response = await searchText({
            query: state.query,
            limit: 60,
            includeNSFW: includeNsfw,
            signal: ac.signal,
          });
          nextPins = response.results.map(pinFromSearchResult);
          total = response.total ?? response.results.length;
        } else if (state.mode === "similar" && state.similarTo !== undefined) {
          const response = await searchSimilar({
            imageId: state.similarTo,
            limit: 60,
            includeNSFW: includeNsfw,
            signal: ac.signal,
          });
          nextPins = response.results.map(pinFromSearchResult);
          total = response.total ?? response.results.length;
        } else {
          const response = await listImages({ limit: 60, includeNSFW: includeNsfw, signal: ac.signal });
          nextPins = response.images.map(pinFromImage);
          total = response.total ?? response.images.length;
        }

        if (token !== currentRequestToken) return;
        pins.set(nextPins);
        resultsMeta.set({ total, durationMs: performance.now() - start, loading: false });
      } catch (err) {
        if ((err as { name?: string }).name === "AbortError") return;
        if (token !== currentRequestToken) return;
        const message = err instanceof ApiError ? err.message : err instanceof Error ? err.message : "Search failed";
        resultsMeta.set({ total: 0, loading: false, error: message });
        pins.set([]);
      }
    })();

    return () => ac.abort();
  });

  const emptyMessage = $derived.by(() => {
    if ($resultsMeta.error) return `Couldn't load: ${$resultsMeta.error}`;
    if ($mode.mode === "search") return `No matches for "${$mode.query ?? ""}".`;
    if ($mode.mode === "similar") return "No similar items found in your library.";
    return "Library is empty. Upload some media via the legacy UI.";
  });
</script>

<div class="grid grid-cols-1 sm:grid-cols-[64px_1fr] min-h-screen">
  <Rail />
  <div class="min-w-0 flex flex-col">
    <Header />
    <SearchBar />
    <QuickRow />

    <div class="px-5 sm:px-9 mt-3 mb-1 text-[12.5px] text-muted-2 flex flex-wrap gap-2 items-center">
      {#if $resultsMeta.loading}
        <span>Loading…</span>
      {:else if $resultsMeta.error}
        <span class="text-bad">{$resultsMeta.error}</span>
      {:else}
        <span>
          {$resultsMeta.total.toLocaleString()} {$resultsMeta.total === 1 ? "result" : "results"}
          {#if $resultsMeta.durationMs && $mode.mode !== "library"}
            · in {Math.round($resultsMeta.durationMs)} ms
          {/if}
        </span>
      {/if}
    </div>

    <Masonry pins={$pins} loading={$resultsMeta.loading} {emptyMessage} />
  </div>
</div>

<Lightbox />
