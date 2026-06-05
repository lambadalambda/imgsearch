<script lang="ts">
  import Rail from "./components/Rail.svelte";
  import Header from "./components/Header.svelte";
  import SearchBar from "./components/SearchBar.svelte";
  import QuickRow from "./components/QuickRow.svelte";
  import StatisticsPane from "./components/StatisticsPane.svelte";
  import Masonry from "./components/Masonry.svelte";
  import Lightbox from "./components/Lightbox.svelte";
  import Upload from "./components/Upload.svelte";
  import Feed from "./components/Feed.svelte";

  import { tick, untrack } from "svelte";
  import { get } from "svelte/store";
  import {
    mode,
    pins,
    resultsMeta,
    includeNSFW,
    libraryMedia,
    librarySort,
    stats,
    topTags,
    pageBump,
    bumpPage,
    dataEpoch,
  } from "./lib/stores";
  import {
    ApiError,
    getStats,
    listImages,
    listTagCloud,
    listVideos,
    searchSimilar,
    searchTags,
    searchText,
  } from "./lib/api";
  import { pinFromImage, pinFromSearchResult, pinFromVideo } from "./lib/utils";
  import type { Pin } from "./lib/types";

  const PAGE_SIZE = 48;
  const RANDOM_SEED_MAX = 0x80000000;

  function newLibrarySeed(): number {
    if (globalThis.crypto?.getRandomValues) {
      const values = new Uint32Array(1);
      globalThis.crypto.getRandomValues(values);
      return values[0] & 0x7fffffff;
    }
    return Math.floor(Math.random() * RANDOM_SEED_MAX);
  }

  function randomKey(key: string, seed: number): number {
    let hash = seed >>> 0;
    for (let i = 0; i < key.length; i += 1) {
      hash = Math.imul(hash ^ key.charCodeAt(i), 16777619) >>> 0;
    }
    return hash;
  }

  function compareRecentPins(left: Pin, right: Pin): number {
    const leftTime = Date.parse(left.createdAt ?? "") || 0;
    const rightTime = Date.parse(right.createdAt ?? "") || 0;
    return rightTime - leftTime || right.key.localeCompare(left.key);
  }

  function combineLibraryPins(imagePins: Pin[], videoPins: Pin[], sort: "random" | "newest", seed: number): Pin[] {
    if (sort === "newest") {
      return [...imagePins, ...videoPins].sort(compareRecentPins);
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

  // Stats are cheap and useful in the search bar, so fetch them immediately.
  // Tag cloud is deferred until after the first page load because its JSON tag
  // scan can otherwise monopolize the single SQLite connection before images.
  void (async () => {
    try {
      const s = await getStats();
      stats.set({ ...s, images: s.standalone_images_total ?? s.images_total, videos: s.videos_total });
    } catch (err) {
      console.warn("stats bootstrap failed", err);
    }
  })();

  let currentRequestToken = 0;
  let currentOffset = 0;
  let lastPageBump = 0;
  let lastDataKey = "";
  let canLoadMore = $state(false);
  let loadingMore = $state(false);
  let firstPageLoaded = $state(false);
  let tagCloudStarted = false;
  let libraryRandomSeed = newLibrarySeed();

  function scrollSimilarResultsToTop(): void {
    if (typeof document === "undefined") return;
    void tick().then(() => {
      const target =
        document.querySelector<HTMLElement>("[data-results-grid]") ??
        document.querySelector<HTMLElement>("[data-results]");
      target?.scrollIntoView({ block: "start" });
    });
  }

  $effect(() => {
    if (!firstPageLoaded || tagCloudStarted) return;
    tagCloudStarted = true;
    const ac = new AbortController();
    void (async () => {
      try {
        const t = await listTagCloud({ limit: 16, signal: ac.signal });
        topTags.set(t.tags);
      } catch (err) {
        if ((err as { name?: string }).name === "AbortError") return;
        console.warn("tag cloud bootstrap failed", err);
      }
    })();
    return () => ac.abort();
  });

  $effect(() => {
    const state = $mode;
    const includeNsfw = $includeNSFW;
    const media = $libraryMedia;
    const sort = $librarySort;
    const bump = $pageBump;
    // Subscribe to dataEpoch so successful uploads (or other refresh events)
    // can force a fresh, replacing fetch without changing mode.
    const epoch = $dataEpoch;
    const dataKey = [
      state.mode,
      state.query ?? "",
      state.similarTo ?? "",
      state.tags?.join("\u0000") ?? "",
      state.tagMode ?? "",
      state.mode === "library" ? media : "",
      state.mode === "library" ? sort : "",
      includeNsfw ? "1" : "0",
      epoch,
    ].join("\u0001");
    const token = ++currentRequestToken;

    // First request for a data key starts at offset 0 and replaces. "Load more"
    // calls bumpPage(); only a new bump on the same key appends.
    let appending = false;
    untrack(() => {
      if (bump > lastPageBump && dataKey === lastDataKey) {
        appending = true;
      } else {
        currentOffset = 0;
        libraryRandomSeed = newLibrarySeed();
      }
      lastPageBump = bump;
      lastDataKey = dataKey;
      const previous = get(resultsMeta);
      resultsMeta.set({ total: previous.total, loading: !appending, error: undefined });
    });
    if (appending) {
      loadingMore = true;
    } else {
      canLoadMore = false;
    }

    const start = performance.now();
    const ac = new AbortController();
    void (async () => {
      try {
        let nextPins: Pin[] = [];
        let total = 0;

        if (state.mode === "search" && state.query) {
          const response = await searchText({
            query: state.query,
            limit: PAGE_SIZE,
            offset: currentOffset,
            includeNSFW: includeNsfw,
            signal: ac.signal,
          });
          nextPins = response.results.map(pinFromSearchResult);
          total = response.total ?? response.results.length;
        } else if (state.mode === "similar" && state.similarTo !== undefined) {
          // Similar search currently returns a single non-paginated batch from
          // the backend; "Load more" still re-runs but the response is the same set.
          const response = await searchSimilar({
            imageId: state.similarTo,
            limit: PAGE_SIZE,
            includeNSFW: includeNsfw,
            signal: ac.signal,
          });
          nextPins = response.results.map(pinFromSearchResult);
          total = response.total ?? response.results.length;
        } else if (state.mode === "tag" && state.tags?.length) {
          const response = await searchTags({
            tags: state.tags,
            mode: state.tagMode ?? "any",
            limit: PAGE_SIZE,
            offset: currentOffset,
            includeNSFW: includeNsfw,
            signal: ac.signal,
          });
          nextPins = response.results.map(pinFromSearchResult);
          total = response.total ?? response.results.length;
        } else {
          const seed = sort === "random" ? libraryRandomSeed : undefined;
          if (media === "images") {
            const response = await listImages({
              limit: PAGE_SIZE,
              offset: currentOffset,
              order: sort,
              seed,
              includeNSFW: includeNsfw,
              signal: ac.signal,
            });
            nextPins = response.images.map(pinFromImage);
            total = response.total ?? response.images.length;
          } else if (media === "videos") {
            const response = await listVideos({
              limit: PAGE_SIZE,
              offset: currentOffset,
              order: sort,
              seed,
              includeNSFW: includeNsfw,
              signal: ac.signal,
            });
            nextPins = response.videos.map(pinFromVideo);
            total = response.total ?? response.videos.length;
          } else {
            const limit = currentOffset + PAGE_SIZE;
            const [imageResponse, videoResponse] = await Promise.all([
              listImages({ limit, offset: 0, order: sort, seed, includeNSFW: includeNsfw, signal: ac.signal }),
              listVideos({ limit, offset: 0, order: sort, seed, includeNSFW: includeNsfw, signal: ac.signal }),
            ]);
            const combined = combineLibraryPins(
              imageResponse.images.map(pinFromImage),
              videoResponse.videos.map(pinFromVideo),
              sort,
              libraryRandomSeed,
            );
            nextPins = combined.slice(currentOffset, currentOffset + PAGE_SIZE);
            total = (imageResponse.total ?? imageResponse.images.length) + (videoResponse.total ?? videoResponse.videos.length);
          }
        }

        if (token !== currentRequestToken) return;

        if (appending) {
          pins.update((existing) => [...existing, ...nextPins]);
          currentOffset += nextPins.length;
        } else {
          pins.set(nextPins);
          currentOffset = nextPins.length;
          if (state.mode === "similar") {
            scrollSimilarResultsToTop();
          }
        }
        canLoadMore = nextPins.length === PAGE_SIZE && currentOffset < total;
        resultsMeta.set({ total, durationMs: performance.now() - start, loading: false });
        if (!firstPageLoaded && !appending) {
          firstPageLoaded = true;
        }
      } catch (err) {
        if ((err as { name?: string }).name === "AbortError") return;
        if (token !== currentRequestToken) return;
        const message = err instanceof ApiError ? err.message : err instanceof Error ? err.message : "Search failed";
        resultsMeta.set({ total: 0, loading: false, error: message });
        if (!appending) {
          pins.set([]);
          currentOffset = 0;
          canLoadMore = false;
        }
      } finally {
        loadingMore = false;
      }
    })();

    return () => ac.abort();
  });

  function loadMore() {
    if (!canLoadMore || loadingMore || get(resultsMeta).loading) return;
    bumpPage();
  }

  const emptyMessage = $derived.by(() => {
    if ($resultsMeta.error) return `Couldn't load: ${$resultsMeta.error}`;
    if ($mode.mode === "search") return `No matches for "${$mode.query ?? ""}".`;
    if ($mode.mode === "similar") return "No similar items found in your library.";
    if ($mode.mode === "tag" && $mode.tags?.length) {
      return `No items tagged ${$mode.tags.join(", ")}.`;
    }
    if ($libraryMedia === "images") return "No images yet. Click Upload to add some media.";
    if ($libraryMedia === "videos") return "No videos yet. Click Upload to add some media.";
    return "Library is empty. Click Upload to add some media.";
  });
</script>

<div class="grid grid-cols-1 sm:grid-cols-[64px_1fr] min-h-screen">
  <Rail />
  <div class="min-w-0 flex flex-col">
    <Header />
    <SearchBar />
    <QuickRow />

    {#if $mode.mode === "stats"}
      <StatisticsPane />
    {:else}
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

      <Masonry
        pins={$pins}
        loading={$resultsMeta.loading}
        {emptyMessage}
        {canLoadMore}
        {loadingMore}
        onLoadMore={loadMore}
      />
    {/if}
  </div>
</div>

<Lightbox />
<Upload />
<Feed />
