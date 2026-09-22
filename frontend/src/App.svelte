<script lang="ts">
  import Rail from "./components/Rail.svelte";
  import Header from "./components/Header.svelte";
  import SearchBar from "./components/SearchBar.svelte";
  import QuickRow from "./components/QuickRow.svelte";
  import StatisticsPane from "./components/StatisticsPane.svelte";
  import SettingsPane from "./components/SettingsPane.svelte";
  import DuplicatesPane from "./components/DuplicatesPane.svelte";
  import Masonry from "./components/Masonry.svelte";
  import Lightbox from "./components/Lightbox.svelte";
  import Upload from "./components/Upload.svelte";
  import Feed from "./components/Feed.svelte";

  import { onMount, tick, untrack } from "svelte";
  import { startAnnotationWatch } from "./lib/annotationWatch";
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
    queryImage,
    clearQueryImage,
    setLibrary,
  } from "./lib/stores";
  import {
    ApiError,
    listImages,
    listTagCloud,
    listVideos,
    searchSimilar,
    searchByImage,
    searchTags,
    searchText,
  } from "./lib/api";
  import { refreshStats } from "./lib/stats";
  import { appendUniquePins, pinFromImage, pinFromSearchResult, pinFromVideo } from "./lib/utils";
  import { combineLibraryPins, newLibrarySeed } from "./lib/library";
  import type { Pin } from "./lib/types";

  const PAGE_SIZE = 48;
  // Stats are cheap and useful in the search bar, so fetch them immediately.
  // Tag cloud is deferred until after the first page load because its JSON tag
  // scan can otherwise monopolize the single SQLite connection before images.
  void refreshStats().catch((err) => {
    console.warn("stats bootstrap failed", err);
  });

  // Poll annotation progress for the pins on screen (meta/issues/121).
  onMount(() => startAnnotationWatch());

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
    const image = $queryImage;
    if (state.mode === "byimage" && !image) {
      // Nothing to search with (for example after a reload): back to the library.
      setLibrary();
      return;
    }
    const dataKey = [
      state.mode,
      state.mode === "byimage" ? String(image?.id ?? "") : "",
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
        } else if (state.mode === "byimage" && image) {
          const response = await searchByImage({
            file: image.file,
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
          // Advance by what the server returned, not by what we kept, so a
          // de-duplicated page never gets re-requested.
          pins.update((existing) => appendUniquePins(existing, nextPins));
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
        // An empty search may just mean indexing hasn't caught up; refresh
        // stats so the empty state can report current progress.
        if (nextPins.length === 0 && !appending && (state.mode === "search" || state.mode === "similar")) {
          void refreshStats().catch(() => {});
        }
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

  // Incomplete embedding progress, if the stats snapshot reports any. An
  // empty search on a half-indexed library is indistinguishable from a true
  // miss, so the empty state explains what is actually happening.
  const embeddingBacklog = $derived.by(() => {
    const s = $stats;
    if (!s) return null;
    const done = s.job_kinds?.["embed_image"]?.done ?? 0;
    const expected = s.queue?.total ?? s.images_total ?? 0;
    return expected > 0 && done < expected ? { done, expected } : null;
  });

  const emptyMessage = $derived.by(() => {
    if ($resultsMeta.error) return `Couldn't load: ${$resultsMeta.error}`;
    const indexingSuffix = embeddingBacklog
      ? ` The library is still indexing (${embeddingBacklog.done.toLocaleString()} of ${embeddingBacklog.expected.toLocaleString()} images embedded), so results will improve as it completes.`
      : "";
    if ($mode.mode === "search") return `No matches for "${$mode.query ?? ""}".${indexingSuffix}`;
    if ($mode.mode === "similar" || $mode.mode === "byimage") return `No similar items found in your library.${indexingSuffix}`;
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
    {:else if $mode.mode === "settings"}
      <SettingsPane />
    {:else if $mode.mode === "duplicates"}
      <DuplicatesPane />
    {:else}
      <div
        data-results-meta
        role="status"
        aria-live="polite"
        class="px-5 sm:px-9 mt-3 mb-1 text-[12.5px] text-muted-2 flex flex-wrap gap-2 items-center"
      >
        {#if $mode.mode === "byimage" && $queryImage}
          <span data-query-image class="inline-flex items-center gap-2 pr-1 mr-1 border-r border-line">
            <img src={$queryImage.previewUrl} alt="" class="w-7 h-7 rounded-[6px] object-cover border border-line" />
            <span class="text-ink-2">Similar to {$queryImage.name}</span>
            <button
              type="button"
              data-query-image-clear
              aria-label="Clear image query"
              onclick={() => {
                clearQueryImage();
                setLibrary();
              }}
              class="inline-grid place-items-center w-5 h-5 bg-transparent border-0 text-muted rounded-full cursor-pointer hover:bg-bg-2 hover:text-ink"
            >
              ×
            </button>
          </span>
        {/if}
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
