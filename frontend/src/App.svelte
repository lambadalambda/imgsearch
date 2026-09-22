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
  import CollectionsPane from "./components/CollectionsPane.svelte";
  import CollectionPicker from "./components/CollectionPicker.svelte";
  import Feed from "./components/Feed.svelte";

  import { onMount, tick, untrack } from "svelte";
  import { startAnnotationWatch } from "./lib/annotationWatch";
  import { get } from "svelte/store";
  import {
    mode,
    collectionEpoch,
    collectionName,
    lightboxPin,
    collectionPickerTarget,
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
    getCollection,
    listCollectionItems,
    listImages,
    listTagCloud,
    listVideos,
    searchSimilar,
    searchByImage,
    searchTags,
    searchText,
  } from "./lib/api";
  import { requestErrorMessage } from "./lib/requestError";
  import { refreshStats } from "./lib/stats";
  import { appendUniquePins, pinFromImage, pinFromSearchResult, pinFromVideo } from "./lib/utils";
  import { hasSearchChrome, resultsCountLabel } from "./lib/searchPresentation";
  import { collectionPresentation } from "./lib/collectionPresentation";
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

  let retryAttempt = $state(0);
  let lastRetryAttempt = 0;
  let failedRequest = $state<{ scope: string; appending: boolean } | null>(null);
  function retryRequest() {
    if (failedRequest?.scope === requestScope) retryAttempt += 1;
  }

  let currentRequestToken = 0;
  let currentOffset = 0;
  let lastPageBump = 0;
  let lastDataKey = "";
  let canLoadMore = $state(false);
  let loadingMore = $state(false);
  let firstPageLoaded = $state(false);
  let tagCloudStarted = false;
  let libraryRandomSeed = newLibrarySeed();
  let collectionItemCount = $state(0);
  let loadedCollectionScope = $state("");
  const collectionScope = $derived(`${$mode.collectionId}:${$includeNSFW}:${$collectionEpoch}:${$dataEpoch}`);
  const collectionCurrent = $derived($mode.mode !== "collection" || loadedCollectionScope === collectionScope);
  // Rendering follows the requested scope immediately, before the fetch effect
  // runs. An append keeps this key and the loaded cards; replacements cannot.
  const requestScope = $derived([
    $mode.mode,
    $mode.collectionId ?? "",
    $mode.mode === "collection" ? $collectionEpoch : "",
    $mode.mode === "byimage" ? String($queryImage?.id ?? "") : "",
    $mode.query ?? "",
    $mode.similarTo ?? "",
    $mode.tags?.join("\u0000") ?? "",
    $mode.tagMode ?? "",
    $mode.mode === "library" ? $libraryMedia : "",
    $mode.mode === "library" ? $librarySort : "",
    $includeNSFW ? "1" : "0",
    $dataEpoch,
  ].join("\u0001"));
  let loadedResultsScope = $state("");
  const resultsCurrent = $derived(collectionCurrent && (!hasSearchChrome($mode.mode) || loadedResultsScope === requestScope));
  const collectionCopy = $derived(collectionPresentation(collectionItemCount, $resultsMeta.total));

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
    const isRetry = retryAttempt > lastRetryAttempt;
    lastRetryAttempt = retryAttempt;
    const retry = untrack(() => failedRequest);
    const state = $mode;
    const detailScope = state.mode === "collection" ? collectionScope : "";
    const includeNsfw = $includeNSFW;
    const media = $libraryMedia;
    const sort = $librarySort;
    const bump = $pageBump;
    const image = $queryImage;
    if (state.mode === "byimage" && !image) {
      // Nothing to search with (for example after a reload): back to the library.
      setLibrary();
      return;
    }
    const dataKey = requestScope;
    const token = ++currentRequestToken;
    if (["collections", "settings", "stats", "duplicates"].includes(state.mode)) {
      pins.set([]);
      lightboxPin.set(null);
      loadedResultsScope = "";
      resultsMeta.set({ total: 0, loading: false });
      failedRequest = null;
      lastDataKey = "";
      return;
    }

    // First request for a data key starts at offset 0 and replaces. "Load more"
    // calls bumpPage(); only a new bump on the same key appends.
    let appending = false;
    untrack(() => {
      if (isRetry && retry?.scope === dataKey) {
        appending = retry.appending;
      } else if (bump > lastPageBump && dataKey === lastDataKey) {
        appending = true;
      } else {
        currentOffset = 0;
        libraryRandomSeed = newLibrarySeed();
      }
      failedRequest = null;
      lastPageBump = bump;
      lastDataKey = dataKey;
      const previous = get(resultsMeta);
      resultsMeta.set({ total: previous.total, loading: !appending, error: undefined });
    });
    if (appending) {
      loadingMore = true;
    } else {
      canLoadMore = false;
      pins.set([]);
      lightboxPin.set(null);
      loadedResultsScope = "";
      if (state.mode === "collection") {
        collectionItemCount = 0;
        loadedCollectionScope = "";
      }
    }

    const start = performance.now();
    const ac = new AbortController();
    const current = () => token === currentRequestToken && !ac.signal.aborted && dataKey === requestScope
      && includeNsfw === get(includeNSFW)
      && (state.mode !== "collection" || (get(mode).mode === "collection"
        && detailScope === `${get(mode).collectionId}:${get(includeNSFW)}:${get(collectionEpoch)}:${get(dataEpoch)}`));
    void (async () => {
      try {
        let nextPins: Pin[] = [];
        let total = 0;

        if (state.mode === "collection" && state.collectionId !== undefined) {
          if (!appending) collectionName.set("");
          const [collection, response] = await Promise.all([
            getCollection(state.collectionId, ac.signal),
            listCollectionItems(state.collectionId, { limit: PAGE_SIZE, offset: currentOffset, includeNSFW: includeNsfw, signal: ac.signal }),
          ]);
          if (!current()) return;
          collectionName.set(collection.name);
          collectionItemCount = response.item_count;
          loadedCollectionScope = detailScope;
          nextPins = response.items.map((item) => item.media_type === "image" ? pinFromImage(item.image) : pinFromVideo(item.video));
          total = response.total;
        } else if (state.mode === "search" && state.query) {
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

        if (!current()) return;

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
        loadedResultsScope = dataKey;
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
        if (!current()) return;
        loadedResultsScope = dataKey;
        if (state.mode === "collection") loadedCollectionScope = detailScope;
        failedRequest = { scope: dataKey, appending };
        resultsMeta.set({ total: appending ? get(resultsMeta).total : 0, loading: false, error: requestErrorMessage(err) });
        if (!appending) {
          pins.set([]);
          currentOffset = 0;
          canLoadMore = false;
        }
      } finally {
        if (token === currentRequestToken) loadingMore = false;
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
    const indexingSuffix = embeddingBacklog
      ? ` The library is still indexing (${embeddingBacklog.done.toLocaleString()} of ${embeddingBacklog.expected.toLocaleString()} images embedded), so results will improve as it completes.`
      : "";
    if ($mode.mode === "search") return `No matches for "${$mode.query ?? ""}".${indexingSuffix}`;
    if ($mode.mode === "similar" || $mode.mode === "byimage") return `No similar items found in your library.${indexingSuffix}`;
    if ($mode.mode === "tag" && $mode.tags?.length) {
      return `No items tagged ${$mode.tags.join(", ")}.`;
    }
    if ($mode.mode === "collection") return collectionCopy.message;
    if ($libraryMedia === "images") return "No images yet. Click Upload to add some media.";
    if ($libraryMedia === "videos") return "No videos yet. Click Upload to add some media.";
    return "Library is empty. Click Upload to add some media.";
  });
</script>

<div inert={$collectionPickerTarget !== null} class="grid grid-cols-1 sm:grid-cols-[160px_minmax(0,1fr)] min-h-screen pb-20 sm:pb-0">
  <Rail />
  <div class="min-w-0 flex flex-col">
    <Header />
    {#if hasSearchChrome($mode.mode)}
      <SearchBar />
      <QuickRow />
    {/if}

    {#if $mode.mode === "stats"}
      <StatisticsPane />
    {:else if $mode.mode === "settings"}
      <SettingsPane />
    {:else if $mode.mode === "duplicates"}
      <DuplicatesPane />
    {:else if $mode.mode === "collections"}
      <CollectionsPane />
    {:else}
      <div
        data-results-meta
        role="status"
        aria-live="polite"
        class="px-5 sm:px-9 mt-1 sm:mt-3 mb-1 text-[12.5px] text-muted-2 flex flex-wrap gap-2 items-center"
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
              class="inline-grid place-items-center w-11 h-11 bg-transparent border-0 text-muted rounded-full cursor-pointer hover:bg-bg-2 hover:text-ink"
            >
              ×
            </button>
          </span>
        {/if}
        {#if !resultsCurrent || $resultsMeta.loading}
          <span>Loading…</span>
        {:else if $resultsMeta.error}
          <span data-request-error class="text-bad">{$resultsMeta.error}</span>
          <button type="button" class="px-4 min-h-11 border border-line rounded-lg text-ink cursor-pointer" onclick={retryRequest}>Retry</button>
        {:else}
          <span>
            {resultsCountLabel($mode.mode, $pins.length, $resultsMeta.total)}
            {#if $mode.mode === "collection" && collectionCurrent} · {collectionCopy.count}{/if}
            {#if $resultsMeta.durationMs && $mode.mode !== "library"}
              · in {Math.round($resultsMeta.durationMs)} ms
            {/if}
          </span>
          {#if ["search", "similar", "byimage"].includes($mode.mode)}
            <span>Match scores are relative, not probabilities.</span>
          {/if}
        {/if}
      </div>

      {#if $mode.mode === "collection" && collectionCurrent && !$resultsMeta.loading && !$resultsMeta.error && $pins.length === 0}
        <section data-results class="px-5 sm:px-9 pb-16 pt-4" aria-label="Results">
          <p data-results-empty class="text-muted mb-3">{emptyMessage}</p>
          {#if collectionCopy.action === "browse"}
            <p class="text-sm text-muted mb-3">Save media from cards, the lightbox, or Feed.</p>
            <button class="px-4 py-2 bg-ink text-white rounded-lg cursor-pointer" onclick={setLibrary}>Browse library</button>
          {:else if collectionCopy.action === "show"}
            <p class="text-sm text-muted mb-3">This turns on the global NSFW setting, showing hidden items throughout the library.</p>
            <button class="px-4 py-2 bg-ink text-white rounded-lg cursor-pointer" onclick={() => includeNSFW.set(true)}>Show hidden items</button>
          {/if}
        </section>
      {:else if !resultsCurrent || !$resultsMeta.error || $pins.length > 0}
        {#if $mode.mode === "collection" && collectionCurrent && !$resultsMeta.loading && !$resultsMeta.error && collectionCopy.action === "show"}
          <p class="px-5 sm:px-9 my-2 text-sm text-muted">{collectionCopy.message} Enable the global NSFW setting to show them.</p>
        {/if}
        <Masonry
          pins={resultsCurrent ? $pins : []}
          loading={!resultsCurrent || $resultsMeta.loading}
          {emptyMessage}
          canLoadMore={resultsCurrent && !$resultsMeta.error && canLoadMore}
          {loadingMore}
          onLoadMore={loadMore}
        />
      {/if}
    {/if}
  </div>
</div>

{#if resultsCurrent}<Lightbox />{/if}
<Upload />
<Feed />
{#if $collectionPickerTarget}
  {#key $collectionPickerTarget}
    <CollectionPicker pin={$collectionPickerTarget} />
  {/key}
{/if}
