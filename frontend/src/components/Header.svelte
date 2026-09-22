<script lang="ts">
  import {
    mode,
    headline,
    includeNSFW,
    libraryMedia,
    librarySort,
    openUpload,
    setLibrary,
    setCollections,
    stats,
  } from "../lib/stores";
  import { filterSummary } from "../lib/searchPresentation";
  import { formatCount } from "../lib/utils";

  let filtersOpen = $state(false);
  const showFilters = $derived(
    $mode.mode !== "settings" && $mode.mode !== "stats",
  );
  const scope = $derived(
    $mode.mode === "search"
      ? "Text search · Library"
      : $mode.mode === "tag"
        ? "Browse tag · Library"
        : $mode.mode === "byimage"
          ? "Image search · Library"
          : $mode.mode === "similar"
            ? "Similar media · Library"
            : "",
  );
  const control =
    "inline-flex min-h-11 items-center gap-2 px-3 border border-line-2 rounded-xl bg-surface text-ink-2 text-[13px]";
</script>

<header class="px-5 sm:px-9 pt-2 sm:pt-6">
  <div class="flex items-center gap-3 min-h-11">
    <div class="flex-1 min-w-0">
      {#if scope}
        <p data-search-scope class="m-0 text-xs text-muted">{scope}</p>
      {:else if $mode.mode === "collection"}
        <button
          class="text-xs text-accent min-h-11 cursor-pointer"
          onclick={setCollections}>← Collections</button
        >
      {/if}
      <h1
        class="font-display m-0 text-[22px] sm:text-[30px] font-semibold leading-tight break-words"
      >
        {$headline}
      </h1>
    </div>
    <button
      type="button"
      data-upload-trigger
      onclick={openUpload}
      class="min-h-11 px-4 bg-ink text-white rounded-full text-sm cursor-pointer"
      >Upload</button
    >
  </div>
  {#if $mode.mode === "library" && $stats}
    <p data-library-totals class="hidden sm:block mt-2 mb-0 text-xs text-muted">
      Library totals · {formatCount($stats.images)} images · {formatCount(
        $stats.videos,
      )} videos
    </p>
  {/if}
  {#if showFilters}
    <div class="flex items-center gap-2 min-h-11 mt-1">
      <button
        type="button"
        class="sm:hidden min-h-11 min-w-11 text-sm font-semibold text-accent cursor-pointer"
        aria-expanded={filtersOpen}
        aria-controls="filter-options"
        onclick={() => (filtersOpen = !filtersOpen)}>Filters</button
      >
      <p data-filter-summary class="m-0 text-[12px] text-muted">
        {filterSummary($mode.mode, $libraryMedia, $librarySort, $includeNSFW)}
      </p>
    </div>
    <div
      id="filter-options"
      data-filter-options
      class="filter-options flex-wrap items-center gap-2 pb-2"
      class:expanded={filtersOpen}
    >
      {#if $mode.mode === "library"}
        <label class={control}>
          <span>Media</span>
          <select
            data-library-media
            bind:value={$libraryMedia}
            aria-label="Filter library media"
            class="min-h-11 bg-transparent font-semibold cursor-pointer"
          >
            <option value="all">Images + videos</option>
            <option value="images">Images only</option>
            <option value="videos">Videos only</option>
          </select>
        </label>
        <label class={control}>
          <span>Sort</span>
          <select
            data-library-sort
            bind:value={$librarySort}
            aria-label="Sort library"
            class="min-h-11 bg-transparent font-semibold cursor-pointer"
          >
            <option value="random">Random</option>
            <option value="newest">Recently added</option>
            <option value="captured">Date taken</option>
          </select>
        </label>
      {/if}
      <label class="{control} cursor-pointer">
        <input
          type="checkbox"
          data-nsfw-toggle
          bind:checked={$includeNSFW}
          class="w-4 h-4 accent-accent"
        />
        <span>{$includeNSFW ? "NSFW: Shown" : "NSFW: Hidden"}</span>
      </label>
      {#if ["search", "tag", "similar", "byimage"].includes($mode.mode)}
        <p class="text-xs text-muted m-0">
          Type and sort are available in <button
            type="button"
            class="min-h-11 underline cursor-pointer"
            onclick={setLibrary}>Library</button
          > only.
        </p>
      {/if}
    </div>
  {/if}
</header>

<style>
  .filter-options {
    display: none;
  }
  .filter-options.expanded {
    display: flex;
  }
  @media (min-width: 640px) {
    .filter-options {
      display: flex;
    }
  }
</style>
