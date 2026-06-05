<script lang="ts">
  import { stats, topTags } from "../lib/stores";
  import { formatCount } from "../lib/utils";

  function safeCount(value: number | undefined | null): number {
    return Number.isFinite(value) ? Number(value) : 0;
  }

  const standaloneImages = $derived(safeCount($stats?.standalone_images_total ?? $stats?.images_total));
  const videos = $derived(safeCount($stats?.videos_total));
  const videoFrames = $derived(safeCount($stats?.video_frame_images_total));
  const totalMedia = $derived(standaloneImages + videos);
  const processingTotal = $derived(safeCount($stats?.queue?.total));
  const processed = $derived(safeCount($stats?.queue?.done));
  const pending = $derived(safeCount($stats?.queue?.pending));
  const active = $derived(safeCount($stats?.queue?.leased));
  const ready = $derived(safeCount($stats?.queue?.runnable));
  const failed = $derived(safeCount($stats?.queue?.failed));
  const missing = $derived(safeCount($stats?.queue?.missing));
  const annotationsMissing = $derived(safeCount($stats?.queue?.annotations_missing));
  const progressPercent = $derived(
    processingTotal > 0 ? Math.max(0, Math.min(100, Math.round((processed / processingTotal) * 100))) : 0,
  );
</script>

{#if $stats}
  <section
    data-stats-pane
    aria-labelledby="statistics-pane-title"
    class="mx-5 sm:mx-9 mt-4 rounded-card border border-line bg-surface shadow-card overflow-hidden"
  >
    <div class="px-4 sm:px-5 py-4 border-b border-line bg-surface-2/55 flex flex-col sm:flex-row sm:items-end gap-1 sm:gap-3">
      <div class="flex-1 min-w-0">
        <p class="m-0 text-[11px] font-semibold uppercase tracking-[0.08em] text-accent-strong">Statistics</p>
        <h2 id="statistics-pane-title" class="m-0 mt-1 font-display text-[20px] leading-tight text-ink">Library health</h2>
      </div>
      <p class="m-0 text-[12.5px] text-muted-2">
        {formatCount(totalMedia)} media items · {formatCount(processed)} indexed
      </p>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-3 divide-y lg:divide-y-0 lg:divide-x divide-line">
      <article class="px-4 sm:px-5 py-4">
        <p class="m-0 text-[12px] font-semibold uppercase tracking-[0.07em] text-muted-2">Media ingested</p>
        <div class="mt-3 grid grid-cols-2 gap-2">
          <div>
            <p class="m-0 text-[24px] font-semibold leading-none text-ink tabular-nums">{formatCount(standaloneImages)}</p>
            <p class="m-0 mt-1 text-[12.5px] text-muted">images</p>
          </div>
          <div>
            <p class="m-0 text-[24px] font-semibold leading-none text-accent tabular-nums">{formatCount(videos)}</p>
            <p class="m-0 mt-1 text-[12.5px] text-muted">videos</p>
          </div>
        </div>
        <p class="m-0 mt-3 text-[12.5px] text-muted-2">
          {formatCount(videoFrames)} video frames sampled for search.
        </p>
      </article>

      <article class="px-4 sm:px-5 py-4">
        <p class="m-0 text-[12px] font-semibold uppercase tracking-[0.07em] text-muted-2">Image processing</p>
        <p class="m-0 mt-3 text-[24px] font-semibold leading-none text-ink tabular-nums">
          {formatCount(processed)} / {formatCount(processingTotal)} processed
        </p>
        <div class="mt-3 h-2 rounded-full bg-bg-2 overflow-hidden" aria-hidden="true">
          <div class="h-full rounded-full bg-accent" style:width={`${progressPercent}%`}></div>
        </div>
        <div class="mt-3 flex flex-wrap gap-1.5 text-[12px] text-muted-2">
          <span class="px-2 py-1 rounded-full bg-bg-2">{formatCount(pending)} queued</span>
          <span class="px-2 py-1 rounded-full bg-bg-2">{formatCount(active)} active</span>
          <span class="px-2 py-1 rounded-full bg-bg-2">{formatCount(ready)} ready</span>
          <span class="px-2 py-1 rounded-full {failed > 0 ? 'bg-[#f7dddd] text-bad' : 'bg-bg-2'}">{formatCount(failed)} failed</span>
        </div>
        {#if missing > 0 || annotationsMissing > 0}
          <p class="m-0 mt-3 text-[12.5px] text-muted-2">
            {#if missing > 0}{formatCount(missing)} missing index jobs{/if}{#if missing > 0 && annotationsMissing > 0} · {/if}{#if annotationsMissing > 0}{formatCount(annotationsMissing)} need annotations{/if}
          </p>
        {/if}
      </article>

      <article class="px-4 sm:px-5 py-4">
        <p class="m-0 text-[12px] font-semibold uppercase tracking-[0.07em] text-muted-2">Most frequent tags</p>
        {#if $topTags.length > 0}
          <div class="mt-3 flex flex-wrap gap-1.5">
            {#each $topTags.slice(0, 6) as entry (entry.tag)}
              <span class="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-full bg-accent-soft text-accent-strong text-[12.5px] font-medium leading-none">
                <span>{entry.tag}</span>
                <span class="text-[11px] text-accent/75 tabular-nums">{formatCount(entry.count)}</span>
              </span>
            {/each}
          </div>
        {:else}
          <p class="m-0 mt-3 text-[12.5px] text-muted-2">Collecting tags after the first page loads.</p>
        {/if}
      </article>
    </div>
  </section>
{/if}
