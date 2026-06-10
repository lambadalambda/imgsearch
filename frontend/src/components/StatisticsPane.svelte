<script lang="ts">
  import { stats, topTags } from "../lib/stores";
  import { refreshStats } from "../lib/stats";
  import { formatCount } from "../lib/utils";
  import type { JobKindStats, StatsResponse } from "../lib/types";

  const REFRESH_INTERVAL_MS = 3000;

  // The pane is a progress dashboard, so it keeps itself current while open.
  // Polling pauses while the tab is hidden and stops when the pane unmounts.
  $effect(() => {
    const refresh = () => {
      if (document.visibilityState !== "visible") return;
      void refreshStats().catch(() => {
        /* keep showing the last snapshot */
      });
    };
    refresh();
    const timer = window.setInterval(refresh, REFRESH_INTERVAL_MS);
    document.addEventListener("visibilitychange", refresh);
    return () => {
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", refresh);
    };
  });

  function safeCount(value: number | undefined | null): number {
    return Number.isFinite(value) ? Number(value) : 0;
  }

  function jobKind(value: StatsResponse["job_kinds"], kind: string): JobKindStats {
    return value?.[kind] ?? { tracked: 0, runnable: 0, pending: 0, leased: 0, done: 0, failed: 0, oldest_runnable_age_seconds: 0 };
  }

  const standaloneImages = $derived(safeCount($stats?.standalone_images_total ?? $stats?.images_total));
  const videos = $derived(safeCount($stats?.videos_total));
  const videoFrames = $derived(safeCount($stats?.video_frame_images_total));
  const totalMedia = $derived(standaloneImages + videos);

  const embed = $derived(jobKind($stats?.job_kinds, "embed_image"));
  const imageAnnotate = $derived(jobKind($stats?.job_kinds, "annotate_image"));
  const videoAnnotate = $derived(jobKind($stats?.job_kinds, "annotate_video"));
  const transcribeVideo = $derived(jobKind($stats?.job_kinds, "transcribe_video"));

  const embedExpected = $derived(safeCount($stats?.queue?.total) || safeCount($stats?.images_total));
  const imageAnnotateExpected = $derived(safeCount($stats?.image_annotation_expected));
  const videoAnnotateExpected = $derived(safeCount($stats?.video_annotation_expected));
  const transcribeVideoExpected = $derived(safeCount($stats?.video_transcription_expected));

  const recentFailures = $derived($stats?.recent_failures ?? []);

  function percent(done: number, total: number): number {
    if (total <= 0) return 0;
    return Math.max(0, Math.min(100, Math.round((done / total) * 100)));
  }
</script>

{#if $stats}
  <section
    data-stats-pane
    aria-labelledby="statistics-pane-title"
    class="mx-5 sm:mx-9 mt-4 flex flex-col gap-4"
  >
    <header class="px-1 sm:px-0 pt-2">
      <p class="m-0 text-[11px] font-semibold uppercase tracking-[0.08em] text-accent-strong">Statistics</p>
      <h2 id="statistics-pane-title" class="m-0 mt-1 font-display text-[22px] leading-tight text-ink">Library health</h2>
      <p class="m-0 mt-1 text-[12.5px] text-muted-2">
        {formatCount(totalMedia)} media items · {formatCount(embed.done)} embedded · {formatCount(imageAnnotate.done)} annotated
      </p>
    </header>

    <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
      <article class="rounded-card border border-line bg-surface shadow-card p-4 sm:p-5">
        <p class="m-0 text-[12px] font-semibold uppercase tracking-[0.07em] text-muted-2">Media ingested</p>
        <div class="mt-4 grid grid-cols-3 gap-3">
          <div>
            <p class="m-0 text-[26px] font-semibold leading-none text-ink tabular-nums">{formatCount(standaloneImages)}</p>
            <p class="m-0 mt-1.5 text-[12.5px] text-muted">standalone images</p>
          </div>
          <div>
            <p class="m-0 text-[26px] font-semibold leading-none text-accent tabular-nums">{formatCount(videos)}</p>
            <p class="m-0 mt-1.5 text-[12.5px] text-muted">videos</p>
          </div>
          <div>
            <p class="m-0 text-[26px] font-semibold leading-none text-muted-2 tabular-nums">{formatCount(videoFrames)}</p>
            <p class="m-0 mt-1.5 text-[12.5px] text-muted">video frames</p>
          </div>
        </div>
      </article>

      <article class="rounded-card border border-line bg-surface shadow-card p-4 sm:p-5">
        <p class="m-0 text-[12px] font-semibold uppercase tracking-[0.07em] text-muted-2">Image embedding</p>
        <p class="m-0 mt-3 text-[26px] font-semibold leading-none text-ink tabular-nums">
          {formatCount(embed.done)} / {formatCount(embedExpected)} processed
        </p>
        <div class="mt-3 h-2 rounded-full bg-bg-2 overflow-hidden" aria-hidden="true">
          <div class="h-full rounded-full bg-accent" style:width={`${percent(embed.done, embedExpected)}%`}></div>
        </div>
        <div class="mt-3 flex flex-wrap gap-1.5 text-[12px] text-muted-2">
          <span class="px-2 py-1 rounded-full bg-bg-2">{formatCount(embed.pending)} queued</span>
          <span class="px-2 py-1 rounded-full bg-bg-2">{formatCount(embed.leased)} active</span>
          <span class="px-2 py-1 rounded-full bg-bg-2">{formatCount(embed.runnable)} ready</span>
          <span class="px-2 py-1 rounded-full {embed.failed > 0 ? 'bg-[#f7dddd] text-bad' : 'bg-bg-2'}">{formatCount(embed.failed)} failed</span>
        </div>
        {#if safeCount($stats?.queue?.missing) > 0 || safeCount($stats?.queue?.annotations_missing) > 0}
          <p class="m-0 mt-3 text-[12.5px] text-muted-2">
            {#if safeCount($stats?.queue?.missing) > 0}{formatCount(safeCount($stats?.queue?.missing))} missing embed jobs{/if}{#if safeCount($stats?.queue?.missing) > 0 && safeCount($stats?.queue?.annotations_missing) > 0} · {/if}{#if safeCount($stats?.queue?.annotations_missing) > 0}{formatCount(safeCount($stats?.queue?.annotations_missing))} need annotations{/if}
          </p>
        {/if}
      </article>

      <article class="rounded-card border border-line bg-surface shadow-card p-4 sm:p-5">
        <p class="m-0 text-[12px] font-semibold uppercase tracking-[0.07em] text-muted-2">Image annotation</p>
        <p class="m-0 mt-3 text-[26px] font-semibold leading-none text-ink tabular-nums">
          {formatCount(imageAnnotate.done)} / {formatCount(imageAnnotateExpected)} processed
        </p>
        <div class="mt-3 h-2 rounded-full bg-bg-2 overflow-hidden" aria-hidden="true">
          <div class="h-full rounded-full bg-accent" style:width={`${percent(imageAnnotate.done, imageAnnotateExpected)}%`}></div>
        </div>
        <div class="mt-3 flex flex-wrap gap-1.5 text-[12px] text-muted-2">
          <span class="px-2 py-1 rounded-full bg-bg-2">{formatCount(imageAnnotate.pending)} queued</span>
          <span class="px-2 py-1 rounded-full bg-bg-2">{formatCount(imageAnnotate.leased)} active</span>
          <span class="px-2 py-1 rounded-full bg-bg-2">{formatCount(imageAnnotate.runnable)} ready</span>
          <span class="px-2 py-1 rounded-full {imageAnnotate.failed > 0 ? 'bg-[#f7dddd] text-bad' : 'bg-bg-2'}">{formatCount(imageAnnotate.failed)} failed</span>
        </div>
        {#if safeCount($stats?.image_annotation_missing) > 0}
          <p class="m-0 mt-3 text-[12.5px] text-muted-2">
            {formatCount(safeCount($stats?.image_annotation_missing))} images still need an annotation job.
          </p>
        {/if}
      </article>

      <article class="rounded-card border border-line bg-surface shadow-card p-4 sm:p-5">
        <p class="m-0 text-[12px] font-semibold uppercase tracking-[0.07em] text-muted-2">Video annotation</p>
        <p class="m-0 mt-3 text-[26px] font-semibold leading-none text-ink tabular-nums">
          {formatCount(videoAnnotate.done)} / {formatCount(videoAnnotateExpected)} processed
        </p>
        <div class="mt-3 h-2 rounded-full bg-bg-2 overflow-hidden" aria-hidden="true">
          <div class="h-full rounded-full bg-accent" style:width={`${percent(videoAnnotate.done, videoAnnotateExpected)}%`}></div>
        </div>
        <div class="mt-3 flex flex-wrap gap-1.5 text-[12px] text-muted-2">
          <span class="px-2 py-1 rounded-full bg-bg-2">{formatCount(videoAnnotate.pending)} queued</span>
          <span class="px-2 py-1 rounded-full bg-bg-2">{formatCount(videoAnnotate.leased)} active</span>
          <span class="px-2 py-1 rounded-full bg-bg-2">{formatCount(videoAnnotate.runnable)} ready</span>
          <span class="px-2 py-1 rounded-full {videoAnnotate.failed > 0 ? 'bg-[#f7dddd] text-bad' : 'bg-bg-2'}">{formatCount(videoAnnotate.failed)} failed</span>
        </div>
        {#if safeCount($stats?.video_annotation_missing) > 0}
          <p class="m-0 mt-3 text-[12.5px] text-muted-2">
            {formatCount(safeCount($stats?.video_annotation_missing))} videos still need an annotation job.
          </p>
        {/if}
      </article>

      <article class="rounded-card border border-line bg-surface shadow-card p-4 sm:p-5">
        <p class="m-0 text-[12px] font-semibold uppercase tracking-[0.07em] text-muted-2">Video transcription</p>
        <p class="m-0 mt-3 text-[26px] font-semibold leading-none text-ink tabular-nums">
          {formatCount(transcribeVideo.done)} / {formatCount(transcribeVideoExpected)} processed
        </p>
        <div class="mt-3 h-2 rounded-full bg-bg-2 overflow-hidden" aria-hidden="true">
          <div class="h-full rounded-full bg-accent" style:width={`${percent(transcribeVideo.done, transcribeVideoExpected)}%`}></div>
        </div>
        <div class="mt-3 flex flex-wrap gap-1.5 text-[12px] text-muted-2">
          <span class="px-2 py-1 rounded-full bg-bg-2">{formatCount(transcribeVideo.pending)} queued</span>
          <span class="px-2 py-1 rounded-full bg-bg-2">{formatCount(transcribeVideo.leased)} active</span>
          <span class="px-2 py-1 rounded-full bg-bg-2">{formatCount(transcribeVideo.runnable)} ready</span>
          <span class="px-2 py-1 rounded-full {transcribeVideo.failed > 0 ? 'bg-[#f7dddd] text-bad' : 'bg-bg-2'}">{formatCount(transcribeVideo.failed)} failed</span>
        </div>
        {#if safeCount($stats?.video_transcription_missing) > 0}
          <p class="m-0 mt-3 text-[12.5px] text-muted-2">
            {formatCount(safeCount($stats?.video_transcription_missing))} videos still need a transcription job.
          </p>
        {/if}
      </article>

      <article class="rounded-card border border-line bg-surface shadow-card p-4 sm:p-5">
        <p class="m-0 text-[12px] font-semibold uppercase tracking-[0.07em] text-muted-2">Most frequent tags</p>
        {#if $topTags.length > 0}
          <div class="mt-3 flex flex-wrap gap-1.5">
            {#each $topTags.slice(0, 8) as entry (entry.tag)}
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

    <article class="rounded-card border border-line bg-surface shadow-card p-4 sm:p-5">
      <div class="flex items-baseline justify-between gap-2 flex-wrap">
        <p class="m-0 text-[12px] font-semibold uppercase tracking-[0.07em] text-muted-2">Recent failures</p>
        {#if recentFailures.length > 0}
          <p class="m-0 text-[12px] text-muted-2">Last {Math.min(recentFailures.length, 10)} jobs that exhausted retries</p>
        {/if}
      </div>
      {#if recentFailures.length === 0}
        <p class="m-0 mt-3 text-[12.5px] text-muted-2">No recent failures.</p>
      {:else}
        <ul class="m-0 mt-3 divide-y divide-line" data-stats-failures>
          {#each recentFailures as failure (failure.job_id)}
            <li class="py-2.5 first:pt-0 last:pb-0 flex flex-col sm:flex-row sm:items-baseline gap-1 sm:gap-3 text-[13px]">
              <div class="flex items-baseline gap-2 min-w-0">
                <span class="text-[11px] font-semibold uppercase tracking-[0.07em] text-muted-2 w-24 flex-none">{failure.kind}</span>
                <span class="font-medium text-ink truncate">{failure.original_name || (failure.media_type === "video" ? `video #${failure.video_id}` : `image #${failure.image_id}`)}</span>
              </div>
              <div class="flex items-baseline gap-2 sm:ml-auto text-[12px] text-muted-2 min-w-0">
                <span>attempt {failure.attempts}</span>
                <span class="text-bad truncate max-w-[40ch] sm:max-w-[60ch]" title={failure.last_error}>{failure.last_error}</span>
              </div>
            </li>
          {/each}
        </ul>
      {/if}
    </article>
  </section>
{/if}
