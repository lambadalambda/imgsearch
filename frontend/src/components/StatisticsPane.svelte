<script lang="ts">
  import { onMount } from "svelte";
  import { stats, topTags } from "../lib/stores";
  import { refreshStats } from "../lib/stats";
  import { retryFailedKind } from "../lib/api";
  import { healthCards, capabilityCopy } from "../lib/health";
  import { formatCount } from "../lib/utils";

  let refreshError = $state(false);
  let refreshing = $state(false);
  let recovery = $state<Record<string, { busy: boolean; error?: boolean; message: string }>>({});
  const cards = $derived($stats ? healthCards($stats) : []);
  const standalone = $derived($stats?.standalone_images_total ?? Math.max(0, ($stats?.images_total ?? 0) - ($stats?.video_frame_images_total ?? 0)));
  const totalMedia = $derived(standalone + ($stats?.videos_total ?? 0));
  const failures = $derived($stats?.recent_failures ?? []);
  let controller: AbortController | undefined;

  async function refreshHealth() {
    if (refreshing) return;
    refreshing = true;
    try {
      await refreshStats(controller?.signal);
      refreshError = false;
    } catch (error) {
      if ((error as Error).name !== "AbortError") refreshError = true;
    } finally { refreshing = false; }
  }

  onMount(() => {
    controller = new AbortController();
    const refresh = () => { if (document.visibilityState === "visible") void refreshHealth(); };
    refresh();
    const timer = window.setInterval(refresh, 3000);
    document.addEventListener("visibilitychange", refresh);
    return () => {
      controller?.abort();
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", refresh);
    };
  });

  async function retry(card: (typeof cards)[number]) {
    if (recovery[card.kind]?.busy) return;
    recovery[card.kind] = { busy: true, message: "Returning failed jobs to Waiting…" };
    try {
      const result = await retryFailedKind(card.kind);
      recovery[card.kind] = { busy: false, message: `${result.retried} failed ${card.action} ${result.retried === 1 ? "job" : "jobs"} returned to Waiting. No missing jobs were created.` };
      void refreshHealth();
    } catch {
      recovery[card.kind] = { busy: false, error: true, message: `Couldn’t retry failed ${card.action}. Try again; no missing jobs will be created.` };
    }
  }
</script>

<section data-stats-pane aria-labelledby="statistics-pane-title" class="mx-5 sm:mx-9 mt-4 flex flex-col gap-4 pb-8">
  <header class="pt-2">
    <p class="m-0 text-xs font-semibold uppercase text-accent-strong">Statistics</p>
    <h2 id="statistics-pane-title" class="m-0 mt-1 font-display text-[22px] text-ink">Library health</h2>
    {#if $stats}
      <p class="mt-1 text-sm text-muted-2">{formatCount(totalMedia)} media items · sampled video frames counted separately below</p>
    {/if}
    <button class="min-h-11 px-3 rounded-lg border border-line text-sm cursor-pointer" disabled={refreshing} onclick={refreshHealth}>
      {refreshError ? "Retry health refresh" : "Refresh health"}
    </button>
    {#if refreshError}
      <p role="status" class="text-sm text-bad">{$stats ? "Couldn’t refresh library health. Showing the last snapshot." : "Couldn’t load library health. Check your connection, then retry."}</p>
    {:else if !$stats}
      <p role="status" class="text-sm text-muted">Loading library health…</p>
    {/if}
  </header>

  {#if $stats}
    <article class="rounded-card border border-line bg-surface p-4 sm:p-5">
      <h3 class="m-0 text-sm font-semibold text-ink">Media ingested</h3>
      <p class="mt-2 text-sm text-muted-2">{formatCount(standalone)} standalone images · {formatCount($stats.videos_total)} videos · {formatCount($stats.video_frame_images_total ?? 0)} sampled video frames</p>
      <p class="mt-2 text-sm text-muted-2">
        {#if $stats.capabilities?.worker === "local"}
          Processing is configured in this server. Configuration is not a worker heartbeat.
        {:else}
          Worker configuration unknown. This API cannot confirm a separate worker’s configuration or whether it is running.
        {/if}
        If Waiting does not change, check that the worker is running and review its logs.
      </p>
    </article>

    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
      {#each cards as card (card.kind)}
        <article id={`health-${card.kind}`} data-health-kind={card.kind} class="min-w-0 rounded-card border border-line bg-surface shadow-card p-4 sm:p-5">
          <h3 class="m-0 text-sm font-semibold text-ink">{card.title}</h3>
          <p class="mt-2 text-xs font-medium text-accent-strong">{capabilityCopy(card.capability)}</p>
          <p class="mt-3 text-[24px] font-semibold tabular-nums text-ink">{formatCount(card.complete)} / {formatCount(card.total)} Complete</p>
          <p class="mt-1 text-sm text-muted-2">{card.scope}</p>
          <div class="mt-3 h-2 rounded-full bg-bg-2 overflow-hidden" aria-hidden="true">
            <div class="h-full rounded-full bg-accent" style:width={`${card.percent}%`}></div>
          </div>
          <div class="mt-3 flex flex-wrap gap-2 text-sm text-muted-2">
            <span>{formatCount(card.waiting)} Waiting</span>
            <span>{card.processing === undefined ? "Unknown" : formatCount(card.processing)} Processing</span>
            <span class:text-bad={card.failed > 0}>{formatCount(card.failed)} Failed</span>
          </div>
          <p class="mt-2 text-xs text-muted-2">{card.note}</p>
          {#if card.capability === "disabled" || card.capability === "unconfigured"}
            <p class="mt-3 text-sm text-muted-2">{card.capability === "disabled" ? "Processing is switched off in this server." : "No processing backend is configured in this server."} Existing counts are historical; remaining media is not necessarily waiting for work.</p>
          {:else if card.missing > 0}
            <p class="mt-3 text-sm text-muted-2">{formatCount(card.missing)} items have no tracked job for this step. Check worker configuration and startup logs; Retry below does not create missing jobs.</p>
          {/if}
          {#if card.kind === "annotate_image" || card.kind === "annotate_video"}
            <a href="/?view=settings" class="inline-flex min-h-11 items-center text-sm text-accent-strong underline">Open annotation settings</a>
          {:else if card.kind === "transcribe_video"}
            <p class="mt-3 text-sm text-muted-2">For transcription, ask the server operator to check the worker’s Parakeet model and ONNX runtime configuration, then restart the worker. There is no transcription switch in Settings.</p>
          {:else}
            <p class="mt-3 text-sm text-muted-2">For embedding failures, check the worker’s model configuration and logs before retrying.</p>
          {/if}
          {#if card.failed > 0 && card.canRetry}
            <p class="mt-2 text-xs text-muted-2">Retry only existing failed {card.action} jobs for the current model. It does not change completed work or create missing jobs. {card.capability === "unknown" ? "Confirm the separate worker supports this step first." : ""}</p>
            <button class="mt-2 min-h-11 px-3 border border-line rounded-lg text-sm cursor-pointer disabled:opacity-50" disabled={recovery[card.kind]?.busy} onclick={() => retry(card)}>Retry failed {card.action}</button>
          {/if}
          {#if recovery[card.kind]}
            <p role="status" class="mt-2 text-sm" class:text-bad={recovery[card.kind].error}>{recovery[card.kind].message}</p>
          {/if}
          <details class="mt-3 text-xs text-muted-2">
            <summary class="min-h-11 flex items-center cursor-pointer">Advanced details</summary>
            <p>{formatCount(card.job.runnable)} ready to run · {formatCount(card.job.pending)} pending (includes delayed retries) · {formatCount(card.job.leased)} leased · {formatCount(card.job.tracked)} tracked jobs</p>
            <p class="mt-2">Ready means eligible for a worker, not complete. Waiting includes pending jobs and expired leases; Processing means an unexpired lease, not a heartbeat. These scheduler subsets overlap and must not be added together.</p>
            <p class="mt-2">Oldest eligible job: {formatCount(card.job.oldest_runnable_age_seconds)} seconds. {formatCount(card.missing)} items without a tracked job.</p>
            {#if card.kind === "transcribe_video"}<p class="mt-2">Operator flags: <code class="break-all">-parakeet-onnx-bundle-dir</code> and <code class="break-all">-parakeet-onnxruntime-lib</code>.</p>{/if}
          </details>
        </article>
      {/each}
    </div>

    <article class="rounded-card border border-line bg-surface p-4 sm:p-5">
      <h3 class="m-0 text-sm font-semibold text-ink">Recent failures</h3>
      {#if failures.length === 0}<p class="mt-2 text-sm text-muted-2">No recent failures.</p>
      {:else}
        <p class="mt-2 text-sm text-muted-2">Up to 10 jobs that exhausted retries. Check the step’s configuration before retrying.</p>
        <ul data-stats-failures class="mt-3 divide-y divide-line">
          {#each failures as failure (failure.job_id)}
            {@const card = cards.find((c) => c.kind === failure.kind)}
            <li class="py-3 min-w-0 text-sm">
              <p class="break-words text-ink">{failure.original_name || `${failure.media_type} #${failure.video_id || failure.image_id}`}</p>
              {#if card}<a href={`#health-${card.kind}`} class="inline-flex min-h-11 items-center text-accent-strong underline">Review {card.title.toLowerCase()} recovery</a>{/if}
              <details class="text-xs text-muted-2"><summary class="cursor-pointer">Advanced error details</summary><p class="mt-2 break-all">{failure.kind} · attempt {failure.attempts} · {failure.last_error}</p></details>
            </li>
          {/each}
        </ul>
      {/if}
    </article>
    <article class="rounded-card border border-line bg-surface p-4 sm:p-5">
      <h3 class="m-0 text-sm font-semibold text-ink">Most frequent tags</h3>
      <div class="mt-3 flex flex-wrap gap-2 text-sm text-muted-2">
        {#each $topTags.slice(0, 8) as entry (entry.tag)}<span>{entry.tag} · {formatCount(entry.count)}</span>{:else}<span>Collecting tags after the first page loads.</span>{/each}
      </div>
    </article>
    {#if $stats.build}<details data-stats-build class="text-xs text-muted-2"><summary class="cursor-pointer min-h-11">Advanced build details</summary><p>imgsearch {$stats.build.version} · {$stats.build.commit} · {$stats.build.date}</p></details>{/if}
  {/if}
</section>
