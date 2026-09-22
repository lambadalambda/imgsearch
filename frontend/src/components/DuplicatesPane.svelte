<script lang="ts">
  import { tick, untrack } from "svelte";
  import ConfirmDialog from "./ConfirmDialog.svelte";
  import DuplicateGroup from "./DuplicateGroup.svelte";
  import { listDuplicates, consolidateDuplicates } from "../lib/api";
  import { includeNSFW, bumpDataEpoch, refreshCollections } from "../lib/stores";
  import { formatCount } from "../lib/utils";
  import type { DuplicatesResponse, ImageRecord, DuplicateConsolidationRequest, DuplicateConsolidationPreview } from "../lib/types";

  let loading = $state(true);
  let error = $state("");
  let distance = $state(4);
  let result = $state<DuplicatesResponse | null>(null);
  let busy = $state(false);
  let conflicts = $state<string[]>([]);
  let conflictPanel = $state<HTMLElement | null>(null);
  let pending = $state<{ request: DuplicateConsolidationRequest; preview: DuplicateConsolidationPreview; keeper: ImageRecord } | null>(null);
  let generation = 0;

  async function load(options: { distance: number; includeNSFW: boolean }): Promise<void> {
    const current = ++generation;
    loading = true;
    error = "";
    pending = null;
    conflicts = [];
    try {
      const next = await listDuplicates(options);
      if (current === generation) result = next;
    } catch (err) {
      if (current === generation) { result = null; error = err instanceof Error ? err.message : "Failed to load possible matches"; }
    } finally {
      if (current === generation) loading = false;
    }
  }

  $effect(() => {
    const options = { distance, includeNSFW: $includeNSFW };
    untrack(() => { void load(options); });
  });

  async function review(items: ImageRecord[], keeper: ImageRecord): Promise<void> {
    const current = generation;
    busy = true;
    error = "";
    conflicts = [];
    const request = { keeper_id: keeper.image_id, group_ids: items.map(item => item.image_id), distance: result?.max_distance ?? distance, include_nsfw: $includeNSFW };
    try {
      const preview = await consolidateDuplicates(request);
      if (current !== generation) return;
      conflicts = preview.conflicts;
      if (conflicts.length === 0) pending = { request, preview, keeper };
      else {
        await tick();
        conflictPanel?.focus();
      }
    } catch (err) {
      if (current === generation) error = err instanceof Error ? err.message : "Preview failed";
    } finally {
      busy = false;
    }
  }

  async function confirm(): Promise<void> {
    const reviewed = pending;
    pending = null;
    if (!reviewed) return;
    busy = true;
    error = "";
    try {
      const completed = await consolidateDuplicates({ ...reviewed.request, confirm: true, preview_token: reviewed.preview.preview_token });
      const removed = new Set(completed.removed_ids);
      if (result) result = { ...result, groups: result.groups.map(group => ({ items: group.items.filter(item => !removed.has(item.image_id)) })).filter(group => group.items.length > 1) };
      bumpDataEpoch();
      refreshCollections();
    } catch (err) {
      error = err instanceof Error ? err.message : "Consolidation failed; review again";
    } finally {
      busy = false;
    }
  }
</script>

<section data-duplicates-pane aria-labelledby="duplicates-pane-title" class="mx-5 sm:mx-9 mt-4 mb-10 flex flex-col gap-4">
  <header class="px-1 sm:px-0 pt-2 flex flex-wrap items-end gap-3 justify-between">
    <div>
      <p class="m-0 text-[11px] font-semibold uppercase tracking-[0.08em] text-accent-strong">Duplicates</p>
      <h2 id="duplicates-pane-title" class="m-0 mt-1 font-display text-[22px] leading-tight text-ink">Possible matches</h2>
      <p class="m-0 mt-1 text-[13px] text-muted-2">dHash measures visual similarity, not byte or pixel equality. Colors, edits and fine detail may differ—even with an identical visual hash. Groups can include transitive matches.</p>
      <p class="m-0 mt-1 text-[12.5px] text-muted-2">Manual cleanup only; the separate unattended-import policy is unchanged.</p>
      {#if result}<p class="m-0 mt-1 text-[12.5px] text-muted-2">{formatCount(result.groups.length)} groups across {formatCount(result.scanned)} hashed images{result.unhashed ? ` · ${formatCount(result.unhashed)} without a usable hash` : ""}</p>{/if}
    </div>
    <label class="inline-flex items-center gap-2 text-[13px] text-ink-2">
      <span>Sensitivity</span>
      <select data-duplicates-distance bind:value={distance} disabled={busy || !!pending} class="bg-surface border border-line-2 rounded-full px-3 py-2 text-ink font-semibold text-[13px] cursor-pointer">
        <option value={0}>Identical visual hash</option>
        <option value={2}>Strict</option>
        <option value={4}>Normal</option>
        <option value={7}>Loose</option>
      </select>
    </label>
  </header>
  {#if error}
    <div role="alert" data-duplicates-error class="px-3 py-2 rounded bg-bad/10 text-bad text-[13px]">
      {error}
      <button type="button" disabled={busy} onclick={() => void load({ distance, includeNSFW: $includeNSFW })} class="underline ml-2">Reload matches</button>
    </div>
  {/if}
  {#if conflicts.length}
    <div bind:this={conflictPanel} tabindex="-1" data-duplicate-conflicts role="alert" class="px-3 py-2 rounded bg-bad/10 text-bad text-[13px]">
      <strong>Resolve metadata conflicts before confirming</strong>
      <ul>{#each conflicts as conflict}<li>{conflict}</li>{/each}</ul>
      <p>Nothing has changed. Open the images to edit their metadata, then review consolidation again.</p>
    </div>
  {/if}
  {#if busy}<p role="status" class="m-0 text-[13px] text-muted">Checking and consolidating…</p>{/if}
  {#if loading}
    <p class="m-0 text-[13px] text-muted">Scanning…</p>
  {:else if result && result.groups.length === 0}
    <p data-duplicates-empty class="m-0 text-[13px] text-muted">No possible matches at this sensitivity.</p>
  {:else if result}
    {#each result.groups as group, index (group.items.map(item => item.image_id).join(","))}
      <DuplicateGroup items={group.items} {index} disabled={busy || !!pending} onreview={review} />
    {/each}
  {/if}
</section>

{#if pending}
  <ConfirmDialog
    title={`Keep ${pending.keeper.original_name}?`}
    detail={`Keep ${pending.keeper.width}×${pending.keeper.height}; remove ${pending.preview.removed_ids.length} other library entries. Preserve memberships in ${pending.preview.affected_collections} affected collections.`}
    note="Original files and thumbnails are retained on disk for safe recovery; this does not free disk space. Collection memberships and compatible manual metadata are merged. Generated annotations stay with the keeper. Library removal cannot be undone here."
    confirmLabel="Consolidate library entries"
    onconfirm={confirm}
    oncancel={() => (pending = null)}
  />
{/if}
