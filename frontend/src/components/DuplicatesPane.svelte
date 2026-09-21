<script lang="ts">
  import { onMount } from "svelte";
  import ConfirmDialog from "./ConfirmDialog.svelte";
  import { listDuplicates, deleteMedia, ApiError } from "../lib/api";
  import { includeNSFW, lightboxPin, bumpDataEpoch } from "../lib/stores";
  import { mediaUrl, pinFromImage, formatCount } from "../lib/utils";
  import type { DuplicatesResponse, ImageRecord } from "../lib/types";

  let loading = $state(true);
  let error = $state("");
  let distance = $state(4);
  let result = $state<DuplicatesResponse | null>(null);
  let busyIds = $state<Set<number>>(new Set());
  let pendingDelete = $state<{ label: string; detail: string; ids: number[] } | null>(null);

  async function load(): Promise<void> {
    loading = true;
    error = "";
    try {
      result = await listDuplicates({ distance, includeNSFW: $includeNSFW });
    } catch (err) {
      error = err instanceof ApiError || err instanceof Error ? err.message : "failed to load duplicates";
    } finally {
      loading = false;
    }
  }

  onMount(() => {
    void load();
  });

  function pixels(item: ImageRecord): number {
    return (item.width || 0) * (item.height || 0);
  }

  function askDelete(items: ImageRecord[], label: string): void {
    if (items.length === 0) return;
    pendingDelete = {
      label,
      detail: items.length === 1 ? items[0].original_name : `${items.length} images`,
      ids: items.map((item) => item.image_id),
    };
  }

  async function confirmDelete(): Promise<void> {
    const request = pendingDelete;
    pendingDelete = null;
    if (!request || !result) return;
    busyIds = new Set([...busyIds, ...request.ids]);
    const removed = new Set<number>();
    try {
      for (const id of request.ids) {
        await deleteMedia("image", id);
        removed.add(id);
      }
    } catch (err) {
      error = err instanceof ApiError || err instanceof Error ? err.message : "delete failed";
    } finally {
      busyIds = new Set([...busyIds].filter((id) => !request.ids.includes(id)));
    }
    if (removed.size > 0) {
      result = {
        ...result,
        groups: result.groups
          .map((group) => ({ items: group.items.filter((item) => !removed.has(item.image_id)) }))
          .filter((group) => group.items.length > 1),
      };
      bumpDataEpoch();
    }
  }

  function open(item: ImageRecord): void {
    lightboxPin.set(pinFromImage(item));
  }
</script>

<section data-duplicates-pane aria-labelledby="duplicates-pane-title" class="mx-5 sm:mx-9 mt-4 mb-10 flex flex-col gap-4">
  <header class="px-1 sm:px-0 pt-2 flex flex-wrap items-end gap-3 justify-between">
    <div>
      <p class="m-0 text-[11px] font-semibold uppercase tracking-[0.08em] text-accent-strong">Duplicates</p>
      <h2 id="duplicates-pane-title" class="m-0 mt-1 font-display text-[22px] leading-tight text-ink">Near-duplicate images</h2>
      <p class="m-0 mt-1 text-[12.5px] text-muted-2">
        {#if result}
          {formatCount(result.groups.length)} groups across {formatCount(result.scanned)} hashed images{result.unhashed
            ? ` · ${formatCount(result.unhashed)} not hashed yet`
            : ""}
        {:else}
          Groups of pictures whose perceptual hashes are within a few bits of each other.
        {/if}
      </p>
    </div>
    <label class="inline-flex items-center gap-2 text-[13px] text-ink-2">
      <span>Sensitivity</span>
      <select
        data-duplicates-distance
        bind:value={distance}
        onchange={() => void load()}
        class="bg-surface border border-line-2 rounded-full px-3 py-1.5 text-ink font-semibold text-[13px] outline-none cursor-pointer"
      >
        <option value={0}>Exact only</option>
        <option value={2}>Strict</option>
        <option value={4}>Normal</option>
        <option value={7}>Loose</option>
      </select>
    </label>
  </header>

  {#if error}
    <p data-duplicates-error role="alert" class="m-0 px-3 py-2 rounded-[10px] bg-bad/10 text-bad text-[13px]">{error}</p>
  {/if}

  {#if loading}
    <p class="m-0 text-[13px] text-muted">Scanning…</p>
  {:else if result && result.groups.length === 0}
    <p data-duplicates-empty class="m-0 text-[13px] text-muted">No near-duplicates found at this sensitivity.</p>
  {:else if result}
    <div class="flex flex-col gap-4">
      {#each result.groups as group, index (group.items.map((item) => item.image_id).join(","))}
        {@const largest = group.items[0]}
        <article
          data-duplicate-group={index}
          class="rounded-card border border-line bg-surface shadow-card p-4 sm:p-5 flex flex-col gap-3"
        >
          <div class="flex flex-wrap items-center justify-between gap-2">
            <p class="m-0 text-[12px] font-semibold uppercase tracking-[0.07em] text-muted-2">
              {group.items.length} copies
            </p>
            <button
              type="button"
              data-duplicate-keep-largest
              disabled={group.items.some((item) => busyIds.has(item.image_id))}
              onclick={() => askDelete(group.items.slice(1), `Keep ${largest.width}×${largest.height} and delete the rest?`)}
              class="px-[12px] py-[7px] bg-surface text-ink-2 border border-line-2 rounded-full text-[12.5px] font-medium leading-none cursor-pointer transition-colors duration-150 ease-soft hover:bg-surface-2 disabled:opacity-50 disabled:cursor-default"
            >
              Keep largest
            </button>
          </div>
          <div class="flex flex-wrap gap-3">
            {#each group.items as item (item.image_id)}
              <figure
                data-duplicate-item={item.image_id}
                class="m-0 w-[180px] flex flex-col gap-1.5 {busyIds.has(item.image_id) ? 'opacity-50' : ''}"
              >
                <button
                  type="button"
                  onclick={() => open(item)}
                  class="block p-0 border border-line rounded-[10px] overflow-hidden bg-bg-2 cursor-pointer w-full aspect-[4/3]"
                  aria-label={`Open ${item.original_name}`}
                >
                  <img src={mediaUrl(item.storage_path)} alt={item.title || item.original_name} loading="lazy" class="block w-full h-full object-cover" />
                </button>
                <figcaption class="flex flex-col gap-1.5 text-[12px] leading-snug text-ink-2 break-all">
                  <span class="block font-medium text-ink truncate" title={item.original_name}>{item.original_name}</span>
                  <span class="tabular-nums text-muted">{item.width}×{item.height}{item === largest ? " · largest" : ""}</span>
                  <button
                    type="button"
                    data-duplicate-delete={item.image_id}
                    disabled={busyIds.has(item.image_id)}
                    onclick={() => askDelete([item], "Delete this image?")}
                    class="self-start px-[10px] py-[6px] bg-surface text-bad border border-bad/40 rounded-full text-[12px] font-medium leading-none cursor-pointer transition-colors duration-150 ease-soft hover:bg-[color-mix(in_oklab,#f4d8d6_60%,white_40%)] disabled:opacity-50 disabled:cursor-default"
                  >
                    Delete
                  </button>
                </figcaption>
              </figure>
            {/each}
          </div>
        </article>
      {/each}
    </div>
  {/if}
</section>

{#if pendingDelete}
  <ConfirmDialog
    title={pendingDelete.label}
    detail={pendingDelete.detail}
    onconfirm={confirmDelete}
    oncancel={() => (pendingDelete = null)}
  />
{/if}
