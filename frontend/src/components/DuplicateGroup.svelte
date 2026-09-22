<script lang="ts">
  import { lightboxPin } from "../lib/stores";
  import { fallbackToOriginal, mediaUrl, pinFromImage } from "../lib/utils";
  import { duplicateFileInfo, resolutionLabel } from "../lib/duplicates";
  import type { ImageRecord } from "../lib/types";

  let { items, index, disabled, onreview }: {
    items: ImageRecord[];
    index: number;
    disabled: boolean;
    onreview: (items: ImageRecord[], keeper: ImageRecord) => void;
  } = $props();
  let selectedID = $state<number | null>(null);
  let comparisonID = $state<number | null>(null);
  let flip = $state(false);
  let showOther = $state(false);
  let keeper = $derived(items.find(item => item.image_id === selectedID) ?? items[0]);
  let others = $derived(items.filter(item => item.image_id !== keeper.image_id));
  let comparison = $derived(others.find(item => item.image_id === comparisonID) ?? others[0]);
  let compared = $derived(flip ? [showOther ? comparison : keeper] : [keeper, comparison]);
</script>

<article data-duplicate-group={index} class="rounded-card border border-line bg-surface shadow-card p-3 sm:p-5 flex flex-col gap-3 min-w-0">
  <div class="flex flex-wrap items-center justify-between gap-2">
    <h3 class="m-0 text-[13px] font-semibold text-ink">{items.length} Possible matches</h3>
    <button type="button" data-duplicate-review {disabled} onclick={() => onreview(items, keeper)} class="px-3 py-2 bg-surface text-ink border border-line-2 rounded-full text-[13px] cursor-pointer disabled:opacity-50">
      Review consolidation
    </button>
  </div>
  <fieldset disabled={disabled} class="m-0 p-0 border-0 min-w-0">
    <legend class="text-[13px] text-muted mb-2">Choose the image to keep. The most pixels is only a starting suggestion.</legend>
    <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
      {#each items as item (item.image_id)}
        <figure data-duplicate-item={item.image_id} class="m-0 min-w-0 flex flex-col gap-1.5">
          <button type="button" onclick={() => lightboxPin.set(pinFromImage(item))} aria-label={`Open ${item.original_name}`} class="block p-0 border border-line rounded-[10px] overflow-hidden bg-bg-2 cursor-pointer w-full aspect-[4/3]">
            <img src={mediaUrl(item.thumbnail_path || item.storage_path)} onerror={(event) => fallbackToOriginal(event.currentTarget, mediaUrl(item.storage_path))} alt={item.title || item.original_name} loading="lazy" class="block w-full h-full object-contain" />
          </button>
          <figcaption class="flex flex-col gap-1 text-[12px] text-ink-2 break-words min-w-0">
            <span class="font-medium text-ink truncate" title={item.original_name}>{item.original_name}</span>
            <span class="tabular-nums text-muted">{item.width}×{item.height}</span>
            {#if resolutionLabel(item, items)}<span class="text-muted">{resolutionLabel(item, items)}</span>{/if}
            <span class="text-muted">{duplicateFileInfo(item)}</span>
            <label class="flex items-center gap-2 min-h-11 cursor-pointer">
              <input type="radio" name={`keeper-${items[0].image_id}`} value={item.image_id} checked={keeper.image_id === item.image_id} onchange={() => { selectedID = item.image_id; showOther = false; }} aria-label={`Keep ${item.original_name}`} />
              {#if keeper.image_id === item.image_id}<strong data-keep-badge class="text-accent-strong">Keep</strong>{:else}<span>Keep this</span>{/if}
            </label>
          </figcaption>
        </figure>
      {/each}
    </div>
  </fieldset>
  <section aria-label={`Compare possible matches in group ${index + 1}`} class="border-t border-line pt-3 min-w-0">
    <div class="flex flex-wrap items-center gap-2 mb-2 text-[13px]">
      <label class="flex items-center gap-2 min-w-0 max-w-full">Compare with
        <select aria-label={`Compare keeper with in group ${index + 1}`} value={comparison.image_id} onchange={event => { comparisonID = Number(event.currentTarget.value); showOther = false; }} class="min-w-0 max-w-[180px] bg-surface border border-line-2 rounded px-2 py-2">
          {#each others as item}<option value={item.image_id}>{item.original_name}</option>{/each}
        </select>
      </label>
      <button type="button" aria-pressed={!flip} onclick={() => { flip = false; }} class="px-3 py-2 border border-line-2 rounded-full">Two-up comparison</button>
      <button type="button" aria-pressed={flip} onclick={() => { flip = true; }} class="px-3 py-2 border border-line-2 rounded-full">Flip comparison</button>
      {#if flip}<button type="button" onclick={() => { showOther = !showOther; }} class="px-3 py-2 border border-line-2 rounded-full">Show {showOther ? "keeper" : "comparison"}</button>{/if}
    </div>
    <p class="text-[12px] text-muted m-0 mb-2">Full images, uncropped. Compare colors, edits and fine detail before removing library entries.</p>
    <div data-duplicate-compare class="grid gap-2 {flip ? 'grid-cols-1' : 'grid-cols-2'}" aria-live="polite">
      {#each compared as item (item.image_id)}
        <figure class="m-0 min-w-0">
          <img src={mediaUrl(item.storage_path)} alt={`${item.image_id === keeper.image_id ? 'Keeper' : 'Comparison'}: ${item.original_name}`} loading="lazy" class="block w-full aspect-square max-h-[420px] object-contain bg-bg-2 rounded" />
          <figcaption class="text-[12px] text-muted break-all mt-1">{item.image_id === keeper.image_id ? "Keep" : "Compare"} · {item.original_name}</figcaption>
        </figure>
      {/each}
    </div>
  </section>
</article>
