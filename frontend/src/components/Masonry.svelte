<script lang="ts">
  import type { Pin as PinType } from "../lib/types";
  import Pin from "./Pin.svelte";

  interface Props {
    pins: PinType[];
    loading?: boolean;
    emptyMessage?: string;
    canLoadMore?: boolean;
    loadingMore?: boolean;
    onLoadMore?: () => void;
  }

  let {
    pins,
    loading = false,
    emptyMessage = "No matches yet.",
    canLoadMore = false,
    loadingMore = false,
    onLoadMore,
  }: Props = $props();

  // CSS columns gives us the masonry effect; gap-x-* sets the column gap and
  // break-inside-avoid keeps each pin self-contained.
  const masonryClass = "columns-1 sm:columns-2 lg:columns-3 xl:columns-4 gap-x-4";
</script>

<section class="px-5 sm:px-9 pb-16 pt-2" aria-label="Results">
  {#if loading && pins.length === 0}
    <div class={masonryClass}>
      {#each Array.from({ length: 12 }) as _, index (index)}
        <div
          class="skeleton break-inside-avoid mb-4 border border-line rounded-card shadow-card"
          style:height={`${180 + ((index * 47) % 220)}px`}
        ></div>
      {/each}
    </div>
  {:else if pins.length === 0}
    <p class="my-6 text-muted text-[14.5px]">{emptyMessage}</p>
  {:else}
    <div class={masonryClass}>
      {#each pins as pin (pin.key)}
        <Pin {pin} />
      {/each}
    </div>

    {#if canLoadMore || loadingMore}
      <div class="flex justify-center mt-2">
        <button
          type="button"
          data-load-more
          class="px-[22px] py-[10px] bg-surface border border-line-2 rounded-full text-ink-2 text-[13.5px] font-semibold leading-none cursor-pointer transition-[background-color,border-color,transform] duration-150 ease-soft hover:not-disabled:bg-surface-2 hover:not-disabled:border-ink hover:not-disabled:-translate-y-px disabled:opacity-60 disabled:cursor-default"
          disabled={loadingMore}
          onclick={() => onLoadMore?.()}
        >
          {loadingMore ? "Loading…" : "Load more"}
        </button>
      </div>
    {/if}
  {/if}
</section>
