<script lang="ts">
  import { onMount } from "svelte";
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

  const ROW_HEIGHT_PX = 8;
  const ROW_GAP_PX = 16;

  let spans = $state<Record<string, number>>({});
  let resizeObserver: ResizeObserver | undefined;
  const observedTargets = new Map<HTMLElement, string>();

  function spanForHeight(height: number): number {
    return Math.max(1, Math.ceil((height + ROW_GAP_PX) / (ROW_HEIGHT_PX + ROW_GAP_PX)));
  }

  function updateSpan(key: string, target: HTMLElement): void {
    const next = spanForHeight(target.getBoundingClientRect().height);
    if (spans[key] === next) return;
    spans = { ...spans, [key]: next };
  }

  function measureCell(node: HTMLElement, key: string) {
    let currentKey = key;
    let target: HTMLElement | undefined;
    let frame = 0;
    let destroyed = false;

    function attach() {
      frame = 0;
      if (destroyed) return;
      const nextTarget = (node.querySelector("[data-pin]") as HTMLElement | null) ?? node;
      if (target && target !== nextTarget) {
        resizeObserver?.unobserve(target);
        observedTargets.delete(target);
      }
      target = nextTarget;
      observedTargets.set(target, currentKey);
      resizeObserver?.observe(target);
      updateSpan(currentKey, target);
    }

    function scheduleAttach() {
      if (frame) cancelAnimationFrame(frame);
      frame = requestAnimationFrame(attach);
    }

    scheduleAttach();

    return {
      update(nextKey: string) {
        currentKey = nextKey;
        if (target) observedTargets.set(target, currentKey);
        scheduleAttach();
      },
      destroy() {
        destroyed = true;
        if (frame) cancelAnimationFrame(frame);
        if (target) {
          resizeObserver?.unobserve(target);
          observedTargets.delete(target);
        }
      },
    };
  }

  onMount(() => {
    resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const target = entry.target as HTMLElement;
        const key = observedTargets.get(target);
        if (key) updateSpan(key, target);
      }
    });
    for (const [target, key] of observedTargets) {
      resizeObserver.observe(target);
      updateSpan(key, target);
    }
    return () => resizeObserver?.disconnect();
  });

  const gridClass = "grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-x-4 gap-y-4 items-start";
  const masonryClass = `${gridClass} [grid-auto-rows:8px]`;
</script>

<section class="px-5 sm:px-9 pb-16 pt-2" aria-label="Results">
  {#if loading && pins.length === 0}
    <div class={gridClass}>
      {#each Array.from({ length: 12 }) as _, index (index)}
        <div
          class="skeleton border border-line rounded-card shadow-card"
          style:height={`${180 + ((index * 47) % 220)}px`}
        ></div>
      {/each}
    </div>
  {:else if pins.length === 0}
    <p class="my-6 text-muted text-[14.5px]">{emptyMessage}</p>
  {:else}
    <div class={masonryClass}>
      {#each pins as pin (pin.key)}
        <div
          class="min-w-0"
          use:measureCell={pin.key}
          style:grid-row-end={`span ${spans[pin.key] ?? 1}`}
        >
          <Pin {pin} />
        </div>
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
