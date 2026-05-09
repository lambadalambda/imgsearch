<script lang="ts">
  import type { Pin as PinType } from "../lib/types";
  import Pin from "./Pin.svelte";

  interface Props {
    pins: PinType[];
    loading?: boolean;
    emptyMessage?: string;
  }

  let { pins, loading = false, emptyMessage = "No matches yet." }: Props = $props();
</script>

<section class="px-5 sm:px-9 pb-16 pt-2" aria-label="Results">
  {#if loading && pins.length === 0}
    <div class="masonry">
      {#each Array.from({ length: 12 }) as _, index (index)}
        <div class="skeleton-card" style:height={`${180 + ((index * 47) % 220)}px`}></div>
      {/each}
    </div>
  {:else if pins.length === 0}
    <p class="empty">{emptyMessage}</p>
  {:else}
    <div class="masonry">
      {#each pins as pin (pin.key)}
        <Pin {pin} />
      {/each}
    </div>
  {/if}
</section>

<style>
  .masonry {
    column-count: 4;
    column-gap: 16px;
  }
  @media (max-width: 1280px) {
    .masonry {
      column-count: 3;
    }
  }
  @media (max-width: 900px) {
    .masonry {
      column-count: 2;
    }
  }
  @media (max-width: 600px) {
    .masonry {
      column-count: 1;
    }
  }
  .skeleton-card {
    break-inside: avoid;
    margin: 0 0 16px;
    background: linear-gradient(
      90deg,
      var(--color-surface-2) 25%,
      var(--color-bg-2) 50%,
      var(--color-surface-2) 75%
    );
    background-size: 200% 100%;
    border: 1px solid var(--color-line);
    border-radius: var(--radius-card);
    box-shadow: var(--shadow-card);
    animation: shimmer 1.4s ease-in-out infinite;
  }
  .empty {
    margin: 24px 0;
    color: var(--color-muted);
    font-size: 14.5px;
  }
</style>
