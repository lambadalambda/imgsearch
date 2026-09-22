<script lang="ts">
  import type { CollectionCover } from "../lib/types";
  import { mediaUrl } from "../lib/utils";

  let { covers }: { covers: CollectionCover[] } = $props();
  let failed = $state<string[]>([]);
</script>

<span aria-hidden="true" class="grid grid-cols-2 grid-rows-2 gap-0.5 w-20 h-20 sm:w-24 sm:h-24 shrink-0 rounded-lg overflow-hidden bg-bg-2">
  {#each Array.from({ length: 4 }, (_, i) => covers[i]) as cover, i (i)}
    {@const path = cover?.thumbnail_path}
    {#if path && !failed.includes(path)}
      <img src={mediaUrl(path)} alt="" loading="lazy" class="w-full h-full object-cover min-h-0" onerror={() => failed = [...failed, path]} />
    {:else}
      <span data-cover-placeholder class="grid place-items-center bg-surface-2 text-muted-2 text-sm">—</span>
    {/if}
  {/each}
</span>
