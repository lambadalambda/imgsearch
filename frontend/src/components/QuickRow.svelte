<script lang="ts">
  import { mode, setLibrary, setQuery, topTags } from "../lib/stores";
</script>

<nav
  class="quickrow scroll-x-soft mx-auto max-w-[920px] mt-3 px-5 sm:px-0 flex flex-wrap sm:flex-nowrap items-center gap-1.5 overflow-x-auto"
  aria-label="Quick collections"
>
  <button
    type="button"
    class="quick {$mode.mode === 'library' ? 'is-active' : ''}"
    onclick={() => setLibrary()}
  >
    All matches
  </button>

  {#each $topTags.slice(0, 8) as entry (entry.tag)}
    <button
      type="button"
      class="quick {$mode.mode === 'search' && $mode.query === entry.tag ? 'is-active' : ''}"
      onclick={() => setQuery(entry.tag)}
      title={`${entry.count.toLocaleString()} matches`}
    >
      Tag · {entry.tag}
    </button>
  {/each}
</nav>

<style>
  .quickrow {
    padding-bottom: 4px;
  }
  .quick {
    display: inline-flex;
    align-items: center;
    padding: 6px 12px;
    background: var(--color-surface);
    border: 1px solid var(--color-line);
    border-radius: 999px;
    color: var(--color-ink-2);
    font: 500 13px/1 var(--font-sans);
    cursor: pointer;
    white-space: nowrap;
    transition: background 120ms var(--ease), border-color 120ms var(--ease), color 120ms var(--ease);
  }
  .quick:hover:not(.is-active) {
    background: var(--color-bg-2);
    border-color: var(--color-line-2);
  }
  .quick.is-active {
    background: var(--color-ink);
    color: #fffdf8;
    border-color: var(--color-ink);
  }
</style>
