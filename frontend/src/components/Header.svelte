<script lang="ts">
  import { mode, headline, includeNSFW, setLibrary } from "../lib/stores";
</script>

<header class="px-5 sm:px-9 pt-5 sm:pt-7 pb-3 sm:pb-4 border-b border-line">
  <div class="flex flex-col sm:flex-row sm:items-end gap-3 sm:gap-4">
    <div class="flex-1 min-w-0 order-1">
      <p class="m-0 text-[13px] text-muted">
        <a href="?" class="hover:text-ink transition-colors" onclick={(event) => { event.preventDefault(); setLibrary(); }}>Library</a>
        <span class="mx-1.5 text-muted-2">/</span>
        {#if $mode.mode === "library"}
          <span>Discover</span>
        {:else if $mode.mode === "search"}
          <a href="?" class="hover:text-ink transition-colors" onclick={(event) => { event.preventDefault(); setLibrary(); }}>Discover</a>
          <span class="mx-1.5 text-muted-2">/</span>
          <span>Search</span>
        {:else}
          <a href="?" class="hover:text-ink transition-colors" onclick={(event) => { event.preventDefault(); setLibrary(); }}>Discover</a>
          <span class="mx-1.5 text-muted-2">/</span>
          <span>Similar</span>
        {/if}
      </p>
      <h1 class="font-display m-0 mt-1.5 text-[22px] sm:text-[30px] font-semibold leading-[1.15] text-ink">
        {$headline}
      </h1>
    </div>

    <div class="flex items-center gap-2 flex-wrap order-2">
      <label class="nsfw-toggle">
        <input type="checkbox" bind:checked={$includeNSFW} />
        <span>{$includeNSFW ? "NSFW: Shown" : "NSFW: Hidden"}</span>
      </label>
      <a class="btn-primary" href="/legacy">Upload (legacy)</a>
    </div>
  </div>
</header>

<style>
  .nsfw-toggle {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 13px;
    border: 1px solid var(--color-line-2);
    border-radius: 999px;
    background: var(--color-surface);
    color: var(--color-ink-2);
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: background 140ms var(--ease), border-color 140ms var(--ease), color 140ms var(--ease);
  }
  .nsfw-toggle:hover {
    background: var(--color-surface-2);
  }
  .nsfw-toggle input {
    accent-color: var(--color-accent);
    inline-size: 14px;
    block-size: 14px;
    margin: 0;
  }
  .btn-primary {
    display: inline-flex;
    align-items: center;
    padding: 8px 14px;
    background: var(--color-ink);
    color: #fffdf8;
    border: 1px solid var(--color-ink);
    border-radius: 999px;
    font-size: 13.5px;
    font-weight: 500;
    text-decoration: none;
    transition: background 140ms var(--ease);
  }
  .btn-primary:hover {
    background: #2d2924;
  }
</style>
