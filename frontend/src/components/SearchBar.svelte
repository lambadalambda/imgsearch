<script lang="ts">
  import { untrack } from "svelte";
  import { mode, setLibrary, setQuery, stats } from "../lib/stores";
  import { formatCount } from "../lib/utils";
  import Icon from "./Icon.svelte";

  let value = $state($mode.mode === "search" ? ($mode.query ?? "") : "");
  let inputEl: HTMLInputElement | undefined = $state();

  // Keep the input mirrored to URL-driven mode changes (back/forward) without
  // creating a cycle: depend on $mode here, but read/write `value` inside
  // untrack so the effect only re-runs when mode changes.
  $effect(() => {
    const current = $mode;
    untrack(() => {
      if (current.mode === "search" && current.query !== undefined && current.query !== value) {
        value = current.query;
      } else if (current.mode !== "search" && value !== "") {
        value = "";
      }
    });
  });

  function submit(event: SubmitEvent) {
    event.preventDefault();
    const next = value.trim();
    if (next) {
      setQuery(next);
    } else {
      setLibrary();
    }
    inputEl?.blur();
  }

  function clear() {
    value = "";
    setLibrary();
    inputEl?.focus();
  }
</script>

<div class="px-5 sm:px-9 pt-5 sm:pt-6 pb-1 sticky top-0 z-20 bg-bg/85 backdrop-blur supports-[backdrop-filter]:bg-bg/75">
  <form class="searchbar mx-auto max-w-[920px]" onsubmit={submit}>
    <span class="searchbar-icon" aria-hidden="true">
      <Icon name="search" class="w-[18px] h-[18px]" />
    </span>
    <input
      id="atelier-search"
      bind:this={inputEl}
      type="search"
      autocomplete="off"
      placeholder="Search by description, paste a tag, or open similar from a pin"
      bind:value
    />
    {#if value}
      <button type="button" class="clear-btn" aria-label="Clear search" onclick={clear}>
        <Icon name="close" class="w-[14px] h-[14px]" />
      </button>
    {/if}
    {#if $stats}
      <div class="hidden md:inline-flex items-center gap-2 ml-2 mr-1">
        <span class="pill" data-tone="muted">{formatCount($stats.images)} images</span>
        <span class="pill" data-tone="accent">{formatCount($stats.videos)} videos</span>
      </div>
    {/if}
  </form>
</div>

<style>
  .searchbar {
    position: relative;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 16px 10px 50px;
    background: var(--color-surface);
    border: 1px solid var(--color-line-2);
    border-radius: 999px;
    box-shadow: var(--shadow-card);
    transition: border-color 140ms var(--ease), box-shadow 140ms var(--ease);
  }
  .searchbar:focus-within {
    border-color: color-mix(in oklab, var(--color-accent) 40%, var(--color-line-2) 60%);
    box-shadow:
      0 1px 2px rgba(40, 30, 18, 0.05),
      0 0 0 3px color-mix(in oklab, var(--color-accent) 16%, transparent 84%);
  }
  .searchbar-icon {
    position: absolute;
    left: 18px;
    top: 50%;
    transform: translateY(-50%);
    color: var(--color-muted-2);
    pointer-events: none;
  }
  .searchbar input {
    flex: 1 1 auto;
    background: transparent;
    border: 0;
    outline: 0;
    color: var(--color-ink);
    font: 16px/1.4 inherit;
    min-width: 0;
    padding: 4px 0;
  }
  .searchbar input::placeholder {
    color: var(--color-muted-2);
  }
  .clear-btn {
    display: inline-grid;
    place-items: center;
    width: 28px;
    height: 28px;
    background: transparent;
    border: 0;
    color: var(--color-muted);
    border-radius: 999px;
    cursor: pointer;
    transition: background 120ms var(--ease), color 120ms var(--ease);
  }
  .clear-btn:hover {
    background: var(--color-bg-2);
    color: var(--color-ink);
  }
  .pill {
    font: 500 12px/1 var(--font-sans);
    padding: 5px 10px;
    border-radius: 999px;
    white-space: nowrap;
  }
  .pill[data-tone="muted"] {
    color: var(--color-muted);
    background: var(--color-bg-2);
  }
  .pill[data-tone="accent"] {
    color: var(--color-accent);
    background: var(--color-accent-soft);
  }
</style>
