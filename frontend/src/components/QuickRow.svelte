<script lang="ts">
  import { mode, setLibrary, setTagSearch, topTags } from "../lib/stores";

  function isActiveTag(tag: string): boolean {
    return (
      $mode.mode === "tag" && $mode.tags?.length === 1 && $mode.tags[0] === tag
    );
  }

  let expanded = $state(false);
  let disclosure: HTMLButtonElement;

  function collapse() {
    // The disclosure is hidden on desktop; keep focus on the still-visible tag.
    if (disclosure?.getClientRects().length) disclosure.focus();
    expanded = false;
  }

  let navEl: HTMLElement | undefined = $state();
  let canScrollRight = $state(false);

  function updateOverflow() {
    if (!navEl) return;
    canScrollRight =
      navEl.scrollLeft + navEl.clientWidth < navEl.scrollWidth - 1;
  }

  function onWheel(event: WheelEvent) {
    if (!navEl || navEl.scrollWidth <= navEl.clientWidth) return;
    if (Math.abs(event.deltaY) <= Math.abs(event.deltaX)) return;
    event.preventDefault();
    navEl.scrollLeft += event.deltaY;
  }

  // Track overflow as chips load or the viewport resizes. The wheel listener
  // is attached manually because Svelte registers `onwheel` passively, which
  // would forbid the preventDefault needed to stop the page scrolling.
  $effect(() => {
    void $topTags;
    void expanded;
    const el = navEl;
    if (!el) return;
    updateOverflow();
    const observer = new ResizeObserver(updateOverflow);
    observer.observe(el);
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => {
      observer.disconnect();
      el.removeEventListener("wheel", onWheel);
    };
  });

  const base =
    "inline-flex items-center min-h-11 px-3 py-1.5 bg-surface border border-line rounded-full text-ink-2 text-[13px] font-medium leading-none whitespace-nowrap cursor-pointer transition-[background-color,border-color,color] duration-100 ease-soft";
  const inactive = "hover:bg-bg-2 hover:border-line-2";
  const active = "bg-ink text-[#fffdf8] border-ink";
</script>

<div data-browse-tags class="px-5 sm:px-9 mt-1 sm:mt-3">
  <div
    class="flex items-center gap-2 min-h-11 sm:min-h-0 sm:mb-1 text-xs text-muted"
  >
    <button
      bind:this={disclosure}
      type="button"
      class="sm:hidden min-h-11 text-sm font-semibold text-accent cursor-pointer"
      aria-expanded={expanded}
      aria-controls="browse-tags"
      onclick={() => (expanded = !expanded)}>Browse tags</button
    >
    <span class="hidden sm:inline font-semibold">Browse tags</span>
    <span id="tag-switch-hint">Replaces current query</span>
  </div>
  <div class="tag-options relative min-w-0" class:expanded>
    <nav
      id="browse-tags"
      bind:this={navEl}
      onscroll={updateOverflow}
      class="scroll-x-soft flex flex-nowrap items-center gap-1.5 overflow-x-auto pb-1"
      aria-label="Browse tags"
      aria-describedby="tag-switch-hint"
    >
      <button
        type="button"
        class="{base} {$mode.mode === 'library' ? active : inactive}"
        onclick={() => {
          setLibrary();
          collapse();
        }}
      >
        Browse library
      </button>

      {#each $topTags.slice(0, 8) as entry (entry.tag)}
        <button
          type="button"
          class="{base} {isActiveTag(entry.tag) ? active : inactive}"
          aria-label={`Browse tag ${entry.tag}`}
          aria-describedby="tag-switch-hint"
          aria-pressed={isActiveTag(entry.tag)}
          onclick={() => {
            setTagSearch([entry.tag]);
            collapse();
          }}
          title="Browse this tag — replaces current query"
        >
          {entry.tag}
        </button>
      {/each}
    </nav>
    {#if canScrollRight}
      <div
        data-quick-overflow
        aria-hidden="true"
        class="pointer-events-none absolute inset-y-0 right-0 w-12 bg-gradient-to-l from-bg to-transparent"
      ></div>
    {/if}
  </div>
</div>

<style>
  .tag-options {
    display: none;
  }
  .tag-options.expanded {
    display: block;
  }
  @media (min-width: 640px) {
    .tag-options {
      display: block;
    }
  }
</style>
