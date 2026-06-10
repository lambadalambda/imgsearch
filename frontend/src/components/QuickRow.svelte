<script lang="ts">
  import { mode, setLibrary, setTagSearch, topTags } from "../lib/stores";

  function isActiveTag(tag: string): boolean {
    return $mode.mode === "tag" && $mode.tags?.length === 1 && $mode.tags[0] === tag;
  }

  let navEl: HTMLElement | undefined = $state();
  let canScrollRight = $state(false);

  function updateOverflow() {
    if (!navEl) return;
    canScrollRight = navEl.scrollLeft + navEl.clientWidth < navEl.scrollWidth - 1;
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
    "inline-flex items-center px-3 py-1.5 bg-surface border border-line rounded-full text-ink-2 text-[13px] font-medium leading-none whitespace-nowrap cursor-pointer transition-[background-color,border-color,color] duration-100 ease-soft";
  const inactive = "hover:bg-bg-2 hover:border-line-2";
  const active = "bg-ink text-[#fffdf8] border-ink";
</script>

<div class="relative mx-auto max-w-[920px] w-full mt-3">
  <nav
    bind:this={navEl}
    onscroll={updateOverflow}
    class="scroll-x-soft px-5 sm:px-0 flex flex-nowrap items-center gap-1.5 overflow-x-auto pb-1"
    aria-label="Quick collections"
  >
    <button
      type="button"
      class="{base} {$mode.mode === 'library' ? active : inactive}"
      onclick={() => setLibrary()}
    >
      All matches
    </button>

    {#each $topTags.slice(0, 8) as entry (entry.tag)}
      <button
        type="button"
        class="{base} {isActiveTag(entry.tag) ? active : inactive}"
        onclick={() => setTagSearch([entry.tag])}
        title={`${entry.count.toLocaleString()} matches`}
      >
        Tag · {entry.tag}
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
