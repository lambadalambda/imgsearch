<script lang="ts">
  import { mode, setLibrary, setTagSearch, topTags } from "../lib/stores";

  function isActiveTag(tag: string): boolean {
    return $mode.mode === "tag" && $mode.tags?.length === 1 && $mode.tags[0] === tag;
  }

  const base =
    "inline-flex items-center px-3 py-1.5 bg-surface border border-line rounded-full text-ink-2 text-[13px] font-medium leading-none whitespace-nowrap cursor-pointer transition-[background-color,border-color,color] duration-100 ease-soft";
  const inactive = "hover:bg-bg-2 hover:border-line-2";
  const active = "bg-ink text-[#fffdf8] border-ink";
</script>

<nav
  class="scroll-x-soft mx-auto max-w-[920px] mt-3 px-5 sm:px-0 flex flex-wrap sm:flex-nowrap items-center gap-1.5 overflow-x-auto pb-1"
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
