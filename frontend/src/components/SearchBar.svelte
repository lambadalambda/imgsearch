<script lang="ts">
  import { untrack } from "svelte";
  import { mode, setLibrary, setQuery, setByImage, stats } from "../lib/stores";
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

  let dragOver = $state(false);

  function imageFileFrom(transfer: DataTransfer | null): File | null {
    if (!transfer) return null;
    for (const item of Array.from(transfer.items ?? [])) {
      if (item.kind === "file" && item.type.startsWith("image/")) {
        const file = item.getAsFile();
        if (file) return file;
      }
    }
    for (const file of Array.from(transfer.files ?? [])) {
      if (file.type.startsWith("image/")) return file;
    }
    return null;
  }

  // A pasted picture (clipboard image or copied file) becomes an image query.
  function handlePaste(event: ClipboardEvent) {
    const file = imageFileFrom(event.clipboardData);
    if (!file) return;
    event.preventDefault();
    setByImage(file, file.name || "pasted image");
    inputEl?.blur();
  }

  function handleDragOver(event: DragEvent) {
    if (!event.dataTransfer?.types?.includes("Files")) return;
    event.preventDefault();
    dragOver = true;
  }

  function handleDrop(event: DragEvent) {
    dragOver = false;
    const file = imageFileFrom(event.dataTransfer);
    if (!file) return;
    event.preventDefault();
    setByImage(file, file.name || "dropped image");
  }

  function handleKeydown(event: KeyboardEvent) {
    if (event.key !== "Escape" || event.isComposing || !value) return;
    event.preventDefault();
    event.stopPropagation();
    clear();
  }
</script>

<div
  class="px-5 sm:px-9 pt-3 sm:pt-6 pb-1 sticky top-0 z-20 bg-bg/85 backdrop-blur supports-[backdrop-filter]:bg-bg/75"
>
  <form
    data-search-form
    class="relative mx-auto max-w-[920px] flex items-center gap-2.5 pl-[50px] pr-4 py-2.5 bg-surface border rounded-full shadow-card transition-[border-color,box-shadow] duration-150 ease-soft focus-within:border-accent/40 focus-within:[box-shadow:0_1px_2px_rgba(40,30,18,0.05),0_0_0_3px_color-mix(in_oklab,var(--color-accent)_16%,transparent_84%)] {dragOver
      ? 'border-accent border-dashed'
      : 'border-line-2'}"
    onsubmit={submit}
    ondragover={handleDragOver}
    ondragleave={() => (dragOver = false)}
    ondrop={handleDrop}
  >
    <span
      class="absolute left-[18px] top-1/2 -translate-y-1/2 text-muted-2 pointer-events-none"
      aria-hidden="true"
    >
      <Icon name="search" class="w-[18px] h-[18px]" />
    </span>
    <input
      id="atelier-search"
      bind:this={inputEl}
      type="text"
      enterkeyhint="search"
      role="searchbox"
      aria-label="Search library"
      autocomplete="off"
      placeholder="Search by description, paste or drop an image, or open similar from a pin"
      bind:value
      onkeydown={handleKeydown}
      onpaste={handlePaste}
      class="flex-1 min-w-0 bg-transparent border-0 outline-none text-ink text-base leading-[1.4] py-1 placeholder:text-muted-2"
    />
    {#if value}
      <button
        type="button"
        class="inline-grid place-items-center w-7 h-7 bg-transparent border-0 text-muted rounded-full cursor-pointer transition-colors duration-100 ease-soft hover:bg-bg-2 hover:text-ink"
        aria-label="Clear search"
        onclick={clear}
      >
        <Icon name="close" class="w-[14px] h-[14px]" />
      </button>
    {/if}
    {#if $stats}
      <div class="hidden md:inline-flex items-center gap-2 ml-2 mr-1">
        <span class="text-[12px] font-medium leading-none px-2.5 py-[5px] rounded-full whitespace-nowrap text-muted bg-bg-2">
          {formatCount($stats.images)} images
        </span>
        <span class="text-[12px] font-medium leading-none px-2.5 py-[5px] rounded-full whitespace-nowrap text-accent bg-accent-soft">
          {formatCount($stats.videos)} videos
        </span>
      </div>
    {/if}
  </form>
</div>
