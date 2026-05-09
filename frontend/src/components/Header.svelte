<script lang="ts">
  import { mode, headline, includeNSFW, openUpload, setLibrary } from "../lib/stores";

  function goHome(event: Event) {
    event.preventDefault();
    setLibrary();
  }
</script>

<header class="px-5 sm:px-9 pt-5 sm:pt-7 pb-3 sm:pb-4 border-b border-line">
  <div class="flex flex-col sm:flex-row sm:items-end gap-3 sm:gap-4">
    <div class="flex-1 min-w-0 order-1">
      <p class="m-0 text-[13px] text-muted">
        <a href="?" class="text-muted hover:text-ink transition-colors no-underline" onclick={goHome}>Library</a>
        <span class="mx-1.5 text-muted-2">/</span>
        {#if $mode.mode === "library"}
          <span>Discover</span>
        {:else if $mode.mode === "search"}
          <a href="?" class="text-muted hover:text-ink transition-colors no-underline" onclick={goHome}>Discover</a>
          <span class="mx-1.5 text-muted-2">/</span>
          <span>Search</span>
        {:else if $mode.mode === "tag"}
          <a href="?" class="text-muted hover:text-ink transition-colors no-underline" onclick={goHome}>Discover</a>
          <span class="mx-1.5 text-muted-2">/</span>
          <span>Tag</span>
        {:else}
          <a href="?" class="text-muted hover:text-ink transition-colors no-underline" onclick={goHome}>Discover</a>
          <span class="mx-1.5 text-muted-2">/</span>
          <span>Similar</span>
        {/if}
      </p>
      <h1 class="font-display m-0 mt-1.5 text-[22px] sm:text-[30px] font-semibold leading-[1.15] text-ink">
        {$headline}
      </h1>
    </div>

    <div class="flex items-center gap-2 flex-wrap order-2">
      <label
        class="inline-flex items-center gap-2 px-[13px] py-[7px] border border-line-2 rounded-full bg-surface text-ink-2 text-[13px] font-medium leading-none cursor-pointer transition-[background-color,border-color,color] duration-150 ease-soft hover:bg-surface-2"
      >
        <input
          type="checkbox"
          bind:checked={$includeNSFW}
          class="m-0 w-[14px] h-[14px] accent-accent"
        />
        <span>{$includeNSFW ? "NSFW: Shown" : "NSFW: Hidden"}</span>
      </label>
      <button
        type="button"
        data-upload-trigger
        onclick={() => openUpload()}
        class="inline-flex items-center gap-1.5 px-[14px] py-2 bg-ink text-[#fffdf8] border border-ink rounded-full text-[13.5px] font-medium leading-none cursor-pointer transition-colors duration-150 ease-soft hover:bg-[#2d2924]"
      >
        <span aria-hidden="true" class="text-[15px] leading-none">+</span>
        <span>Upload</span>
      </button>
    </div>
  </div>
</header>
