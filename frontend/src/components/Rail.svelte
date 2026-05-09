<script lang="ts">
  import Icon from "./Icon.svelte";
  import { listVideos } from "../lib/api";
  import {
    feedSeed,
    includeNSFW,
    mode,
    openFeed,
    openUpload,
    setLibrary,
    uploadOpen,
  } from "../lib/stores";
  import { canPlayMime, pinFromVideo } from "../lib/utils";

  type RailItem = {
    id: "library" | "search" | "tags" | "feed" | "upload";
    label: string;
    onClick?: () => void;
    disabled?: boolean;
    activeWhen?: (state: { mode: string }) => boolean;
  };

  let feedLoading = $state(false);

  async function openRandomFeed() {
    if (feedLoading) return;
    feedLoading = true;
    try {
      const firstPage = await listVideos({ limit: 1, offset: 0, includeNSFW: $includeNSFW });
      const total = firstPage.total ?? firstPage.videos.length;
      if (total <= 0) return;

      const seenOffsets = new Set<number>();
      const maxAttempts = Math.min(total, 6);
      for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
        let offset = Math.floor(Math.random() * total);
        while (seenOffsets.has(offset) && seenOffsets.size < total) {
          offset = Math.floor(Math.random() * total);
        }
        seenOffsets.add(offset);

        const page = offset === 0
          ? firstPage
          : await listVideos({ limit: 1, offset, includeNSFW: $includeNSFW });
        const record = page.videos[0] ?? firstPage.videos[0];
        if (!record) continue;

        const seed = pinFromVideo(record);
        if (!canPlayMime(seed.mimeType)) continue;
        openFeed(seed);
        return;
      }
    } catch (err) {
      console.warn("random Feed launch failed", err);
    } finally {
      feedLoading = false;
    }
  }

  const items: RailItem[] = [
    {
      id: "library",
      label: "Library",
      onClick: () => setLibrary(),
      activeWhen: (s) => s.mode === "library" || s.mode === "similar" || s.mode === "tag",
    },
    {
      id: "search",
      label: "Search",
      onClick: () => document.querySelector<HTMLInputElement>("#atelier-search")?.focus(),
      activeWhen: (s) => s.mode === "search",
    },
    { id: "tags", label: "Tags (soon)", disabled: true },
    {
      id: "feed",
      label: "Feed — random video",
      onClick: openRandomFeed,
      activeWhen: () => $feedSeed !== null,
    },
    {
      id: "upload",
      label: "Upload",
      onClick: () => openUpload(),
      activeWhen: () => $uploadOpen,
    },
  ];

  const railBtnBase =
    "grid place-items-center w-10 h-10 border-0 rounded-[10px] cursor-pointer text-muted bg-transparent transition-[background-color,color] duration-150 ease-soft hover:bg-black/[0.04] hover:text-ink disabled:opacity-45 disabled:cursor-not-allowed disabled:hover:bg-transparent disabled:hover:text-muted";
  const activeBtn =
    "bg-surface text-accent shadow-[0_1px_2px_rgba(40,30,18,0.06)] hover:bg-surface hover:text-accent";
</script>

<aside
  class="row-start-1 col-start-1 sm:row-span-3 sm:col-span-1 sm:h-screen sm:sticky sm:top-0 flex sm:flex-col items-center gap-1 px-3 sm:px-0 py-2 sm:py-4 bg-bg-2 border-b sm:border-b-0 sm:border-r border-line z-30"
  aria-label="Primary navigation"
>
  <a
    href="?"
    aria-label="imgsearch home"
    class="flex-none grid place-items-center w-8 h-8 mr-2 sm:mr-0 sm:mb-3 rounded-[9px] bg-gradient-to-br from-accent to-accent-strong text-white text-[13px] font-bold leading-none shadow-[0_4px_12px_rgba(160,60,30,0.28)] no-underline"
    onclick={(event) => {
      event.preventDefault();
      setLibrary();
    }}
  >
    i
  </a>

  {#each items as item (item.id)}
    {@const isActive = item.activeWhen?.($mode) ?? false}
    <button
      type="button"
      class="{railBtnBase} {isActive ? activeBtn : ''}"
      disabled={item.disabled || (item.id === "feed" && feedLoading)}
      title={item.label}
      aria-label={item.label}
      aria-busy={item.id === "feed" && feedLoading ? "true" : undefined}
      data-rail-item={item.id}
      data-upload-trigger={item.id === "upload" ? "" : undefined}
      onclick={() => item.onClick?.()}
    >
      <Icon name={item.id} class="w-[19px] h-[19px]" />
    </button>
  {/each}

  <span class="hidden sm:block flex-1"></span>

  <a
    href="/legacy"
    class={railBtnBase}
    title="Legacy UI"
    aria-label="Open legacy UI"
  >
    <Icon name="external" class="w-[18px] h-[18px]" />
  </a>
  <button
    type="button"
    class={railBtnBase}
    disabled
    aria-label="Settings (soon)"
    title="Settings (soon)"
  >
    <Icon name="settings" class="w-[19px] h-[19px]" />
  </button>
</aside>
