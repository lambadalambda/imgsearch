<script lang="ts">
  import { tick } from "svelte";
  import { hasSearchChrome } from "../lib/searchPresentation";
  import Icon from "./Icon.svelte";
  import { listVideos } from "../lib/api";
  import {
    feedSeed,
    includeNSFW,
    mode,
    openFeed,
    setLibrary,
    setCollections,
    setSettings,
    setStats,
    setDuplicates,
  } from "../lib/stores";
  import { pinFromVideo } from "../lib/utils";

  type RailItem = {
    id:
      | "collections"
      | "library"
      | "search"
      | "stats"
      | "duplicates"
      | "feed"
      | "settings";
    label: string;
    onClick?: () => void;
    activeWhen?: (state: { mode: string }) => boolean;
  };

  function activate(item: RailItem) {
    if (moreOpen) moreButton?.focus();
    moreOpen = false;
    item.onClick?.();
  }

  let moreOpen = $state(false);
  let moreButton: HTMLButtonElement;

  async function focusSearch() {
    if (!hasSearchChrome($mode.mode)) setLibrary();
    await tick();
    document.querySelector<HTMLInputElement>("#atelier-search")?.focus();
  }

  function closeMore(event: KeyboardEvent) {
    if (event.key !== "Escape" || !moreOpen) return;
    moreOpen = false;
    moreButton?.focus();
  }

  let feedLoading = $state(false);
  let feedNotice = $state("");

  async function openRandomFeed() {
    if (feedLoading) return;
    feedLoading = true;
    feedNotice = "";
    try {
      const firstPage = await listVideos({
        limit: 1,
        offset: 0,
        includeNSFW: $includeNSFW,
      });
      const total = firstPage.total ?? firstPage.videos.length;
      if (total <= 0) {
        feedNotice = "No videos available for this feed. Try again after adding videos.";
        return;
      }

      const seenOffsets = new Set<number>();
      const maxAttempts = Math.min(total, 6);
      for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
        let offset = Math.floor(Math.random() * total);
        while (seenOffsets.has(offset) && seenOffsets.size < total) {
          offset = Math.floor(Math.random() * total);
        }
        seenOffsets.add(offset);

        const page =
          offset === 0
            ? firstPage
            : await listVideos({ limit: 1, offset, includeNSFW: $includeNSFW });
        const record = page.videos[0] ?? firstPage.videos[0];
        if (!record) continue;

        const seed = pinFromVideo(record);
        openFeed(seed);
        return;
      }
      feedNotice = "No videos available for this feed. Try again after adding videos.";
    } catch (err) {
      feedNotice = "Could not open feed. Please try again.";
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
      activeWhen: (s) => s.mode === "library" || s.mode === "tag",
    },
    {
      id: "search",
      label: "Search",
      onClick: focusSearch,
      activeWhen: (s) => ["search", "byimage", "similar"].includes(s.mode),
    },
    {
      id: "collections",
      label: "Collections",
      onClick: setCollections,
      activeWhen: (s) => s.mode === "collections" || s.mode === "collection",
    },
    {
      id: "settings",
      label: "Settings",
      onClick: setSettings,
      activeWhen: (s) => s.mode === "settings",
    },
    {
      id: "stats",
      label: "Statistics",
      onClick: () => setStats(),
      activeWhen: (s) => s.mode === "stats",
    },
    {
      id: "duplicates",
      label: "Duplicates",
      onClick: () => setDuplicates(),
      activeWhen: (s) => s.mode === "duplicates",
    },
    {
      id: "feed",
      label: "Feed",
      onClick: openRandomFeed,
      activeWhen: () => $feedSeed !== null,
    },
  ];

  const railBtnBase =
    "flex items-center gap-3 min-h-11 min-w-11 px-3 rounded-xl cursor-pointer text-muted hover:bg-surface hover:text-ink text-sm";
  const activeBtn = "bg-surface text-accent shadow-sm";
</script>

<svelte:window
  onkeydown={closeMore}
  onclick={(event) => {
    if (
      moreOpen &&
      event.target instanceof Element &&
      !event.target.closest("[data-navigation]")
    )
      moreOpen = false;
  }}
/>

<nav data-navigation aria-label="Primary navigation" class="rail">
  <a
    href="?"
    aria-label="imgsearch home"
    class="hidden sm:flex items-center min-h-11 px-3 mb-4 text-accent font-bold"
    onclick={(event) => {
      event.preventDefault();
      setLibrary();
    }}>imgsearch</a
  >
  <div class="grid grid-cols-4 sm:flex sm:flex-col gap-1">
    {#each items.slice(0, 3) as item (item.id)}
      {@const isActive = item.activeWhen?.($mode) ?? false}
      <button
        type="button"
        class="{railBtnBase} flex-col sm:flex-row justify-center sm:justify-start gap-1 sm:gap-3 py-1 {isActive
          ? activeBtn
          : ''}"
        aria-label={item.label}
        aria-current={isActive ? "page" : undefined}
        data-rail-item={item.id}
        onclick={() => activate(item)}
      >
        <Icon name={item.id} class="w-5 h-5 shrink-0" />
        <span class="text-xs sm:text-sm">{item.label}</span>
      </button>
    {/each}
    <button
      bind:this={moreButton}
      type="button"
      class="{railBtnBase} sm:hidden flex-col justify-center gap-1 py-1 {moreOpen ||
      ['settings', 'stats', 'duplicates'].includes($mode.mode)
        ? activeBtn
        : ''}"
      aria-expanded={moreOpen}
      aria-controls="rail-secondary"
      onclick={() => (moreOpen = !moreOpen)}
    >
      <span aria-hidden="true" class="h-5 leading-4 text-xl">···</span>
      <span class="text-xs">More</span>
    </button>
  </div>
  <div id="rail-secondary" class="secondary" class:expanded={moreOpen}>
    {#each items.slice(3) as item (item.id)}
      {@const isActive = item.activeWhen?.($mode) ?? false}
      <button
        type="button"
        class="{railBtnBase} {isActive ? activeBtn : ''}"
        aria-disabled={item.id === "feed" && feedLoading}
        aria-busy={item.id === "feed" && feedLoading ? "true" : undefined}
        aria-current={isActive ? "page" : undefined}
        aria-label={item.label}
        title={item.id === "feed" ? "Feed — random video" : item.label}
        data-rail-item={item.id}
        onclick={() => activate(item)}
      >
        <Icon name={item.id} class="w-5 h-5 shrink-0" />
        <span>{item.label}</span>
      </button>
    {/each}
  </div>
</nav>

{#if feedNotice}
  <div data-feed-launch-notice class="fixed bottom-20 sm:bottom-4 right-4 left-4 sm:left-auto sm:max-w-sm z-[1500] p-4 bg-surface text-ink border border-line rounded-xl shadow-lg">
    <p role="status" class="m-0 text-sm">{feedNotice}</p>
    <button type="button" class="mt-2 min-h-11 px-3 border border-line rounded-full cursor-pointer" onclick={() => (feedNotice = "")}>Dismiss</button>
  </div>
{/if}

<style>
  .rail {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 6px 8px max(6px, env(safe-area-inset-bottom));
    background: var(--color-bg-2);
    border-top: 1px solid var(--color-line);
    z-index: 30;
  }
  .secondary {
    display: none;
  }
  .secondary.expanded {
    display: flex;
    flex-direction: column;
    position: absolute;
    bottom: 100%;
    right: 8px;
    min-width: 180px;
    padding: 8px;
    border: 1px solid var(--color-line);
    border-radius: 12px;
    background: var(--color-bg-2);
    box-shadow: var(--shadow-elev);
  }
  @media (min-width: 640px) {
    .rail {
      position: sticky;
      top: 0;
      height: 100vh;
      padding: 16px 8px;
      border-top: 0;
      border-right: 1px solid var(--color-line);
    }
    .secondary,
    .secondary.expanded {
      display: flex;
      flex-direction: column;
      gap: 4px;
      position: static;
      min-width: 0;
      padding: 0;
      margin-top: 12px;
      border: 0;
      box-shadow: none;
    }
  }
</style>
