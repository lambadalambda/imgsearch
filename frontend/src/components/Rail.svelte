<script lang="ts">
  import Icon from "./Icon.svelte";
  import {
    feedSeed,
    mode,
    openUpload,
    setLibrary,
    uploadOpen,
  } from "../lib/stores";

  type RailItem = {
    id: "library" | "search" | "tags" | "feed" | "upload";
    label: string;
    onClick?: () => void;
    disabled?: boolean;
    activeWhen?: (state: { mode: string }) => boolean;
  };

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
      label: "Feed — open from a video card",
      // No global launcher — Feed is always seeded by a specific video pin.
      // The button stays clickable so it can serve as a hint; we focus the
      // first video Feed button on the page if one exists.
      onClick: () => {
        const seedBtn = document.querySelector<HTMLElement>(
          '[data-pin-action="feed"]',
        );
        if (seedBtn) {
          seedBtn.focus();
          seedBtn.scrollIntoView({ block: "center", behavior: "smooth" });
        }
      },
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
      disabled={item.disabled}
      title={item.label}
      aria-label={item.label}
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
