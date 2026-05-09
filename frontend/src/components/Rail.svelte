<script lang="ts">
  import Icon from "./Icon.svelte";
  import { mode, setLibrary } from "../lib/stores";

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
      activeWhen: (s) => s.mode === "library" || s.mode === "similar",
    },
    {
      id: "search",
      label: "Search",
      onClick: () => document.querySelector<HTMLInputElement>("#atelier-search")?.focus(),
      activeWhen: (s) => s.mode === "search",
    },
    { id: "tags", label: "Tags (soon)", disabled: true },
    { id: "feed", label: "Feed (soon)", disabled: true },
    {
      id: "upload",
      label: "Upload (legacy)",
      onClick: () => {
        window.location.href = "/legacy";
      },
    },
  ];
</script>

<aside
  class="row-start-1 col-start-1 sm:row-span-3 sm:col-span-1 sm:h-screen sm:sticky sm:top-0 flex sm:flex-col items-center gap-1 px-3 sm:px-0 py-2 sm:py-4 bg-bg-2 border-b sm:border-b-0 sm:border-r border-line z-30"
  aria-label="Primary navigation"
>
  <a
    href="?"
    class="rail-mark mr-2 sm:mr-0 sm:mb-3"
    aria-label="imgsearch home"
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
      class="rail-btn {isActive ? 'is-active' : ''}"
      class:disabled={item.disabled}
      disabled={item.disabled}
      title={item.label}
      aria-label={item.label}
      onclick={() => item.onClick?.()}
    >
      <Icon name={item.id} class="w-[19px] h-[19px]" />
    </button>
  {/each}

  <span class="hidden sm:block flex-1"></span>

  <a
    href="/legacy"
    class="rail-btn"
    title="Legacy UI"
    aria-label="Open legacy UI"
  >
    <Icon name="external" class="w-[18px] h-[18px]" />
  </a>
  <button type="button" class="rail-btn" disabled aria-label="Settings (soon)" title="Settings (soon)">
    <Icon name="settings" class="w-[19px] h-[19px]" />
  </button>
</aside>

<style>
  .rail-mark {
    display: grid;
    place-items: center;
    width: 32px;
    height: 32px;
    border-radius: 9px;
    background: linear-gradient(135deg, var(--color-accent) 0%, var(--color-accent-strong) 100%);
    color: #fff;
    font-weight: 700;
    font-size: 13px;
    box-shadow: 0 4px 12px rgba(160, 60, 30, 0.28);
    text-decoration: none;
    flex: 0 0 auto;
  }

  .rail-btn {
    display: grid;
    place-items: center;
    width: 40px;
    height: 40px;
    border: 0;
    background: transparent;
    color: var(--color-muted);
    border-radius: 10px;
    cursor: pointer;
    transition: background 140ms var(--ease), color 140ms var(--ease);
  }
  .rail-btn:hover:not(:disabled) {
    background: rgba(0, 0, 0, 0.04);
    color: var(--color-ink);
  }
  .rail-btn.is-active {
    background: var(--color-surface);
    color: var(--color-accent);
    box-shadow: 0 1px 2px rgba(40, 30, 18, 0.06);
  }
  .rail-btn.disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
</style>
