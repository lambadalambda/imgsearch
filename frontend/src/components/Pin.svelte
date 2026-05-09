<script lang="ts">
  import type { Pin } from "../lib/types";
  import { formatDuration, formatPercent, tagTone } from "../lib/utils";
  import { setSimilar, setTagSearch, lightboxPin, pins } from "../lib/stores";
  import { deleteMedia, reannotate, toggleNSFW, ApiError } from "../lib/api";

  interface Props {
    pin: Pin;
  }

  let { pin }: Props = $props();

  // Local state for the overflow menu + per-action feedback. We keep this
  // per-pin so multiple cards can show their state independently.
  let menuOpen = $state(false);
  let actionPending = $state(false);
  let actionError = $state<string | null>(null);
  let isHidden = $state(false);
  let nsfwLocal = $state<boolean | null>(null);

  function open() {
    lightboxPin.set(pin);
  }

  function findSimilar() {
    setSimilar(pin.imageId);
  }

  function closeMenu() {
    menuOpen = false;
  }

  async function runAction(label: string, fn: () => Promise<unknown>): Promise<boolean> {
    if (actionPending) return false;
    actionPending = true;
    actionError = null;
    try {
      await fn();
      return true;
    } catch (err) {
      const msg =
        err instanceof ApiError
          ? err.message
          : err instanceof Error
            ? err.message
            : `${label} failed`;
      actionError = msg;
      return false;
    } finally {
      actionPending = false;
    }
  }

  async function flagNSFW() {
    closeMenu();
    const kind = pin.mediaType;
    const id = kind === "video" && pin.videoId !== undefined ? pin.videoId : pin.imageId;
    const wasFlagged = nsfwLocal ?? pin.isNSFW ?? false;
    nsfwLocal = !wasFlagged; // optimistic
    const ok = await runAction(wasFlagged ? "unflag" : "flag", () => toggleNSFW(kind, id));
    if (!ok) {
      nsfwLocal = wasFlagged; // revert on failure
    }
  }

  async function reannotateAction() {
    closeMenu();
    const kind = pin.mediaType;
    const id = kind === "video" && pin.videoId !== undefined ? pin.videoId : pin.imageId;
    await runAction("re-annotate", () => reannotate(kind, id));
  }

  async function deleteAction() {
    closeMenu();
    const label = pin.mediaType === "video" ? "video" : "image";
    const confirmed = window.confirm(
      `Delete this ${label}? This cannot be undone.\n\n${pin.filename}`,
    );
    if (!confirmed) return;
    const kind = pin.mediaType;
    const id = kind === "video" && pin.videoId !== undefined ? pin.videoId : pin.imageId;
    const ok = await runAction("delete", () => deleteMedia(kind, id));
    if (!ok) return;
    isHidden = true;
    pins.update((existing) => existing.filter((p) => p.key !== pin.key));
  }

  function handleTagClick(tag: string) {
    setTagSearch([tag]);
  }

  function handleMenuKey(event: KeyboardEvent) {
    if (event.key === "Escape" && menuOpen) {
      event.stopPropagation();
      closeMenu();
    }
  }

  // Derived helpers
  const aspectRatio = $derived(pin.width && pin.height ? pin.width / pin.height : 4 / 3);
  const matchLabel = $derived(formatPercent(pin.matchScore));
  const durationLabel = $derived(formatDuration(pin.durationMs));
  const tagsToShow = $derived(pin.tags.slice(0, 5));
  const hiddenTagCount = $derived(Math.max(0, pin.tags.length - tagsToShow.length));
  const nsfwFlagged = $derived(nsfwLocal ?? pin.isNSFW ?? false);

  // Tailwind class fragments to keep template tidy.
  const cornerBtn =
    "[font:500_11.5px/1_var(--font-sans)] bg-[rgba(255,253,249,0.96)] text-ink border border-black/[0.06] backdrop-blur-md px-[9px] py-[5px] rounded-full cursor-pointer transition-transform duration-100 ease-soft hover:-translate-y-px";
  const cornerPrimary = "bg-ink text-[#fffdf8] border-ink";
</script>

{#if !isHidden}
  <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
  <article
    data-pin
    data-pin-anchor={pin.isAnchor ? "true" : undefined}
    data-pin-nsfw={nsfwFlagged ? "true" : undefined}
    class="relative break-inside-avoid mb-4 bg-surface border border-line rounded-card overflow-hidden shadow-card cursor-pointer transition-[transform,box-shadow] duration-200 ease-soft hover:-translate-y-0.5 hover:shadow-elev group {pin.isAnchor
      ? 'border-accent/60 [box-shadow:0_0_0_1px_var(--color-accent-soft),var(--shadow-card)]'
      : ''}"
    onkeydown={handleMenuKey}
  >
    <button
      type="button"
      data-pin-media
      class="block relative w-full bg-bg-2 p-0 border-0 cursor-zoom-in"
      onclick={open}
      aria-label={`Open ${pin.filename}`}
    >
      <img
        src={pin.thumbUrl}
        alt={pin.title}
        loading="lazy"
        decoding="async"
        class="block w-full h-auto"
        style:aspect-ratio={aspectRatio}
      />
      {#if matchLabel}
        <span
          data-pin-match
          class="absolute bottom-2 left-2 text-[11px] font-semibold leading-none text-ink bg-[rgba(255,253,249,0.92)] backdrop-blur-sm px-2 py-1 rounded-full tabular-nums"
          title={pin.matchTimestampMs
            ? `Match at ${formatDuration(pin.matchTimestampMs)}`
            : undefined}
        >
          {matchLabel}
        </span>
      {/if}
      {#if pin.mediaType === "video" && durationLabel}
        <span
          class="absolute bottom-2 right-2 text-[11px] font-semibold leading-none text-[#fffdf9] bg-black/70 backdrop-blur-sm px-2 py-1 rounded-full tabular-nums"
          aria-label={`Duration ${durationLabel}`}
        >
          ▶ {durationLabel}
        </span>
      {/if}
      {#if pin.isAnchor}
        <span
          class="absolute top-2 left-2 text-[10.5px] font-semibold leading-none text-accent-strong bg-accent-soft px-[9px] py-1 rounded-full uppercase tracking-[0.04em]"
        >
          Anchor
        </span>
      {/if}
    </button>

    <div
      class="absolute top-2 right-2 z-[2] inline-flex gap-1 opacity-0 -translate-y-1 transition-[opacity,transform] duration-150 ease-soft group-hover:opacity-100 group-hover:translate-y-0 group-focus-within:opacity-100 group-focus-within:translate-y-0 [@media(hover:none)]:opacity-100 [@media(hover:none)]:translate-y-0"
    >
      <button
        type="button"
        data-pin-action="similar"
        class={cornerBtn}
        onclick={(event) => {
          event.stopPropagation();
          findSimilar();
        }}
        aria-label="Find similar"
      >
        Similar
      </button>
      <button
        type="button"
        class="{cornerBtn} {cornerPrimary}"
        onclick={(event) => {
          event.stopPropagation();
          open();
        }}
        aria-label={pin.mediaType === "video" ? "Play video" : "Open image"}
      >
        {pin.mediaType === "video" ? "Play" : "Open"}
      </button>
      <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
      <!-- svelte-ignore a11y_click_events_have_key_events -->
      <details
        class="relative"
        bind:open={menuOpen}
        onclick={(event) => event.stopPropagation()}
      >
        <summary
          data-pin-action="more"
          class="{cornerBtn} no-marker px-[9px] py-[5px] leading-none select-none {menuOpen
            ? cornerPrimary
            : ''}"
          aria-label="More actions"
          title="More actions"
        >
          <span aria-hidden="true" class="text-[14px] leading-none font-bold">…</span>
        </summary>
        <div
          class="absolute right-0 top-[calc(100%+6px)] min-w-[168px] bg-surface border border-line-2 rounded-[12px] p-1.5 grid gap-px z-[5] shadow-elev"
          role="menu"
        >
          <button
            type="button"
            role="menuitem"
            data-pin-menu="nsfw"
            class="text-left bg-transparent border-0 text-ink-2 px-2.5 py-2 rounded-lg [font:500_13px/1.2_var(--font-sans)] cursor-pointer transition-colors duration-100 ease-soft hover:not-disabled:bg-bg-2 hover:not-disabled:text-ink disabled:opacity-50 disabled:cursor-default"
            onclick={flagNSFW}
            disabled={actionPending}
          >
            {nsfwFlagged ? "Unflag NSFW" : "Flag NSFW"}
          </button>
          <button
            type="button"
            role="menuitem"
            data-pin-menu="reannotate"
            class="text-left bg-transparent border-0 text-ink-2 px-2.5 py-2 rounded-lg [font:500_13px/1.2_var(--font-sans)] cursor-pointer transition-colors duration-100 ease-soft hover:not-disabled:bg-bg-2 hover:not-disabled:text-ink disabled:opacity-50 disabled:cursor-default"
            onclick={reannotateAction}
            disabled={actionPending}
          >
            Re-annotate
          </button>
          <button
            type="button"
            role="menuitem"
            data-pin-menu="delete"
            class="text-left bg-transparent border-0 text-bad px-2.5 py-2 rounded-lg [font:500_13px/1.2_var(--font-sans)] cursor-pointer transition-colors duration-100 ease-soft hover:not-disabled:bg-[color-mix(in_oklab,#f4d8d6_60%,white_40%)] disabled:opacity-50 disabled:cursor-default"
            onclick={deleteAction}
            disabled={actionPending}
          >
            Delete…
          </button>
        </div>
      </details>
    </div>

    {#if actionError}
      <p
        class="absolute inset-x-2 bottom-2 m-0 px-2.5 py-1.5 bg-bad/90 text-[#fffdf8] rounded-lg text-[12px] font-medium leading-snug text-center z-[4] backdrop-blur-sm"
        role="alert"
      >
        {actionError}
      </p>
    {/if}
    {#if nsfwFlagged}
      <span
        class="absolute {pin.isAnchor ? 'top-9' : 'top-2'} left-2 z-[2] text-[10.5px] font-semibold leading-none text-bad bg-[color-mix(in_oklab,#fae5e4_90%,white_10%)] border border-bad/40 px-[9px] py-1 rounded-full uppercase tracking-[0.04em]"
        aria-label="Flagged NSFW"
      >
        NSFW
      </span>
    {/if}

    <div class="px-[13px] pt-3 pb-[13px] grid gap-[7px]">
      <h3
        class="m-0 text-[14.5px] font-semibold leading-snug text-ink tracking-[-0.005em] line-clamp-2 [-webkit-box-orient:vertical] overflow-hidden"
      >
        {pin.title}
      </h3>
      {#if tagsToShow.length}
        <div class="flex gap-[5px] flex-wrap">
          {#each tagsToShow as tag (tag)}
            {@const tone = tagTone(tag)}
            <button
              type="button"
              class="text-[11.5px] font-medium leading-none text-ink-2 bg-bg-2 border border-transparent px-[9px] py-[3px] rounded-full cursor-pointer transition-[background-color,color,border-color] duration-100 ease-soft hover:bg-accent-soft hover:text-accent {tone ===
              'plum'
                ? 'tone-plum'
                : tone === 'moss'
                  ? 'tone-moss'
                  : tone === 'gold'
                    ? 'tone-gold'
                    : ''}"
              onclick={() => handleTagClick(tag)}
              title={`Search "${tag}"`}
            >
              {tag}
            </button>
          {/each}
          {#if hiddenTagCount > 0}
            <span
              class="text-[11.5px] font-medium leading-none text-muted bg-transparent border border-line px-[9px] py-[3px] rounded-full cursor-default"
              aria-label={`${hiddenTagCount} more tags`}
            >
              +{hiddenTagCount}
            </span>
          {/if}
        </div>
      {/if}
    </div>
  </article>
{/if}
