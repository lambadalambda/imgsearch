<script lang="ts">
  import { lightboxPin, pins, setSimilar, setTagSearch } from "../lib/stores";
  import { formatDuration, formatPercent } from "../lib/utils";
  import { focusTrap } from "../lib/focus";
  import { ApiError, deleteMedia, reannotate, toggleNSFW } from "../lib/api";
  import ConfirmDialog from "./ConfirmDialog.svelte";
  import Icon from "./Icon.svelte";

  function close() {
    lightboxPin.set(null);
  }

  // Media actions, mirroring the card overflow menu (meta/issues/084).
  let actionPending = $state(false);
  let actionError = $state<string | null>(null);
  let confirmingDelete = $state(false);

  async function runAction(label: string, fn: () => Promise<unknown>): Promise<boolean> {
    if (actionPending) return false;
    actionPending = true;
    actionError = null;
    try {
      await fn();
      return true;
    } catch (err) {
      actionError =
        err instanceof ApiError ? err.message : err instanceof Error ? err.message : `${label} failed`;
      return false;
    } finally {
      actionPending = false;
    }
  }

  /** Update a pin's NSFW flag everywhere it is rendered: the masonry card
   *  derives its badge from the store entry, so the change reflects there. */
  function setPinNSFW(key: string, value: boolean) {
    pins.update((list) => list.map((p) => (p.key === key ? { ...p, isNSFW: value } : p)));
    lightboxPin.update((p) => (p && p.key === key ? { ...p, isNSFW: value } : p));
  }

  function actionTarget(pin: { mediaType: "image" | "video"; imageId: number; videoId?: number }) {
    const kind = pin.mediaType;
    const id = kind === "video" && pin.videoId !== undefined ? pin.videoId : pin.imageId;
    return { kind, id };
  }

  async function nsfwAction() {
    const pin = $lightboxPin;
    if (!pin) return;
    const { kind, id } = actionTarget(pin);
    const next = !(pin.isNSFW ?? false);
    setPinNSFW(pin.key, next); // optimistic
    const ok = await runAction(next ? "flag" : "unflag", () => toggleNSFW(kind, id));
    if (!ok) setPinNSFW(pin.key, !next);
  }

  async function reannotateAction() {
    const pin = $lightboxPin;
    if (!pin) return;
    const { kind, id } = actionTarget(pin);
    await runAction("re-annotate", () => reannotate(kind, id));
  }

  async function confirmDelete() {
    confirmingDelete = false;
    const pin = $lightboxPin;
    if (!pin) return;
    const { kind, id } = actionTarget(pin);
    const ok = await runAction("delete", () => deleteMedia(kind, id));
    if (!ok) return;
    pins.update((existing) => existing.filter((p) => p.key !== pin.key));
    close();
  }

  // Position of the open pin within the current results, for prev/next
  // navigation. -1 when the pin is no longer in the list (e.g. deleted).
  const pinIndex = $derived(
    $lightboxPin ? $pins.findIndex((p) => p.key === $lightboxPin.key) : -1,
  );
  const canPrev = $derived(pinIndex > 0);
  const canNext = $derived(pinIndex >= 0 && pinIndex < $pins.length - 1);

  function showPrev() {
    if (canPrev) lightboxPin.set($pins[pinIndex - 1]);
  }

  function showNext() {
    if (canNext) lightboxPin.set($pins[pinIndex + 1]);
  }

  function searchTag(tag: string) {
    setTagSearch([tag]);
    close();
  }

  function handleBackdropClick(event: MouseEvent) {
    if (event.target === event.currentTarget) {
      close();
    }
  }

  $effect(() => {
    if ($lightboxPin) {
      document.body.classList.add("modal-open");
    } else {
      document.body.classList.remove("modal-open");
    }
    return () => {
      document.body.classList.remove("modal-open");
    };
  });

  function onKey(event: KeyboardEvent) {
    if (!$lightboxPin) return;
    if (event.key === "Escape") {
      close();
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      showPrev();
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      showNext();
    }
  }
</script>

<svelte:window on:keydown={onKey} />

{#if $lightboxPin}
  {@const pin = $lightboxPin}
  <!-- The backdrop closes on click; the modal contents stop propagation -->
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div
    data-lightbox
    data-lightbox-key={pin.key}
    data-lightbox-index={pinIndex}
    class="fixed inset-0 z-[1200] grid place-items-center bg-black/[0.78] p-6 backdrop-blur-md"
    role="dialog"
    aria-modal="true"
    aria-label={pin.title}
    tabindex="-1"
    onclick={handleBackdropClick}
  >
    <div
      use:focusTrap
      class="relative bg-surface rounded-[20px] shadow-[0_30px_80px_rgba(0,0,0,0.45)] w-[min(1100px,calc(100vw-32px))] max-h-[calc(100vh-32px)] grid grid-rows-[minmax(180px,55vh)_minmax(0,1fr)] overflow-hidden md:grid-rows-1 md:grid-cols-[minmax(0,1fr)_320px]"
    >
      <button
        type="button"
        class="absolute top-3 right-3 z-[5] grid place-items-center w-9 h-9 bg-black/[0.74] text-[#fffdf8] border-0 rounded-full cursor-pointer transition-colors duration-100 ease-soft hover:bg-black/[0.92]"
        aria-label="Close"
        onclick={close}
      >
        <Icon name="close" class="w-4 h-4" />
      </button>

      <div class="relative bg-[#181613] grid place-items-center min-h-0 overflow-hidden">
        {#if canPrev || canNext}
          <button
            type="button"
            data-lightbox-prev
            disabled={!canPrev}
            aria-label="Previous result"
            onclick={showPrev}
            class="absolute left-3 top-1/2 -translate-y-1/2 z-[4] grid place-items-center w-9 h-9 bg-black/[0.55] text-[#fffdf8] border-0 rounded-full cursor-pointer transition-colors duration-100 ease-soft hover:bg-black/[0.8] disabled:opacity-35 disabled:cursor-default"
          >
            <Icon name="chevron-left" class="w-4 h-4" />
          </button>
          <button
            type="button"
            data-lightbox-next
            disabled={!canNext}
            aria-label="Next result"
            onclick={showNext}
            class="absolute right-3 top-1/2 -translate-y-1/2 z-[4] grid place-items-center w-9 h-9 bg-black/[0.55] text-[#fffdf8] border-0 rounded-full cursor-pointer transition-colors duration-100 ease-soft hover:bg-black/[0.8] disabled:opacity-35 disabled:cursor-default"
          >
            <Icon name="chevron-right" class="w-4 h-4" />
          </button>
        {/if}
        {#if pin.mediaType === "video"}
          <!-- User-uploaded media has no captions track; suppress the a11y nag -->
          <!-- svelte-ignore a11y_media_has_caption -->
          <video
            src={pin.mediaUrl}
            controls
            autoplay
            playsinline
            preload="auto"
            poster={pin.thumbUrl}
            class="block max-w-full max-h-full w-auto h-auto object-contain"
          ></video>
        {:else}
          <img
            src={pin.mediaUrl}
            alt={pin.title}
            class="block max-w-full max-h-full w-auto h-auto object-contain"
          />
        {/if}
      </div>

      <div class="p-[18px_22px_22px] overflow-auto overscroll-contain">
        <!-- Derived titles can be a full sentence; clamp so a long one cannot
             balloon into a screen-high headline (full text repeats in the
             description below). -->
        <h2
          class="pr-12 font-display text-[18px] md:text-[24px] font-semibold text-ink leading-tight m-0 line-clamp-3 [-webkit-box-orient:vertical] overflow-hidden"
          title={pin.title}
        >
          {pin.title}
        </h2>
        <p class="text-sm text-muted mt-1 break-all m-0">{pin.filename}</p>

        {#if pin.summary && pin.summary !== pin.title && pin.summary !== pin.fullDescription}
          <p class="mt-4 mb-0 text-[14px] leading-relaxed text-ink-2">
            {pin.summary}
          </p>
        {/if}

        {#if pin.fullDescription}
          <section class="mt-4">
            <h3 class="m-0 text-[12px] uppercase tracking-[0.08em] text-muted-2 font-semibold">
              Description
            </h3>
            <p data-lightbox-description class="mt-1.5 mb-0 text-[13.5px] leading-relaxed text-ink-2 whitespace-pre-wrap">
              {pin.fullDescription}
            </p>
          </section>
        {/if}

        <div class="flex flex-wrap items-center gap-2 mt-[14px]">
          {#if pin.matchScore !== undefined}
            <span
              class="text-[12px] font-medium leading-none bg-bg-2 text-ink px-[9px] py-[5px] rounded-full tabular-nums"
            >
              {formatPercent(pin.matchScore)} match{pin.matchTimestampMs
                ? ` at ${formatDuration(pin.matchTimestampMs)}`
                : ""}
            </span>
          {/if}
          {#if pin.mediaType === "video" && pin.durationMs}
            <span
              class="text-[12px] font-medium leading-none bg-bg-2 text-ink px-[9px] py-[5px] rounded-full tabular-nums"
            >
              ▶ {formatDuration(pin.durationMs)}
            </span>
          {/if}
          {#if pin.tags?.length}
            <div class="flex flex-wrap gap-[5px]">
              {#each pin.tags as tag (tag)}
                <button
                  type="button"
                  data-lightbox-tag
                  class="text-[12px] font-medium leading-none bg-surface-2 text-ink-2 border border-transparent px-[9px] py-1 rounded-full cursor-pointer transition-colors duration-100 ease-soft hover:bg-accent-soft hover:text-accent"
                  onclick={() => searchTag(tag)}
                  title={`Search "${tag}"`}
                >
                  {tag}
                </button>
              {/each}
            </div>
          {/if}
        </div>

        <div class="mt-[18px] flex flex-wrap gap-2">
          <button
            type="button"
            class="px-[14px] py-[9px] bg-ink text-[#fffdf8] border border-ink rounded-full text-[13.5px] font-medium leading-none cursor-pointer transition-colors duration-150 ease-soft hover:bg-[#2d2924]"
            onclick={() => {
              setSimilar(pin.imageId);
              close();
            }}
          >
            Find similar
          </button>
          <a
            href={pin.mediaUrl}
            target="_blank"
            rel="noreferrer noopener"
            class="px-[14px] py-[9px] bg-surface text-ink-2 border border-line-2 rounded-full text-[13.5px] font-medium leading-none cursor-pointer no-underline transition-colors duration-150 ease-soft hover:bg-surface-2"
          >
            Open original
          </a>
          <button
            type="button"
            data-lightbox-action="nsfw"
            disabled={actionPending}
            onclick={nsfwAction}
            class="px-[14px] py-[9px] bg-surface text-ink-2 border border-line-2 rounded-full text-[13.5px] font-medium leading-none cursor-pointer transition-colors duration-150 ease-soft hover:bg-surface-2 disabled:opacity-50 disabled:cursor-default"
          >
            {pin.isNSFW ? "Unflag NSFW" : "Flag NSFW"}
          </button>
          <button
            type="button"
            data-lightbox-action="reannotate"
            disabled={actionPending}
            onclick={reannotateAction}
            class="px-[14px] py-[9px] bg-surface text-ink-2 border border-line-2 rounded-full text-[13.5px] font-medium leading-none cursor-pointer transition-colors duration-150 ease-soft hover:bg-surface-2 disabled:opacity-50 disabled:cursor-default"
          >
            Re-annotate
          </button>
          <button
            type="button"
            data-lightbox-action="delete"
            disabled={actionPending}
            onclick={() => (confirmingDelete = true)}
            class="px-[14px] py-[9px] bg-surface text-bad border border-bad/40 rounded-full text-[13.5px] font-medium leading-none cursor-pointer transition-colors duration-150 ease-soft hover:bg-[color-mix(in_oklab,#f4d8d6_60%,white_40%)] disabled:opacity-50 disabled:cursor-default"
          >
            Delete…
          </button>
        </div>
        {#if actionError}
          <p role="alert" class="mt-3 mb-0 px-3 py-2 rounded-[10px] bg-bad/10 text-bad text-[13px]">
            {actionError}
          </p>
        {/if}
      </div>
    </div>
  </div>

  {#if confirmingDelete}
    <ConfirmDialog
      title={`Delete this ${pin.mediaType === "video" ? "video" : "image"}?`}
      detail={pin.filename}
      onconfirm={confirmDelete}
      oncancel={() => (confirmingDelete = false)}
    />
  {/if}
{/if}
