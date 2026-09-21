<script lang="ts">
  import { lightboxPin, pins, setSimilar, setTagSearch } from "../lib/stores";
  import { formatDuration, formatPercent } from "../lib/utils";
  import { focusTrap } from "../lib/focus";
  import { createMediaActions } from "../lib/mediaActions";
  import ConfirmDialog from "./ConfirmDialog.svelte";
  import Icon from "./Icon.svelte";

  function close() {
    lightboxPin.set(null);
  }

  // Media actions shared with the card overflow menu (meta/issues/084, 096).
  const actions = createMediaActions();
  const actionPending = actions.pending;
  const actionError = actions.error;
  let confirmingDelete = $state(false);

  function nsfwAction() {
    if ($lightboxPin) void actions.flagNSFW($lightboxPin);
  }

  function reannotateAction() {
    if ($lightboxPin) void actions.reannotate($lightboxPin);
  }

  async function confirmDelete() {
    confirmingDelete = false;
    if ($lightboxPin) await actions.remove($lightboxPin);
  }

  // Manual title and tag editing (meta/issues/110).
  let editing = $state(false);
  let draftTitle = $state("");
  let draftTags = $state<string[]>([]);
  let newTag = $state("");
  let tagInputEl: HTMLInputElement | undefined = $state();

  function startEdit() {
    const pin = $lightboxPin;
    if (!pin) return;
    draftTitle = pin.title;
    draftTags = [...pin.tags];
    newTag = "";
    editing = true;
  }

  function cancelEdit() {
    editing = false;
  }

  function addDraftTag() {
    const value = newTag.trim();
    newTag = "";
    if (!value) return;
    if (draftTags.some((t) => t.toLowerCase() === value.toLowerCase())) return;
    draftTags = [...draftTags, value];
    tagInputEl?.focus();
  }

  function removeDraftTag(tag: string) {
    draftTags = draftTags.filter((t) => t !== tag);
  }

  function onTagInputKey(event: KeyboardEvent) {
    if (event.key === "Enter" || event.key === ",") {
      event.preventDefault();
      addDraftTag();
    }
  }

  async function saveEdit() {
    const pin = $lightboxPin;
    if (!pin) return;
    addDraftTag();
    const ok = await actions.edit(pin, { title: draftTitle.trim(), tags: draftTags });
    if (ok) editing = false;
  }

  // Leaving the pin (prev/next/close) drops an unsaved draft.
  $effect(() => {
    void $lightboxPin?.key;
    editing = false;
  });

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
    // A focused <video controls> owns the arrow keys for seeking.
    const target = event.target instanceof HTMLElement ? event.target : null;
    const onVideo = target?.tagName === "VIDEO";
    const typing = target?.tagName === "INPUT" || target?.tagName === "TEXTAREA";
    if (event.key === "Escape") {
      if (editing) {
        cancelEdit();
        return;
      }
      close();
    } else if (onVideo || typing) {
      return;
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
        {#if editing}
          <label class="block pr-12">
            <span class="sr-only">Title</span>
            <input
              data-lightbox-title-input
              type="text"
              bind:value={draftTitle}
              maxlength="300"
              placeholder="Title"
              class="w-full box-border font-display text-[18px] md:text-[22px] font-semibold text-ink leading-tight bg-surface border border-line-2 rounded-[10px] px-3 py-2 outline-none focus:border-accent/60"
            />
          </label>
        {:else}
          <h2
            class="pr-12 font-display text-[18px] md:text-[24px] font-semibold text-ink leading-tight m-0 line-clamp-3 [-webkit-box-orient:vertical] overflow-hidden"
            title={pin.title}
          >
            {pin.title}
          </h2>
        {/if}
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
          {#if editing}
            <div data-lightbox-tag-editor class="flex flex-wrap items-center gap-[5px]">
              {#each draftTags as tag (tag)}
                <span
                  data-lightbox-draft-tag={tag}
                  class="inline-flex items-center gap-1 text-[12px] font-medium leading-none bg-surface-2 text-ink-2 border border-transparent pl-[9px] pr-1 py-[3px] rounded-full"
                >
                  {tag}
                  <button
                    type="button"
                    data-lightbox-tag-remove={tag}
                    aria-label={`Remove tag ${tag}`}
                    onclick={() => removeDraftTag(tag)}
                    class="grid place-items-center w-4 h-4 rounded-full border-0 bg-transparent text-muted-2 cursor-pointer hover:bg-bad/15 hover:text-bad"
                  >
                    ×
                  </button>
                </span>
              {/each}
              <input
                data-lightbox-tag-input
                bind:this={tagInputEl}
                bind:value={newTag}
                type="text"
                maxlength="64"
                placeholder="Add tag"
                aria-label="Add tag"
                onkeydown={onTagInputKey}
                onblur={addDraftTag}
                class="text-[12px] leading-none bg-surface border border-line-2 rounded-full px-[9px] py-[5px] outline-none min-w-[96px] focus:border-accent/60"
              />
            </div>
          {:else if pin.tags?.length}
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
          {#if editing}
            <button
              type="button"
              data-lightbox-edit-save
              disabled={$actionPending}
              onclick={saveEdit}
              class="px-[14px] py-[9px] bg-ink text-[#fffdf8] border border-ink rounded-full text-[13.5px] font-medium leading-none cursor-pointer transition-colors duration-150 ease-soft hover:bg-[#2d2924] disabled:opacity-50 disabled:cursor-default"
            >
              Save
            </button>
            <button
              type="button"
              data-lightbox-edit-cancel
              onclick={cancelEdit}
              class="px-[14px] py-[9px] bg-surface text-ink-2 border border-line-2 rounded-full text-[13.5px] font-medium leading-none cursor-pointer transition-colors duration-150 ease-soft hover:bg-surface-2"
            >
              Cancel
            </button>
          {:else}
            <button
              type="button"
              data-lightbox-action="edit"
              onclick={startEdit}
              class="px-[14px] py-[9px] bg-surface text-ink-2 border border-line-2 rounded-full text-[13.5px] font-medium leading-none cursor-pointer transition-colors duration-150 ease-soft hover:bg-surface-2"
            >
              Edit
            </button>
          {/if}
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
            disabled={$actionPending}
            onclick={nsfwAction}
            class="px-[14px] py-[9px] bg-surface text-ink-2 border border-line-2 rounded-full text-[13.5px] font-medium leading-none cursor-pointer transition-colors duration-150 ease-soft hover:bg-surface-2 disabled:opacity-50 disabled:cursor-default"
          >
            {pin.isNSFW ? "Unflag NSFW" : "Flag NSFW"}
          </button>
          <button
            type="button"
            data-lightbox-action="reannotate"
            disabled={$actionPending}
            onclick={reannotateAction}
            class="px-[14px] py-[9px] bg-surface text-ink-2 border border-line-2 rounded-full text-[13.5px] font-medium leading-none cursor-pointer transition-colors duration-150 ease-soft hover:bg-surface-2 disabled:opacity-50 disabled:cursor-default"
          >
            Re-annotate
          </button>
          <button
            type="button"
            data-lightbox-action="delete"
            disabled={$actionPending}
            onclick={() => (confirmingDelete = true)}
            class="px-[14px] py-[9px] bg-surface text-bad border border-bad/40 rounded-full text-[13.5px] font-medium leading-none cursor-pointer transition-colors duration-150 ease-soft hover:bg-[color-mix(in_oklab,#f4d8d6_60%,white_40%)] disabled:opacity-50 disabled:cursor-default"
          >
            Delete…
          </button>
        </div>
        {#if $actionError}
          <p role="alert" class="mt-3 mb-0 px-3 py-2 rounded-[10px] bg-bad/10 text-bad text-[13px]">
            {$actionError}
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
