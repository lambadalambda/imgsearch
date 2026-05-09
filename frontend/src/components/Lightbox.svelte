<script lang="ts">
  import { lightboxPin, setSimilar } from "../lib/stores";
  import { formatDuration, formatPercent } from "../lib/utils";
  import Icon from "./Icon.svelte";

  function close() {
    lightboxPin.set(null);
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
    if (event.key === "Escape" && $lightboxPin) {
      close();
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
    class="lightbox"
    role="dialog"
    aria-modal="true"
    aria-label={pin.title}
    tabindex="-1"
    onclick={handleBackdropClick}
  >
    <div class="lightbox-card">
      <button type="button" class="lightbox-close" aria-label="Close" onclick={close}>
        <Icon name="close" class="w-4 h-4" />
      </button>

      <div class="lightbox-media">
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
          ></video>
        {:else}
          <img src={pin.mediaUrl} alt={pin.title} />
        {/if}
      </div>

      <div class="lightbox-meta">
        <h2 class="font-display text-2xl font-semibold text-ink leading-tight">{pin.title}</h2>
        <p class="text-sm text-muted mt-1 break-all">{pin.filename}</p>

        <div class="meta-row">
          {#if pin.matchScore !== undefined}
            <span class="meta-pill">{formatPercent(pin.matchScore)} match{pin.matchTimestampMs ? ` at ${formatDuration(pin.matchTimestampMs)}` : ""}</span>
          {/if}
          {#if pin.mediaType === "video" && pin.durationMs}
            <span class="meta-pill">▶ {formatDuration(pin.durationMs)}</span>
          {/if}
          {#if pin.tags?.length}
            <div class="meta-tags">
              {#each pin.tags as tag (tag)}
                <span class="meta-tag">{tag}</span>
              {/each}
            </div>
          {/if}
        </div>

        <div class="meta-actions">
          <button
            type="button"
            class="action primary"
            onclick={() => {
              setSimilar(pin.imageId);
              close();
            }}
          >
            Find similar
          </button>
          <a class="action" href={pin.mediaUrl} target="_blank" rel="noreferrer noopener">
            Open original
          </a>
        </div>
      </div>
    </div>
  </div>
{/if}

<style>
  .lightbox {
    position: fixed;
    inset: 0;
    z-index: 1200;
    display: grid;
    place-items: center;
    background: rgba(20, 17, 12, 0.78);
    padding: 24px;
    backdrop-filter: blur(6px);
  }
  .lightbox-card {
    position: relative;
    background: var(--color-surface);
    border-radius: 20px;
    box-shadow: 0 30px 80px rgba(0, 0, 0, 0.45);
    width: min(1100px, calc(100vw - 32px));
    max-height: calc(100vh - 32px);
    display: grid;
    grid-template-rows: minmax(0, 1fr) auto;
    overflow: hidden;
  }
  @media (min-width: 900px) {
    .lightbox-card {
      grid-template-columns: minmax(0, 1fr) 320px;
      grid-template-rows: 1fr;
    }
  }
  .lightbox-close {
    position: absolute;
    top: 12px;
    right: 12px;
    z-index: 5;
    display: grid;
    place-items: center;
    width: 36px;
    height: 36px;
    background: rgba(20, 17, 12, 0.74);
    color: #fffdf8;
    border: 0;
    border-radius: 999px;
    cursor: pointer;
    transition: background 120ms var(--ease);
  }
  .lightbox-close:hover {
    background: rgba(20, 17, 12, 0.92);
  }
  .lightbox-media {
    background: #181613;
    display: grid;
    place-items: center;
    min-height: 0;
  }
  .lightbox-media img,
  .lightbox-media video {
    max-width: 100%;
    max-height: calc(100vh - 200px);
    width: auto;
    height: auto;
    display: block;
    object-fit: contain;
  }
  @media (min-width: 900px) {
    .lightbox-media img,
    .lightbox-media video {
      max-height: calc(100vh - 64px);
    }
  }
  .lightbox-meta {
    padding: 18px 22px 22px;
    overflow: auto;
  }
  .meta-row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    margin-top: 14px;
  }
  .meta-pill {
    font: 500 12px/1 var(--font-sans);
    background: var(--color-bg-2);
    color: var(--color-ink);
    padding: 5px 9px;
    border-radius: 999px;
    font-variant-numeric: tabular-nums;
  }
  .meta-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
  }
  .meta-tag {
    font: 500 12px/1 var(--font-sans);
    background: var(--color-surface-2);
    color: var(--color-ink-2);
    padding: 4px 9px;
    border-radius: 999px;
  }
  .meta-actions {
    margin-top: 18px;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }
  .action {
    padding: 9px 14px;
    border: 1px solid var(--color-line-2);
    background: var(--color-surface);
    color: var(--color-ink-2);
    border-radius: 999px;
    font: 500 13.5px/1 var(--font-sans);
    cursor: pointer;
    text-decoration: none;
    transition: background 140ms var(--ease), border-color 140ms var(--ease);
  }
  .action:hover {
    background: var(--color-surface-2);
  }
  .action.primary {
    background: var(--color-ink);
    color: #fffdf8;
    border-color: var(--color-ink);
  }
  .action.primary:hover {
    background: #2d2924;
  }
</style>
