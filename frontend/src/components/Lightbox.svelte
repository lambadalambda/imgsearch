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
    data-lightbox
    class="fixed inset-0 z-[1200] grid place-items-center bg-black/[0.78] p-6 backdrop-blur-md"
    role="dialog"
    aria-modal="true"
    aria-label={pin.title}
    tabindex="-1"
    onclick={handleBackdropClick}
  >
    <div
      class="relative bg-surface rounded-[20px] shadow-[0_30px_80px_rgba(0,0,0,0.45)] w-[min(1100px,calc(100vw-32px))] max-h-[calc(100vh-32px)] grid grid-rows-[minmax(0,1fr)_auto] overflow-hidden md:grid-rows-1 md:grid-cols-[minmax(0,1fr)_320px]"
    >
      <button
        type="button"
        class="absolute top-3 right-3 z-[5] grid place-items-center w-9 h-9 bg-black/[0.74] text-[#fffdf8] border-0 rounded-full cursor-pointer transition-colors duration-100 ease-soft hover:bg-black/[0.92]"
        aria-label="Close"
        onclick={close}
      >
        <Icon name="close" class="w-4 h-4" />
      </button>

      <div class="bg-[#181613] grid place-items-center min-h-0">
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
            class="block max-w-full w-auto h-auto max-h-[calc(100vh-200px)] md:max-h-[calc(100vh-64px)] object-contain"
          ></video>
        {:else}
          <img
            src={pin.mediaUrl}
            alt={pin.title}
            class="block max-w-full w-auto h-auto max-h-[calc(100vh-200px)] md:max-h-[calc(100vh-64px)] object-contain"
          />
        {/if}
      </div>

      <div class="p-[18px_22px_22px] overflow-auto">
        <h2 class="font-display text-[24px] font-semibold text-ink leading-tight m-0">
          {pin.title}
        </h2>
        <p class="text-sm text-muted mt-1 break-all m-0">{pin.filename}</p>

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
                <span
                  class="text-[12px] font-medium leading-none bg-surface-2 text-ink-2 px-[9px] py-1 rounded-full"
                >
                  {tag}
                </span>
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
        </div>
      </div>
    </div>
  </div>
{/if}
