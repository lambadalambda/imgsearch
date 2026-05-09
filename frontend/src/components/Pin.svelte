<script lang="ts">
  import type { Pin } from "../lib/types";
  import { formatDuration, formatPercent, tagTone } from "../lib/utils";
  import { setSimilar, lightboxPin } from "../lib/stores";

  interface Props {
    pin: Pin;
  }

  let { pin }: Props = $props();

  function open() {
    lightboxPin.set(pin);
  }

  function findSimilar() {
    setSimilar(pin.imageId);
  }

  // Derived helpers
  const aspectRatio = $derived(pin.width && pin.height ? pin.width / pin.height : 4 / 3);
  const matchLabel = $derived(formatPercent(pin.matchScore));
  const durationLabel = $derived(formatDuration(pin.durationMs));
  const tagsToShow = $derived(pin.tags.slice(0, 5));
  const hiddenTagCount = $derived(Math.max(0, pin.tags.length - tagsToShow.length));
</script>

<article class="pin group" class:is-anchor={pin.isAnchor}>
  <button type="button" class="pin-media" onclick={open} aria-label={`Open ${pin.filename}`}>
    <img
      src={pin.thumbUrl}
      alt={pin.title}
      loading="lazy"
      decoding="async"
      style:aspect-ratio={aspectRatio}
    />
    {#if matchLabel}
      <span class="pin-match" title={pin.matchTimestampMs ? `Match at ${formatDuration(pin.matchTimestampMs)}` : undefined}>
        {matchLabel}
      </span>
    {/if}
    {#if pin.mediaType === "video" && durationLabel}
      <span class="pin-duration" aria-label={`Duration ${durationLabel}`}>
        ▶ {durationLabel}
      </span>
    {/if}
    {#if pin.isAnchor}
      <span class="pin-anchor">Anchor</span>
    {/if}
  </button>

  <div class="pin-corner">
    <button type="button" class="corner-btn" onclick={findSimilar} aria-label="Find similar">
      Similar
    </button>
    <button type="button" class="corner-btn primary" onclick={open} aria-label={pin.mediaType === "video" ? "Play video" : "Open image"}>
      {pin.mediaType === "video" ? "Play" : "Open"}
    </button>
  </div>

  <div class="pin-body">
    <h3 class="pin-title">{pin.title}</h3>
    {#if tagsToShow.length}
      <div class="pin-tags">
        {#each tagsToShow as tag (tag)}
          {@const tone = tagTone(tag)}
          <button
            type="button"
            class="tagchip"
            class:tone-plum={tone === "plum"}
            class:tone-moss={tone === "moss"}
            class:tone-gold={tone === "gold"}
            onclick={() => setSimilar(pin.imageId) /* tag-search not yet implemented */}
            title={`Find similar to "${tag}"`}
          >
            {tag}
          </button>
        {/each}
        {#if hiddenTagCount > 0}
          <span class="tagchip muted" aria-label={`${hiddenTagCount} more tags`}>+{hiddenTagCount}</span>
        {/if}
      </div>
    {/if}
  </div>
</article>

<style>
  .pin {
    break-inside: avoid;
    margin: 0 0 16px;
    background: var(--color-surface);
    border: 1px solid var(--color-line);
    border-radius: var(--radius-card);
    overflow: hidden;
    box-shadow: var(--shadow-card);
    transition: transform 200ms var(--ease), box-shadow 200ms var(--ease);
    position: relative;
  }
  .pin:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-elev);
  }
  .pin.is-anchor {
    border-color: color-mix(in oklab, var(--color-accent) 60%, var(--color-line) 40%);
    box-shadow:
      0 0 0 1px var(--color-accent-soft),
      var(--shadow-card);
  }
  .pin-media {
    display: block;
    position: relative;
    width: 100%;
    border: 0;
    background: var(--color-bg-2);
    padding: 0;
    cursor: zoom-in;
  }
  .pin-media img {
    display: block;
    width: 100%;
    height: auto;
  }
  .pin-corner {
    position: absolute;
    top: 8px;
    right: 8px;
    display: inline-flex;
    gap: 4px;
    opacity: 0;
    transform: translateY(-4px);
    transition: opacity 160ms var(--ease), transform 160ms var(--ease);
    z-index: 2;
  }
  .pin:hover .pin-corner,
  .pin:focus-within .pin-corner {
    opacity: 1;
    transform: none;
  }
  @media (hover: none) and (pointer: coarse) {
    .pin-corner {
      opacity: 1;
      transform: none;
    }
  }
  .corner-btn {
    font: 500 11.5px/1 var(--font-sans);
    background: rgba(255, 253, 249, 0.96);
    color: var(--color-ink);
    border: 1px solid rgba(0, 0, 0, 0.06);
    backdrop-filter: blur(8px);
    padding: 5px 9px;
    border-radius: 999px;
    cursor: pointer;
    transition: transform 120ms var(--ease), background 120ms var(--ease), color 120ms var(--ease);
  }
  .corner-btn:hover {
    transform: translateY(-1px);
  }
  .corner-btn.primary {
    background: var(--color-ink);
    color: #fffdf8;
    border-color: var(--color-ink);
  }
  .pin-match {
    position: absolute;
    bottom: 8px;
    left: 8px;
    font: 600 11px/1 var(--font-sans);
    color: var(--color-ink);
    background: rgba(255, 253, 249, 0.92);
    backdrop-filter: blur(6px);
    padding: 4px 8px;
    border-radius: 999px;
    font-variant-numeric: tabular-nums;
  }
  .pin-duration {
    position: absolute;
    bottom: 8px;
    right: 8px;
    font: 600 11px/1 var(--font-sans);
    color: #fffdf9;
    background: rgba(20, 17, 12, 0.7);
    backdrop-filter: blur(6px);
    padding: 4px 8px;
    border-radius: 999px;
    font-variant-numeric: tabular-nums;
  }
  .pin-anchor {
    position: absolute;
    top: 8px;
    left: 8px;
    font: 600 10.5px/1 var(--font-sans);
    color: var(--color-accent-strong);
    background: var(--color-accent-soft);
    padding: 4px 9px;
    border-radius: 999px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }
  .pin-body {
    padding: 12px 13px 13px;
    display: grid;
    gap: 7px;
  }
  .pin-title {
    margin: 0;
    font: 600 14.5px/1.35 var(--font-sans);
    color: var(--color-ink);
    letter-spacing: -0.005em;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
  .pin-tags {
    display: flex;
    gap: 5px;
    flex-wrap: wrap;
  }
  .tagchip {
    font: 500 11.5px/1 var(--font-sans);
    color: var(--color-ink-2);
    padding: 3px 9px;
    border-radius: 999px;
    background: var(--color-bg-2);
    border: 1px solid transparent;
    cursor: pointer;
    transition: background 120ms var(--ease), color 120ms var(--ease), border-color 120ms var(--ease);
  }
  .tagchip:hover:not(.muted) {
    background: var(--color-accent-soft);
    color: var(--color-accent);
  }
  .tagchip.muted {
    background: transparent;
    border-color: var(--color-line);
    color: var(--color-muted);
    cursor: default;
  }
</style>
