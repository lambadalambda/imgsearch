<script lang="ts">
  type IconName =
    | "library"
    | "search"
    | "tags"
    | "feed"
    | "upload"
    | "settings"
    | "close"
    | "external"
    | "spark"
    | "image"
    | "video"
    | "check";

  interface Props {
    name: IconName;
    class?: string;
    title?: string;
  }

  let { name, class: className = "", title }: Props = $props();
</script>

{#if name === "library"}
  <svg viewBox="0 0 24 24" class={className} aria-hidden={!title} role={title ? "img" : undefined}>
    {#if title}<title>{title}</title>{/if}
    <g fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
      <rect x="3" y="3" width="7" height="7" rx="1.4" />
      <rect x="14" y="3" width="7" height="7" rx="1.4" />
      <rect x="3" y="14" width="7" height="7" rx="1.4" />
      <rect x="14" y="14" width="7" height="7" rx="1.4" />
    </g>
  </svg>
{:else if name === "search"}
  <svg viewBox="0 0 24 24" class={className} aria-hidden={!title} role={title ? "img" : undefined}>
    {#if title}<title>{title}</title>{/if}
    <g fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
      <circle cx="11" cy="11" r="7" />
      <path d="m20 20-3.5-3.5" />
    </g>
  </svg>
{:else if name === "tags"}
  <svg viewBox="0 0 24 24" class={className} aria-hidden={!title} role={title ? "img" : undefined}>
    {#if title}<title>{title}</title>{/if}
    <g fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
      <path d="M20 13.41 11.41 22a2 2 0 0 1-2.83 0L2 15.41V4a2 2 0 0 1 2-2h11.41L22 8.59a2 2 0 0 1 0 2.83Z" />
      <circle cx="7" cy="7" r="1.4" />
    </g>
  </svg>
{:else if name === "feed"}
  <svg viewBox="0 0 24 24" class={className} aria-hidden={!title} role={title ? "img" : undefined}>
    {#if title}<title>{title}</title>{/if}
    <polygon points="5 3 19 12 5 21 5 3" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" />
  </svg>
{:else if name === "upload"}
  <svg viewBox="0 0 24 24" class={className} aria-hidden={!title} role={title ? "img" : undefined}>
    {#if title}<title>{title}</title>{/if}
    <g fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </g>
  </svg>
{:else if name === "settings"}
  <svg viewBox="0 0 24 24" class={className} aria-hidden={!title} role={title ? "img" : undefined}>
    {#if title}<title>{title}</title>{/if}
    <g fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.7 1.7 0 0 0 .3 1.85l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.7 1.7 0 0 0-1.85-.3 1.7 1.7 0 0 0-1 1.55V21a2 2 0 1 1-4 0v-.06a1.7 1.7 0 0 0-1-1.55 1.7 1.7 0 0 0-1.85.3l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.7 1.7 0 0 0 .3-1.85 1.7 1.7 0 0 0-1.55-1H3a2 2 0 1 1 0-4h.06a1.7 1.7 0 0 0 1.55-1 1.7 1.7 0 0 0-.3-1.85L4.25 6.8a2 2 0 1 1 2.83-2.83l.06.06a1.7 1.7 0 0 0 1.85.3H9a1.7 1.7 0 0 0 1-1.55V3a2 2 0 1 1 4 0v.06a1.7 1.7 0 0 0 1 1.55 1.7 1.7 0 0 0 1.85-.3l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.7 1.7 0 0 0-.3 1.85V10a1.7 1.7 0 0 0 1.55 1H21a2 2 0 1 1 0 4h-.06a1.7 1.7 0 0 0-1.55 1Z" />
    </g>
  </svg>
{:else if name === "close"}
  <svg viewBox="0 0 24 24" class={className} aria-hidden={!title} role={title ? "img" : undefined}>
    {#if title}<title>{title}</title>{/if}
    <g fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
      <path d="M18 6 6 18" />
      <path d="m6 6 12 12" />
    </g>
  </svg>
{:else if name === "external"}
  <svg viewBox="0 0 24 24" class={className} aria-hidden={!title} role={title ? "img" : undefined}>
    {#if title}<title>{title}</title>{/if}
    <g fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
      <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
      <polyline points="15 3 21 3 21 9" />
      <line x1="10" y1="14" x2="21" y2="3" />
    </g>
  </svg>
{:else if name === "spark"}
  <svg viewBox="0 0 24 24" class={className} aria-hidden={!title} role={title ? "img" : undefined}>
    {#if title}<title>{title}</title>{/if}
    <path d="M12 2 14.5 9.5 22 12l-7.5 2.5L12 22l-2.5-7.5L2 12l7.5-2.5Z" fill="currentColor" />
  </svg>
{:else if name === "image"}
  <svg viewBox="0 0 24 24" class={className} aria-hidden={!title} role={title ? "img" : undefined}>
    {#if title}<title>{title}</title>{/if}
    <g fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <circle cx="8.5" cy="8.5" r="1.5" />
      <polyline points="21 15 16 10 5 21" />
    </g>
  </svg>
{:else if name === "video"}
  <svg viewBox="0 0 24 24" class={className} aria-hidden={!title} role={title ? "img" : undefined}>
    {#if title}<title>{title}</title>{/if}
    <g fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
      <polygon points="23 7 16 12 23 17 23 7" />
      <rect x="1" y="5" width="15" height="14" rx="2" />
    </g>
  </svg>
{:else if name === "check"}
  <svg viewBox="0 0 24 24" class={className} aria-hidden={!title} role={title ? "img" : undefined}>
    {#if title}<title>{title}</title>{/if}
    <polyline points="20 6 9 17 4 12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
  </svg>
{/if}
