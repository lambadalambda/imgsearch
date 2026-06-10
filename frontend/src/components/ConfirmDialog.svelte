<script lang="ts">
  import { focusTrap } from "../lib/focus";

  interface Props {
    title: string;
    /** Secondary line, e.g. the filename being deleted. */
    detail?: string;
    confirmLabel?: string;
    onconfirm: () => void;
    oncancel: () => void;
  }

  let { title, detail, confirmLabel = "Delete", onconfirm, oncancel }: Props = $props();

  function onKey(event: KeyboardEvent) {
    if (event.key === "Escape") {
      // Keep the Escape from also closing whatever opened this dialog.
      event.stopPropagation();
      oncancel();
    }
  }
</script>

<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
  data-confirm-dialog
  role="alertdialog"
  aria-modal="true"
  aria-label={title}
  tabindex="-1"
  class="fixed inset-0 z-[1500] grid place-items-center bg-black/[0.55] p-6 backdrop-blur-sm"
  onclick={(event) => {
    event.stopPropagation();
    if (event.target === event.currentTarget) oncancel();
  }}
  onkeydown={onKey}
>
  <div
    use:focusTrap
    class="w-[min(420px,calc(100vw-32px))] bg-surface rounded-[16px] shadow-elev p-5 flex flex-col gap-3"
  >
    <h2 class="m-0 font-display text-[19px] font-semibold text-ink leading-snug">{title}</h2>
    {#if detail}
      <p class="m-0 text-[13px] text-muted break-all">{detail}</p>
    {/if}
    <p class="m-0 text-[13px] text-muted-2">This cannot be undone.</p>
    <div class="flex justify-end gap-2 mt-1">
      <button
        type="button"
        data-confirm-cancel
        onclick={oncancel}
        class="px-[14px] py-[9px] bg-surface text-ink-2 border border-line-2 rounded-full text-[13.5px] font-medium leading-none cursor-pointer transition-colors duration-150 ease-soft hover:bg-surface-2"
      >
        Cancel
      </button>
      <button
        type="button"
        data-confirm-accept
        onclick={onconfirm}
        class="px-[14px] py-[9px] bg-bad text-[#fffdf8] border border-bad rounded-full text-[13.5px] font-medium leading-none cursor-pointer transition-colors duration-150 ease-soft hover:bg-[#8c2d2d]"
      >
        {confirmLabel}
      </button>
    </div>
  </div>
</div>
