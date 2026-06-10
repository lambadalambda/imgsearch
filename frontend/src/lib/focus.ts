/** Svelte action for modal dialogs: moves focus into the node on mount,
 *  keeps Tab/Shift+Tab cycling inside it, and restores focus to the
 *  previously focused element when the node is destroyed. */
export function focusTrap(node: HTMLElement) {
  const previous =
    document.activeElement instanceof HTMLElement ? document.activeElement : null;

  const selector = [
    "a[href]",
    "button:not([disabled])",
    "input:not([disabled])",
    "select:not([disabled])",
    "textarea:not([disabled])",
    "video[controls]",
    '[tabindex]:not([tabindex="-1"])',
  ].join(",");

  function focusables(): HTMLElement[] {
    return Array.from(node.querySelectorAll<HTMLElement>(selector)).filter(
      (el) => el.getClientRects().length > 0,
    );
  }

  (focusables()[0] ?? node).focus();

  function onKeydown(event: KeyboardEvent) {
    if (event.key !== "Tab") return;
    const els = focusables();
    if (els.length === 0) {
      event.preventDefault();
      return;
    }
    const current = document.activeElement;
    const index = current instanceof HTMLElement ? els.indexOf(current) : -1;
    const next = event.shiftKey
      ? index <= 0
        ? els.length - 1
        : index - 1
      : index === -1 || index === els.length - 1
        ? 0
        : index + 1;
    event.preventDefault();
    els[next].focus();
  }

  node.addEventListener("keydown", onKeydown);
  return {
    destroy() {
      node.removeEventListener("keydown", onKeydown);
      previous?.focus();
    },
  };
}
