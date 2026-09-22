export function collectionPresentation(itemCount: number, visibleCount: number): {
  count: string;
  message: string;
  action: "browse" | "show" | null;
} {
  const hidden = Math.max(0, itemCount - visibleCount);
  const count = `${itemCount.toLocaleString()} saved ${itemCount === 1 ? "item" : "items"}`;
  if (itemCount === 0) return { count, message: "No saved items yet.", action: "browse" };
  if (hidden > 0) return {
    count: `${count} · ${hidden.toLocaleString()} hidden`,
    message: visibleCount === 0
      ? "All saved items are hidden by the NSFW filter."
      : `${hidden.toLocaleString()} ${hidden === 1 ? "item" : "items"} hidden by the NSFW filter.`,
    action: "show",
  };
  return { count, message: "", action: null };
}
