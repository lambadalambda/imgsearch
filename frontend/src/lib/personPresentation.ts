export function personFacesLabel(faceCount: number): string {
  return `${faceCount.toLocaleString()} ${faceCount === 1 ? "face" : "faces"}`;
}

/** Drag-and-drop merge is only valid between two different people. */
export function canMergePeople(sourceId: number, targetId: number): boolean {
  return (
    Number.isSafeInteger(sourceId) && sourceId > 0 &&
    Number.isSafeInteger(targetId) && targetId > 0 &&
    sourceId !== targetId
  );
}

export function personItemsPresentation(itemCount: number, visibleCount: number): {
  count: string;
  message: string;
  action: "browse" | "show" | null;
} {
  const hidden = Math.max(0, itemCount - visibleCount);
  const count = `${itemCount.toLocaleString()} grouped ${itemCount === 1 ? "item" : "items"}`;
  if (itemCount === 0) return { count, message: "No grouped photos or videos yet.", action: "browse" };
  if (hidden > 0) return {
    count: `${count} · ${hidden.toLocaleString()} hidden`,
    message: visibleCount === 0
      ? "All grouped items are hidden by the NSFW filter."
      : `${hidden.toLocaleString()} ${hidden === 1 ? "item" : "items"} hidden by the NSFW filter.`,
    action: "show",
  };
  return { count, message: "", action: null };
}
