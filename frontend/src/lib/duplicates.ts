import type { ImageRecord } from "./types";

const pixels = (item: ImageRecord): number => item.width * item.height;

export function resolutionLabel(item: ImageRecord, items: ImageRecord[]): string {
  if (items.some(other => other.image_id !== item.image_id && other.width === item.width && other.height === item.height)) return "Same resolution";
  if (items.some(other => other.image_id !== item.image_id && pixels(other) === pixels(item))) return "Same pixel count";
  return pixels(item) === Math.max(...items.map(pixels)) ? "Most pixels (not necessarily best quality)" : "";
}

export function duplicateFileInfo(item: ImageRecord): string {
  const format = item.mime_type.replace(/^image\//, "").toUpperCase() || "Unknown format";
  const size = item.file_size_bytes;
  if (size === undefined) return `${format} · Size unavailable`;
  if (size < 1024) return `${format} · ${size} B`;
  const unit = size < 1024 * 1024 ? "KiB" : "MiB";
  const divisor = unit === "KiB" ? 1024 : 1024 * 1024;
  return `${format} · ${Number((size / divisor).toFixed(1))} ${unit}`;
}
