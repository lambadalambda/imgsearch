import { getStats } from "./api";
import { stats } from "./stores";

/** Fetch /api/stats and publish the snapshot to the shared store. */
export async function refreshStats(signal?: AbortSignal): Promise<void> {
  const s = await getStats(signal);
  stats.set({
    ...s,
    images: s.standalone_images_total ?? s.images_total,
    videos: s.videos_total,
  });
}
