import type { CapabilityState, JobKindStats, StatsResponse } from "./types";

const emptyJob: JobKindStats = { tracked: 0, runnable: 0, pending: 0, leased: 0, done: 0, failed: 0, oldest_runnable_age_seconds: 0 };
export const capabilityCopy = (state: CapabilityState): string => ({ configured: "Configured", disabled: "Disabled", unconfigured: "Not configured", unknown: "Worker configuration unknown" })[state];

export function healthCards(stats: StatsResponse) {
  const definitions = [
    { kind: "embed_image", title: "Image embedding", action: "embedding", total: stats.queue?.total ?? stats.images_total, scope: "Standalone images + sampled video frames. Videos are represented by their frames.", capability: (stats.capabilities?.worker === "local" ? "configured" : "unknown") as CapabilityState },
    { kind: "annotate_image", title: "Image annotation", action: "image annotation", total: stats.image_annotation_expected ?? stats.standalone_images_total, scope: "Standalone images only; sampled video frames are excluded.", capability: stats.capabilities?.annotations ?? "unknown" },
    { kind: "annotate_video", title: "Video annotation", action: "video annotation", total: stats.video_annotation_expected ?? stats.videos_total, scope: "Videos, not individual sampled frames.", capability: stats.capabilities?.annotations ?? "unknown" },
    { kind: "transcribe_video", title: "Video transcription", action: "transcription", total: stats.video_transcription_expected ?? stats.videos_total, scope: "Videos, including those without recognized speech.", capability: stats.capabilities?.transcription ?? "unknown" },
  ] as const;
  return definitions.map((definition) => {
    const job = stats.job_kinds?.[definition.kind] ?? (stats.capabilities ? { ...emptyJob, waiting: 0, processing: 0 } : emptyJob);
    return {
      ...definition, job,
      // Older servers do not expose lease-aware counts. Don't pretend leases
      // prove active processing; keep those raw counts under Advanced details.
      waiting: job.waiting ?? job.pending,
      processing: job.processing,
      complete: job.done,
      failed: job.failed,
      missing: Math.max(0, definition.total - job.tracked),
      percent: definition.total > 0 ? Math.min(100, Math.round(100 * job.done / definition.total)) : 0,
      // Capability metadata and bounded retry ship together. Older APIs may
      // silently ignore the kind query, so never offer recovery without it.
      canRetry: !!stats.capabilities && (definition.capability === "configured" || definition.capability === "unknown"),
      note: definition.kind === "transcribe_video"
        ? `${stats.video_transcription_empty ?? 0} completed without recognized speech. Stored empty transcripts cannot distinguish no audio from no recognized speech.`
        : "Complete means this processing step finished, not that every library step is complete.",
    };
  });
}
