# Video summaries and sampling

Settings → **Video summaries and sampling** controls two independent choices. Saving either does not enqueue work or reprocess existing media.

## Frame count

**Frames per new video** accepts 1–32, default **5**. Each new video upload snapshots the saved count and extracts evenly spaced frames with ffmpeg. A count of five samples approximately 10%,30%,50%,70%,90% of the video's duration.

- Applies to both native and external annotation, and to the frame embeddings used for timestamped search.
- Changes apply to subsequent new uploads without restarting. An upload already sampling keeps its captured count.
- Existing videos—including duplicate uploads—keep their stored frames. Re-annotation does not resample. Older videos may have ten or another number of stored samples.
- `-video-frame-count` supplies the initial value (also 1–32). Legacy saved settings lacking the field still inherit that flag until a settings save persists the effective value. Thereafter the saved value takes precedence.

API field: `annotation.video_frame_count`. Omitted fields retain the current effective value. Invalid numeric values are rejected before saving.

## External API video input

API field: `annotation.video_mode`:

| Value | Behavior |
| --- | --- |
| `frame_descriptions` | Default. Generate/reuse compact annotations for sampled frames, then synthesize them with one representative image. |
| `direct_frames` | Send all stored sampled frame images directly in one multimodal request, skipping per-frame description generation. |

The direct option applies only to the **external OpenAI-compatible backend**. Native annotation always uses frame descriptions, preserving its existing lower-peak-visual-context path. Switching to native retains the saved external preference for a later switch back.

Direct input contains every stored sample in chronological order, including repeated images at different timeline positions. Each image is preceded by its frame index and timestamp. The prompt also receives duration, meaningful filename context, available transcript context and the library tag vocabulary. It asks for a title, summary, full description, tags and an NSFW classification across **any** supplied frame—not just recurring content.

All images use the existing image preparation/resolution policy. The complete image/timestamp sequence and context survive compact-output, transport and response-format retries. Missing images or provider errors fail visibly; the implementation never silently drops samples or switches to a different input mode. Backend and mode are pinned for the video job so a concurrent settings change cannot switch pipelines halfway through preparation.

### Limits and trade-offs

- The external model **and server** must support multiple images. OpenAI-compatible transport alone does not establish this; Settings' **Test connection** does not certify multi-image support.
- Direct mode may reduce generation calls but increases per-request visual context, payload and memory requirements. No universal speed/quality improvement is assumed. Large stored sample sets can exceed provider limits.
- Direct mode uses the **Video summaries** output-token limits. **Video frames** output limits apply only to the description pipeline. Output budgets do not limit visual input size.
- Sampling and frame embeddings remain intact. Direct mode creates video-level text/tags rather than separate frame descriptions; frame-specific textual tag coverage can differ even though visual similarity vectors do not change.
- Existing cached frame annotations are neither cleared nor overwritten. Existing NSFW frame tags continue to hide videos conservatively, alongside the new video-level classification. Single-video reannotation retains previous metadata while pending or failed, avoiding loss of its only NSFW signal before a replacement result arrives. User tag edits retain their existing merge semantics.
- Only transcript text available when the job starts is supplied. This feature neither enables transcription nor introduces an audio dependency or automatic re-summarization after transcription.

Mode changes apply to subsequent annotation jobs after the worker refreshes settings. To update an existing video's summary, explicitly re-annotate it; it will use that video's existing stored samples. Bulk re-annotation is never triggered by saving these settings.
